import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import lovely_tensors as lt
from torchmetrics import PeakSignalNoiseRatio
from accelerate import Accelerator
import torch.backends.cudnn as cudnn
from ot_vae_lightning.networks.cnn import ConvBlock
from ot_vae_lightning.data import dataset_split
from decimal import Decimal
from ot_vae_lightning.ot.gaussian_transport import GaussianTransport


lt.monkey_patch()  # debug sugar
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

MEAN, STD = torch.tensor((0.485, 0.456, 0.406)), torch.tensor((0.229, 0.224, 0.225))


class MultiLevelEncDec(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        h = x
        loss = 0
        for i, block in enumerate(self.encoder):
            h = block(h)
            hat = self.decoder[-(i+1):](h)
            loss += F.mse_loss(hat, x)
        return loss


@torch.no_grad()
def test_transport(encoder, decoder, dl, dl_degraded, test_x, test_y, denorm, device, save_path):
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    # keep track of 1st and 2nd order statistics of every layer of the model
    gaussian_transport = nn.ModuleList([
        GaussianTransport(block.out_features, diag=False, pg_star=0., stochastic=False, make_pd=True, verbose=True)
        for block in encoder
    ]).to(device)
    encoder.eval(); decoder.eval(); gaussian_transport.eval()
    assert len(gaussian_transport) == len(encoder) and len(encoder) == len(decoder)

    def update_transport(samples, dist='source'):
        h = samples.to(device)
        for i in range(len(encoder)):
            h = encoder[i](h)
            update = {f'{dist}_samples': h.permute(0, 2, 3, 1).flatten(0, 2)}
            gaussian_transport[i].update(**update)

    for y, _ in tqdm(dl_degraded, desc=f"source", leave=True): update_transport(y, 'source')
    for x, _ in tqdm(dl, desc=f"target", leave=True): update_transport(x, 'target')

    for tr in gaussian_transport:
        print(f'W2 transport cost: {Decimal(tr._compute_helper().item()):.2E}')
        print(f'num obs {Decimal(tr.n_obs_source.item()):.2E}')
        print(f'transport operator', tr.transport_operator)

    y = x_transported = test_y
    bs = y.size(0)
    for i in range(len(encoder)):
        enc = encoder[:-i] if i > 0 else encoder
        h_transported = enc(x_transported)
        h, w = h_transported.shape[-2:]

        # gaussian transport
        h_transported = gaussian_transport[-(i+1)].transport(
            h_transported.permute(0, 2, 3, 1).flatten(0, 2)
        ).unflatten(0, (bs, h, w)).permute(0, 3, 1, 2)
        x_transported = decoder[i:](h_transported)

    collage = torch.cat([y, decoder(encoder(y)), x_transported, decoder(encoder(test_x)), test_x], dim=2)[:4]
    torchvision.utils.save_image(denorm(collage), save_path, nrow=4)
    print("saved", save_path)


if __name__ == "__main__":
    # create output dir
    ckpt_path = None
    now = datetime.datetime.now()
    out_dir = f'logs/{now.strftime("%d-%m-%Y-%H:%M")}'
    os.makedirs(out_dir, exist_ok=True)

    # We use hugging face's accelerator library to deal with ddp
    ddp = Accelerator()

    # We prepare a denormalization transform to plot images
    to_tensor = T.ToTensor()
    norm = T.Normalize(MEAN, STD)
    denorm = T.Compose([
        T.Normalize(torch.zeros_like(MEAN), 1./STD),
        T.Normalize(-MEAN, torch.ones_like(STD))
    ])
    random_resize, resize = T.RandomResizedCrop(224), T.Compose([T.CenterCrop(224), T.Resize(224)])
    blur = T.GaussianBlur((9, 9), sigma=(4, 4))  # degradation operator

    # Load the ImageNet dataset
    imagenet_train = torchvision.datasets.ImageNet("~/data/ImageNet", transform=T.Compose([to_tensor, norm, random_resize]))
    imagenet_val = torchvision.datasets.ImageNet("~/data/ImageNet", split="val", transform=T.Compose([to_tensor, norm, resize]))
    imagenet_val_degraded = torchvision.datasets.ImageNet("~/data/ImageNet", split="val", transform=T.Compose([to_tensor, norm, resize, blur]))

    # Split the samples into non-overlapping splits unseen during the auto-encoder training
    imagenet_val, imagenet_val_degraded = dataset_split([imagenet_val, imagenet_val_degraded], split=0.5, seed=42)

    # Create a data loader for the dataset
    train_dl = DataLoader(imagenet_train, batch_size=32, shuffle=True,  num_workers=10, pin_memory=True)
    val_dl = DataLoader(imagenet_val, batch_size=100, shuffle=False, num_workers=10)
    val_dl_degraded = DataLoader(imagenet_val_degraded, batch_size=100, shuffle=False, num_workers=10)
    test_x = next(iter(val_dl))[0]
    test_y = blur(test_x)

    cfg = dict(normalization="batch", activation="leaky", residual="add", equalized_lr=1.)
    encoder = nn.Sequential(                        # 3,  224, 224
        ConvBlock(3,   64,  down_sample=2, **cfg),  # 64, 112, 112
        ConvBlock(64,  128, down_sample=2, **cfg),  # 128, 56, 56
        ConvBlock(128, 256, down_sample=2, **cfg),  # 256, 28, 28
        ConvBlock(256, 512, down_sample=2, **cfg),  # 512, 14, 14
        ConvBlock(512, 512, down_sample=2, **cfg),  # 512, 7, 7
    )

    decoder = nn.Sequential(                        # 512, 7, 7
        ConvBlock(512, 512, up_sample=2, **cfg),    # 512, 14, 14
        ConvBlock(512, 256, up_sample=2, **cfg),    # 256, 28, 28
        ConvBlock(256, 128, up_sample=2, **cfg),    # 128, 56, 56
        ConvBlock(128, 64,  up_sample=2, **cfg),    # 64, 112, 112
        ConvBlock(64,  3,   up_sample=2, **cfg),    # 3,  224, 224
    )

    if ckpt_path is not None:
        encoder.load_state_dict(torch.load(f'{ckpt_path}/encoder_weights.ckpt'))
        decoder.load_state_dict(torch.load(f'{ckpt_path}/decoder_weights.ckpt'))

    model = MultiLevelEncDec(encoder, decoder)

    # Define the loss function and optimizer
    metric = PeakSignalNoiseRatio()
    train_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(train_params, betas=(0, 0.99))

    model, optimizer, train_dl, metric = ddp.prepare(model, optimizer, train_dl, metric)

    for epoch in range(100):
        # training
        if epoch > 0 or ckpt_path is None:
            encoder.train(); decoder.train()
            for x, _ in (pbar := tqdm(train_dl, desc=f"epoch {epoch}", leave=True, disable=not ddp.is_local_main_process)):
                optimizer.zero_grad()
                loss = model(x)
                ddp.backward(loss)
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})

        # checkpoint saving
        ddp.wait_for_everyone()
        if ddp.is_local_main_process:
            save_dir = f'{out_dir}/checkpoints/epoch_{str(epoch).zfill(3)}'
            os.makedirs(save_dir, exist_ok=True)
            ddp.save(ddp.unwrap_model(model).encoder.state_dict(), f'{save_dir}/encoder_weights.ckpt')
            ddp.save(ddp.unwrap_model(model).decoder.state_dict(), f'{save_dir}/decoder_weights.ckpt')
            ddp.save_state(f'{save_dir}/full_state.ckpt')

            # Using validation data to test transport
            os.makedirs(f'{out_dir}/images', exist_ok=True)
            test_transport(
                ddp.unwrap_model(model).encoder, ddp.unwrap_model(model).decoder, val_dl, val_dl_degraded, test_x, test_y,
                denorm, device=ddp.device, save_path=f"{out_dir}/images/transport_epoch_{epoch}.png"
            )
