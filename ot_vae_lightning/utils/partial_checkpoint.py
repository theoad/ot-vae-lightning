import os
from collections import OrderedDict

import torch
from ot_vae_lightning.utils import human_format
from pytorch_lightning.utilities.memory import get_model_size_mb


class PartialCheckpoint:
    """
    Why add yet another checkpoint loading method on top of lightning's `ckpt_path` and torch.load() ?
    lightning's `ckpt_path` loads weights for all attributes in a strict manner as well as
    hparams, optimizer states, metric states and so on.
    This custom checkpoint loading method loads only weights for specific attributes.
    It does not interfere with lightning's `ckpt_path` (`resume_from_checkpoint` prior to 1.7.0).
    """

    def __init__(
            self,
            checkpoint_path: str,
            attr_name: str = None,
            replace_str: str = "",
            strict: bool = True,
            freeze: bool = False
    ):
        """
        :param checkpoint_path: absolute/relative path to the checkpoint file.
        :param attr_name: the class attribute name which is loaded.
        :param replace_str: the string that replaces the target attribute (ex: encoder.conv1.weight -> conv1.weight).
        :param strict: if ``True`` the weights will be loaded in a strict manner.
        :param freeze: if ``True`` the weights will be freezed - requires_grad(False) -.
        """
        self.attr_name = attr_name
        self.checkpoint_path = checkpoint_path
        self.replace_str = replace_str
        self.strict = strict
        self.freeze = freeze
        assert os.path.exists(checkpoint_path), f'Error: Path {checkpoint_path} not found.'

    @property
    def state_dict(self):
        checkpoint = torch.load(self.checkpoint_path)
        if self.attr_name is None or all([self.attr_name not in k for k in checkpoint['state_dict'].keys()]):
            return checkpoint['state_dict']

        state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if self.attr_name == '.'.join(key.split('.')[:self.attr_name.count('.') + 1]):
                state_dict[key.replace(f"{self.attr_name}.", self.replace_str, 1)] = checkpoint['state_dict'][key]

        return state_dict

    def load_attribute(self, module: torch.nn.Module, attr: str):
        for sub_attr in attr.rsplit('.'):
            module = getattr(module, sub_attr)
        module.load_state_dict(self.state_dict, strict=self.strict)
        if self.freeze:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        print(f'[info]: self.{attr} [{human_format(sum(p.numel() for p in module.parameters()))} '
              f'parameters - {int(get_model_size_mb(module))}Mb] loaded successfully '
              f'{"and freezed" if self.freeze else ""}')
