from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from src.model.HiFiGAN.generator import Generator
from src.model.HiFiGAN.mpd import MPD
from src.model.HiFiGAN.msd import MSD


class HiFiGan(nn.Module):
    def __init__(
        self, generator_config: Dict = {}, mpd_config: Dict = {}, msd_config: Dict = {}
    ):
        super().__init__()

        self.generator = Generator(**generator_config)
        self.mpd = MPD(**mpd_config)
        self.msd = MSD(**msd_config)

    def forward(self, mel_spec):
        return self.generator(mel_spec)
