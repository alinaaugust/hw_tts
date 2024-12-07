import torch
import torch.nn.functional as F
from torch import nn

from src.loss.fm_loss import FeatureMatchingLoss
from src.loss.gan_loss import DiscriminatorGanLoss, GeneratorGanLoss
from src.loss.melspec_loss import MelSpectrogramLoss


class HiFiGanLoss(nn.Module):
    def __init__(self, fm_lambda: int, mel_lambda: int):
        super().__init__()
        self.fm_loss = FeatureMatchingLoss(fm_lambda)
        self.mel_loss = MelSpectrogramLoss(mel_lambda)
        self.generator_gan_loss = GeneratorGanLoss()
        self.discriminator_gan_loss = DiscriminatorGanLoss()
