import torch.nn.functional as F
from torch import nn


class MelSpectrogramLoss(nn.Module):
    def __init__(self, mel_lambda: int = 45):
        super().__init__()
        self.mel_lambda = mel_lambda

    def forward(self, pred_melspec, true_melspec):
        return self.mel_lambda * F.l1_loss(pred_melspec, true_melspec)
