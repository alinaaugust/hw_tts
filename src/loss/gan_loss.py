import torch
import torch.nn.functional as F
from torch import nn


class DiscriminatorGanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_pred, output_true):
        loss = 0.0
        for pred, true in zip(output_pred, output_true):
            loss_pred = torch.mean(pred**2)
            loss_true = torch.mean((true - 1) ** 2)
            loss += loss_pred + loss_true
        return loss


class GeneratorGanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_pred):
        loss = 0.0
        for pred in output_pred:
            loss += torch.mean((pred - 1) ** 2)
        return loss
