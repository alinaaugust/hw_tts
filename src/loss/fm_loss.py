import torch.nn.functional as F
from torch import nn


class FeatureMatchingLoss(nn.Module):
    def __init__(self, fm_lambda: int = 2):
        super().__init__()
        self.fm_lambda = fm_lambda

    def forward(self, pred_fmaps, true_fmaps):
        loss = 0.0
        for discr_pred_fmap, discr_true_fmap in zip(pred_fmaps, true_fmaps):
            for pred, true in zip(discr_pred_fmap, discr_true_fmap):
                loss += F.l1_loss(pred, true)
        return self.fm_lambda * loss
