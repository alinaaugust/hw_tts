from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm


class DiscriminatorS(nn.Module):
    def __init__(
        self,
        disc_idx: int,
        n_channels: List[int] = [1, 128, 128, 256, 512, 1024, 1024, 1024],
        kernel_sizes: List[int] = [15, 41, 41, 41, 41, 41, 5],
        strides: List[int] = [1, 2, 2, 4, 4, 1, 1],
        groups: List[int] = [1, 4, 16, 16, 16, 16, 1],
    ):
        super().__init__()
        self.disc_idx = disc_idx
        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.groups = groups
        self.norm = spectral_norm if not self.disc_idx else weight_norm
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

        self.layers = [
            nn.Sequential(
                self.norm(
                    nn.Conv1d(
                        in_channels=self.channels[i],
                        out_channels=self.channels[i + 1],
                        kernel_size=self.kernel_sizes[i],
                        stride=self.strides[i],
                        groups=self.groups[i],
                        padding=(self.kernel_sizes[i] - 1) // 2,
                    )
                ),
                nn.LeakyReLU(),
            )
            for i in range(len(self.channels))
        ]

        self.layers.append(
            self.norm(
                nn.Conv1d(
                    in_channels=self.channels[-1],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        if self.disc_idx == 1:
            x = self.pooling(x)
        elif self.disc_idx == 2:
            x = self.pooling(self.pooling(x))

        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps


class MSD(nn.Module):
    def __init__(
        self,
        indices: List[int] = [0, 1, 2],
        n_channels: List[int] = [1, 128, 128, 256, 512, 1024, 1024, 1024],
        kernel_sizes: List[int] = [15, 41, 41, 41, 41, 41, 5],
        strides: List[int] = [1, 2, 2, 4, 4, 1, 1],
        groups: List[int] = [1, 4, 16, 16, 16, 16, 1],
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(i, n_channels, kernel_sizes, strides, groups)
                for i in indices
            ]
        )

    def forward(self, y, y_hat):
        output, output_hat = [], []
        feature_maps, feature_maps_hat = [], []
        for discriminator in self.discriminators:
            out, out_features = discriminator(y)
            out_hat, out_hat_features = discriminator(y_hat)
            output.append(out)
            output_hat.append(out_hat)
            feature_maps.append(out_features)
            feature_maps_hat.append(out_hat_features)
        return output, output_hat, feature_maps, feature_maps_hat
