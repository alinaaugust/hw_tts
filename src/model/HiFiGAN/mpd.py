from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.kernel_size = 5
        self.stride = 3
        self.channels = [1, 32, 128, 512, 1024]
        self.layers = [
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=self.channels[i],
                        out_channels=self.channels[i + 1],
                        kernel_size=(self.kernel_size, 1),
                        stride=(self.stride, 1),
                        padding=((self.kernel_size - 1) // 2, 0),
                    )
                ),
                nn.LeakyReLU(),
            )
            for i in range(len(self.channels))
        ]

        self.layers.append(
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=self.channels[-1],
                        out_channels=1024,
                        kernel_size=(5, 1),
                        padding="same",
                    )
                ),
                nn.LeakyReLU(),
            )
        )

        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=1024, out_channels=1, kernel_size=(3, 1), padding="same"
                )
            )
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        batch_size, n_channels, time = x.shape

        if time % self.period != 0:
            x = F.pad(x, (0, self.period - (time % self.period)), "reflect")
        x = x.reshape(batch_size, n_channels, time // self.period, self.period)

        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps


class MPD(nn.Module):
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in periods]
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
