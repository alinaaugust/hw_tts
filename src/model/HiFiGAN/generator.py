from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from src.model.HiFiGAN.mrf_block import MRF


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        hidden_size: int = 512,
        kernels_upsample: List[int] = [16, 16, 4, 4],
        kernel_sizes: List[int] = [3, 7, 11],
        dilations: List[List[List[int]]] = [
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
        ],
    ):
        super().__init__()
        self.first_conv = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=7,
                dilation=1,
                padding="same",
            )
        )

        self.layers = []
        for l in range(len(kernels_upsample)):
            self.layers.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    weight_norm(
                        nn.ConvTranspose1d(
                            in_channels=hidden_size // (2**l),
                            out_channels=hidden_size // (2 ** (l + 1)),
                            kernel_size=kernels_upsample[l],
                            stride=kernels_upsample[l] // 2,
                            padding=kernels_upsample[l] // 4,
                        )
                    ),
                    MRF(hidden_size // (2 ** (l + 1)), kernel_sizes, dilations),
                )
            )
        self.layers = nn.Sequential(*self.layers)

        self.last_conv = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(
                nn.Conv1d(
                    in_channels=hidden_size // (2 ** (len(kernels_upsample))),
                    out_channels=1,
                    kernel_size=7,
                    padding="same",
                )
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return x
