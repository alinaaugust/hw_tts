from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int, dilations: List[List[int]]):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.layers = []
        for m in range(len(self.dilations)):
            partial_layers = []
            for l in range(len(self.dilations[0])):
                partial_layers.append(nn.LeakyReLU())
                partial_layers.append(
                    weight_norm(
                        nn.Conv1d(
                            in_channels=self.n_channels,
                            out_channels=self.n_channels,
                            kernel_size=self.kernel_size,
                            dilation=self.dilations[m][l],
                            padding="same",
                        )
                    )
                )
            self.layers.append(nn.Sequential(*partial_layers))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class MRF(nn.Module):
    def __init__(
        self, n_channels: int, kernel_sizes: List[int], dilations: List[List[List[int]]]
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.blocks = nn.ModuleList(
            [
                ResBlock(self.n_channels, self.kernel_sizes[i], self.dilations[i])
                for i in range(len(self.kernel_sizes))
            ]
        )

    def forward(self, x):
        output = sum([block(x) for block in self.blocks])
        return output
