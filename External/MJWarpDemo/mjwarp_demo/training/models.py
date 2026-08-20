from __future__ import annotations

import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple[int, ...] = (256, 256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.extend((nn.Linear(previous, action_dim), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: tuple[int, ...] = (256, 256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class RiskPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, output_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(sequence)
        return self.head(hidden[-1])


class _DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TinyUNet(nn.Module):
    def __init__(self, class_count: int, base_channels: int = 16) -> None:
        super().__init__()
        self.down1 = _DoubleConv(3, base_channels)
        self.down2 = _DoubleConv(base_channels, base_channels * 2)
        self.bottom = _DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decode2 = _DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decode1 = _DoubleConv(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, class_count, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.down1(image)
        second = self.down2(self.pool(first))
        bottom = self.bottom(self.pool(second))
        decoded_second = self.decode2(torch.cat((self.up2(bottom), second), dim=1))
        decoded_first = self.decode1(torch.cat((self.up1(decoded_second), first), dim=1))
        return self.output(decoded_first)
