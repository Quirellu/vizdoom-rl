import torch
from torch import nn


class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Flatten(),

            nn.LazyLinear(256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = x.float() / 255.0

        value = self.net(x)

        return value.squeeze(-1)