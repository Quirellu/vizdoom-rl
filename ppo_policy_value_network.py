import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),

            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels)
        )

        self.activation = nn.SiLU()

    def forward(self, x):

        residual = x

        x = self.block(x)

        x = x + residual

        x = self.activation(x)

        return x

class PolicyValueNetwork(nn.Module):

    def __init__(self, action_dim, input_channels = 3):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(input_channels, 32, 8, 4),
            nn.SiLU(),

            ResidualBlock(32),

            nn.Conv2d(32, 64, 4, 2),
            nn.SiLU(),

            ResidualBlock(64),

            nn.Conv2d(64, 64, 3, 1),
            nn.SiLU(),

            ResidualBlock(64),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten()
        )

        self.policy_head = nn.Sequential(

            nn.Linear(64, 256),
            nn.SiLU(),

            nn.Linear(256, action_dim)
        )

        self.value_head = nn.Sequential(

            nn.Linear(64, 256),
            nn.SiLU(),

            nn.Linear(256, 1)
        )


    def forward(self, x):

        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = x / 255.0

        x = self.features(x)

        logits = self.policy_head(x)

        value = self.value_head(x)

        return logits, value
