import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter


class PolicyValueNetwork(nn.Module):

    def __init__(self, action_dim):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.shared = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(512, action_dim)

        self.value_head = nn.Linear(512, 1)


    def forward(self, x):

        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = x / 255.0

        x = self.features(x)

        x = self.shared(x)

        logits = self.policy_head(x)

        value = self.value_head(x)

        return logits, value
