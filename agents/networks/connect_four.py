import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockCNN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x += residual
        x = F.relu(x)
        return x


class ConnectFourCNN(nn.Module):
    def __init__(self, return_logits: bool = False, **kwargs) -> None:
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res1 = ResidualBlockCNN(128)
        self.res2 = ResidualBlockCNN(128)
        self.res3 = ResidualBlockCNN(128)

        policy_layers = [
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7),
        ]
        if not return_logits:
            policy_layers.append(nn.Softmax(dim=1))
        self.policy_head = nn.Sequential(*policy_layers)

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle flat input (B, 126) by reshaping to (B, 3, 6, 7)
        if x.dim() == 2:
            x = x.view(-1, 3, 6, 7)

        # Shape: (B, 3, 6, 7)
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        features = self.res3(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
