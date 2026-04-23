from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.utils import OPENSPIEL_BACKGAMMON_ACTION_SPACE_SIZE


class ResidualBlockCNN1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x += residual
        x = F.relu(x)
        return x


class BackgammonCNN(nn.Module):
    def __init__(self, return_logits: bool = False, **kwargs) -> None:
        super().__init__()

        self.board_in = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.board_conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # board_feature dim: (batch_size, 4 * 24)
        self.board_out = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=4, kernel_size=3, stride=1, padding="same", bias=False
            ),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Global: stats and dice input
        self.global_in = nn.Sequential(nn.Linear(20, 64), nn.ReLU())

        combined_feature_dim = 4 * 24 + 64

        policy_out_dim = OPENSPIEL_BACKGAMMON_ACTION_SPACE_SIZE
        policy_layers = [
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, policy_out_dim),
        ]
        if not return_logits:
            policy_layers.append(nn.Softmax(dim=1))
        self.policy_head = nn.Sequential(*policy_layers)

        self.value_head = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, 200)
        # Split & normalize the state vector
        board_state, stats, dice_flat = self._process_state(x)
        # Feature - board (batch_size, 96)
        feat_board = self.board_in(board_state)
        feat_board = self.board_conv_block_1(feat_board)
        feat_board = self.board_out(feat_board)
        # Feature - global (batch_size, 64)
        feat_global = self.global_in(torch.cat((stats, dice_flat), dim=1))
        feat_comb = torch.cat((feat_board, feat_global), dim=1)
        policy = self.policy_head(feat_comb)
        value = self.value_head(feat_comb)
        return policy, value

    def _process_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x.shape: (batch_size, 200)
        batch_size = x.shape[0]

        # Use .clone() to prevent in-place mutation of the original 'x' tensor
        board_state = x[:, :192].clone()
        stats = x[:, 192:198].clone()
        dice_state = x[:, 198:200]  # No clone needed, we don't modify it in-place

        # Normalization
        board_state[:, 3::4] = board_state[:, 3::4] / 12.0
        stats[:, 0] = stats[:, 0] / 15.0
        stats[:, 1] = stats[:, 1] / 15.0
        stats[:, 3] = stats[:, 3] / 15.0
        stats[:, 4] = stats[:, 4] / 15.0

        # From (B, 192) -> (B, 24, 8) -> (B, 8, 24)
        board_state = board_state.view(batch_size, 24, 8).permute(0, 2, 1)

        # Dice state to one-hot: (B, 2) -> (B, 2, 7) -> (B, 14)
        # Clamped to prevent F.one_hot from crashing on negative values provided by torch.randn during JIT tracing
        dice_int = dice_state.to(torch.int64).clamp(min=0, max=6)
        dice_onehot = F.one_hot(dice_int, num_classes=7).float()
        dice_flat = dice_onehot.view(x.shape[0], -1)

        return board_state, stats, dice_flat
