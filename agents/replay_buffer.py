import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque


from .game_spec import GameSpec


class ReplayBuffer(Dataset):
    def __init__(self, max_size=50000, game_spec: GameSpec | None = None):
        # deque automatically handles maximum capacity, removing the oldest data when exceeded
        self.buffer = deque(maxlen=max_size)
        self.game_spec = game_spec

    def push(self, trajectory):
        """Push a game's worth of data returned from C++ into the pool and perform mirror data augmentation"""
        for state, pi, v in trajectory:
            self.buffer.append((state, pi, v))
            # Data augmentation via GameSpec
            if self.game_spec:
                augmented = self.game_spec.augment(state, pi)
                for s_aug, pi_aug in augmented:
                    self.buffer.append((s_aug, pi_aug, v))

    def snapshot_to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert current buffer contents to contiguous tensors for efficient multi-worker DataLoader.

        This creates a one-time copy of the deque into contiguous memory, enabling:
        - num_workers > 0 (deque cannot be shared across processes)
        - Faster batching from contiguous memory
        """
        states = torch.FloatTensor(np.array([s for s, _, _ in self.buffer]))
        pis = torch.FloatTensor(np.array([p for _, p, _ in self.buffer]))
        vs = torch.FloatTensor(np.array([v for _, _, v in self.buffer]).reshape(-1, 1))
        return states, pis, vs

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, pi, v = self.buffer[idx]
        # DataLoader requires returning Tensors
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(pi),
            torch.FloatTensor([v]),  # Keep shape as (1,)
        )
