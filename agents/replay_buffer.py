import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque


class ReplayBuffer(Dataset):
    def __init__(self, max_size=50000):
        # deque automatically handles maximum capacity, removing the oldest data when exceeded
        self.buffer = deque(maxlen=max_size)

    def push(self, trajectory):
        """Push a game's worth of data returned from C++ into the pool and perform mirror data augmentation"""
        for state, pi, v in trajectory:
            self.buffer.append((state, pi, v))
            # Data augmentation: flipping the state and policy (pi)
            self.buffer.append((np.flip(state, axis=2).copy(), np.flip(pi).copy(), v))

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
