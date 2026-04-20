import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .replay_buffer import ReplayBuffer


class AlphaZeroLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        replay_buffer: ReplayBuffer,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model = model
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Tell PyTorch Lightning (PL) to save hyperparameters (excluding model and buffer objects)
        self.save_hyperparameters(ignore=["model", "replay_buffer"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, target_pis, target_vs = batch

        out_pi, out_v = self(states)

        # Policy Loss: Cross Entropy (Assuming out_pi hasn't gone through Softmax or LogSoftmax yet)
        # If your network ends with raw linear output, using F.cross_entropy directly is more stable
        loss_pi = -torch.mean(torch.sum(target_pis * torch.log(out_pi + 1e-8), dim=1))

        # Value Loss: Mean Squared Error (MSE)
        loss_v = F.mse_loss(out_v, target_vs)

        loss = loss_pi + loss_v

        # PL automatically logs and displays these values on the progress bar
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_pi", loss_pi)
        self.log("loss_v", loss_v)

        return loss

    def configure_optimizers(self):
        # Weight Decay is recommended to prevent overfitting
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        # Bind the Dataloader within the Module
        # Note: num_workers=0 is used to prevent multiprocessing from messing with the deque.
        # Consider multiprocessing only if the dataset becomes extremely large.
        return DataLoader(
            self.replay_buffer, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
