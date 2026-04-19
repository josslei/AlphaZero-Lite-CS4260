import sys
import os
import datetime
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# Add the project root to sys.path so we can import from agents and core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks.connect_four import ConnectFourCNN
from agents.alphazero import AlphaZeroLightning
from agents.replay_buffer import ReplayBuffer
from agents.mcts import SelfPlayEngine

# Set float32 matmul precision for better performance on RTX 4060 (Tensor Cores)
torch.set_float32_matmul_precision("medium")


def execute_self_play(model_path, num_games=256, num_threads=128, num_iters=400, batch_size=64):
    """
    Simulates self-play games using the Pure C++ SelfPlayEngine.
    Crank up concurrency to fully utilize high-end GPUs like RTX 4060.
    """
    print(f"Launching C++ SelfPlayEngine: {num_games} games, {num_threads} threads, {num_iters} iters (batch={batch_size})...")

    engine = SelfPlayEngine(
        model_path=model_path,
        batch_size=batch_size,
        num_threads=num_threads,
        num_iters=num_iters,
        temperature=1.0,
        c_puct=1.0
    )

    trajectories = engine.generate_games(num_games=num_games, game_name="connect_four")

    # Explicitly cleanup engine to free C++/LibTorch GPU memory
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trajectories


class SelfPlayCallback(Callback):
    def __init__(self, output_dir, num_games=256, mcts_threads=128):
        super().__init__()
        self.output_dir = output_dir
        self.num_games = num_games
        self.mcts_threads = mcts_threads

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n>>> [Epoch {trainer.current_epoch}] C++ MCTS Self-Play: START")
        
        # 1. Export the latest model for C++ LibTorch use
        export_path = os.path.join(self.output_dir, "current_model.pt")
        pl_module.eval()
        
        example_input = torch.randn(1, 3, 6, 7, device=pl_module.device)
        traced_model = torch.jit.trace(pl_module.model, example_input)
        traced_model.save(export_path)
        pl_module.train()

        # 2. Invoke the C++ Self-Play engine
        new_data = execute_self_play(
            model_path=export_path, 
            num_games=self.num_games, 
            num_threads=self.mcts_threads,
            num_iters=400,
            batch_size=64
        )
        
        # 3. Load the new data into the Replay Buffer
        for trajectory in new_data:
            pl_module.replay_buffer.push(trajectory)
            
        print(f"<<< [Epoch {trainer.current_epoch}] C++ MCTS Self-Play: END (Pool Size: {len(pl_module.replay_buffer)})")
        print(f">>> [Epoch {trainer.current_epoch}] Neural Network Training: START")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"<<< [Epoch {trainer.current_epoch}] Neural Network Training: END\n")


def main():
    # 0. Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    output_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting training run: {run_name}")
    print(f"Outputs will be saved to: {run_dir}")

    # 1. Initialize components
    model = ConnectFourCNN()
    buffer = ReplayBuffer(max_size=50000)

    lit_model = AlphaZeroLightning(model=model, replay_buffer=buffer, lr=0.001, batch_size=64)

    # 2. Bootstrap: Initial cold start
    print("Generating initial dataset (Pure C++)...")
    lit_model.eval()
    init_model_path = os.path.join(run_dir, "current_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)
    example_input = torch.randn(1, 3, 6, 7, device=device)
    traced_model = torch.jit.trace(lit_model.model, example_input)
    traced_model.save(init_model_path)
    lit_model.train()
    
    # Execute initial self-play games (use high concurrency even for bootstrap)
    init_data = execute_self_play(init_model_path, num_games=64, num_threads=64, num_iters=100, batch_size=32)
    for t in init_data:
        buffer.push(t)

    print(f"Initial dataset ready! Experience pool size: {len(buffer)}")

    # 3. Configure PyTorch Lightning Trainer
    csv_logger = CSVLogger(save_dir=run_dir, name="logs")
    tb_logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="alphazero-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=100,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            SelfPlayCallback(output_dir=run_dir, num_games=256, mcts_threads=128),
            checkpoint_callback,
        ],
        logger=[csv_logger, tb_logger],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    # 4. Start the closed-loop training
    trainer.fit(lit_model)


if __name__ == "__main__":
    main()
