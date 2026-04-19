import sys
import os
import datetime
from rich.progress import track

# Add the project root to sys.path so we can import from agents and core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
import numpy as np
import pyspiel

from agents.networks.connect_four import ConnectFourCNN
from agents.alphazero import AlphaZeroLightning
from agents.replay_buffer import ReplayBuffer
from agents.mcts import CppMCTS


def execute_self_play(model_path, num_games=50, num_threads=8):
    """
    Simulates self-play games using the C++ MCTS backend to generate training data.
    """
    game = pyspiel.load_game("connect_four")
    # Use the C++ backend for fast batched multi-threaded tree search
    mcts = CppMCTS(
        model_path=model_path, 
        num_iters=800, 
        temperature=1.0, 
        num_threads=num_threads, 
        batch_size=8
    )
    
    all_trajectories = []
    
    for _ in track(range(num_games), description="Processing..."):
        state = game.new_initial_state()
        trajectory = []
        
        while not state.is_terminal():
            # Get action probabilities from MCTS
            probs_dict = mcts.search(state)

            # Convert dict {action: prob} to array
            pi = np.zeros(7, dtype=np.float32)
            for a, p in probs_dict.items():
                pi[a] = p
                
            # Convert OpenSpiel state to tensor (Shape: 3, 6, 7)
            obs = np.array(state.observation_tensor(), dtype=np.float32).reshape(3, 6, 7)
            
            # Store state, policy, and current player id (to assign values later)
            trajectory.append((obs, pi, state.current_player()))
            
            # Sample an action according to the probabilities
            actions = list(probs_dict.keys())
            probs = np.array(list(probs_dict.values()), dtype=np.float64)
            probs /= probs.sum()
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
            
        # The game has ended. Assign returns from the perspective of the player who made the move.
        returns = state.returns()
        final_trajectory = []
        
        for obs, pi, player_id in trajectory:
            v = returns[player_id]
            final_trajectory.append((obs, pi, v))
            
        all_trajectories.append(final_trajectory)
        
    return all_trajectories


class SelfPlayCallback(Callback):
    def __init__(self, output_dir, num_games=50, mcts_threads=8):
        super().__init__()
        self.output_dir = output_dir
        self.num_games = num_games
        self.mcts_threads = mcts_threads

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n--- [Epoch {trainer.current_epoch}] Launching C++ MCTS Self-Play ---")
        
        # 1. Export the latest model for C++ LibTorch use
        export_path = os.path.join(self.output_dir, "current_model.pt")
        pl_module.eval()
        
        # Ensure the input tensor is on the correct device (CPU/GPU)
        example_input = torch.randn(1, 3, 6, 7, device=pl_module.device)
        traced_model = torch.jit.trace(pl_module.model, example_input)
        traced_model.save(export_path)
        pl_module.train()

        # 2. Invoke the Self-Play engine
        new_data = execute_self_play(
            model_path=export_path, 
            num_games=self.num_games, 
            num_threads=self.mcts_threads
        )
        
        # 3. Load the new data into the Replay Buffer
        for trajectory in new_data:
            pl_module.replay_buffer.push(trajectory)
            
        print(f"Self-play complete. Current experience pool size: {len(pl_module.replay_buffer)}")


def main():
    # 0. Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting training run: {run_name}")
    print(f"Outputs will be saved to: {run_dir}")

    # 1. Initialize components
    model = ConnectFourCNN()
    buffer = ReplayBuffer(max_size=50000)
    
    lit_model = AlphaZeroLightning(
        model=model, 
        replay_buffer=buffer,
        lr=0.001,
        batch_size=64
    )

    # 2. Bootstrap: Initial cold start (ensures the DataLoader is not empty for the first training round)
    print("Generating initial dataset...")
    lit_model.eval()
    init_model_path = os.path.join(run_dir, "current_model.pt")
    
    # Trace model for C++ engine
    traced_model = torch.jit.trace(lit_model.model, torch.randn(1, 3, 6, 7))
    traced_model.save(init_model_path)
    lit_model.train()

    # Execute initial self-play games
    init_data = execute_self_play(init_model_path, num_games=10, num_threads=8)
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
        mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=100,
        # Key setting: Re-evaluate dataset size every epoch, as self-play continuously adds data
        reload_dataloaders_every_n_epochs=1, 
        callbacks=[
            SelfPlayCallback(output_dir=run_dir, num_games=50, mcts_threads=8),
            checkpoint_callback
        ],
        logger=[csv_logger, tb_logger],
        accelerator="auto", # Automatically identifies MPS (Mac), CUDA, or CPU
        devices=1,
        log_every_n_steps=10
    )

    # 4. Start the closed-loop training
    trainer.fit(lit_model)

if __name__ == "__main__":
    main()
