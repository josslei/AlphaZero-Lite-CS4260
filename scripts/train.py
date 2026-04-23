import sys
import os
import datetime
import torch
import numpy as np
import yaml
import argparse
import pyspiel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# Add the project root to sys.path so we can import from agents and core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast
import torch.nn as nn

from agents.networks.factory import get_model
from agents.alphazero import AlphaZeroLightning
from agents.replay_buffer import ReplayBuffer
from agents.mcts import SelfPlayEngine
from agents.game_spec import get_game_spec


def execute_self_play(
    config,
    model_path,
    obs_flat_size,
    num_games=None,
    num_threads=None,
    num_iters=None,
    batch_size=None,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray, float]]], dict]:
    """
    Simulates self-play games using the Pure C++ SelfPlayEngine.
    Values are taken from config['mcts'] unless overridden.
    """
    mcts_cfg = config["mcts"]
    num_games = num_games or mcts_cfg["num_games_per_epoch"]
    num_threads = num_threads or mcts_cfg["num_threads"]
    num_iters = num_iters or mcts_cfg.get("num_iters", 800)
    batch_size = batch_size or mcts_cfg["batch_size"]

    game_name = config["game"]["name"]

    print(
        f"Launching C++ SelfPlayEngine: {num_games} games, {num_threads} threads, {num_iters} iters (batch={batch_size})..."
    )

    use_fp16 = config["system"].get("use_fp16", False)
    use_undo = mcts_cfg.get("use_undo", False)

    engine = SelfPlayEngine(
        model_path=model_path,
        batch_size=batch_size,
        obs_flat_size=obs_flat_size,
        num_threads=num_threads,
        num_iters=num_iters,
        temperature=mcts_cfg["temperature"],
        c_puct=mcts_cfg["c_puct"],
        dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
        dirichlet_epsilon=mcts_cfg.get("dirichlet_epsilon", 0.25),
        use_fp16=use_fp16,
        use_undo=use_undo,
    )

    trajectories = engine.generate_games(num_games=num_games, game_name=game_name)
    cpp_metrics = engine.get_metrics()

    # Explicitly cleanup engine to free C++/LibTorch GPU memory
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trajectories, cpp_metrics


def resolve_num_iters(config, current_epoch: int) -> int:
    """
    Determines the number of MCTS iterations for the current epoch based on the schedule.
    Schedule format: [[start_epoch, iters], ...]
    """
    mcts_cfg = config["mcts"]
    schedule = mcts_cfg.get("schedule")
    if not schedule:
        return mcts_cfg.get("num_iters", 800)

    # Sort schedule by start_epoch just in case it's not ordered
    sorted_schedule = sorted(schedule, key=lambda x: x[0])

    current_iters = sorted_schedule[0][1]
    for start_epoch, iters in sorted_schedule:
        if current_epoch >= start_epoch:
            current_iters = iters
        else:
            break

    return current_iters


class ModelExportCallback(Callback):
    """
    Ensures the top-k best models tracked by the CheckpointCallback
    are also exported as TorchScript (.pt) files for inference.
    """

    def __init__(self, config, obs_flat_size):
        super().__init__()
        self.config = config
        self.obs_flat_size = obs_flat_size

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 1. Export the top-k models
        checkpoint_callback = cast(ModelCheckpoint, trainer.checkpoint_callback)
        best_k_models = checkpoint_callback.best_k_models  # Dict[path, score]

        use_fp16 = self.config["system"].get("use_fp16", False)
        input_dtype = torch.float16 if use_fp16 else torch.float32
        example_input = torch.randn(
            1, self.obs_flat_size, device=pl_module.device, dtype=input_dtype
        )

        # Track valid .pt paths to keep
        valid_pt_paths = set()

        for ckpt_path in best_k_models.keys():
            # Generate the corresponding .pt path
            # From: /path/to/checkpoints/alphazero-epoch=02-train_loss=0.50.ckpt
            # To:   /path/to/checkpoints/alphazero-epoch=02-train_loss=0.50.pt
            pt_path = ckpt_path.replace(".ckpt", ".pt")
            valid_pt_paths.add(pt_path)

            if not os.path.exists(pt_path):
                print(f"Exporting top-k checkpoint to TorchScript: {pt_path}")
                # Load weights from the checkpoint into a temporary state dict
                checkpoint = torch.load(ckpt_path, map_location=pl_module.device)

                # We need to load these weights into the inner model
                # PL state_dict prefixes weights with "model."
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v

                # Use a clean model instance to avoid side effects
                # Ensure the exported model outputs probabilities (Softmax)
                export_params = self.config["model"]["params"].copy()
                export_params["return_logits"] = False
                model_to_export = get_model(self.config["model"]["architecture"], export_params)
                model_to_export.load_state_dict(new_state_dict)
                model_to_export.to(pl_module.device)
                model_to_export.eval()
                if use_fp16:
                    model_to_export.half()

                traced_model = cast(
                    torch.jit.ScriptModule, torch.jit.trace(model_to_export, example_input)
                )
                traced_model = torch.jit.optimize_for_inference(traced_model)
                traced_model.save(pt_path)

        # 2. Cleanup orphaned .pt files
        # ModelCheckpoint deletes old .ckpt files automatically, but we must delete .pt files
        ckpt_dir = checkpoint_callback.dirpath
        if ckpt_dir and os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".pt") and f.startswith("alphazero-"):
                    full_path = os.path.join(ckpt_dir, f)
                    if full_path not in valid_pt_paths:
                        print(f"Removing orphaned TorchScript checkpoint: {full_path}")
                        try:
                            os.remove(full_path)
                        except OSError:
                            print(f"[WARNING] Failed to remove model {full_path}")


class SelfPlayCallback(Callback):
    def __init__(self, config, output_dir, obs_flat_size):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.obs_flat_size = obs_flat_size

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Cast to our specific class so Pyright sees .model and .replay_buffer
        az_module = cast(AlphaZeroLightning, pl_module)

        # Determine num_iters based on schedule
        num_iters = resolve_num_iters(self.config, trainer.current_epoch)

        print(
            f"\n>>> [Epoch {trainer.current_epoch}] C++ MCTS Self-Play: START (iters={num_iters})"
        )

        # Export the latest model for C++ LibTorch use
        export_path = os.path.join(self.output_dir, "current_model.pt")

        # We need a model that outputs probabilities (Softmax) for MCTS
        export_params = self.config["model"]["params"].copy()
        export_params["return_logits"] = False
        inference_model = get_model(self.config["model"]["architecture"], export_params)
        inference_model.load_state_dict(az_module.model.state_dict())
        inference_model.to(az_module.device)
        inference_model.eval()

        use_fp16 = self.config["system"].get("use_fp16", False)
        if use_fp16:
            inference_model.half()

        input_dtype = torch.float16 if use_fp16 else torch.float32
        example_input = torch.randn(
            1, self.obs_flat_size, device=az_module.device, dtype=input_dtype
        )

        # Cast to ScriptModule to resolve .save() attribute error
        traced_model = cast(torch.jit.ScriptModule, torch.jit.trace(inference_model, example_input))
        traced_model = torch.jit.optimize_for_inference(traced_model)
        traced_model.save(export_path)

        # Invoke the C++ Self-Play engine using scheduled iters
        new_data, cpp_metrics = execute_self_play(
            config=self.config,
            model_path=export_path,
            obs_flat_size=self.obs_flat_size,
            num_iters=num_iters,
        )

        # 2. Log MCTS depth metrics returned from C++
        pl_module.log(
            "self_play/mcts_avg_depth",
            cpp_metrics.get("avg_search_depth", 0.0),
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            "self_play/mcts_max_depth",
            cpp_metrics.get("max_search_depth", 0.0),
            on_epoch=True,
            sync_dist=True,
        )

        # 3. Calculate and log average game length
        if new_data:
            avg_game_length = sum(len(traj) for traj in new_data) / len(new_data)
            pl_module.log(
                "self_play/avg_game_length", float(avg_game_length), on_epoch=True, sync_dist=True
            )

        # Load the new data into the Replay Buffer
        for trajectory in new_data:
            az_module.replay_buffer.push(trajectory)

        # 4. Log Buffer Size
        pl_module.log(
            "self_play/buffer_size",
            float(len(az_module.replay_buffer)),
            on_epoch=True,
            sync_dist=True,
        )

        # 5. Calculate Root Value of the initial board state (zero overhead)
        game = pyspiel.load_game(self.config["game"]["name"])
        init_state = game.new_initial_state()

        # Advance past initial chance nodes (e.g. initial dice roll in backgammon)
        while init_state.is_chance_node():
            outcomes = init_state.chance_outcomes()
            action, _ = outcomes[0]  # Deterministic enough for a root value sample
            init_state.apply_action(action)

        obs_tensor = torch.tensor(
            init_state.observation_tensor(), dtype=torch.float32, device=az_module.device
        ).unsqueeze(0)

        if self.config["system"].get("use_fp16", False):
            obs_tensor = obs_tensor.half()

        with torch.no_grad():
            _, value = inference_model(obs_tensor)  # Use the latest model

        pl_module.log("self_play/init_root_value", value.item(), on_epoch=True, sync_dist=True)

        print(
            f"<<< [Epoch {trainer.current_epoch}] C++ MCTS Self-Play: END (Pool Size: {len(az_module.replay_buffer)})"
        )
        print(f">>> [Epoch {trainer.current_epoch}] Neural Network Training: START")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"<<< [Epoch {trainer.current_epoch}] Neural Network Training: END\n")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero General Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # 0. Load Configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    output_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    run_dir = os.path.join(output_root, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save a copy of the config for reproducibility
    with open(os.path.join(run_dir, "config_dump.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"Starting experiment: {experiment_name}")
    print(f"Outputs will be saved to: {run_dir}")

    # Set performance precision
    torch.set_float32_matmul_precision(config["system"].get("precision", "medium"))

    # 1. Initialize components
    game_spec = get_game_spec(config["game"]["name"])
    obs_flat_size = game_spec.obs_flat_size

    model = get_model(config["model"]["architecture"], config["model"]["params"])
    buffer = ReplayBuffer(max_size=config["training"]["replay_buffer_size"], game_spec=game_spec)

    lit_model = AlphaZeroLightning(
        model=model,
        replay_buffer=buffer,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 0),
    )

    # 2. Bootstrap: Initial cold start
    # Bootstrap settings are now under config['mcts']['bootstrap']
    bootstrap_cfg = config["mcts"].get("bootstrap", {})
    if bootstrap_cfg.get("enabled", False):
        print(f"Generating initial dataset (Pure C++)...")
        init_model_path = os.path.join(run_dir, "current_model.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the exported model outputs probabilities (Softmax)
        export_params = config["model"]["params"].copy()
        export_params["return_logits"] = False
        inference_model = get_model(config["model"]["architecture"], export_params)
        inference_model.load_state_dict(lit_model.model.state_dict())
        inference_model.to(device)
        inference_model.eval()

        use_fp16 = config["system"].get("use_fp16", False)
        if use_fp16:
            inference_model.half()

        input_dtype = torch.float16 if use_fp16 else torch.float32
        example_input = torch.randn(1, obs_flat_size, device=device, dtype=input_dtype)
        traced_model = cast(torch.jit.ScriptModule, torch.jit.trace(inference_model, example_input))
        traced_model = torch.jit.optimize_for_inference(traced_model)
        traced_model.save(init_model_path)

        # Execute initial self-play games using bootstrap overrides
        init_data, _ = execute_self_play(
            config=config,
            model_path=init_model_path,
            obs_flat_size=obs_flat_size,
            num_games=bootstrap_cfg.get("num_games"),
            num_threads=bootstrap_cfg.get("num_threads"),
            num_iters=bootstrap_cfg.get("num_iters"),
            batch_size=bootstrap_cfg.get("batch_size"),
        )
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

    use_fp16 = config["system"].get("use_fp16", False)
    precision = "16-mixed" if use_fp16 and torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=config["training"]["max_epochs"],
        reload_dataloaders_every_n_epochs=1,
        precision=precision,
        callbacks=[
            SelfPlayCallback(config=config, output_dir=run_dir, obs_flat_size=obs_flat_size),
            ModelExportCallback(config=config, obs_flat_size=obs_flat_size),
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ],
        logger=[csv_logger, tb_logger],
        accelerator=config["system"].get("accelerator", "auto"),
        devices=config["system"].get("devices", 1),
        log_every_n_steps=10,
    )

    # 4. Start the closed-loop training
    trainer.fit(lit_model)


if __name__ == "__main__":
    main()
