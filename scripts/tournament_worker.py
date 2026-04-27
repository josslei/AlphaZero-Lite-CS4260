"""
Full Tournament Worker
======================
Standalone script invoked as a background subprocess by FullTournamentCallback.
It loads the exported TorchScript model, runs a full tournament via C++ TournamentEngine
against all configured opponents, and writes results to TensorBoard and a JSON file.

Usage (launched automatically by FullTournamentCallback; can also be run manually):
    python scripts/tournament_worker.py \
        --model_path  /path/to/eval_model.pt \
        --config      /path/to/config.yaml \
        --log_dir     /path/to/tb_log_dir \
        --epoch       20 \
        --device      cuda   # or cpu
"""

import os
import sys
import json
import argparse
import yaml

# Allow importing from agents/ regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcts import TournamentEngine


def run_worker(model_path: str, config: dict, log_dir: str, epoch: int, device: str):
    from torch.utils.tensorboard import SummaryWriter

    full_cfg = config.get("full_tournament", {})
    game_name = config["game"]["name"]
    mcts_cfg = config["mcts"]

    num_games: int = full_cfg.get("num_games", 100)
    num_threads: int = min(
        full_cfg.get("num_threads", 16),
        num_games,  # Never spawn more threads than games
    )
    mcts_iters: int = full_cfg.get("mcts_iters", 200)
    batch_size: int = full_cfg.get("batch_size", mcts_cfg.get("batch_size", 16))
    c_puct: float = mcts_cfg.get("c_puct", 1.0)
    use_undo: bool = full_cfg.get("use_undo", mcts_cfg.get("use_undo", False))
    opponents: list = full_cfg.get("opponents", ["random", "greedy"])

    use_fp16: bool = config["system"].get("use_fp16", False)
    # Override device if caller requests CPU explicitly to avoid competing with training GPU
    use_gpu = (device == "cuda")
    if not use_gpu:
        use_fp16 = False  # FP16 on CPU is not useful

    from agents.game_spec import get_game_spec
    game_spec = get_game_spec(game_name)
    obs_flat_size = game_spec.obs_flat_size

    print(
        f"[FullTournament Worker] epoch={epoch} game={game_name} "
        f"opponents={opponents} games={num_games} threads={num_threads} "
        f"iters={mcts_iters} device={device}"
    )

    opening_temp_moves: int = full_cfg.get("opening_temp_moves", 2)

    engine = TournamentEngine(
        model_path=model_path,
        batch_size=batch_size,
        obs_flat_size=obs_flat_size,
        num_threads=num_threads,
        num_iters=mcts_iters,
        temperature=0.0,   # Fully deterministic evaluation
        c_puct=c_puct,
        use_fp16=use_fp16,
        use_undo=use_undo,
        opening_temp_moves=opening_temp_moves,
        chance_aware=mcts_cfg.get("chance_aware", False),  # [New] Ablation switch
    )

    all_results: dict = {}
    writer = SummaryWriter(log_dir=log_dir)

    for opp_type in opponents:
        print(f"  > Running vs {opp_type} ...")
        results = engine.play_tournament(
            num_games=num_games,
            game_name=game_name,
            opponent=opp_type,
        )

        wins   = results["wins"]
        losses = results["losses"]
        draws  = results["draws"]
        win_rate  = wins  / max(1, num_games)
        loss_rate = losses / max(1, num_games)
        draw_rate = draws  / max(1, num_games)

        # Log everything to TensorBoard at the epoch step of the triggering epoch
        tag = opp_type
        writer.add_scalar(f"full_eval/win_rate_vs_{tag}",     win_rate,                       epoch)
        writer.add_scalar(f"full_eval/loss_rate_vs_{tag}",    loss_rate,                      epoch)
        writer.add_scalar(f"full_eval/draw_rate_vs_{tag}",    draw_rate,                      epoch)
        writer.add_scalar(f"full_eval/total_time_s_{tag}",    results["total_time_s"],         epoch)
        writer.add_scalar(f"full_eval/games_per_sec_{tag}",   results["games_per_sec"],        epoch)
        writer.add_scalar(f"full_eval/avg_game_length_{tag}", results["avg_game_length"],      epoch)
        writer.add_scalar(f"full_eval/avg_batch_size_{tag}",  results["avg_batch_size"],       epoch)
        writer.add_scalar(f"full_eval/avg_mcts_depth_{tag}",  results["avg_mcts_depth"],       epoch)
        writer.add_scalar(f"full_eval/iters_saved_{tag}",     results["iters_saved"],          epoch)

        all_results[opp_type] = {
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": win_rate, "loss_rate": loss_rate, "draw_rate": draw_rate,
            **{k: results[k] for k in ["total_time_s", "games_per_sec", "avg_game_length",
                                        "avg_batch_size", "avg_mcts_depth", "iters_saved"]},
        }

        print(
            f"    VS {opp_type.upper()}: "
            f"Win={win_rate:.0%} Loss={loss_rate:.0%} Draw={draw_rate:.0%} | "
            f"{results['games_per_sec']:.2f} games/s | "
            f"Avg len: {results['avg_game_length']:.1f} moves"
        )

    writer.flush()
    writer.close()

    # Persist results to a JSON file for debugging / external tooling
    results_path = os.path.join(log_dir, f"full_tournament_epoch_{epoch:03d}.json")
    with open(results_path, "w") as f:
        json.dump({"epoch": epoch, "results": all_results}, f, indent=2)

    print(f"[FullTournament Worker] Done. Results written to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Full Tournament Worker")
    parser.add_argument("--model_path", type=str, required=True,  help="Path to TorchScript .pt model")
    parser.add_argument("--config",     type=str, required=True,  help="Path to YAML config")
    parser.add_argument("--log_dir",    type=str, required=True,  help="TensorBoard log directory")
    parser.add_argument("--epoch",      type=int, required=False, default=-1, help="Evaluation identifier or final epoch (for TB step)")
    parser.add_argument("--device",     type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for the tournament engine (cuda or cpu)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_worker(
        model_path=args.model_path,
        config=config,
        log_dir=args.log_dir,
        epoch=args.epoch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
