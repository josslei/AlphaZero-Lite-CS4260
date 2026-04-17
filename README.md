# Multi-Game AlphaZero with OpenSpiel & Streamlit

A custom implementation of the AlphaZero algorithm designed to master multiple games using the OpenSpiel framework, featuring an interactive Web UI for human-AI play.

## 🚀 Project Overview
This project implements the AlphaZero algorithm from scratch, leveraging DeepMind's **OpenSpiel** for game mechanics and **PyTorch** for deep learning. The architecture is designed to handle diverse game structures, starting with:
- **Connect Four:** 2D visual structure using Convolutional Neural Networks (CNN).
- **Backgammon:** 1D array structure using Multi-Layer Perceptrons (MLP).

An interactive **Streamlit** dashboard allows users to select games, play against trained models, and visualize AI decision-making (MCTS priors and value estimates).

## 🛠 Tech Stack
- **RL Framework:** Custom AlphaZero (Self-play, MCTS, Policy/Value Network Updates).
- **Game Engine:** [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`pyspiel`).
- **Deep Learning:** PyTorch.
- **Web UI:** Streamlit.
- **Configuration:** YAML-based hyperparameter management.

## 📁 Directory Structure & Design Philosophy

```text
.
├── ui/                     # INTERACTIVE WEB GUI
│   ├── app.py              # Main entry point (Welcome/Game Selection)
│   ├── components.py       # Reusable UI components (board rendering, charts)
│   └── pages/              # Streamlit multipage routing for specific games
│       ├── 1_connect_four.py  
│       └── 2_backgammon.py    
├── agents/                 # AI ALGORITHM CORE
│   ├── __init__.py
│   ├── alphazero.py        # Controller for self-play loop and network updates
│   ├── mcts.py             # Pure Monte Carlo Tree Search logic (game-agnostic)
│   ├── networks.py         # Neural architectures (CNN for C4, MLP for Backgammon)
│   ├── replay_buffer.py    # Experience replay storage and sampling
│   └── env_utils.py        # Bridge: Translates OpenSpiel `State` into Tensors
├── scripts/                # EXECUTION ENTRY POINTS
│   ├── train_alphazero.py  # Main training script (reads configs, runs self-play)
│   ├── evaluate_models.py  # Tournament-style evaluation (New vs. Old checkpoints)
│   └── test_env.py         # Quick sanity checks for OpenSpiel game mechanics
├── configs/                # HYPERPARAMETERS
│   ├── c4_config.yaml      # Connect Four specific parameters
│   └── bg_config.yaml      # Backgammon specific parameters
├── outputs/                # ARTIFACTS
│   └── checkpoints/        # Saved model weights (.pth)
├── requirements.txt        # Dependency management
├── .gitignore              # Git exclusion rules
└── README.md               # Project documentation
```

### Design Philosophy
1.  **Modularity:** The `agents/` core is decoupled from the UI and the execution scripts. MCTS logic is game-agnostic, relying on the `env_utils.py` bridge to handle game-specific state representations.
2.  **Configuration-Driven:** To ensure reproducibility and experiment tracking, no hyperparameters are hardcoded. All settings—from learning rates to MCTS simulations—are loaded from `configs/`.
3.  **Visualization-First:** The `ui/` layer is treated as a first-class citizen, allowing for immediate qualitative analysis of agent behavior beyond simple win-rate metrics.
4.  **Extensibility:** The `networks.py` and `env_utils.py` files are designed to be extended for new games (e.g., Chess, Go) without modifying the core AlphaZero controller.

## 🚦 Getting Started
(Details on installation and execution to be added upon implementation.)
