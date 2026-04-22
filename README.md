# Multi-Game AlphaZero with OpenSpiel & Flet

A custom implementation of the AlphaZero algorithm designed to master multiple games using the OpenSpiel framework, featuring an interactive Cross-Platform UI (Flet) for human-AI play.

## 🚀 Project Overview
This project implements the AlphaZero algorithm from scratch, leveraging DeepMind's **OpenSpiel** for game mechanics and **PyTorch** for deep learning. The architecture is designed to handle diverse game structures, starting with:
- **Connect Four:** 2D visual structure using Convolutional Neural Networks (CNN).
- **Backgammon:** 1D array structure using Multi-Layer Perceptrons (MLP).

An interactive **Flet** (Python + Flutter) dashboard allows users to select games, play against trained models, and visualize AI decision-making.

## 🛠 Tech Stack
- **RL Framework:** Custom AlphaZero (Self-play, MCTS, Policy/Value Network Updates).
- **Game Engine:** [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`pyspiel`).
- **Deep Learning:** PyTorch.
- **Web/Desktop UI:** Flet.
- **Configuration:** YAML-based hyperparameter management.

## 📁 Directory Structure & Design Philosophy

```text
.
├── ui/                     # INTERACTIVE UI (Flet)
│   ├── app.py              # Main entry point (Routing & Window Config)
│   └── views/              # Page-specific views
│       ├── home.py         # Game selection hub
│       ├── connect_four.py # Interactive Connect Four board
│       └── backgammon.py   # Interactive Backgammon board
├── agents/                 # AI ALGORITHM CORE
│   ├── __init__.py
│   ├── alphazero.py        # Controller for self-play loop and network updates
│   ├── mcts.py             # Pure Monte Carlo Tree Search logic (game-agnostic)
│   ├── networks.py         # Neural architectures (CNN for C4, MLP for Backgammon)
│   ├── replay_buffer.py    # Experience replay storage and sampling
│   └── utils.py            # Utility functions
├── scripts/                # EXECUTION ENTRY POINTS
├── configs/                # HYPERPARAMETERS
├── outputs/                # ARTIFACTS
├── requirements.txt        # Dependency management
├── .gitignore              # Git exclusion rules
└── README.md               # Project documentation
```

### Design Philosophy
1.  **Modularity:** The `agents/` core is decoupled from the UI.
2.  **Modern UI:** Leveraging Flet for a responsive, desktop-class interactive experience.
3.  **Visualization-First:** The UI allows for immediate qualitative analysis of agent behavior.

### Observation Pipeline Convention

The training and inference pipeline always passes **flat observation vectors** (as returned by open_spiel's `ObservationTensor()`) to the model. The **model network itself** is responsible for:

1. **Reshaping** the flat input to its preferred format (e.g., `(B, 126) → (B, 3, 6, 7)` for the Connect Four CNN).
2. **Normalization** of observation values, if needed by the game (e.g., for games with non-binary observation values).

This convention ensures that both the C++ MCTS self-play engine and the Python inference code can feed raw flat observations to the TorchScript-traced model, and all transforms are handled transparently inside the model's `forward()` pass. Nothing outside the model needs to know about game-specific observation structure or value ranges.

## 🚦 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the UI: `python3 ui/app.py`
