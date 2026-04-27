# AlphaZero-Lite: High-Performance Reinforcement Learning

A modular, high-performance implementation of the AlphaZero algorithm for board games (Connect Four, Backgammon) using a C++ MCTS backend and PyTorch.

## 🚀 Key Features

- **Blazing Fast MCTS**: Core MCTS and Game Logic implemented in C++ with multi-threading.
- **Zero-GIL Evaluation**: Tournament engine runs fully in C++, bypassing the Python Global Interpreter Lock.
- **Batch Inference**: Thread-safe evaluation system that batches neural network requests across multiple MCTS workers for maximum GPU throughput.
- **Asynchronous Tournaments**: Comprehensive evaluation against benchmark agents (Greedy, Minimax, Random) runs in detached background processes.
- **Game Support**: Native support for Connect Four and Backgammon via OpenSpiel.

## 🛠 Setup & Installation

### 1. Environment Setup
We recommend using [Conda](https://docs.conda.io/en/latest/) for dependency management:

```bash
# Create the environment
conda create -n rl python=3.12 -y
conda activate rl

# Install core dependencies
pip install torch torchvision torchaudio
pip install pytorch-lightning tensorboard pybind11 pyyaml absl-py open-spiel rich
```

### 2. C++ Backend Dependencies
The backend requires **LibTorch** (PyTorch C++ API). 
- **macOS (Homebrew)**: `brew install libtorch`
- **Linux/Manual**: Download the LibTorch zip from [pytorch.org](https://pytorch.org/get-started/locally/) and set `LIBTORCH_PATH`.

### 3. Compiling the MCTS Engine
The project includes a build script to compile the C++ extension and bind it to Python via PyBind11:

```bash
chmod +x build_cppmcts.sh
./build_cppmcts.sh
```
*This will generate `agents/mcts_backend.so` (or `.dylib` on Mac), enabling high-speed self-play.*

## 🏁 How to Train

Training is orchestrated via YAML configuration files.

### Start Training (Connect Four)
```bash
python scripts/train.py --config configs/connect_four_base.yaml
```

### Start Training (Backgammon)
```bash
python scripts/train.py --config configs/backgammon_base.yaml
```

## 📊 Evaluation & Monitoring

### TensorBoard
Monitor training progress, loss, win rates, and engine performance (games/sec, MCTS depth):
```bash
tensorboard --logdir runs/
```

### Tournament Types
1. **Light Tournament**: Runs every few epochs directly in the training loop against a Greedy opponent.
2. **Full Tournament**: Runs asynchronously every $N$ epochs (configured in YAML) against multiple benchmark agents. It executes in a separate process to avoid slowing down training.

## 📁 Project Structure

- `agents/cpp/`: Core C++ implementation of MCTS and Batch Evaluator.
- `agents/networks/`: PyTorch neural network architectures.
- `configs/`: YAML files defining hyperparameters and evaluation settings.
- `scripts/`: Entry points for training and background tournament workers.
- `agents/game_spec.py`: Utilities for handling different game observation shapes.
