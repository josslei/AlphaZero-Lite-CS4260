import torch
import numpy as np
from typing import Any, Mapping
import pyspiel
from .mcts import State, PPD

if torch.cuda.is_available():
    default_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    default_device = torch.device("mps")
else:
    default_device = torch.device("cpu")


class PyspielStateWrapper(State[int]):
    def __init__(self, state: pyspiel.State):
        self.state = state

    def legal_actions(self) -> list[int]:
        return self.state.legal_actions()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def rewards(self) -> float:
        if not self.is_terminal():
            return 0.0
        # For MCTS.py backpropagate, we need value for the "current" player at terminal node.
        # Since it's terminal, it's the player whose turn it would be.
        # Connect Four always alternates.
        next_player = len(self.state.history()) % 2
        return self.state.returns()[next_player]

    def apply_action(self, action: int) -> None:
        self.state.apply_action(action)

    def is_chance_node(self) -> bool:
        return self.state.is_chance_node()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return self.state.chance_outcomes()

    def clone(self) -> "PyspielStateWrapper":
        return PyspielStateWrapper(self.state.clone())


class AlphaZeroEvaluator:
    def __init__(self, model_path: str, obs_flat_size: int, device: torch.device = default_device):
        self.device = torch.device(device)
        self.obs_flat_size = obs_flat_size
        try:
            # Try loading as a traced model first (as exported by train.py)
            self.model = torch.jit.load(model_path, map_location=self.device)
            print(f"Loaded traced model from {model_path}")
        except Exception as e:
            print(f"Failed to load as traced model: {e}")
            raise e

        self.model.eval()

    def __call__(self, state_wrapper: PyspielStateWrapper) -> tuple[PPD[int], float]:
        state = state_wrapper.state

        # 1. Convert to observation tensor
        # Every game now feeds a flat observation tensor to the model.
        # The model network (if traced correctly) reshapes it internally,
        # but for consistency we feed (1, obs_flat_size).
        obs = state.observation_tensor()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).view(
            1, self.obs_flat_size
        )

        # 2. Model Inference
        with torch.no_grad():
            policy_logits, value = self.model(obs_tensor)

        # 3. Process Policy
        policy_probs = policy_logits.cpu().numpy()[0]
        v = value.cpu().item()

        # 4. Mask legal moves and re-normalize
        legal_actions = state.legal_actions()
        mask = np.zeros_like(policy_probs)
        mask[legal_actions] = 1.0

        masked_policy = policy_probs * mask
        sum_p = np.sum(masked_policy)
        if sum_p > 0:
            masked_policy /= sum_p
        else:
            # Fallback to uniform if model gives 0 to all legal moves
            masked_policy[legal_actions] = 1.0 / len(legal_actions)

        policy_dict = {action: float(masked_policy[action]) for action in legal_actions}

        return policy_dict, v
