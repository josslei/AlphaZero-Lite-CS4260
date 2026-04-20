import time
import random
import os
import glob
from typing import Any

from agents.mcts import MCTS
from agents.utils import select_alphazero, State, PPD
from agents.inference import PyspielStateWrapper, AlphaZeroEvaluator


class Agent:
    def __init__(self):
        pass

    def get_best_move(self, state: State, think_time_limit=1.0) -> Any:
        raise NotImplementedError

    def update_state(self, new_state: State):
        raise NotImplementedError

    def stop(self):
        pass


class RandomPolicyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.cur_state: State | None = None

    def get_best_move(self, state: State, think_time_limit=1.0) -> Any:
        self.cur_state = state
        if self.cur_state is None:
            return None
        legal_moves = self.cur_state.legal_actions()
        if not legal_moves:
            return None

        # Simulating thinking time so the UI doesn't freeze instantly
        time.sleep(think_time_limit)

        # For now, return a random legal move
        return random.choice(legal_moves)

    def update_state(self, new_state: State):
        self.cur_state = new_state


class AlphaZeroAgent(Agent):
    def __init__(self, game_name: str, model_path: str | None = None, num_iters: int = 800):
        super().__init__()
        self.cur_state: State | None = None
        self.num_iters = num_iters
        self.game_name = game_name

        if model_path is None:
            model_path = self._find_latest_model()

        if model_path and os.path.exists(model_path):
            self.evaluator = AlphaZeroEvaluator(model_path)
            self.mcts = MCTS(
                select_fn=select_alphazero,
                evaluate_fn=self.evaluator,
                num_iters=self.num_iters,
                temperature=0.0,  # Greedy for inference
            )
            print(f"AlphaZeroAgent initialized with model: {model_path}")
        else:
            print(
                f"AlphaZeroAgent ({self.game_name}): No model found. Falling back to random moves."
            )
            self.mcts = None

    def _find_latest_model(self) -> str | None:
        # Fixed place for each game: outputs/<game_name>/<game_name>.pt
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs"))
        model_path = os.path.join(base_path, self.game_name, f"{self.game_name}.pt")
        return model_path if os.path.exists(model_path) else None

    def get_best_move(self, state: State, think_time_limit=1.0) -> Any:
        self.cur_state = state

        if self.mcts is None:
            # Fallback to random if no model
            time.sleep(think_time_limit)
            return random.choice(self.cur_state.legal_actions())

        start_time = time.time()

        # Wrap the state
        state_wrapper = PyspielStateWrapper(self.cur_state.clone())

        # Run MCTS search
        action_probs = self.mcts.search(state_wrapper)

        # Pick the best action
        if not action_probs:
            return random.choice(self.cur_state.legal_actions())

        # Sort actions by probability for debugging
        sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
        debug_str = ", ".join([f"Col {a}: {p:.1%}" for a, p in sorted_actions[:3]])

        best_action = sorted_actions[0][0]

        # Ensure we spend at least some time thinking for UI feel
        elapsed = time.time() - start_time
        if elapsed < think_time_limit:
            time.sleep(think_time_limit - elapsed)

        return best_action

    def update_state(self, new_state: State):
        self.cur_state = new_state
