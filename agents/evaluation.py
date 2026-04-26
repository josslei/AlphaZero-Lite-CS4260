import numpy as np
import pyspiel
from typing import Any, Dict

from agents.inference import PyspielStateWrapper
from agents.mcts import MCTS
from agents.utils import select_alphazero


class BaseAgent:
    """
    Abstract base class for all evaluation benchmark agents.
    """
    def __init__(self, **kwargs):
        pass

    def get_action(self, state: pyspiel.State, player_id: int) -> int:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """
    Uniform random agent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, state: pyspiel.State, player_id: int) -> int:
        legal_actions = state.legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available for RandomAgent.")
        return int(np.random.choice(legal_actions))


class MinimaxAgent(BaseAgent):
    """
    Minimax evaluation agent.
    """
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def get_action(self, state: pyspiel.State, player_id: int) -> int:
        # Minimax left empty per user request for investigation later
        raise NotImplementedError("MinimaxAgent policy not yet implemented.")


class GreedyAgent(BaseAgent):
    """
    Greedy evaluation agent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, state: pyspiel.State, player_id: int) -> int:
        # Greedy left empty per user request for investigation later
        raise NotImplementedError("GreedyAgent policy not yet implemented.")


class AlphaZeroAgent(BaseAgent):
    """
    AlphaZero evaluation agent powered by the specified engine.
    """
    def __init__(self, evaluator: Any, engine: str = "python", mcts_iters: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.mcts_iters = mcts_iters
        self.evaluator = evaluator

        if self.engine == "python":
            self.mcts = MCTS(
                select_fn=select_alphazero,
                evaluate_fn=self.evaluator,
                num_iters=self.mcts_iters,
                temperature=0.0,  # Greedy deterministic selection for evaluation matches
            )
        elif self.engine == "cpp":
            # C++ implementation will be connected here later.
            raise NotImplementedError("C++ Evaluation MCTS not yet exposed to Python")
        else:
            raise ValueError(f"Unknown AlphaZero engine: {self.engine}")

    def get_action(self, state: pyspiel.State, player_id: int) -> int:
        if self.engine == "python":
            state_wrapper = PyspielStateWrapper(state.clone())
            action_probs = self.mcts.search(state_wrapper)
            # action_probs is a single-element list of (action, prob) because temperature=0
            if action_probs:
                return action_probs[0][0]
            
            # Fallback
            legal_actions = state.legal_actions()
            return int(np.random.choice(legal_actions))
        
        raise NotImplementedError


def create_agent(config: Dict, evaluator: Any = None) -> BaseAgent:
    """
    Dynamically creates an agent based on the provided configuration dictionary.
    """
    agent_type = config.get("type", "").lower()
    
    if agent_type == "random":
        return RandomAgent(**config)
    elif agent_type == "minimax":
        return MinimaxAgent(**config)
    elif agent_type == "greedy":
        return GreedyAgent(**config)
    elif agent_type == "alphazero":
        # AlphaZero agent requires the neural network evaluator
        return AlphaZeroAgent(evaluator=evaluator, **config)
    else:
        raise ValueError(f"Unknown agent type requested in evaluation config: {agent_type}")
