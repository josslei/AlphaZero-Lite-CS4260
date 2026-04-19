import math
import random
import numpy as np
from typing import Any
from copy import deepcopy
from agents.mcts import Node, State, PPD


def select_traditional(child_node: Node, parent_visit_count: int) -> float:
    c_param = 1.414  # Exploration constant (commonly sqrt(2))

    # 1. Force exploration of completely unvisited nodes
    if child_node.visit_count == 0:
        return float("inf")

    # 2. Exploitation term (Expected win rate: Q)
    exploitation = child_node.mean_value

    # 3. Exploration term (Increases if child is rarely visited compared to parent)
    exploration = c_param * math.sqrt(math.log(parent_visit_count) / child_node.visit_count)

    return exploitation + exploration


def select_alphazero(child_node: Node, parent_visit_count: int) -> float:
    c_puct = 1.0  # Exploration constant (can be tuned or decayed over time)

    # 1. Exploitation term (Expected win rate from neural net & searches: Q)
    exploitation = child_node.mean_value

    # 2. Exploration term
    # Notice how prior_prob (P) scales the exploration bonus.
    # The "+ 1" in the denominator prevents division by zero.
    exploration = (
        c_puct
        * child_node.prior_prob
        * (math.sqrt(parent_visit_count) / (1 + child_node.visit_count))
    )

    return exploitation + exploration


def evaluate_traditional[A](state: State[A]) -> tuple[PPD[A], float]:
    # --- 1. Generate Uniform Policy ---
    legal_actions = state.legal_actions()
    if not legal_actions:
        return {}, 0.0

    uniform_prob = 1.0 / len(legal_actions)
    policy = {action: uniform_prob for action in legal_actions}

    # --- 2. Perform Random Rollout to get Value ---
    simulation_state = deepcopy(state)

    while not simulation_state.is_terminal():
        # Pick a completely random action
        actions = simulation_state.legal_actions()
        if not actions:
            break
        random_action = random.choice(actions)
        simulation_state.apply_action(random_action)

    # Get the final game result
    value = simulation_state.rewards()

    return policy, value


class AlphaZeroEvaluator[S, A]:
    def __init__(self, neural_network: Any):
        self.neural_network = neural_network

    def __call__(self, state: S) -> tuple[PPD[A], float]:
        """
        Implementation depends on neural_network and state.to_tensor() which are not yet defined.
        The logic below follows the requested pseudocode structure.
        """
        # 1. Convert game state to a tensor suitable for Neural Net input
        # state_tensor = state.to_tensor()

        # 2. Ask the Neural Network for predictions
        # raw_policy_vector, value = self.neural_network.predict(state_tensor)

        # 3. Mask illegal moves and Normalize
        # legal_actions = state.legal_actions()
        # ... (remaining implementation depends on neural network output format) ...
        return ({}, 0)
