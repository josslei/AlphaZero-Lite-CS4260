import random
import numpy as np
from typing import Optional, Self, Callable, Protocol, Mapping, MutableMapping
from copy import deepcopy

try:
    from . import mcts_backend
    USE_CPP = True
except ImportError:
    print("Warning: C++ MCTS backend not found.")
    USE_CPP = False


type PPD[A] = Mapping[A, float]  # Policy Probability Distribution


class State[A](Protocol):
    def legal_actions(self) -> list[A]: ...
    def is_terminal(self) -> bool: ...
    def rewards(self) -> float: ...
    def apply_action(self, action: A) -> None: ...
    def clone(self) -> Self: ...


class Node[A]:
    def __init__(
        self,
        parent: Optional[Self] = None,
        children: Optional[MutableMapping[A, Self]] = None,
        prior_prob: float = 1.0,
    ):
        # Tree node properties
        self.parent = parent
        self.children = children if children is not None else {}
        self.is_expanded = False

        # Statistics
        self.visit_count: int = 0  # (N) number of times this node was visited
        self.total_value: float = 0  # (W) sum of values from all sub-tree evaluations
        self.mean_value: float = 0.0  # (Q) W / N (Expected reward)
        self.prior_prob: float = (
            prior_prob  # (P) Policy probability from Neural Net (1.0 for Traditional)
        )


class MCTS[S: State, A]:
    def __init__(
        self,
        select_fn: Callable[[Node, int], float],
        evaluate_fn: Callable[[S], tuple[PPD[A], float]],
        num_iters: int,
        temperature: float,
    ):
        self.select_fn = select_fn
        self.evaluate_fn = evaluate_fn
        self.num_iters = num_iters
        self.temperature = temperature

    def search(self, s_init: S):
        root = Node[A](parent=None, prior_prob=1.0)

        # Expand root node using eval fn to get initial prior probs
        policy, _ = self.evaluate_fn(s_init)
        self.expand_node(root, s_init, policy)

        for _ in range(self.num_iters):
            cur_node = root
            cur_state = deepcopy(s_init)

            # Step 1: Selection
            while cur_node.is_expanded and (not cur_state.is_terminal()):
                best_action, next_node = self.select_best_child(cur_node, self.select_fn)
                cur_state.apply_action(best_action)
                cur_node = next_node

            # Step 2: Expansion & Evaluation
            value = 0.0
            if cur_state.is_terminal():
                value = cur_state.rewards()
            else:
                policy, value = self.evaluate_fn(cur_state)
                self.expand_node(cur_node, cur_state, policy)

            # Step 3: Backpropagation
            self.backpropagate(cur_node, value)

        # Step 4: Policy Generation
        return self.calculate_action_probabilities(root)

    def select_best_child(
        self, node: Node[A], score_function: Callable[[Node, int], float]
    ) -> tuple[A, Node[A]]:
        assert node.children, f"Node {node} is marked as expanded but has no children!"
        best_action, best_child = max(
            node.children.items(), key=lambda item: score_function(item[1], node.visit_count)
        )
        return best_action, best_child

    def expand_node(self, node: Node[A], state: S, policy: PPD[A]) -> None:
        legal_actions: list[A] = state.legal_actions()
        for action in legal_actions:
            if action not in node.children:
                prob = policy.get(action, 0.0)
                node.children[action] = Node(parent=node, prior_prob=prob)
        node.is_expanded = True

    def backpropagate(self, node: Node[A], value: float) -> None:
        cur_node: Node | None = node
        while cur_node is not None:
            cur_node.visit_count += 1
            cur_node.total_value += value
            cur_node.mean_value = cur_node.total_value / cur_node.visit_count
            cur_node = cur_node.parent
            value = -value

    def calculate_action_probabilities(self, root: Node[A]) -> Mapping[A, float]:
        if not root.children:
            return {}

        if self.temperature <= 1e-3:
            max_visits = max(child.visit_count for child in root.children.values())
            best_actions = [a for a, c in root.children.items() if c.visit_count == max_visits]
            best_action = random.choice(best_actions)
            return {action: (1.0 if action == best_action else 0.0) for action in root.children}

        actions = list(root.children.keys())
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
        weights = visits ** (1.0 / self.temperature)
        total_weight = np.sum(weights)

        if total_weight == 0:
            probs = np.ones_like(visits) / len(visits)
        else:
            probs = weights / total_weight

        return dict(zip(actions, probs))


class SelfPlayEngine:
    def __init__(self, model_path: str, batch_size: int, num_threads: int, num_iters: int, temperature: float, c_puct: float = 1.0):
        if not USE_CPP:
            raise RuntimeError("C++ MCTS backend is not available.")
        self.engine = mcts_backend.SelfPlayEngine(model_path, batch_size, num_threads, num_iters, temperature, c_puct)
            
    def generate_games(self, num_games: int, game_name: str = "connect_four"):
        return self.engine.generate_games(num_games, game_name)
