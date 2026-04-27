import random
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Callable, Protocol, Self

import numpy as np

from .utils import PPD, State

try:
    from . import mcts_backend

    USE_CPP = True
except ImportError:
    mcts_backend = None
    USE_CPP = False


class Node[A]:
    def __init__(
        self,
        parent: Self | None = None,
        children: MutableMapping[A, Self] | None = None,
        prior_prob: float = 1.0,
        player_id: int = 0,
    ):
        # Tree node properties
        self.parent = parent
        self.children = children if children is not None else {}
        self.is_expanded = False
        self.player_id = player_id

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
        # We must align the root state past any initial environment chance mechanisms
        self.advance_chance_nodes(s_init)

        root = Node[A](parent=None, prior_prob=1.0, player_id=s_init.current_player())
        root.visit_count = 1

        # Expand root node using eval fn to get initial prior probs
        policy, _ = self.evaluate_fn(s_init)
        self.expand_node(root, s_init, policy)

        for _ in range(self.num_iters):
            cur_node = root
            cur_state = s_init.clone()

            # Step 1: Selection
            while cur_node.is_expanded and (not cur_state.is_terminal()):
                legal_actions = cur_state.legal_actions()
                best_action, next_node = self.select_best_child(
                    cur_node, self.select_fn, legal_actions
                )
                
                cur_p = cur_state.current_player()
                cur_state.apply_action(best_action)
                self.advance_chance_nodes(cur_state)
                cur_node = next_node
                
                if not cur_state.is_terminal():
                    cur_node.player_id = cur_state.current_player()
                else:
                    cur_node.player_id = cur_p

            # Step 2: Expansion & Evaluation
            value = 0.0
            if cur_state.is_terminal():
                if cur_node.player_id < 0:
                    cur_node.player_id = cur_node.parent.player_id if cur_node.parent is not None else 0
                value = cur_state.returns()[cur_node.player_id]
            else:
                policy, value = self.evaluate_fn(cur_state)
                self.expand_node(cur_node, cur_state, policy)
                cur_node.player_id = cur_state.current_player()

            # Step 3: Backpropagation
            self.backpropagate(cur_node, value)

        # Step 4: Policy Generation
        return self.calculate_action_probabilities(root, s_init.legal_actions())

    def advance_chance_nodes(self, state: S) -> None:
        while state.is_chance_node() and not state.is_terminal():
            outcomes = state.chance_outcomes()
            actions = [outcome[0] for outcome in outcomes]
            probs = [outcome[1] for outcome in outcomes]

            # Normalize probabilities to avoid numpy rounding check errors
            probs_arr = np.array(probs, dtype=np.float64)
            probs_arr /= np.sum(probs_arr)

            sampled_action = np.random.choice(actions, p=probs_arr)
            state.apply_action(sampled_action)

    def select_best_child(
        self, node: Node[A], score_function: Callable[[Node, int], float], legal_actions: list[A]
    ) -> tuple[A, Node[A]]:
        best_action = None
        best_child = None
        best_score = -float("inf")

        for action in legal_actions:
            if action not in node.children:
                # Lazy instantiation for open-loop paths not encountered during initial expansion
                node.children[action] = Node(
                    parent=node, prior_prob=1.0 / max(1, len(legal_actions))
                )

            child = node.children[action]
            score = score_function(child, node.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert (
            best_action is not None and best_child is not None
        ), "No valid action found in legal_actions."
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
            
            if cur_node.parent is not None and cur_node.parent.player_id != cur_node.player_id:
                value = -value
                
            cur_node = cur_node.parent

    def calculate_action_probabilities(
        self, root: Node[A], legal_actions: list[A]
    ) -> list[tuple[A, float]]:
        if not root.children:
            return []

        legal_set = set(legal_actions)
        if self.temperature <= 1e-3:
            # Deterministic max visit
            best_action: A | None = None
            max_visits = -1
            for action, child in root.children.items():
                if action not in legal_set:
                    continue
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_action = action

            if best_action is not None:
                return [(best_action, 1.0)]
            return []

        # Temperature scaling
        weights: list[float] = []
        actions: list[A] = []
        for action, child in root.children.items():
            if action not in legal_set:
                continue
            weights.append(child.visit_count ** (1.0 / self.temperature))
            from typing import cast

            actions.append(cast(A, action))

        total_weight = sum(weights)
        if total_weight > 0:
            probs = [w / total_weight for w in weights]
        else:
            probs = [1.0 / len(actions)] * len(actions) if actions else []

        return list(zip(actions, probs))


class SelfPlayEngine:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        obs_flat_size: int,
        num_threads: int,
        num_iters: int,
        temperature: float,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_fp16: bool = False,
        use_undo: bool = False,
    ):
        if not USE_CPP or mcts_backend is None:
            raise RuntimeError("C++ MCTS backend is not available.")

        # Accessing C++ class via getattr to avoid static analysis errors
        engine_cls: type = getattr(mcts_backend, "SelfPlayEngine")
        self.engine = engine_cls(
            model_path,
            batch_size,
            obs_flat_size,
            num_threads,
            num_iters,
            temperature,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            use_fp16,
            use_undo,
        )

    def generate_games(self, num_games: int, game_name: str = "connect_four"):
        return self.engine.generate_games(num_games, game_name)

    def get_metrics(self):
        return self.engine.get_metrics()


class TournamentEngine:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        obs_flat_size: int,
        num_threads: int,
        num_iters: int,
        temperature: float = 0.0,
        c_puct: float = 1.0,
        use_fp16: bool = False,
        use_undo: bool = False,
    ):
        if not USE_CPP or mcts_backend is None:
            raise RuntimeError("C++ MCTS backend is not available.")

        # Accessing C++ class via getattr
        engine_cls: type = getattr(mcts_backend, "TournamentEngine")
        self.engine = engine_cls(
            model_path,
            batch_size,
            obs_flat_size,
            num_threads,
            num_iters,
            temperature,
            c_puct,
            use_fp16,
            use_undo,
        )

    def play_tournament(self, num_games: int, game_name: str = "connect_four", opponent: str = "greedy"):
        return self.engine.play_tournament(num_games, game_name, opponent)
