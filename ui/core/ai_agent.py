import pyspiel
import time
import random  # Placeholder for the actual AI
from typing import Any


class Agent:
    def __init__(self):
        raise NotImplementedError

    def async_continuous_search(self):
        pass

    def get_best_move(self, think_time_limit=1.0):
        raise NotImplementedError

    def update_state(self, new_state):
        raise NotImplementedError

    def stop(self):
        pass


class RandomPolicyAgent(Agent):
    def __init__(self):
        self.cur_state: Any = None

    def get_best_move(self, think_time_limit=1.0):
        legal_moves = self.cur_state.legal_actions()
        if not legal_moves:
            return None

        # Simulating thinking time so the UI doesn't freeze instantly
        time.sleep(think_time_limit)

        # For now, return a random legal move
        return random.choice(legal_moves)

    def update_state(self, new_state):
        self.cur_state = new_state
