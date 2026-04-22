import pyspiel
import numpy as np


class BackgammonEngine:
    def __init__(self):
        self.game = pyspiel.load_game("backgammon")
        self.state = self.game.new_initial_state()
        self._ensure_not_chance()

    def _ensure_not_chance(self):
        """Automatically apply chance actions (dice rolls)."""
        while self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            # In a real game, this might be random, but for now we just pick the first or use a fixed seed if needed.
            # However, for the UI, we should probably let it be random.
            action, prob = outcomes[np.random.choice(len(outcomes))]
            self.state.apply_action(action)

    def reset(self, force_first_player: int | None = None):
        """
        Reset the game. If force_first_player is set (0 or 1), re-roll
        the opening dice until that player is assigned the first turn.
        This is equivalent to that player winning the opening dice roll.
        """
        self.state = self.game.new_initial_state()
        self._ensure_not_chance()
        if force_first_player is not None:
            # Re-roll until the desired player goes first (max 100 attempts to avoid infinite loop)
            for _ in range(100):
                if self.state.current_player() == force_first_player:
                    break
                self.state = self.game.new_initial_state()
                self._ensure_not_chance()

    def get_legal_moves(self):
        return self.state.legal_actions()

    def get_legal_moves_with_names(self):
        """Returns a list of (action_id, action_name) tuples with friendlier descriptions."""
        actions = self.state.legal_actions()
        res = []
        for a in actions:
            raw_name = self.state.action_to_string(a)
            # OpenSpiel format: "ID - 24/22 13/10"
            if " - " in raw_name:
                parts = raw_name.split(" - ", 1)
                move_str = parts[1]
            else:
                move_str = raw_name

            # Format "13/11" as "13 -> 11" for better readability
            # but keep "Bar/22" as "Bar -> 22"
            friendly_parts = []
            for m in move_str.split(" "):
                if "/" in m:
                    # Handle multiple jumps like 24/22/21
                    steps = m.split("/")
                    friendly_parts.append(" -> ".join(steps))
                else:
                    friendly_parts.append(m)

            friendly_name = " & ".join(friendly_parts)
            res.append((a, friendly_name))
        return res

    def apply_move(self, action):
        if action in self.get_legal_moves():
            self.state.apply_action(action)
            self._ensure_not_chance()
            return True
        return False

    def is_game_over(self):
        return self.state.is_terminal()

    def get_winner(self):
        if not self.is_game_over():
            return None
        returns = self.state.returns()
        if returns[0] > 0:
            return 1
        elif returns[1] > 0:
            return 2
        return None

    def get_current_player(self):
        # OpenSpiel 0 -> 1, 1 -> 2
        return self.state.current_player() + 1

    def get_board_grid(self):
        """
        Returns a dictionary containing the board state for the BackgammonBoard component.
        """
        obs = self.state.observation_tensor(0)
        # obs layout (200 values):
        # 0-95: P0 checkers on 24 points (4 values per point: 1, 2, 3, count-3)
        # 96-191: P1 checkers on 24 points
        # 192: Bar P0
        # 193: Score P0
        # 194: Cur player == 0
        # 195: Bar P1
        # 196: Score P1
        # 197: Cur player == 1
        # 198: Dice 0
        # 199: Dice 1

        points = [0] * 24
        for i in range(24):
            # P0
            p0_vals = obs[i * 4 : i * 4 + 4]
            p0_count = 0
            if p0_vals[0] > 0:
                p0_count = 1
            elif p0_vals[1] > 0:
                p0_count = 2
            elif p0_vals[2] > 0:
                p0_count = 3
            elif p0_vals[3] > 0:
                p0_count = int(p0_vals[3] + 3)

            # P1
            p1_vals = obs[96 + i * 4 : 96 + i * 4 + 4]
            p1_count = 0
            if p1_vals[0] > 0:
                p1_count = 1
            elif p1_vals[1] > 0:
                p1_count = 2
            elif p1_vals[2] > 0:
                p1_count = 3
            elif p1_vals[3] > 0:
                p1_count = int(p1_vals[3] + 3)

            if p0_count > 0:
                points[i] = p0_count
            elif p1_count > 0:
                points[i] = -p1_count

        return {
            "points": points,
            "bar_p1": int(obs[192]),
            "bar_p2": int(obs[195]),
            "off_p1": int(obs[193]),
            "off_p2": int(obs[196]),
            "dice": [int(obs[198]), int(obs[199])],
        }
