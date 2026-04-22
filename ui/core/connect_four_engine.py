import pyspiel


class ConnectFourEngine:
    def __init__(self):
        """Initializes the OpenSpiel Connect Four game and state."""
        self.game = pyspiel.load_game("connect_four")
        self.state = self.game.new_initial_state()

    def reset(self, **kwargs):
        """Resets the game to the starting state."""
        self.state = self.game.new_initial_state()

    def get_legal_moves(self):
        """
        Returns a list of valid column indices (0-6) where a piece can be dropped.
        """
        # In OpenSpiel, actions for connect_four correspond directly to column indices.
        return self.state.legal_actions()

    def apply_move(self, column):
        """
        Attempts to drop a piece into the specified column.
        Returns True if successful, False if the move was invalid.
        """
        if column in self.get_legal_moves():
            self.state.apply_action(column)
            return True
        return False

    def is_game_over(self):
        """Returns True if the game has ended (win, loss, or draw)."""
        return self.state.is_terminal()

    def get_winner(self):
        """
        Returns the winning player index (1 or 2).
        Returns None if the game is still ongoing or ends in a draw.
        """
        if not self.is_game_over():
            return None

        # OpenSpiel returns are a list like [1.0, -1.0] for P1 win.
        returns = self.state.returns()
        if returns[0] == 1.0:
            return 1  # Player 1 (Red)
        elif returns[1] == 1.0:
            return 2  # Player 2 (Yellow)

        return None  # Draw

    def get_current_player(self):
        """
        Returns the current player to move.
        1 for Player 1 (Red), 2 for Player 2 (Yellow).
        """
        # OpenSpiel uses 0 and 1 internally. We map to 1 and 2 for clarity in our UI.
        return self.state.current_player() + 1

    def get_board_grid(self):
        """
        Translates OpenSpiel's internal state into a 2D array for Flet.
        Returns a list of 6 rows, where each row is a list of 7 integers:
        0 (empty), 1 (Player 1), 2 (Player 2).
        """
        grid = []

        # str(self.state) returns a string like:
        # .......
        # .......
        # ...x...
        # ..ox...
        # .ooxx..
        board_str = str(self.state).strip().split("\n")

        for line in board_str:
            # Ensure we only process the 7-character board lines
            if len(line) == 7:
                row = []
                for char in line:
                    if char == "x":
                        row.append(1)
                    elif char == "o":
                        row.append(2)
                    else:
                        row.append(0)
                grid.append(row)

        # Safety fallback: if OpenSpiel's string format ever changes,
        # return a blank 6x7 board so the UI doesn't crash.
        if len(grid) != 6:
            return [[0 for _ in range(7)] for _ in range(6)]

        return grid
