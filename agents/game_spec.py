from abc import ABC, abstractmethod
import numpy as np
from agents.utils import (
    OPENSPIEL_BACKGAMMON_ACTION_SPACE_SIZE,
    OPENSPIEL_BACKGAMMON_OBSERVATION_SIZE,
)


class GameSpec(ABC):
    """Encapsulates all game-specific details for the AlphaZero pipeline."""

    @property
    @abstractmethod
    def game_name(self) -> str:
        """open_spiel game identifier, e.g. 'connect_four'"""

    @property
    @abstractmethod
    def obs_flat_size(self) -> int:
        """Flat observation tensor length from open_spiel, e.g. 126 for connect_four"""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of distinct actions, e.g. 7"""

    def augment(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return augmented (state, pi) pairs. State is flat. Default: no augmentation."""
        return []


class ConnectFourSpec(GameSpec):
    @property
    def game_name(self) -> str:
        return "connect_four"

    @property
    def obs_flat_size(self) -> int:
        return 126  # 3 * 6 * 7

    @property
    def num_actions(self) -> int:
        return 7

    def augment(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        # state is flat (126,) — reshape to apply spatial flip, then flatten back
        board = state.reshape(3, 6, 7)
        flipped = np.flip(board, axis=2).copy().flatten()
        return [(flipped, np.flip(pi).copy())]


class BackgammonSpec(GameSpec):
    @property
    def game_name(self) -> str:
        return "backgammon"

    @property
    def obs_flat_size(self) -> int:
        return OPENSPIEL_BACKGAMMON_OBSERVATION_SIZE

    @property
    def num_actions(self) -> int:
        return OPENSPIEL_BACKGAMMON_ACTION_SPACE_SIZE

    def augment(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


def get_game_spec(name: str) -> GameSpec:
    if name == "connect_four":
        return ConnectFourSpec()
    elif name == "backgammon":
        return BackgammonSpec()
    else:
        raise ValueError(f"Unsupported game: {name}")
