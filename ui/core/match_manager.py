import asyncio
from enum import Enum, auto
from typing import Any, Callable

import flet as ft
from core.ai_agent import Agent


class GameMode(Enum):
    HUMAN_VS_HUMAN = auto()
    HUMAN_VS_AI = auto()
    AI_VS_AI = auto()


class Player:
    """General player strategy interface."""

    async def get_move(self, state: Any) -> Any:
        """Called when it is this player's turn to provide an action."""
        raise NotImplementedError

    def inform_move(self, state: Any, is_my_move: bool):
        """
        Called after ANY move is finalized in the game.
        Useful for resetting move futures or updating AI state.
        """
        pass


class HumanPlayer(Player):
    def __init__(self):
        self.move_future: asyncio.Future[Any] | None = None

    async def get_move(self, state: Any) -> Any:
        # Create a new future and wait for the UI to set its result
        self.move_future = asyncio.Future()
        return await self.move_future

    def set_move(self, action: Any):
        # Result set by the UI event (e.g., column click)
        if self.move_future and not self.move_future.done():
            self.move_future.set_result(action)


class AIPlayer(Player):
    def __init__(self, agent: Agent):
        self.agent = agent

    async def get_move(self, state: Any) -> Any:
        # Run agent's decision logic in a background thread
        return await asyncio.to_thread(self.agent.get_best_move)

    def inform_move(self, state: Any, is_my_move: bool):
        # General hook for AI synchronization/pondering
        self.agent.update_state(state)


class MatchManager:
    """
    A general, game-agnostic Orchestrator for matches.
    Operates on any 'engine' that follows the OpenSpiel-like state pattern.
    """

    def __init__(self, engine: Any, on_update: Callable[[], None], page: ft.Page):
        self.engine = engine
        self.on_update = on_update  # UI-specific rendering callback
        self.page = page
        self.players: dict[int, Player] = {}
        self.current_task: asyncio.Task | None = None
        self.on_game_over: Callable[[Any], None] | None = None
        self.on_ai_thinking: Callable[[bool], None] | None = None

    def start_game(self, p1_strategy: Player, p2_strategy: Player):
        """Starts the game loop as a background task."""
        self.players = {0: p1_strategy, 1: p2_strategy}
        # Use existing engine instance to maintain external references
        self.engine.reset()

        if self.current_task:
            self.current_task.cancel()

        self.current_task = asyncio.create_task(self._match_loop())

    async def _match_loop(self):
        """The core loop that drives the game until terminal."""
        while not self.engine.is_game_over():
            # Identify current player
            p_id = self.engine.state.current_player()
            current_player = self.players.get(p_id)

            if not current_player:
                raise ValueError(f"No Player strategy assigned for player ID: {p_id}")

            # Notify UI if AI is thinking
            if self.on_ai_thinking:
                self.on_ai_thinking(isinstance(current_player, AIPlayer))

            # Get move from the strategy (Human or AI)
            action = await current_player.get_move(self.engine.state)

            # Apply action to engine
            if action is not None:
                self.engine.state.apply_action(action)

                # Inform all players of the move
                for pid, p in self.players.items():
                    p.inform_move(self.engine.state, is_my_move=(pid == p_id))

                # Trigger UI update
                if self.on_update:
                    self.on_update()
                self.page.update()

                # Small sleep to allow UI to render human move before AI starts thinking
                await asyncio.sleep(0.1)

        # Loop ended, game is over
        if self.on_game_over:
            self.on_game_over(self.engine.get_winner())
