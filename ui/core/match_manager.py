import flet as ft
import asyncio
from typing import Optional, Any, Callable, Dict
from enum import Enum, auto

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
        'state' is the NEW state resulting from the move.
        Useful for AI pondering or internal state synchronization.
        """
        pass

class HumanPlayer(Player):
    def __init__(self):
        self.move_future: Optional[asyncio.Future[Any]] = None

    async def get_move(self, state: Any) -> Any:
        self.move_future = asyncio.get_running_loop().create_future()
        try:
            return await self.move_future
        finally:
            self.move_future = None

    def handle_click(self, action: Any):
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
        self.on_update = on_update # UI-specific rendering callback
        self.page = page
        self.players: Dict[int, Player] = {}
        self.current_task: Optional[asyncio.Task] = None
        self.on_game_over: Optional[Callable[[Any], None]] = None
        self.on_ai_thinking: Optional[Callable[[bool], None]] = None

    def start_match(self, players: Dict[int, Player]):
        """Starts a match with a mapping of player IDs to Player strategies."""
        self.stop_match()
        self.players = players
        self.current_task = asyncio.create_task(self._match_loop())

    def stop_match(self):
        if self.current_task:
            self.current_task.cancel()
            self.current_task = None

    async def _match_loop(self):
        while not self.engine.is_game_over():
            # Identify current player
            p_id = self.engine.state.current_player()
            current_player = self.players.get(p_id)
            
            if not current_player:
                raise ValueError(f"No Player strategy assigned for player ID: {p_id}")

            # Inform UI about thinking state (e.g. disable input if current is AI)
            if self.on_ai_thinking:
                self.on_ai_thinking(isinstance(current_player, AIPlayer))

            # Request move from the Player strategy
            move = await current_player.get_move(self.engine.state)

            # Apply move to engine
            if self.engine.apply_move(move):
                # Notify ALL players of the NEW state
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