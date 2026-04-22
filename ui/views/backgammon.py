import flet as ft
from core.backgammon_engine import BackgammonEngine
from components.backgammon_board import BackgammonBoard
from views.game_template import CreateGameView
from core.ai_agent import AlphaZeroAgent
from typing import Any


def BackgammonView(page: ft.Page, p1_global: Any, p2_global: Any):
    return CreateGameView(
        page=page,
        route="/backgammon",
        title="Backgammon",
        game_name="backgammon",
        engine_class=BackgammonEngine,
        board_factory=lambda pg, click: BackgammonBoard(pg, click, scale=0.6),
        p1_global=p1_global,
        p2_global=p2_global,
        agent_class=AlphaZeroAgent,
    )
