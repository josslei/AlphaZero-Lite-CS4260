import flet as ft
from core.connect_four_engine import ConnectFourEngine
from components.connect_four_board import ConnectFourBoard
from views.game_template import CreateGameView
from core.ai_agent import AlphaZeroAgent
from typing import Any


def ConnectFourView(page: ft.Page, p1_global: Any, p2_global: Any):
    return CreateGameView(
        page=page,
        route="/connect_four",
        title="Connect Four",
        game_name="connect_four",
        engine_class=ConnectFourEngine,
        board_factory=ConnectFourBoard,
        p1_global=p1_global,
        p2_global=p2_global,
        agent_class=AlphaZeroAgent,
    )
