import flet as ft
from core.connect_four_engine import ConnectFourEngine
from components.connect_four_board import ConnectFourBoard
from views.game_template import CreateGameView
from core.ai_agent import AlphaZeroAgent


def ConnectFourView(page: ft.Page):
    return CreateGameView(
        page=page,
        route="/connect_four",
        title="Connect Four",
        game_name="connect_four",
        engine_class=ConnectFourEngine,
        board_factory=ConnectFourBoard,
        agent_class=AlphaZeroAgent,
    )
