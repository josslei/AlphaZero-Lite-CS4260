import flet as ft
from typing import Any, Callable, Type
from core.match_manager import MatchManager, HumanPlayer, AIPlayer, GameMode
from core.ai_agent import RandomPolicyAgent
from components.side_panel import GameSidePanel

def CreateGameView(
    page: ft.Page,
    route: str,
    title: str,
    engine_class: Type[Any],
    board_factory: Callable[[ft.Page, Callable[[Any], Any]], Any],
    agent_class: Type[Any] = RandomPolicyAgent
):
    """
    A general factory for creating game views (Connect Four, Backgammon, etc.)
    """
    engine = engine_class()
    ai_agent = agent_class()
    
    # Standard players
    h1 = HumanPlayer()
    h2 = HumanPlayer()
    ai = AIPlayer(ai_agent)

    match_state = {"mode": GameMode.HUMAN_VS_AI}

    def on_game_over(winner):
        print(f"{title} Match Finished! Winner: {winner}")
        page.update()

    def toggle_board_input(is_ai_turn):
        board_ui.disabled = is_ai_turn
        page.update()

    def update_board_ui():
        # General assumption: engine has a get_board_grid() or similar
        board_updater(engine.get_board_grid())

    # 1. Setup MatchManager
    manager = MatchManager(engine, on_update=update_board_ui, page=page)
    manager.on_game_over = on_game_over
    manager.on_ai_thinking = toggle_board_input

    def start_game_based_on_mode(mode: GameMode):
        engine.reset()
        update_board_ui()
        if mode == GameMode.HUMAN_VS_HUMAN:
            manager.start_match({0: h1, 1: h2})
        elif mode == GameMode.HUMAN_VS_AI:
            manager.start_match({0: h1, 1: ai})
        page.update()

    # 2. Setup Board with Human Input Routing
    async def route_click_to_human(action):
        h1.handle_click(action)
        h2.handle_click(action)

    board_ui, board_updater = board_factory(page, route_click_to_human)

    # 3. Handle UI interactions
    def handle_restart(e):
        start_game_based_on_mode(match_state["mode"])

    def handle_mode_change(new_mode: GameMode):
        match_state["mode"] = new_mode
        start_game_based_on_mode(new_mode)

    # Initial start
    start_game_based_on_mode(GameMode.HUMAN_VS_AI)

    return ft.View(
        route=route,
        appbar=ft.AppBar(title=ft.Text(title), bgcolor=ft.Colors.SURFACE_CONTAINER),
        controls=[
            ft.Row(
                [
                    GameSidePanel(page, on_restart=handle_restart, on_mode_change=handle_mode_change),
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            ft.Container(
                                content=board_ui,
                                bgcolor=ft.Colors.SURFACE_BRIGHT,
                                border_radius=10,
                                padding=40,
                                expand=True,
                                alignment=ft.alignment.Alignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True,
                    ),
                ],
                expand=True,
            )
        ],
    )
