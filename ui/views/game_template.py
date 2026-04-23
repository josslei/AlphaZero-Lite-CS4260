import flet as ft
from typing import Any, Callable, Type
from core.match_manager import MatchManager, HumanPlayer, AIPlayer, GameMode
from core.ai_agent import RandomPolicyAgent, AlphaZeroAgent
from components.side_panel import GameSidePanel
from components.player_profile import PlayerProfile
from components.move_selector import MoveSelector


def CreateGameView(
    page: ft.Page,
    route: str,
    title: str,
    game_name: str,
    engine_class: Type[Any],
    board_factory: Callable[[ft.Page, Callable[[Any], Any]], Any],
    p1_global: Any,
    p2_global: Any,
    agent_class: Type[Any] = RandomPolicyAgent,
    use_move_selector: bool = False,
):
    """
    A general factory for creating game views (Connect Four, Backgammon, etc.)
    """
    engine = engine_class()

    # Initialize agent
    if agent_class == AlphaZeroAgent:
        ai_agent = AlphaZeroAgent(game_name=game_name)
    else:
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

    # 1. Setup MatchManager
    def update_board_ui():
        # Update turn indicator
        current_p = engine.get_current_player() - 1
        p1_profile.visible_indicator(current_p == 0)
        p2_profile.visible_indicator(current_p == 1)

        # General assumption: engine has a get_board_grid() or similar
        data = engine.get_board_grid()
        if isinstance(data, dict):
            # For games with complex boards (e.g. Backgammon),
            # the engine returns a dict of arguments for the updater.
            # We remove extra keys like 'dice' if the updater doesn't take them,
            # but BackgammonBoard.update_board might not take 'dice'.
            # Let's check BackgammonBoard signature.
            import inspect

            sig = inspect.signature(board_updater)
            filtered_data = {k: v for k, v in data.items() if k in sig.parameters}
            board_updater(**filtered_data)
        else:
            board_updater(data)

        update_move_panel()
        page.update()

    manager = MatchManager(engine, on_update=update_board_ui, page=page)
    manager.on_game_over = on_game_over
    manager.on_ai_thinking = toggle_board_input

    # 2. Setup Profiles with persistence sync
    def on_p1_name_change(new_name):
        p1_global.name = new_name

    def on_p2_name_change(new_name):
        p2_global.name = new_name

    p1_profile = PlayerProfile(
        p1_global.get_display_name(), p1_global.icon, on_name_change=on_p1_name_change
    )
    p2_profile = PlayerProfile(
        p2_global.get_display_name(), p2_global.icon, on_name_change=on_p2_name_change
    )

    # 2.5 Setup Move Selection Panel (For complex games like Backgammon)
    def on_move_submitted(aid):
        h1.set_move(aid)
        h2.set_move(aid)

    move_selector = MoveSelector(on_move_selected=on_move_submitted)

    def update_move_panel():
        current_p = engine.get_current_player() - 1
        is_human_turn = not isinstance(manager.players.get(current_p), AIPlayer)

        if is_human_turn and hasattr(engine, "get_legal_moves_with_names"):
            moves = engine.get_legal_moves_with_names()
            move_selector.update_moves(moves)
        else:
            move_selector.visible = False

    def handle_iters_change(new_iters):
        if isinstance(ai_agent, AlphaZeroAgent):
            ai_agent.update_iters(new_iters)

    # 3. Setup Side Panel
    side_panel = GameSidePanel(
        page,
        on_restart=lambda e: start_game_based_on_mode(match_state["mode"]),
        on_mode_change=lambda mode: handle_mode_change(mode),
        on_iters_change=handle_iters_change,
        initial_iters=getattr(ai_agent, "num_iters", 100),
    )

    def start_game_based_on_mode(mode: GameMode):
        # In Human vs AI, force the human (player 0) to go first
        if mode == GameMode.HUMAN_VS_AI:
            engine.reset(force_first_player=0)
        else:
            engine.reset()

        # P1 logic
        if mode == GameMode.HUMAN_VS_HUMAN:
            p1_name = p1_global.name if p1_global.name else "Human Player 1"
        else:
            p1_name = p1_global.name if p1_global.name else "Human Player"

        p1_profile.update_profile(name=p1_name, icon=ft.Icons.PERSON, is_human=True)

        # P2 logic
        if mode == GameMode.HUMAN_VS_HUMAN:
            p2_name = p2_global.name if p2_global.name else "Human Player 2"
            p2_icon = ft.Icons.PERSON
            p2_is_human = True
        else:
            p2_name = "AlphaZero"  # AI name is fixed unless we want it editable
            p2_icon = ft.Icons.SMART_TOY
            p2_is_human = False

        p2_profile.update_profile(name=p2_name, icon=p2_icon, is_human=p2_is_human)

        update_board_ui()
        page.update()

        if mode == GameMode.HUMAN_VS_HUMAN:
            manager.start_game(h1, h2)
        elif mode == GameMode.HUMAN_VS_AI:
            manager.start_game(h1, ai)
        page.update()

    # 4. Setup Board with Human Input Routing
    async def route_click_to_human(action):
        h1.set_move(action)
        h2.set_move(action)

    board_ui, board_updater = board_factory(page, route_click_to_human)

    # 5. Handle UI interactions
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
                    side_panel,
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            # Use ft.Container for top padding, then P2
                            ft.Container(content=p2_profile, alignment=ft.Alignment(-1, -1)),
                            # The main board area expands to fill all available space
                            ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Container(
                                            content=board_ui,
                                            padding=20,
                                        ),
                                        bgcolor=ft.Colors.SURFACE_BRIGHT,
                                        border_radius=10,
                                        expand=True,
                                        alignment=ft.alignment.Alignment.CENTER,
                                    ),
                                    # Move Panel on the right
                                    ft.Container(
                                        content=move_selector,
                                        width=200,  # Fixed width for the list
                                        expand=False,
                                        visible=use_move_selector,
                                    ),
                                ],
                                expand=True,
                                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                            ),
                            # P1 at the bottom
                            ft.Container(content=p1_profile, alignment=ft.Alignment(-1, 1)),
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
