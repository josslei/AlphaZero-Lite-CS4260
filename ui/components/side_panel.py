import flet as ft
from typing import Callable, Any
from core.match_manager import GameMode


def GameSidePanel(
    page: ft.Page,
    on_restart: Callable[[Any], None],
    on_mode_change: Callable[[GameMode], None] | None = None,
    on_iters_change: Callable[[int], None] | None = None,
    initial_iters: int = 100,
):

    # Map for Enum <-> String display
    MODE_MAP = {
        GameMode.HUMAN_VS_HUMAN: "Human vs. Human",
        GameMode.HUMAN_VS_AI: "Human vs. AI",
    }
    # Inverse map to get Enum from the Dropdown string value
    STR_TO_MODE = {v: k for k, v in MODE_MAP.items()}

    def iters_changed(e):
        if on_iters_change:
            val = e.control.value
            if val == "":
                return
            if val.isdigit():
                on_iters_change(int(val))

    # AI Config Section (MCTS iterations)
    iters_input = ft.TextField(
        label="MCTS Iterations",
        value=str(initial_iters),
        on_change=iters_changed,
        width=210,
        text_size=14,
        input_filter=ft.NumbersOnlyInputFilter(),
        keyboard_type=ft.KeyboardType.NUMBER,
        visible=True,
    )

    def mode_changed(e):
        selected_str = e.control.value
        mode_enum = STR_TO_MODE.get(selected_str)

        # Toggle iterations input visibility
        iters_input.visible = mode_enum == GameMode.HUMAN_VS_AI
        iters_input.update()

        if on_mode_change and mode_enum:
            on_mode_change(mode_enum)
        print(f"Play mode changed to: {selected_str} ({mode_enum})")

    panel_controls: list[ft.Control] = [
        ft.Text("Game Settings", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(height=20),
        # Play Mode Selection
        ft.Text("Play Mode:", weight=ft.FontWeight.W_500, size=14),
        ft.Dropdown(
            options=[
                ft.dropdown.Option(MODE_MAP[GameMode.HUMAN_VS_HUMAN]),
                ft.dropdown.Option(MODE_MAP[GameMode.HUMAN_VS_AI]),
            ],
            value=MODE_MAP[GameMode.HUMAN_VS_AI],
            on_select=mode_changed,
            width=210,
            text_size=14,
        ),
        ft.Container(height=10),
        iters_input,
        ft.Column(expand=True, controls=[]),  # Spacer to push button down
        # Restart/Reset Game Button
        ft.ElevatedButton(
            "Restart Game",
            icon=ft.Icons.RESTART_ALT,
            on_click=on_restart,
            width=210,
            style=ft.ButtonStyle(
                color=ft.Colors.ON_ERROR,
                bgcolor=ft.Colors.ERROR,
            ),
        ),
    ]

    return ft.Container(
        width=250,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        padding=20,
        border_radius=10,
        content=ft.Column(
            controls=panel_controls,
            alignment=ft.MainAxisAlignment.START,
        ),
    )
