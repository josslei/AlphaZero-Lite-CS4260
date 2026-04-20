import flet as ft
from typing import Callable, Any, Coroutine


def ConnectFourBoard(page: ft.Page, on_column_click: Callable[[int], Coroutine[Any, Any, None]]):
    """
    Creates the visual Connect Four board.
    Returns: (UI_Container, update_function)
    """

    # Define our theme colors
    COLOR_BOARD = ft.Colors.BLUE_700
    COLOR_EMPTY = ft.Colors.SURFACE_BRIGHT
    COLOR_P1 = ft.Colors.RED_500
    COLOR_P2 = ft.Colors.YELLOW_500

    # Create a 6x7 2D list to store references to the UI circle Containers
    ui_slots: list[list[Any]] = [[None for _ in range(7)] for _ in range(6)]

    board_columns: list[ft.Control] = []

    # Build the grid vertically (Column by Column)
    for col_idx in range(7):
        column_slots = []
        for row_idx in range(6):
            # Create a single circular slot
            slot = ft.Container(
                width=40,
                height=40,
                border_radius=20,
                bgcolor=COLOR_EMPTY,
            )
            ui_slots[row_idx][col_idx] = slot
            column_slots.append(slot)

        # Wrap the 6 slots in an ft.Column and make the whole strip clickable
        col_container = ft.Container(
            content=ft.Column(
                controls=column_slots,
                spacing=10,
                tight=True,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=10,
            # Use page.run_task to correctly execute the async move handler
            on_click=lambda e, c=col_idx: page.run_task(on_column_click, c),
            ink=True,
            border_radius=8,
        )
        board_columns.append(col_container)

    # Wrap all 7 columns in the main blue board background
    board_ui = ft.Container(
        content=ft.Row(
            controls=board_columns,
            spacing=0,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        ),
        bgcolor=COLOR_BOARD,
        padding=10,
        border_radius=15,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.3, ft.Colors.SHADOW),
        ),
    )

    # The function we expose to the Controller to sync the Flet UI with OpenSpiel
    def update_grid(grid_data: list[list[int]]):
        for row_idx in range(6):
            for col_idx in range(7):
                val = grid_data[row_idx][col_idx]
                slot_ui = ui_slots[row_idx][col_idx]

                if slot_ui:
                    if val == 1:
                        slot_ui.bgcolor = COLOR_P1
                    elif val == 2:
                        slot_ui.bgcolor = COLOR_P2
                    else:
                        slot_ui.bgcolor = COLOR_EMPTY

    return board_ui, update_grid
