import flet as ft
from typing import Callable, List, Tuple


class MoveSelector(ft.Container):
    """
    A reusable component for selecting moves from a list of legal actions.
    Useful for games with complex action spaces like Backgammon.
    """

    def __init__(
        self,
        on_move_selected: Callable[[int], None],
        title: str = "Legal Moves",
        width: int = 200,
        bgcolor: str = ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius: int = 0,
        padding: int = 10,
    ):
        self.on_move_selected = on_move_selected
        self.move_list = ft.Column(
            scroll=ft.ScrollMode.ADAPTIVE,
            expand=True,
            spacing=5,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        super().__init__(
            content=ft.Column([ft.Text(title, weight=ft.FontWeight.BOLD), self.move_list]),
            width=width,
            bgcolor=bgcolor,
            border_radius=border_radius,
            padding=padding,
            expand=True,
            visible=False,
        )

    def update_moves(self, moves: List[Tuple[int, str]]):
        """Updates the list of buttons based on the provided moves."""
        self.move_list.controls.clear()
        for action_id, action_name in moves:
            # Use a closure to capture the specific action_id
            self.move_list.controls.append(
                ft.FilledButton(
                    content=ft.Text(action_name, size=12, no_wrap=True),
                    on_click=lambda e, aid=action_id: self.on_move_selected(aid),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=5),
                        alignment=ft.Alignment.CENTER_LEFT,
                        padding=10,
                    ),
                    tooltip=action_name,
                )
            )
        self.visible = len(moves) > 0
