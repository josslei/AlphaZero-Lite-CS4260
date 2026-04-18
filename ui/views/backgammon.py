import flet as ft
from components.side_panel import GameSidePanel

def BackgammonView(page: ft.Page):
    def handle_restart(e):
        print("Restarting Backgammon game...")

    return ft.View(
        route="/backgammon",
        appbar=ft.AppBar(
            title=ft.Text("Backgammon"),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
        ),
        controls=[
            ft.Row(
                [
                    # Side Panel
                    GameSidePanel(page, on_restart=handle_restart),
                    
                    # Main Game Area Placeholder
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            ft.Container(
                                content=ft.Text("Game Board Placeholder", color=ft.Colors.ON_SURFACE_VARIANT),
                                bgcolor=ft.Colors.SURFACE_BRIGHT,
                                border_radius=10,
                                padding=100,
                                expand=True,
                                alignment=ft.Alignment(0, 0),
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
