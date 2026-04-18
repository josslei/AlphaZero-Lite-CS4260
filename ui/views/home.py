import flet as ft

def HomeView(page: ft.Page):
    def go_connect_four(e):
        page.go("/connect_four")

    def go_backgammon(e):
        page.go("/backgammon")

    return ft.View(
        route="/",
        appbar=ft.AppBar(
            title=ft.Text("AlphaZero Games"),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            center_title=True,
        ),
        controls=[
            ft.Column(
                [
                    ft.Text("Choose a Game", size=30, weight=ft.FontWeight.BOLD),
                    ft.Button("Connect Four", on_click=go_connect_four, width=200),
                    ft.Button("Backgammon", on_click=go_backgammon, width=200),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True
            )
        ],
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
