import flet as ft
from typing import Callable, Any

def GameSidePanel(page: ft.Page, on_restart: Callable[[Any], None]):
    
    def mode_changed(e):
        # You can eventually link this to your AlphaZero backend state
        print(f"Play mode changed to: {e.control.value}")
        
    # Explicitly type the controls list to avoid variance issues with ft.Control
    panel_controls: list[ft.Control] = [
        ft.Text("Game Settings", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(height=20),
        
        # 1. Play Mode Selection
        ft.Text("Play Mode:", weight=ft.FontWeight.W_500, size=14),
        ft.Dropdown(
            options=[
                ft.dropdown.Option("Human vs. Human"),
                ft.dropdown.Option("Human vs. AI"),
            ],
            value="Human vs. Human", # Default value
            on_select=mode_changed,
            width=200,
            text_size=14,
        ),
        
        ft.Container(height=10), # Spacer
        
        # 2. Restart/Reset Game Button
        ft.Button(
            "Restart Game", 
            icon=ft.Icons.RESTART_ALT,
            on_click=on_restart,
            width=200,
            style=ft.ButtonStyle(
                color=ft.Colors.ON_ERROR,
                bgcolor=ft.Colors.ERROR,
            ),
        ),
    ]

    return ft.Container(
        width=240, # Fixed width for the side panel
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        padding=20,
        border_radius=10,
        content=ft.Column(
            controls=panel_controls,
            alignment=ft.MainAxisAlignment.START,
        )
    )
