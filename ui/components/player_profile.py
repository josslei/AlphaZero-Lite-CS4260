import flet as ft
from typing import Callable, Optional, Any


class PlayerProfile(ft.Container):
    def __init__(
        self,
        default_name: str,
        default_icon: Any,
        on_name_change: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self.default_name = default_name
        self.on_name_change = on_name_change

        # 1. Container Style
        self.height = 80
        self.padding = ft.padding.all(10)
        self.border_radius = 8
        self.bgcolor = ft.Colors.TRANSPARENT
        self.border = ft.border.all(1, ft.Colors.OUTLINE_VARIANT)

        # 2. UI Components
        self.icon_img = ft.Icon(default_icon, size=30)

        self.name_text = ft.Text(
            self.default_name,
            size=16,
            weight=ft.FontWeight.W_500,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.name_input = ft.TextField(
            value=self.default_name,
            text_size=14,
            height=30,
            width=150,
            content_padding=5,
            visible=False,
            on_submit=self._save_name,
            on_blur=self._save_name,
        )

        self.edit_btn = ft.IconButton(
            icon=ft.Icons.EDIT_OUTLINED,
            icon_size=16,
            padding=0,
            on_click=self._edit_name,
            tooltip="Edit Name",
        )

        self.indicator_dot = ft.Container(
            width=8,
            height=8,
            bgcolor=ft.Colors.GREEN,
            border_radius=4,
        )
        self.turn_text = ft.Text("Your Turn", size=10)

        # Turn indicator row - Fixed Height to prevent shifting
        self.turn_row = ft.Row(
            [self.indicator_dot, self.turn_text], spacing=5, height=15, opacity=0.0  # Fixed height
        )

        # 3. Layout Structure: [ Icon ] | [ Column: (Name+Edit), (Turn) ]
        self.content = ft.Row(
            [
                # Left: Icon
                ft.Container(
                    content=self.icon_img,
                    alignment=ft.Alignment(0, 0),
                ),
                # Right: Content Column
                ft.Column(
                    [
                        # Row 1: Name and Edit Button - Wrapped in fixed-height container
                        ft.Container(
                            content=ft.Row(
                                [self.name_text, self.name_input, self.edit_btn],
                                spacing=5,
                                alignment=ft.MainAxisAlignment.START,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            height=35,  # Fixed height to absorb edit button/input differences
                            alignment=ft.Alignment(-1, 0),
                        ),
                        # Row 2: Turn indicator
                        self.turn_row,
                    ],
                    spacing=0,
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )

    async def _edit_name(self, e):
        self.name_text.visible = False
        self.edit_btn.visible = False
        self.name_input.visible = True
        self.name_input.value = self.name_text.value
        await self.name_input.focus()
        self.update()

    async def _save_name(self, e):
        if self.name_input.visible:
            new_name = self.name_input.value.strip()
            if new_name:
                self.current_name = new_name
                self.name_text.value = new_name
                if self.on_name_change:
                    self.on_name_change(new_name)

            self.name_text.visible = True
            self.edit_btn.visible = True
            self.name_input.visible = False
            self.update()

    def update_profile(self, name: str, icon: Any, is_human: bool):
        self.name_text.value = name
        self.icon_img.icon = icon
        self.edit_btn.visible = is_human
        self.name_text.visible = True
        self.name_input.visible = False

    def visible_indicator(self, visible: bool):
        self.turn_row.opacity = 1.0 if visible else 0.0
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST if visible else ft.Colors.TRANSPARENT
        self.border = (
            ft.border.all(2, ft.Colors.PRIMARY)
            if visible
            else ft.border.all(1, ft.Colors.OUTLINE_VARIANT)
        )
