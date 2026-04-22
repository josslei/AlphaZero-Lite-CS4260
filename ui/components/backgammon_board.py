import flet as ft
import flet.canvas as cv
from typing import Callable, Any, Coroutine
import math


def BackgammonBoard(
    page: ft.Page, on_point_click: Callable[[int], Coroutine[Any, Any, None]], scale: float = 1.0
):
    """
    Creates the visual Backgammon board.
    """

    # --- Scaled Dimensions ---
    POINT_W = 45 * scale
    POINT_H = 220 * scale
    CHECKER_SIZE = 40 * scale
    BAR_W = 60 * scale
    SPACER_H = 40 * scale
    BOARD_PADDING = 15 * scale
    BOARD_SPACING = 10 * scale
    POINT_SPACING = max(1.0, 2 * scale)
    DICE_SIZE = 40 * scale
    FONT_SIZE_LABEL = (12 * scale) if scale < 0.8 else (14 * scale)
    FONT_SIZE_DICE = 20 * scale

    # Theme Colors
    COLOR_BOARD = ft.Colors.GREEN_800
    COLOR_BAR = ft.Colors.GREEN_900
    COLOR_POINT_DARK = ft.Colors.RED_700
    COLOR_POINT_LIGHT = ft.Colors.WHITE
    COLOR_P1 = ft.Colors.WHITE  # Player 1 Checkers
    COLOR_P2 = ft.Colors.RED  # Player 2 Checkers

    # Store references to the internal Columns where checkers will be drawn
    ui_points: list[ft.Column | None] = [None for _ in range(24)]

    # --- Helper: Create a single point (track) ---
    def create_point_slot(index: int, is_top: bool):
        bg_color = COLOR_POINT_DARK if index % 2 == 0 else COLOR_POINT_LIGHT

        # The column holding the circular checkers
        checker_col = ft.Column(
            spacing=0,
            alignment=ft.MainAxisAlignment.START if is_top else ft.MainAxisAlignment.END,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            width=POINT_W,
            height=POINT_H,
            tight=False,
        )
        ui_points[index] = checker_col

        # Wrap the column in a container to provide internal alignment and fixed size
        checker_container = ft.Container(
            content=checker_col,
            width=POINT_W,
            height=POINT_H,
            # Using explicit Alignment constructor
            alignment=ft.alignment.Alignment(0, -1) if is_top else ft.alignment.Alignment(0, 1),
        )

        # Define triangle path
        if is_top:
            path_data = [
                cv.Path.MoveTo(0, 0),
                cv.Path.LineTo(POINT_W, 0),
                cv.Path.LineTo(POINT_W / 2, POINT_H),
                cv.Path.Close(),
            ]
        else:
            path_data = [
                cv.Path.MoveTo(0, POINT_H),
                cv.Path.LineTo(POINT_W, POINT_H),
                cv.Path.LineTo(POINT_W / 2, 0),
                cv.Path.Close(),
            ]

        return ft.Container(
            width=POINT_W,
            height=POINT_H,
            content=ft.Stack(
                [
                    cv.Canvas(
                        [
                            cv.Path(
                                path_data,
                                paint=ft.Paint(color=bg_color, style=ft.PaintingStyle.FILL),
                            )
                        ],
                        width=POINT_W,
                        height=POINT_H,
                    ),
                    checker_container,
                ]
            ),
            ink=True,
            on_click=lambda e, i=index: page.run_task(on_point_click, i),
        )

    # --- Build the 4 Quadrants ---
    half_width = (6 * POINT_W) + (5 * POINT_SPACING)
    board_half_height = (2 * POINT_H) + SPACER_H

    top_left = ft.Row(
        [create_point_slot(i, True) for i in range(12, 18)], spacing=POINT_SPACING, tight=True
    )
    top_right = ft.Row(
        [create_point_slot(i, True) for i in range(18, 24)], spacing=POINT_SPACING, tight=True
    )
    bottom_left = ft.Row(
        [create_point_slot(i, False) for i in range(11, 5, -1)], spacing=POINT_SPACING, tight=True
    )
    bottom_right = ft.Row(
        [create_point_slot(i, False) for i in range(5, -1, -1)], spacing=POINT_SPACING, tight=True
    )

    left_half = ft.Column(
        [top_left, ft.Container(height=SPACER_H, width=half_width), bottom_left],
        spacing=0,
        tight=True,
        width=half_width,
    )
    right_half = ft.Column(
        [top_right, ft.Container(height=SPACER_H, width=half_width), bottom_right],
        spacing=0,
        tight=True,
        width=half_width,
    )

    # --- Build the Center Bar (Indices 24 & 25) ---
    bar_p1_col = ft.Column(
        spacing=0,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        height=POINT_H,
        width=BAR_W,
    )
    bar_p2_col = ft.Column(
        spacing=0,
        alignment=ft.MainAxisAlignment.END,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        height=POINT_H,
        width=BAR_W,
    )

    center_bar = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=bar_p1_col,
                    height=POINT_H,
                    width=BAR_W,
                    on_click=lambda e: page.run_task(on_point_click, 24),
                    ink=True,
                ),
                ft.Container(height=SPACER_H, width=BAR_W),
                ft.Container(
                    content=bar_p2_col,
                    height=POINT_H,
                    width=BAR_W,
                    on_click=lambda e: page.run_task(on_point_click, 25),
                    ink=True,
                ),
            ],
            spacing=0,
            tight=True,
        ),
        width=BAR_W,
        height=board_half_height,
        bgcolor=COLOR_BAR,
    )

    # --- Build the Bear-Off Trays (Indices 26 & 27) ---
    BEAROFF_EXTRA = 10 * scale
    bearoff_total_w = BAR_W + BEAROFF_EXTRA
    off_p1_col = ft.Column(
        spacing=0,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        height=POINT_H,
        width=BAR_W,
    )
    off_p2_col = ft.Column(
        spacing=0,
        alignment=ft.MainAxisAlignment.END,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        height=POINT_H,
        width=BAR_W,
    )

    bear_off_tray = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=off_p1_col,
                    height=POINT_H,
                    on_click=lambda e: page.run_task(on_point_click, 26),
                    ink=True,
                ),
                ft.Container(height=SPACER_H),
                ft.Container(
                    content=off_p2_col,
                    height=POINT_H,
                    on_click=lambda e: page.run_task(on_point_click, 27),
                    ink=True,
                ),
            ],
            spacing=0,
            tight=True,
        ),
        width=bearoff_total_w,
        height=board_half_height,
        bgcolor=COLOR_BAR,
        border=ft.border.only(left=ft.border.BorderSide(max(1, 2 * scale), ft.Colors.BLACK54)),
        padding=ft.padding.only(right=BEAROFF_EXTRA),
    )

    # --- Compute exact board width from child dimensions ---
    content_width = (2 * half_width) + BAR_W + bearoff_total_w + (3 * BOARD_SPACING)
    total_board_width = content_width + (2 * BOARD_PADDING)

    # --- Wrap it all together ---
    board_ui = ft.Column(
        [
            ft.Container(
                content=ft.Row(
                    controls=[left_half, center_bar, right_half, bear_off_tray],
                    spacing=BOARD_SPACING,
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    tight=True,
                ),
                bgcolor=COLOR_BOARD,
                padding=BOARD_PADDING,
                border_radius=10 * scale,
                width=total_board_width,
                height=board_half_height + (2 * BOARD_PADDING),
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=10 * scale,
                    color=ft.Colors.with_opacity(0.3, ft.Colors.SHADOW),
                ),
            ),
            # Dice Display
            ft.Container(
                content=ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10 * scale,
                    tight=True,
                    controls=[
                        ft.Container(
                            content=ft.Text(
                                "",
                                size=FONT_SIZE_DICE,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.BLACK,
                            ),
                            width=DICE_SIZE,
                            height=DICE_SIZE,
                            bgcolor=ft.Colors.WHITE,
                            border_radius=5 * scale,
                            alignment=ft.alignment.Alignment.CENTER,
                            visible=False,
                            border=ft.border.all(1, ft.Colors.BLACK38),
                        ),
                        ft.Container(
                            content=ft.Text(
                                "",
                                size=FONT_SIZE_DICE,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.BLACK,
                            ),
                            width=DICE_SIZE,
                            height=DICE_SIZE,
                            bgcolor=ft.Colors.WHITE,
                            border_radius=5 * scale,
                            alignment=ft.alignment.Alignment.CENTER,
                            visible=False,
                            border=ft.border.all(1, ft.Colors.BLACK38),
                        ),
                    ],
                ),
                padding=10 * scale,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
        width=total_board_width,
    )

    def draw_checkers(container: ft.Column, count: int):
        container.controls.clear()
        if count == 0:
            return
        is_p1 = count > 0
        checker_color = COLOR_P1 if is_p1 else COLOR_P2
        text_color = ft.Colors.BLACK if is_p1 else ft.Colors.WHITE
        num_checkers = abs(count)
        display_num = min(num_checkers, 5)
        for c in range(display_num):
            is_last = c == display_num - 1
            label = str(num_checkers) if (is_last and num_checkers > 5) else ""
            container.controls.append(
                ft.Container(
                    width=CHECKER_SIZE,
                    height=CHECKER_SIZE,
                    border_radius=CHECKER_SIZE / 2,
                    bgcolor=checker_color,
                    border=ft.border.all(1, ft.Colors.BLACK38),
                    alignment=ft.alignment.Alignment.CENTER,
                    content=(
                        ft.Text(
                            label, color=text_color, weight=ft.FontWeight.BOLD, size=FONT_SIZE_LABEL
                        )
                        if label
                        else None
                    ),
                )
            )

    def update_board(
        points: list[int],
        bar_p1: int = 0,
        bar_p2: int = 0,
        off_p1: int = 0,
        off_p2: int = 0,
        dice: list[int] | None = None,
    ):
        for i in range(24):
            col = ui_points[i]
            if col is not None:
                draw_checkers(col, points[i])
        draw_checkers(bar_p1_col, bar_p1)
        draw_checkers(bar_p2_col, -bar_p2)
        draw_checkers(off_p1_col, off_p1)
        draw_checkers(off_p2_col, -off_p2)

        # Update Dice
        dice_container = board_ui.controls[1]
        assert isinstance(dice_container, ft.Container)
        dice_row = dice_container.content
        assert isinstance(dice_row, ft.Row)
        d0 = dice_row.controls[0]
        d1 = dice_row.controls[1]
        assert isinstance(d0, ft.Container) and isinstance(d1, ft.Container)
        if dice and len(dice) >= 2 and dice[0] > 0:
            d0.content.value = str(dice[0])  # type: ignore[union-attr]
            d0.visible = True
            d1.content.value = str(dice[1])  # type: ignore[union-attr]
            d1.visible = True
        else:
            d0.visible = False
            d1.visible = False

    return board_ui, update_board
