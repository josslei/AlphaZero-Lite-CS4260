import flet as ft

# Import all of our modularized views
from views.home import HomeView
from views.connect_four import ConnectFourView
from views.backgammon import BackgammonView

async def main(page: ft.Page):
    page.title = "AlphaZero Games"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1000
    page.window.height = 600
    page.window.resizable = False

    def route_change(e):
        print(f"Route changed: {page.route}")
        page.views.clear()
        
        # Always add HomeView as the base underlying view
        page.views.append(HomeView(page))
        
        # Append the specific game view on top if navigating to them
        if page.route == "/connect_four":
            page.views.append(ConnectFourView(page))
        elif page.route == "/backgammon":
            page.views.append(BackgammonView(page))
            
        page.update()

    async def view_pop(e):
        # Prevent popping the last view, which would cause an IndexError
        if len(page.views) > 1:
            page.views.pop()
            top_view = page.views[-1]
            await page.push_route(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    
    # Initialize navigation manually to avoid the startup blank screen
    route_change(None)

    page.window.visible = True
    page.update()

if __name__ == "__main__":
    ft.run(main=main, view=ft.AppView.FLET_APP_HIDDEN)