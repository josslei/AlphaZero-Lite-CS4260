from typing import Any

class SelfPlayEngine:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        num_threads: int,
        num_iters: int,
        temperature: float,
        c_puct: float = 1.0,
    ) -> None: ...
    def generate_games(
        self, num_games: int, game_name: str = "connect_four"
    ) -> list[list[tuple[list[float], list[float], float]]]: ...
