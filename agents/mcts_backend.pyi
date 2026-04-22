from typing import Any

class SelfPlayEngine:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        obs_flat_size: int,
        num_threads: int,
        num_iters: int,
        temperature: float,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_fp16: bool = False,
        use_undo: bool = False,
    ) -> None: ...
    def generate_games(
        self, num_games: int, game_name: str = "connect_four"
    ) -> list[list[tuple[list[float], list[float], float]]]: ...
