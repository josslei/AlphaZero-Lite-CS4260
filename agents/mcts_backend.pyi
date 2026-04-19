from typing import Any, Dict

class CppMCTS:
    def __init__(
        self,
        model_path: str,
        num_iters: int,
        temperature: float,
        num_threads: int = 4,
        batch_size: int = 8,
        c_puct: float = 1.0,
    ) -> None: ...

    def search(self, s_init: Any) -> Dict[Any, float]: ...
