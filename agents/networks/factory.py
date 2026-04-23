from .connect_four import ConnectFourCNN
from .backgammon import BackgammonCNN

MODEL_REGISTRY = {
    "ConnectFourCNN": ConnectFourCNN,
    "BackgammonCNN": BackgammonCNN,
}


def get_model(name: str, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    params = params or {}
    return MODEL_REGISTRY[name](**params)
