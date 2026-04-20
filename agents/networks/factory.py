from .connect_four import ConnectFourCNN

MODEL_REGISTRY = {
    "ConnectFourCNN": ConnectFourCNN,
}


def get_model(name: str, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    params = params or {}
    return MODEL_REGISTRY[name](**params)
