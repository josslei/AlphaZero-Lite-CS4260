from .connect_four import ConnectFourCNN

MODEL_REGISTRY = {
    "ConnectFourCNN": ConnectFourCNN,
}


def get_model(name: str, params):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    # If the model constructor doesn't take params, we call it without them
    # For now, ConnectFourCNN takes no args.
    return MODEL_REGISTRY[name]()
