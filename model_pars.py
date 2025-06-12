import torch

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = total_params * 4 / (1024 ** 2)  # Assuming 32-bit floats (4 bytes per parameter)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Model size (MB): {model_size:.2f}")

