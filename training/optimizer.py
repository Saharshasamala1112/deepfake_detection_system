import torch.optim as optim

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-4)