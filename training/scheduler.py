import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer):
    return lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)