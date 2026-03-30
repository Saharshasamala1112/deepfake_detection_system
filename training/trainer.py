class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, batch):
        self.model.train()

        images, labels = batch

        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images).squeeze()

        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()