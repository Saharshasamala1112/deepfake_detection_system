import torch
from torch.utils.data import DataLoader

from models.multimodal_model import MultiModalModel
from datasets.image_dataset import ImageDataset
from training.trainer import Trainer
from training.train_loop import train_one_epoch
from training.losses import get_loss
from training.optimizer import get_optimizer
from utils.device import get_device


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            preds = (outputs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = get_device()

    model = MultiModalModel().to(device)

    train_dataset = ImageDataset("data/images/Train")
    val_dataset = ImageDataset("data/images/Val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = get_optimizer(model)
    criterion = get_loss()
    trainer = Trainer(model, optimizer, criterion, device)

    for epoch in range(15):  # 🔥 increased epochs
        loss = train_one_epoch(trainer, train_loader)
        acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model trained & saved!")


if __name__ == "__main__":
    main()