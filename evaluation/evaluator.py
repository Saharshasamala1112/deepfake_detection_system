import torch

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            out = model(
                batch["image"].to(device),
                batch["video"].to(device),
                batch["audio"].to(device),
                batch["freq"].to(device)
            )

            preds.extend((out > 0.5).cpu().numpy())
            labels.extend(batch["label"].numpy())

    return preds, labels