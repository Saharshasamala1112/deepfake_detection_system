def train_one_epoch(trainer, dataloader):
    total_loss = 0

    for batch in dataloader:
        loss = trainer.train_step(batch)
        total_loss += loss

    return total_loss / len(dataloader)