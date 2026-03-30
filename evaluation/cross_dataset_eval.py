def cross_dataset_eval(model, loaders, device):
    results = {}

    for name, loader in loaders.items():
        preds, labels = evaluate(model, loader, device)
        results[name] = (preds, labels)

    return results