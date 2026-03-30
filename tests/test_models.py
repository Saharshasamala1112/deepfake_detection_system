import torch
from models.multimodal_model import MultiModalModel

def test_model_forward():
    model = MultiModalModel()

    sample = {
        "image": torch.randn(1, 3, 224, 224),
        "video": torch.randn(1, 10, 1792),
        "audio": torch.randn(1, 1, 128, 128),
        "freq": torch.randn(1, 512)
    }

    output = model(
        sample["image"],
        sample["video"],
        sample["audio"],
        sample["freq"]
    )

    assert output.shape == (1, 1)