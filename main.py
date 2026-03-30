from utils.device import get_device
from models.multimodal_model import MultiModalModel

def main():
    device = get_device()
    model = MultiModalModel().to(device)
    print("Model Ready on", device)

if __name__ == "__main__":
    main()