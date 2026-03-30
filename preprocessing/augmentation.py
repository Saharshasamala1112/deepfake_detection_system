import torchvision.transforms as T

def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])