from datasets.image_dataset import ImageDataset
from datasets.video_dataset import VideoDataset
from datasets.audio_dataset import AudioDataset
from datasets.multimodal_dataset import MultiModalDataset

def prepare_dataset(config):
    image_ds = ImageDataset(config["image"])
    video_ds = VideoDataset(config["video"])
    audio_ds = AudioDataset(config["audio"])

    return MultiModalDataset(image_ds, video_ds, audio_ds)