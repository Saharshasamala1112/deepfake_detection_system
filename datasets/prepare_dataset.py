from datasets.image_dataset import ImageDataset
from datasets.video_dataset import VideoDataset
from datasets.audio_dataset import AudioDataset
from datasets.multimodal_dataset import MultiModalDataset


def build_dataset(config):
    image_ds = ImageDataset(config["image_path"])
    video_ds = VideoDataset(config["video_path"])
    audio_ds = AudioDataset(config["audio_path"])

    dataset = MultiModalDataset(image_ds, video_ds, audio_ds)
    return dataset