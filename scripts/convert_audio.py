import os
import subprocess

def convert_video_to_audio(video_path, output_path):
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_path}"
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    video_dir = "data/video"
    audio_dir = "data/audio"

    os.makedirs(audio_dir, exist_ok=True)

    for file in os.listdir(video_dir):
        if file.endswith((".mp4", ".MOV")):
            input_path = os.path.join(video_dir, file)
            output_path = os.path.join(audio_dir, file.replace(".mp4", ".wav"))

            convert_video_to_audio(input_path, output_path)