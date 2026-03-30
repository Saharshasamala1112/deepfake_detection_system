def preprocess_input(input_type, path):
    if input_type == "image":
        return load_image(path)
    elif input_type == "video":
        return load_video(path)
    elif input_type == "audio":
        return load_audio(path)