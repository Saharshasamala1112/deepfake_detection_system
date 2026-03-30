def full_pipeline(video_path):
    frames = decode_video(video_path)
    faces = [detect_face(f) for f in frames]
    aligned = [align_face(f) for f in faces]
    filtered = filter_low_quality(aligned)

    audio = extract_audio(video_path)
    spec = compute_spectrogram(audio, 16000)

    return filtered, spec