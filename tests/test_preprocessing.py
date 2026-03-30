from preprocessing.video_decoder import decode_video

def test_video_decode():
    frames = decode_video("data/video/1.mp4")
    assert len(frames) > 0