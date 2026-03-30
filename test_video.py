from inference.video_predictor import VideoPredictor

vp = VideoPredictor()

pred, score = vp.predict("your_video.mp4")

print(pred, score)