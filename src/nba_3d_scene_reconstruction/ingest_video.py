import cv2
import os
from nba_3d_scene_reconstruction.tracking.schemas import VideoManifest

def processNewVideo(video_path:str) -> VideoManifest:
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise FileNotFoundError("Could not Open File")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    segment_id = "segment_001"
    frames_dir = os.path.join("artifacts", clip_id, segment_id, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = capture.read()

        # Stop looping when the video ends
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"{frame_count:06d}.jpg")
        if not cv2.imwrite(frame_path, frame):
            capture.release()
            raise OSError(f"Could not write frame: {frame_path}")
        frame_count += 1

    capture.release()

    return VideoManifest(
        clip_id=clip_id,
        segment_id=segment_id,
        source_path=video_path,
        frames_dir=frames_dir,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )
