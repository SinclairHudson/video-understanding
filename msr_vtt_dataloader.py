import cv2
import os
import pandas as pd
from typing import List
from tqdm import tqdm
from PIL import Image

def get_video_frames(video_path, proportions=[0, 0.25, 0.5, 0.75, 1.0]):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    frames = []
    for proportion in proportions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((total_frames - 1) * proportion))
        ret, f = cap.read()
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

    # Release the video capture object
    cap.release()

    return frames

def get_video_frames_based_on_L1(video_path, new_frame_threshold = 180) -> List[Image.Image]:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    to_keep_frames = []
    # Read the first frame
    ret, f = cap.read()
    to_keep_frames.append(f)

    # Read the rest of the frames
    while ret:
        ret, f = cap.read()
        if ret and cv2.norm(f, to_keep_frames[-1], cv2.NORM_L1) / (width * height) > new_frame_threshold:
            to_keep_frames.append(f)

    # Release the video capture object
    cap.release()
    print(len(to_keep_frames)/total_frames)
    print(len(to_keep_frames))
    print(total_frames)
    print("------------------")

    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in to_keep_frames]


# Example usage:
if __name__ == '__main__':
    datasets_path = "/media/sinclair/datasets/MSRVTT/videos/all/"
    subset_df = pd.read_csv("/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    subset = subset_df["video_id"].tolist()
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames_based_on_L1(os.path.join(datasets_path, file)+".mp4")

