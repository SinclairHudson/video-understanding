import cv2
import os
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

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

def get_video_frames_triples(video_path, proportions=[0, 0.5, 1.0], delta=0.5) -> List[List[Image.Image]]:
    """
    delta is the time difference between first, second and third frame in a triple. seconds
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    def clamp_middle_frame(middle_frame):
        return int(max(delta * fps, min(total_frames - (delta * fps), middle_frame)))

    # Read the first frame
    triples = []
    for proportion in proportions:
        tri = []
        ideal_mid_frame = int((total_frames - 1) * proportion)
        mid_frame = clamp_middle_frame(ideal_mid_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, mid_frame - fps * delta)))
        ret, f = cap.read()
        tri.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, f = cap.read()
        tri.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(mid_frame + fps * delta, total_frames - 1)))
        ret, f = cap.read()
        tri.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

        triples.append(tri)

    # Release the video capture object
    cap.release()

    return triples

def get_video_frames_based_on_L1(video_path, new_frame_threshold = 180) -> List[Image.Image]:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in to_keep_frames]

def get_length_seconds(video_path) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps

def get_length_histogram():
    datasets_path = "/media/sinclair/datasets/MSRVTT/videos/all/"
    subset_df = pd.read_csv("/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    lengths = []
    subset = subset_df["video_id"].tolist()
    for file in tqdm(subset):
        video_path = os.path.join(datasets_path, file)+".mp4"
        lengths.append(get_length_seconds(video_path))
    subset_df["length"] = lengths
    subset_df.to_csv("msrvtt_jsfusion_with_lengths.csv", index=False)
    plt.tight_layout()
    plt.hist(lengths, bins=32, range=(0, 32))
    plt.xticks(fontsize=18)  # Set x-axis tick label font size
    plt.yticks(fontsize=18)  # Set y-axis tick label font size
    plt.xlabel("Video Length (seconds)", fontsize=18)
    plt.ylabel("count", fontsize=18)
    plt.show()

def get_number_of_frames_per_clip_L1(threshs=[50, 100, 150, 200, 250]):
    datasets_path = "/media/sinclair/datasets/MSRVTT/videos/all/"
    subset_df = pd.read_csv("/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    subset = subset_df["video_id"].tolist()
    for i, file in enumerate(tqdm(subset)):
        for thresh in threshs:
            video_path = os.path.join(datasets_path, file)+".mp4"
            frames = get_video_frames_based_on_L1(video_path, thresh)
            subset_df.at[i, "frames_"+str(thresh)] = len(frames)
    subset_df.to_csv("msrvtt_jsfusion_with_frames_per_clip.csv", index=False)

def show_frames_per_clip_L1():
    df = pd.read_csv("msrvtt_jsfusion_with_frames_per_clip.csv")
    threshs = [50, 100, 150, 200, 250]
    for thresh in threshs:
        x ,y = np.unique(df["frames_"+str(thresh)],return_counts=True)
        plt.plot(x, y, label="c="+str(thresh))

    plt.tight_layout()
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)
    # plt.yscale('log', base=)
    plt.xlabel("Number of Frames", fontsize=26)
    plt.ylabel("Count", fontsize=26)
    plt.xlim(0, 250)
    plt.ylim(0, 200)
    plt.show()



# Example usage:
if __name__ == '__main__':
    # get_length_histogram()
    # get_number_of_frames_per_clip_L1()
    # get_number_of_frames_per_clip_L1([50, 100, 150, 200, 250])
    show_frames_per_clip_L1()

