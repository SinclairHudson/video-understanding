import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from msr_vtt_dataloader import get_video_frames
from moviepy.editor import VideoFileClip

def my_l1_norm(a, b):
    l1 = np.abs(a.astype(np.float32) - b.astype(np.float32))
    l1[l1 < 20] = 0  # don't want small changes to contribute to the sum
    l1[l1 > 90] = 90 # don't want some pixels to contribute too much
    return np.sum(l1) / np.prod(l1.shape), l1

def pre_comp_transform(frame, w, h):
    frame = cv2.resize(frame, (w//8, h//8))
    frame = cv2.GaussianBlur(frame, (13, 13), 0)
    return frame

def coarse_to_fine_clip_partitioning(name, video_file_path, l1_threshold=25):
    cap = cv2.VideoCapture(video_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    coarse_breaks = []

    ret, prev = cap.read()
    prev = pre_comp_transform(prev, width, height)

    print("detecting coarse breaks")
    for i in tqdm(range(int(fps), frame_count, int(fps))):  # skip every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = pre_comp_transform(frame, width, height)
        l1, _ = my_l1_norm(frame, prev)
        if l1 > l1_threshold:
            coarse_breaks.append(i)  # in this second, there's assumed to be a clip break

        prev = frame

    # break values are the LEADING frame
    fine_breaks = []
    break_values = []
    # not every coarse clip break will contain a fine clip break
    print("refining breaks")
    for coarse_break in tqdm(coarse_breaks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, coarse_break - int(fps))
        ret, prev = cap.read()
        prev = pre_comp_transform(prev, width, height)
        max_found = 0
        max_found_frame = 0
        for i in range(coarse_break - int(fps) + 1, coarse_break):
            # examine frame by frame within this
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = pre_comp_transform(frame, width, height)
            l1, _ = my_l1_norm(frame, prev)
            prev = frame  # update prev
            if l1 > max_found:
                max_found = l1
                max_found_frame = i

        if max_found > l1_threshold:
            fine_breaks.append(max_found_frame)
            break_values.append(max_found)
    np.save(f"{name}_breaks.npy", np.array(fine_breaks))
    np.save(f"{name}_break_values.npy", np.array(break_values))

def visualize_clip_breaks_detected(video_file_path, break_file_path):
    breaks = np.load(break_file_path)
    break_vals = np.load("break_values.npy")
    cap = cv2.VideoCapture(video_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for i, break_frame in enumerate(breaks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, break_frame-1)
        ret, prev = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, break_frame)
        ret, frame = cap.read()
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 3)
        plt.imshow(np.abs(frame.astype(np.float32) - prev.astype(np.float32))/255)
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(pre_comp_transform(prev, width, height), cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(pre_comp_transform(frame, width, height), cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 6)
        plt.imshow(my_l1_norm(pre_comp_transform(frame, width, height), pre_comp_transform(prev, width, height))[1]/255)
        print(break_vals[i])
        plt.show()

    pass

def break_into_clips(name, source_file, breaks_file):
    """
    Break source video into multiple different videos, each corresponding to a clip.
    Clips are split based on frame numbers in the breaks file.
    """
    breaks = np.load(breaks_file)
    full_video = VideoFileClip(source_file)
    cap = cv2.VideoCapture(source_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    subclips = []

    num_breaks = len(breaks)
    for i in tqdm(range(num_breaks-1)):
        subclips.append(full_video.subclip(breaks[i]/fps, breaks[i+1]/fps))

    os.mkdir(f"{name}_tmp")
    for i, subclip in enumerate(tqdm(subclips)):
        subclip.write_videofile(f"{name}_tmp/clip_{i}.mp4")

def describe_clips(name: str, clips_path):
    from llava_handler import call_llava
    prompt = "Please describe what is going on in this image."
    df = pd.DataFrame(columns=["clip_file", "description"])
    output_file = f"descriptions/{name}_clip_descriptions.csv"
    for i, clip in enumerate(tqdm(os.listdir(clips_path))):
        frames = get_video_frames(f"{clips_path}/{clip}", proportions=[0, 0.5, 1.0])
        descriptions = []
        for frame in frames:
            description = call_llava(prompt, [frame])
            descriptions.append(description)
        df.loc[len(df)] = [clip, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv(output_file, index=False)
    df.to_csv(output_file, index=False)



if __name__ == '__main__':
    coarse_to_fine_clip_partitioning("soccer", "videos/France v Argentina 2018 FIFA World Cup Full Match.mp4")
    # visualize_clip_breaks_detected("videos/NHL Oct.02_2023 PS Montreal Canadiens - Toronto Maple Leafs.mp4", "breaks.npy")
    break_into_clips("soccer", "videos/France v Argentina 2018 FIFA World Cup Full Match.mp4", "soccer_breaks.npy")
    describe_clips("soccer", "soccer_tmp")

