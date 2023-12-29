import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def my_l1_norm(a, b):
    return np.sum(np.maximum(np.abs(a.astype(np.float32) - b.astype(np.float32)), 60)) / np.prod(a.shape)

def pre_comp_transform(frame, w, h):
    frame = cv2.resize(frame, (w//8, h//8))
    frame = cv2.GaussianBlur(frame, (13, 13), 0)
    return frame

def coarse_to_fine_clip_partitioning(video_file_path, l1_threshold=63):

    cap = cv2.VideoCapture(video_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    downscale_factor = 8
    downscale_width = width // downscale_factor
    downscale_height = height // downscale_factor

    coarse_breaks = []

    ret, prev = cap.read()
    prev = pre_comp_transform(prev, width, height)

    print("detecting coarse breaks")
    for i in tqdm(range(int(fps), frame_count, int(fps))):  # skip every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = pre_comp_transform(frame, width, height)
        if my_l1_norm(frame, prev) > l1_threshold:
            coarse_breaks.append(i)  # in this second, there's assumed to be a clip break

        prev = frame
        if i > 40000:
            break

    fine_breaks = []
    break_values = []
    # not every coarse clip break will contain a fine clip break
    print("refining breaks")
    for coarse_break in tqdm(coarse_breaks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, coarse_break)
        ret, prev = cap.read()
        prev = pre_comp_transform(prev, width, height)
        for i in range(coarse_break, coarse_break + int(fps)):
            # examine frame by frame within this
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = pre_comp_transform(frame, width, height)
            l1 = my_l1_norm(frame, prev)
            if l1 > l1_threshold:
                fine_breaks.append(i)
                break_values.append(l1)
                break  # assume that there's only one fine break in a coarse break
    # TODO save clip breaks
    np.save("breaks.npy", np.array(fine_breaks))
    np.save("break_values.npy", np.array(break_values))

def visualize_clip_breaks_detected(video_file_path, break_file_path):
    # TODO make a matplotlib visual, 3 x 2
    breaks = np.load(break_file_path)
    break_vals = np.load("break_values.npy")
    cap = cv2.VideoCapture(video_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        plt.imshow(np.abs(pre_comp_transform(frame, width, height).astype(np.float32) - pre_comp_transform(prev, width, height).astype(np.float32))/255)
        print(break_vals[i])
        plt.show()

    pass

if __name__ == '__main__':
    coarse_to_fine_clip_partitioning("videos/NHL Oct.02_2023 PS Montreal Canadiens - Toronto Maple Leafs.mp4")
    visualize_clip_breaks_detected("videos/NHL Oct.02_2023 PS Montreal Canadiens - Toronto Maple Leafs.mp4", "breaks.npy")

