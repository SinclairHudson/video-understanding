import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_clip_breaks(source_video_path, break_file_path):
    # 3 x 1 grid of 
    breaks = np.load(break_file_path)

    cap = cv2.VideoCapture(source_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(0, len(breaks)//6):

        for j in range(6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, breaks[(i*6)+j]-1)
            ret, prev = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, breaks[(i*6)+j])
            ret, frame = cap.read()
            # concatenate these two frames so they are side by side
            frame = np.concatenate((prev, frame), axis=1)
            # matplotlib 3x1 grid, plot the frame
            plt.subplot(3, 2, j+1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    pass
    plt.show

if __name__ == '__main__':
    show_clip_breaks("videos/France v Argentina 2018 FIFA World Cup Full Match.mp4", "soccer_breaks.npy")
