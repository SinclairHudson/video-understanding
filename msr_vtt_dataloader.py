import cv2
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

# Example usage:
if __name__ == '__main__':
    video_path = "/media/sinclair/datasets/MSRVTT/videos/all/video0.mp4"
    proportions = [0, 0.25, 0.5, 0.75, 1.0]
    returned_frames = get_video_frames(video_path, proportions)

    for i, frame in enumerate(returned_frames):
        frame.save("frame_{}.jpg".format(i))

