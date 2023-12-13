# this file will create a csv that contains textual descriptions for all videos in the MSR-VTT dataset
import os
from tqdm import tqdm

import pandas as pd
from llava_handler import call_llava
from msr_vtt_dataloader import get_video_frames


datasets_path = "/media/sinclair/datasets/MSRVTT/videos/all/"

subset_df = pd.read_csv("/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv")
subset = subset_df["video_id"].tolist()

def experiment_3strat_0temp():
    prompt = "Please describe the objects in this image. Be as descriptive as possible. "
    proportions = [0, 0.5, 1]
    df = pd.DataFrame(columns=["video_id", "description"])
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames(os.path.join(datasets_path, file)+".mp4", proportions)
        descriptions = []
        for frame in frames:
            description = call_llava(prompt, [frame])
            descriptions.append(description)

        df.loc[len(df)] = [file, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv("msr-vtt_descriptions_3strat_0temp.csv", index=False)

    df.to_csv("msr-vtt_descriptions_3strat_0temp.csv", index=False)

def experiment_5strat_0temp():
    prompt = "Please describe the objects in this image. Be as descriptive as possible. "
    proportions = [0, 0.25, 0.5, 0.75, 1]
    df = pd.DataFrame(columns=["video_id", "description"])
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames(os.path.join(datasets_path, file)+".mp4", proportions)
        descriptions = []
        for frame in frames:
            description = call_llava(prompt, [frame])
            descriptions.append(description)

        df.loc[len(df)] = [file, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv("msr-vtt_descriptions_5strat_0temp.csv", index=False)

    df.to_csv("msr-vtt_descriptions_5strat_0temp.csv", index=False)

def experiment_3strat_03temp():
    prompt = "Please describe the objects in this image. Be as descriptive as possible. "
    proportions = [0, 0.5, 1]
    df = pd.DataFrame(columns=["video_id", "description"])
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames(os.path.join(datasets_path, file)+".mp4", proportions)
        descriptions = []
        for frame in frames:
            description = call_llava(prompt, [frame], temperature=0.3, max_new_tokens=512/4, num_beams=4)
            descriptions.append(description)

        df.loc[len(df)] = [file, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv("msr-vtt_descriptions_3strat_0temp.csv", index=False)

    df.to_csv("msr-vtt_descriptions_3strat_0temp.csv", index=False)

if __name__ == "__main__":
    experiment_5strat_0temp()
