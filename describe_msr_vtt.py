# this file will create a csv that contains textual descriptions for all videos in the MSR-VTT dataset
import os
from tqdm import tqdm

import pandas as pd
from llava_handler import call_llava
from msr_vtt_dataloader import get_video_frames, get_video_frames_based_on_L1, get_video_frames_triples


datasets_path = "/media/sinclair/datasets/MSRVTT/videos/all/"

subset_df = pd.read_csv("/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv")
subset = subset_df["video_id"].tolist()

def experiment_3strat_0temp():
    output_file = "msr-vtt_descriptions_3strat_0temp.csv"
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
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)

def experiment_5strat_0temp():
    output_file = "msr-vtt_descriptions_5strat_0temp.csv"
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
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)

def experiment_3strat_02temp():
    output_file = "msr-vtt_descriptions_3strat_02temp.csv"
    prompt = "Please describe the objects in this image. Be as descriptive as possible. "
    proportions = [0, 0.5, 1]
    df = pd.DataFrame(columns=["video_id", "description"])
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames(os.path.join(datasets_path, file)+".mp4", proportions)
        descriptions = []
        for frame in frames:
            for _ in range(2):
                # go 2 times per image
                description = call_llava(prompt, [frame], temperature=0.2, max_new_tokens=512//2)
                descriptions.append(description)

        df.loc[len(df)] = [file, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)

def experiment_L1_180():
    output_file = "msr-vtt_descriptions_L1180.csv"
    prompt = "Please describe the objects in this image. Be as descriptive as possible. "
    df = pd.DataFrame(columns=["video_id", "description"])
    for i, file in enumerate(tqdm(subset)):
        frames = get_video_frames_based_on_L1(os.path.join(datasets_path, file)+".mp4", new_frame_threshold=180)
        descriptions = []
        for frame in frames:
            description = call_llava(prompt, [frame])
            descriptions.append(description)

        df.loc[len(df)] = [file, " ".join(descriptions)]
        if i % 100 == 0:
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)

def experiment_3strat_triples():
    output_file = "msr-vtt_descriptions_3strat_triples.csv"
    prompt = "These are 3 frames from the same video. Please describe the objects in this video clip."
    proportions = [0, 0.5, 1]
    df = pd.DataFrame(columns=["video_id", "triple_1_desc", "triple_2_desc", "triple_3_desc"])
    for i, file in enumerate(tqdm(subset)):
        triples = get_video_frames_triples(os.path.join(datasets_path, file)+".mp4", proportions)
        descriptions = []
        for triple in triples:
            description = call_llava(prompt, triple)
            descriptions.append(description)
        df.loc[len(df)] = [file].extend(descriptions)
        if i % 100 == 0:
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    experiment_L1_180()
