import os
import random
from os.path import exists

import matplotlib.pyplot as plt
from deep_utils import StringUtils
from joblib import Parallel, delayed
from moviepy import VideoFileClip
import os
import subprocess
import zipfile
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
from moviepy import VideoFileClip
from datasets import load_dataset

# Function to visualize random video frames
def visualize_random_videos(videos_dir: str, num_videos: int = 8) -> None:
    # Get all video files with .mp4 extension
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

    # Randomly sample 'num_videos' videos
    random_videos = random.sample(video_files, num_videos)

    # Create a subplot to display the video frames
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()  # Flatten the axes for easy indexing

    # Loop through the selected videos and display their first frame
    for i, video_file in enumerate(random_videos):
        video_path = os.path.join(videos_dir, video_file)

        # Load the video and extract the first 2 seconds for preview
        clip = VideoFileClip(video_path).subclip(0, 2)

        # Get the first frame of the video
        frame = clip.get_frame(0)

        # Display the frame on the subplot
        axes[i].imshow(frame)
        axes[i].axis('off')  # Hide the axes for cleaner visualization
        axes[i].set_title(f"Video {i + 1}")  # Set the title with the video number

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def download_kaggle_dataset(dataset_name: str, download_dir: str) -> None:
    """
    Downloads a dataset from Kaggle and saves it to the specified directory.

    :param dataset_name: The name of the Kaggle dataset (e.g., 'vishnutheepb/msrvtt').
    :param download_dir: Directory where the dataset will be saved.
    """
    # Make sure the directory exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Download dataset using Kaggle CLI
    print(f"Downloading Kaggle dataset: {dataset_name}...")
    command = f"kaggle datasets download {dataset_name} -p {download_dir}"
    subprocess.run(command, shell=True, check=True)
    print(f"Dataset {dataset_name} downloaded to {download_dir}")


def unzip_file(zip_path: str, extract_dir: str) -> None:
    """
    Unzips a .zip file into the specified directory.

    :param zip_path: Path to the zip file.
    :param extract_dir: Directory to extract files to.
    """
    print(f"Unzipping file: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Files extracted to {extract_dir}")


def download_hf_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Downloads a dataset from HuggingFace.

    :param dataset_name: The name of the dataset on HuggingFace (e.g., 'AlexZigma/msr-vtt').
    :return: A DataFrame containing the dataset.
    """
    print(f"Downloading HuggingFace dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset)
    print(f"HuggingFace dataset {dataset_name} loaded successfully.")
    return df


def convert_video_to_gif(video_path: str, gif_path: str, size: tuple = (64, 64), num_frames: int = 10) -> None:
    """
    Converts a video file (MP4) to a GIF with specified size and number of frames.

    :param video_path: Path to the input video (MP4).
    :param gif_path: Path to save the output GIF.
    :param size: Desired size for the GIF (default is 64x64).
    :param num_frames: The number of frames to sample for the GIF (default is 10).
    """
    try:
        # Load the video file
        clip = VideoFileClip(video_path)

        # Resize the video to the desired size
        clip = clip.resized(height=size[1], width=size[0])
        
        # Sample frames evenly from the video and convert to GIF
        clip = clip.subclipped(0, clip.duration).resized(size).with_fps(clip.fps).with_duration(clip.duration / num_frames)

        clip.write_gif(gif_path)

        print(f"Converted {video_path} to GIF and saved as {gif_path}")
    except Exception as e:
        StringUtils.print(f"Error converting video {video_path} to GIF: {e}")


def create_training_data(df: pd.DataFrame, videos_dir: str, output_dir: str, size: tuple = (64, 64), num_frames: int = 10) -> None:
    """
    Creates a training folder containing GIFs and corresponding caption text files.

    :param df: DataFrame containing video data.
    :param videos_dir: Directory where videos are stored.
    :param output_dir: Directory where the training data will be saved.
    :param size: Desired size for the GIF (default is 64x64).
    :param num_frames: The number of frames to sample for the GIF (default is 10).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Starting the conversion of videos to GIFs and creating caption text files...")

    gif_data = []

    # Use tqdm to show a progress bar while iterating over the rows of the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Videos", ncols=100):
        video_id = row['video_id']
        caption = row['caption']
        
        # Define paths
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        gif_path = os.path.join(output_dir, f"{video_id}.gif")
        caption_path = os.path.join(output_dir, f"{video_id}.txt")
        if not exists(gif_path):
            gif_data.append((video_path, gif_path, size, num_frames))
        # Convert video to GIF with size and frame limit
        if not exists(caption_path):
            # Save the caption in a text file
            with open(caption_path, 'w') as caption_file:
                caption_file.write(caption)

    Parallel(n_jobs=10)(delayed(convert_video_to_gif)(*data) for data in gif_data)
    print(f"Training data successfully created in {output_dir}")


def main():
    # Step 1: Download the Kaggle dataset
    kaggle_dataset_name = 'vishnutheepb/msrvtt'
    download_dir = './msrvtt_data'
    # download_kaggle_dataset(kaggle_dataset_name, download_dir)

    # Step 2: Unzip the Kaggle dataset
    zip_file_path = os.path.join(download_dir, 'msrvtt.zip')
    unzip_dir = os.path.join(download_dir, 'msrvtt')
    unzip_file(zip_file_path, unzip_dir)

    # Step 3: Define the path to the TrainValVideo directory where the videos are located
    videos_dir = os.path.join(unzip_dir, 'TrainValVideo')

    # Step 4: Download the HuggingFace MSR-VTT dataset
    hf_dataset_name = 'AlexZigma/msr-vtt'
    df = download_hf_dataset(hf_dataset_name)

    # Step 5: Create a training folder
    basename = os.path.basename(os.getcwd())
    output_dir = "../training_data" if basename == "data_generation" else "./training_data" if basename == "text2video-from-scratch" else os.path.abspath("training_data")
    
    create_training_data(df, videos_dir, output_dir, size=(64, 64), num_frames=10)

    visualize_random_videos(videos_dir)  # Display 8 random videos' frames

if __name__ == "__main__":
    main()
