"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ur5e",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    #modify for my own data format
    
    # Path to your dataset
    dataset_root = Path(data_dir)
    meta_path = dataset_root / "meta.json"
    episodes_dir = dataset_root / "episodes"

    # Optionally, load meta.json if you need any info from it
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    # Track statistics
    total_episodes = 0
    skipped_episodes = 0
    corrupted_images = 0

    # Iterate through each episode directory
    for episode_dir in sorted(episodes_dir.iterdir()):
        if not episode_dir.is_dir():
            continue

        total_episodes += 1
        obs_dir = episode_dir / "observation" / "image"
        robot_state_path = episode_dir / "observation" / "state.npy"
        action_path = episode_dir / "action.npy"
        timestamps_path = episode_dir / "timestamps.npy"

        # Load robot state, actions, and timestamps
        robot_states = np.load(robot_state_path)  # shape: (T, 7)
        actions = np.load(action_path)            # shape: (T, 7)
        timestamps = np.load(timestamps_path)     # shape: (T,)

        # Get sorted list of image files
        image_files = sorted(obs_dir.glob("*.png"))

        # Check all lengths match
        T = len(image_files)
        if not (len(robot_states) == len(actions) == len(timestamps) == T):
            #print lengths
            print(f"Length mismatch in {episode_dir.name}:")
            print(f"  robot_states: {len(robot_states)}")
            print(f"  actions: {len(actions)}")
            print(f"  timestamps: {len(timestamps)}")
            print(f"  images: {T}")
            print(f"Length mismatch in {episode_dir.name}, skipping.")
            skipped_episodes += 1
            continue

        # Add frames to dataset
        episode_valid = True
        valid_frames = []
        
        for t in range(T):
            try:
                # Try to open and validate the image
                with Image.open(image_files[t]) as img:
                    # Verify the image can be loaded completely
                    img.verify()
                
                # Reopen for actual use (verify() closes the file)
                image = np.array(Image.open(image_files[t]))
                
                # Check if image has expected dimensions
                if image.shape != (256, 256, 3):
                    print(f"Warning: Image {image_files[t]} has unexpected shape {image.shape}, reshaping...")
                    # Resize if needed
                    img_pil = Image.fromarray(image) if len(image.shape) == 3 else Image.open(image_files[t])
                    img_pil = img_pil.resize((256, 256))
                    image = np.array(img_pil)
                    
                    # Ensure 3 channels
                    if len(image.shape) == 2:  # Grayscale
                        image = np.stack([image] * 3, axis=-1)
                    elif image.shape[-1] == 4:  # RGBA
                        image = image[..., :3]  # Remove alpha channel
                
                valid_frames.append((t, image))
                
            except (OSError, IOError, Exception) as e:
                print(f"Corrupted/invalid image {image_files[t]}: {e}")
                corrupted_images += 1
                # Skip this frame but continue with the episode
                continue
        
        # Only save episode if we have valid frames
        if valid_frames:
            for t, image in valid_frames:
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": np.zeros_like(image),  # If you don't have wrist images, duplicate or set to zeros
                        "state": robot_states[t],
                        "actions": actions[t],
                    }
                )
            # Save episode, use episode_dir name as task
            dataset.save_episode(task=episode_dir.name)
            print(f"Processed episode {episode_dir.name}: {len(valid_frames)}/{T} valid frames")
        else:
            print(f"Skipping episode {episode_dir.name}: no valid images")
            skipped_episodes += 1

    # Print summary
    print(f"\nConversion Summary:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Successfully processed: {total_episodes - skipped_episodes}")
    print(f"  Skipped episodes: {skipped_episodes}")
    print(f"  Corrupted images: {corrupted_images}")

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
