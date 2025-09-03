#!/usr/bin/env python3
# scripts/make_episode_from_csv.py

import os, csv, ast, json, shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# --------------------------- EDIT THESE --------------------------- #
CSV_PATH         = "/home/shreya/Downloads/brush_aug_26_40_demos/brush_with_20250826_133059/robot_states.csv"
IMAGES_SRC_DIR   = "/home/shreya/Downloads/brush_aug_26_40_demos/brush_with_20250826_133059"  # folder with ~397 frames
OUTPUT_DATA_ROOT = "my_dataset"            # dataset root to create
EPISODE_ID       = "000001"                # episode folder name
CAM_KEY          = "image"             # camera key in dataset
ALIGN_MODE       = "truncate"              # 'truncate' or 'resample'
# ------------------------------------------------------------------ #

def parse_literal(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def iso_to_unix(ts: str) -> float:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts).timestamp()

def list_images(src_dir: str):
    p = Path(src_dir)
    exts = (".jpg", ".jpeg", ".png")
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    return files

def even_resample_indices(T_src: int, T_tgt: int) -> np.ndarray:
    # Map T_src → T_tgt using even spacing
    if T_tgt <= 0:
        raise ValueError("T_tgt must be > 0")
    idx = (np.arange(T_tgt) * (T_src / T_tgt)).astype(int)
    return np.clip(idx, 0, T_src - 1)

def main():
    # 1) Read CSV and build arrays
    times, states, actions = [], [], []

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            obs  = parse_literal(row["observations"])        # dict
            cart = parse_literal(row["cartesian_action"])    # list(7)
            jp   = obs.get("joint_pos")

            if not (isinstance(jp, list) and len(jp) == 7):
                raise ValueError(f"Row {i}: observations['joint_pos'] must be length 7, got {jp}")
            if not (isinstance(cart, list) and len(cart) == 7):
                raise ValueError(f"Row {i}: cartesian_action must be length 7, got {cart}")

            states.append(jp)
            actions.append(cart)
            times.append(iso_to_unix(row["timestamp"]))

    times   = np.asarray(times, dtype=np.float64)
    states  = np.asarray(states, dtype=np.float32)   # (T,7)
    actions = np.asarray(actions, dtype=np.float32)  # (T,7)

    # Sort by timestamp to be safe
    order = np.argsort(times)
    times, states, actions = times[order], states[order], actions[order]

    # 2) Gather images and align lengths
    img_files = list_images(IMAGES_SRC_DIR)
    N_imgs, N_csv = len(img_files), len(times)

    if N_csv == 0:
        raise RuntimeError("No CSV rows parsed.")

    if N_csv == N_imgs:
        idx = np.arange(N_imgs)
    elif ALIGN_MODE == "truncate":
        N = min(N_csv, N_imgs)
        idx = np.arange(N)
        times, states, actions = times[:N], states[:N], actions[:N]
        img_files = img_files[:N]
    elif ALIGN_MODE == "resample":
        idx = even_resample_indices(N_csv, N_imgs)
        times, states, actions = times[idx], states[idx], actions[idx]
        # keep all images
    else:
        raise ValueError("ALIGN_MODE must be 'truncate' or 'resample'")

    T = len(img_files)
    assert len(times) == len(states) == len(actions) == T, "Alignment failed"

    # 3) Build episode folder
    epi_dir  = Path(OUTPUT_DATA_ROOT) / "episodes" / EPISODE_ID
    cam_dir  = epi_dir / "observation" / CAM_KEY
    os.makedirs(cam_dir, exist_ok=True)

    # 4) Copy/rename images to 000000.jpg …
    for i, src in enumerate(img_files):
        dst = cam_dir / f"{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    # 5) Save arrays
    np.save(epi_dir / "action.npy", actions)
    np.save(epi_dir / "timestamps.npy", times)
    np.save(epi_dir / "observation" / "state.npy", states)

    # 6) Compute FPS + write meta.json
    fps_mean = float(1.0 / np.mean(np.diff(times))) if T > 1 else 0.0
    fps_median = float(1.0 / np.median(np.diff(times))) if T > 1 else 0.0
    meta = {
        "version": 1,
        "fps": round(fps_median if fps_median > 0 else fps_mean, 2),
        "description": "UR3e episode (state=joint_pos, action=cartesian+gripper)",
        "cameras": [CAM_KEY],
        "observation_keys": [f"observation/{CAM_KEY}", "observation/state"],
        "action_key": "action"
    }
    with open(Path(OUTPUT_DATA_ROOT) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Episode written to: {epi_dir}")
    print(f"Frames: {T} | fps(mean): {fps_mean:.2f}, fps(median): {fps_median:.2f}")
    print("Saved:",
          epi_dir / "observation" / "state.npy",
          epi_dir / "action.npy",
          epi_dir / "timestamps.npy")

if __name__ == "__main__":
    main()
