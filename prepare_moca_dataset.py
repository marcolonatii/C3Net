#!/usr/bin/env python3
"""
Prepare MoCA-Mask fine-tuning dataset for C3Net.

Reads frames from --frames_dir and GT masks (every 5 frames) from --masks_dir,
copies only the annotated frames + masks into --out_dir/train/{Imgs,GT},
then generates edge maps with EdgeGenerator into --out_dir/train/Edges.

Usage:
    cd /home/marcol01/C3Net
    python prepare_moca_dataset.py \
        --frames_dir /Experiments/marcol01/frames_train \
        --masks_dir  /Experiments/marcol01/masks_train \
        --out_dir    /Experiments/marcol01/moca_c3net_dataset
"""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.edge_generator import EdgeGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_dir",  required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--edge_width", type=int, default=1)
    args = parser.parse_args()

    frames_root = Path(args.frames_dir)
    masks_root  = Path(args.masks_dir)
    out_root    = Path(args.out_dir)

    imgs_dir  = out_root / "train" / "Imgs"
    gt_dir    = out_root / "train" / "GT"
    edge_dir  = out_root / "train" / "Edges"
    for d in (imgs_dir, gt_dir, edge_dir):
        d.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(p for p in frames_root.iterdir() if p.is_dir())
    copied = 0
    skipped = 0

    for video_dir in tqdm(video_dirs, desc="Videos"):
        video_name = video_dir.name
        mask_video_dir = masks_root / video_name
        if not mask_video_dir.exists():
            skipped += 1
            continue

        # collect GT masks
        mask_paths = sorted(mask_video_dir.glob("*.png"))
        for mask_path in mask_paths:
            stem = mask_path.stem  # e.g. "00000"

            # find matching frame (.jpg or .png)
            frame_path = video_dir / f"{stem}.jpg"
            if not frame_path.exists():
                frame_path = video_dir / f"{stem}.png"
            if not frame_path.exists():
                continue

            unique_stem = f"{video_name.replace('.', '_')}_{stem}"
            shutil.copy2(frame_path, imgs_dir / f"{unique_stem}{frame_path.suffix}")
            shutil.copy2(mask_path,  gt_dir   / f"{unique_stem}.png")
            copied += 1

    print(f"\nCopied {copied} frame/mask pairs ({skipped} videos skipped).")

    # Generate edge maps from GT masks
    gen = EdgeGenerator(edge_width=args.edge_width)
    mask_files = sorted(gt_dir.glob("*.png"))
    print(f"Generating edges for {len(mask_files)} masks ...")
    failed = 0
    for mf in tqdm(mask_files, desc="Edges"):
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed += 1
            continue
        edge, _ = gen.extract_edges(mask, validate=False)
        cv2.imwrite(str(edge_dir / mf.name), edge)

    print(f"Done. Edges saved to {edge_dir}. Failed: {failed}")
    print(f"\nDataset ready at: {out_root}")
    print("Add to configs/default.yaml under training.datasets:")
    print(f"  - \"{out_root}\"")


if __name__ == "__main__":
    main()
