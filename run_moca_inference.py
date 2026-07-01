#!/usr/bin/env python3
"""
Run C3Net inference on all MoCA-Mask test frames and save per-video binary masks.

Preserves the per-video folder structure expected by the VOS evaluation pipeline:
    <out_dir>/<video_name>/<frame_stem>.png

Usage
-----
  cd /home/marcol01/C3Net
  conda activate c3net
  python run_moca_inference.py \
      --frames_dir /Experiments/marcol01/frames \
      --checkpoint checkpoints/model_best.pth \
      --out_dir /Experiments/marcol01/c3net_masks \
      --device cuda:0

Optional
--------
  --threshold 0.5   # binarisation threshold (default 0.5)
  --batch_size 8    # images per GPU batch (default 4)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

# ── make C3Net importable from its own root ──────────────────────────────────
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from models.c3net import C3Net
from utils.image_processor import CODImageProcessor


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    model = C3Net(config["model"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def infer_frame(
    model: torch.nn.Module,
    processor: CODImageProcessor,
    image_path: Path,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (binary_mask, orig_rgb).
    binary_mask: uint8 (H, W) with values 0/255.
    orig_rgb:    uint8 (H, W, 3) original frame.
    """
    orig = Image.open(image_path).convert("RGB")
    H, W = orig.size[1], orig.size[0]

    processed = processor(str(image_path))
    x = processed.image.to(device).unsqueeze(0)   # (1, 3, 392, 392)

    outputs = model(x)
    seg = outputs["final_mask"]                    # (1, 1, h, w) – sigmoid activated

    seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
    binary = (seg.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return binary, np.array(orig)


_OVERLAY_COLOR = np.array([0, 200, 0], dtype=np.uint8)  # green


def make_overlay(frame_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a green overlay where mask > 0 onto frame_rgb."""
    out = frame_rgb.copy()
    fg = mask > 0
    out[fg] = ((1 - alpha) * out[fg] + alpha * _OVERLAY_COLOR).astype(np.uint8)
    return out


def sorted_frames(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.suffix in _IMG_EXTS]
    paths.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True,
                   help="Root directory containing per-video frame subfolders.")
    p.add_argument("--checkpoint", default=str(_REPO / "checkpoints" / "model_best.pth"),
                   help="Path to C3Net checkpoint (.pth).")
    p.add_argument("--config", default=str(_REPO / "configs" / "default.yaml"),
                   help="Path to C3Net config YAML.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for binary masks.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold (default 0.5).")
    p.add_argument("--overlay_alpha", type=float, default=0.45,
                   help="Opacity of the green overlay (0=invisible, 1=solid). Default 0.45.")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Load config and model
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"[init] Loading model from {args.checkpoint} …")
    model = load_model(args.checkpoint, config, device)

    img_cfg = config["model"]["image_processing"]
    processor = CODImageProcessor(
        target_size=img_cfg["target_size"],
        normalize_mean=tuple(img_cfg["normalize_mean"]),
        normalize_std=tuple(img_cfg["normalize_std"]),
    )
    print("  Model ready.")

    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)

    # Each subdirectory is one video
    video_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    if not video_dirs:
        # Single video mode — frames_root itself contains frames
        video_dirs_and_names = [(frames_root, frames_root.name)]
    else:
        video_dirs_and_names = [(d, d.name) for d in video_dirs]

    print(f"[info] {len(video_dirs_and_names)} video(s) to process.\n")

    for vid_dir, vid_name in video_dirs_and_names:
        frames = sorted_frames(vid_dir)
        if not frames:
            print(f"[skip] {vid_name}: no frames.")
            continue

        out_video_dir = out_root / vid_name
        out_video_dir.mkdir(parents=True, exist_ok=True)

        for fpath in tqdm(frames, desc=vid_name, leave=False):
            out_path = out_video_dir / (fpath.stem + ".png")
            if out_path.exists():
                continue  # resume-friendly

            mask, frame_rgb = infer_frame(model, processor, fpath, device, args.threshold)
            overlay = make_overlay(frame_rgb, mask, alpha=args.overlay_alpha)
            Image.fromarray(overlay).save(str(out_path))

        print(f"[done] {vid_name}  →  {out_video_dir}")

    print(f"\n[done] All videos saved to {out_root}")


if __name__ == "__main__":
    main()
