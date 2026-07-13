#!/usr/bin/env python3
"""Render collected .npz levels to MP4 (or GIF) so you can eyeball the gameplay.

Usage:
  python view_npz.py smoke_test                 # render every level_*.npz -> .mp4 next to it
  python view_npz.py smoke_test --gif            # write .gif instead of .mp4
  python view_npz.py smoke_test/level_00000.npz  # a single file
  python view_npz.py smoke_test --fps 10 --scale 6
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np


def render(path, fps, scale, ext):
    data = np.load(path)
    if "observations" not in data:
        # mp4-format collection stores frames in a sibling .mp4, not in the npz
        # (the npz only holds actions/rewards/dones). Nothing to render.
        sibling = os.path.splitext(path)[0] + ".mp4"
        if os.path.exists(sibling):
            print(f"{os.path.basename(path)}: mp4-format run — frames already in {sibling} (open it directly)")
        else:
            print(f"{os.path.basename(path)}: no 'observations' array and no sibling .mp4 — nothing to render")
        return
    obs = data["observations"]  # (T, 64, 64, 3) uint8
    actions = data["actions"] if "actions" in data else None
    rewards = data["rewards"] if "rewards" in data else None
    dones = data["dones"] if "dones" in data else None

    # nearest-neighbour upscale so 64x64 isn't a postage stamp
    if scale > 1:
        frames = obs.repeat(scale, axis=1).repeat(scale, axis=2)
    else:
        frames = obs

    out = os.path.splitext(path)[0] + "." + ext
    if ext == "mp4":
        imageio.mimwrite(out, frames, fps=fps, quality=8)
    else:
        imageio.mimwrite(out, frames, fps=fps)

    stats = f"{os.path.basename(path)}: {obs.shape[0]} frames {obs.shape[1]}x{obs.shape[2]}"
    if rewards is not None:
        stats += f", reward sum={rewards.sum():.1f}"
    if dones is not None:
        stats += f", episode ends={int(dones.sum())}"
    print(f"{stats}  ->  {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a directory of level_*.npz, or a single .npz file")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=int, default=4, help="integer upscale factor (64x64 -> 256x256 at 4)")
    ap.add_argument("--gif", action="store_true", help="write .gif instead of .mp4")
    args = ap.parse_args()

    ext = "gif" if args.gif else "mp4"
    if os.path.isdir(args.target):
        paths = sorted(glob.glob(os.path.join(args.target, "*.npz")))
    else:
        paths = [args.target]

    if not paths:
        raise SystemExit(f"No .npz files found under {args.target}")

    for p in paths:
        render(p, args.fps, args.scale, ext)


if __name__ == "__main__":
    main()
