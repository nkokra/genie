from pathlib import Path
import numpy as np
import imageio.v2 as imageio

import torch.nn.functional as F
import torch.nn as nn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data_collection" / "coinrun_data"


def load_frames(level_mp4: Path) -> np.ndarray:
    """Load one level's frames as a (T, 64, 64, 3) uint8 array.
    """
    with imageio.get_reader(level_mp4) as reader:
        return np.stack([frame for frame in reader]).astype(np.uint8)

# One .npz per level in both formats, so this enumerates every level.
level_files = sorted(DATA_DIR.glob("*.mp4"))
if not level_files:
    raise FileNotFoundError(f"No level_*.mp4 found under {DATA_DIR}")

# Load the first level's frames as a numpy array.
frames = load_frames(level_files[0])
print(f"{level_files[0].name}: frames {frames.shape} {frames.dtype}")

