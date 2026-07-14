from pathlib import Path
import numpy as np
import imageio.v2 as imageio

import torch.nn.functional as F
import torch.nn as nn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data_collection" / "coinrun_data"

def load_frames(level_mp4: Path) -> np.ndarray:
    with imageio.get_reader(level_mp4) as reader:
        return np.stack([frame for frame in reader]).astype(np.float32)

print("Loading data...")
level_files = sorted(DATA_DIR.glob("*.mp4"))
if not level_files:
    raise FileNotFoundError(f"No level_*.mp4 found under {DATA_DIR}")

# Load the first 10 levels' frames into a numpy array.
data = torch.from_numpy(np.stack([load_frames(level_files[i]) for i in range(10)]))
#print(data.shape)
#print(type(data))

patch_size = 4

# Patchify each 64x64x3 frame into 256 patches of 4x4x3 = 48 values.
# reshape splits H and W into (grid, patch); permute groups each patch's pixels
# together; final reshape flattens to (patches, patch_vector).
B, T, H, W, C = data.shape
p = patch_size
data = (
    data.reshape(B, T, H // p, p, W // p, p, C)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(B, T, (H // p) * (W // p), p * p * C)
)
print("Finished loading and shaping data")
#print(data.shape)  # [10, 1000, 256, 48]

embedding_dim = 16 # is this d_model?? Should be 512?
attention_dim = 8
causal_mask = torch.triu(torch.ones(1000, 1000, dtype=bool), diagonal=1)

E = torch.zeros([C * (patch_size ** 2), embedding_dim])
K_spatial = torch.zeros([embedding_dim, attention_dim])
Q_spatial = torch.zeros([embedding_dim, attention_dim])
V_spatial = torch.zeros([256, embedding_dim])

K_temp = torch.zeros([embedding_dim, attention_dim])
Q_temp = torch.zeros([embedding_dim, attention_dim])
V_temp = torch.zeros([1000, embedding_dim])

spatial_embedding = torch.zeros([1, 1, 256, embedding_dim])
temporal_embedding = torch.zeros([1, 1000, 1, embedding_dim])

pixel_embedding = data @ E
O = pixel_embedding + spatial_embedding + temporal_embedding

query = O @ Q_spatial
key = O @ K_spatial
scores = torch.softmax(((query @ key.transpose(-2, -1)) / (attention_dim ** 0.5)), dim=-1)
O = scores @ V_spatial

query = O.transpose(-3, -2) @ Q_temp
key = O.transpose(-3, -2) @ K_temp
scores = torch.softmax(((query @ key.transpose(-2, -1)).masked_fill(causal_mask, float("-inf")) / (attention_dim ** 0.5)), dim=-1)
O = (scores @ V_temp).transpose(-3, -2)

print(O.shape)
