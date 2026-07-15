from pathlib import Path
import numpy as np
import imageio.v2 as imageio

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data_collection" / "coinrun_data"

# Apple Silicon GPU (Metal) when present, else CPU. All model weights and the batch
# are created on / moved to this device.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CoinRunWindows(Dataset):
    """Map-style dataset: one index per level. __getitem__ returns a random
    `context_length`-frame contiguous window from that level -> [context_length, 256, 48]."""
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return self.data.shape[0]                        # number of levels

    def __getitem__(self, level_idx):
        num_frames = self.data.shape[1]
        start = torch.randint(0, num_frames - self.context_length + 1, (1,)).item()
        return self.data[level_idx, start:start + self.context_length]

def load_frames(level_mp4: Path) -> np.ndarray:
    with imageio.get_reader(level_mp4) as reader:
        return np.stack([frame for frame in reader]).astype(np.float32)

def attention(K, Q, V, O, mask=None):
    query = O @ Q
    key = O @ K
    value = O @ V
    if mask is None:
        scores = torch.softmax(((query @ key.transpose(-2, -1)) / (K.shape[-1] ** 0.5)), dim=-1)
    else:
        scores = torch.softmax(((query @ key.transpose(-2, -1)).masked_fill(mask, float("-inf")) / (K.shape[-1] ** 0.5)), dim=-1)
    sublayer = scores @ value
    return sublayer

def multi_head_attention(O, Wq, Wk, Wv, Wo, mask=None):
    """Run each head's scaled-dot-product attention on the shared input O, concat the
    per-head outputs along the feature axis, then apply the output projection Wo."""
    heads = [attention(Wk[i], Wq[i], Wv[i], O, mask=mask) for i in range(len(Wq))]
    return torch.concat(heads, dim=-1) @ Wo

def layer_norm(x, eps=1e-4):
    """Normalize each token's feature vector (last dim) to zero mean / unit std.
    No learnable gain/bias yet - that arrives with step (4)."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + eps)

def xavier(shape):
    """Xavier/Glorot-uniform weight as a leaf tensor that requires grad. Good default for
    linear projections: keeps activation variance stable as signals pass through the stack."""
    w = torch.empty(shape, device=DEVICE)          # create ON device so it stays a leaf
    nn.init.xavier_uniform_(w)
    return w.requires_grad_(True)

def small_normal(shape, std=0.02):
    """Small zero-mean normal init (leaf, requires grad) - the usual choice for additive
    position embeddings."""
    w = torch.empty(shape, device=DEVICE)          # create ON device so it stays a leaf
    nn.init.normal_(w, std=std)
    return w.requires_grad_(True)

def init_st_block(d_model, num_heads):
    """All weights for one ST-transformer block, Xavier-initialized and requiring grad."""
    kq_dim = d_model // num_heads
    v_dim = d_model // num_heads
    heads = lambda out_dim: [xavier([d_model, out_dim]) for _ in range(num_heads)]
    return {
        "Wq_spatial":  heads(kq_dim), "Wk_spatial":  heads(kq_dim), "Wv_spatial":  heads(v_dim),
        "Wo_spatial":  xavier([d_model, d_model]),
        "Wq_temporal": heads(kq_dim), "Wk_temporal": heads(kq_dim), "Wv_temporal": heads(v_dim),
        "Wo_temporal": xavier([d_model, d_model]),
        "W_ff1": xavier([d_model, 4 * d_model]),
        "W_ff2": xavier([4 * d_model, d_model]),
    }

def st_transformer_block(O, params, causal_mask):
    """One ST-transformer block: spatial attention -> temporal (causal) attention -> FFN,
    each wrapped as a pre-norm residual sub-layer:  x = x + sublayer(layer_norm(x))."""
    # Spatial self-attention: mixes the patches within each frame (no causal mask).
    O = O + multi_head_attention(
        layer_norm(O),
        params["Wq_spatial"], params["Wk_spatial"], params["Wv_spatial"], params["Wo_spatial"],
    )
    # Temporal self-attention: mixes across time per patch. Transpose so time is the
    # attended (second-to-last) axis, then transpose the result back.
    temporal = multi_head_attention(
        layer_norm(O).transpose(-3, -2),
        params["Wq_temporal"], params["Wk_temporal"], params["Wv_temporal"], params["Wo_temporal"],
        mask=causal_mask,
    )
    O = O + temporal.transpose(-3, -2)
    # Position-wise feed-forward.
    normed = layer_norm(O)
    O = O + F.gelu(normed @ params["W_ff1"]) @ params["W_ff2"]
    return O

def train_video_tokenizer(data, patch_size):
    B, T, H, W, C = data.shape
    p = patch_size
    data = (
        data.reshape(B, T, H // p, p, W // p, p, C)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, T, (H // p) * (W // p), p * p * C)
    )
    print("Finished loading and shaping data")

    batch_size = 1        # smoke test: activation memory scales linearly with this.
                          # 5 blew past 16 GB RAM during backward; 1 keeps peak ~3-4 GB.
    context_length = 16   # contiguous frames per sample

    dataset = CoinRunWindows(data, context_length)
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * 100)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    batch = next(iter(loader)).to(DEVICE)   # [1, 16, 256, 48]  (for training: `for batch in loader:`)
    print(f"Selected a batch of data: {tuple(batch.shape)} on {batch.device}")

    d_model = 512
    num_heads = 8
    num_blocks = 8   # Genie's ST-transformer stacks several of these
    latent_dim = 32
    codebook_size = 1024
    causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=bool, device=DEVICE), diagonal=1)

    # Patch embedding (48 -> d_model) plus additive spatial/temporal position embeddings.
    E = xavier([C * (patch_size ** 2), d_model])
    spatial_embedding = small_normal([1, 1, (H // patch_size) ** 2, d_model])
    temporal_embedding = small_normal([1, context_length, 1, d_model])
    O = batch @ E + spatial_embedding + temporal_embedding

    # Stack of ST-transformer blocks, each with its own independent weights.
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    print("ST-transformer output:", tuple(O.shape))

    # VQ-VAE codebook
    # Rows must be diverse (nonzero) or every patch collapses onto the same code index.
    vq_codebook = torch.randn([codebook_size, latent_dim], device=DEVICE, requires_grad=True)
    W_latent = xavier([d_model, latent_dim])   # renamed from F: F is torch.nn.functional
    latents = O @ W_latent                          # [B, T, N, latent_dim]  project d_model -> 32
    print(latents.shape)

    # Nearest-neighbour quantization
    flat = latents.reshape(-1, latent_dim)                        # [B*T*N, latent_dim]
    dists = torch.cdist(flat, vq_codebook)                        # [B*T*N, codebook_size]  pairwise L2
    indices = dists.argmin(dim=-1).reshape(latents.shape[:-1])    # [B, T, N]  chosen code per patch
    quantized = vq_codebook[indices]                             # [B, T, N, latent_dim]

    W_dmodel = xavier([latent_dim, d_model])
    # Straight-through estimator: forward value equals the quantized code, but the gradient
    # is routed to `latents` (the encoder) as if quantization were the identity. argmin is
    # non-differentiable, so without this the reconstruction loss can never reach the encoder.
    # Keep `quantized` (the raw codebook gather) separate for the VQ losses below.
    quantized_ste = latents + (quantized - latents).detach()
    O = quantized_ste @ W_dmodel
    O = O + spatial_embedding + temporal_embedding

    # Decoder
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    W_final = xavier([d_model, C * (patch_size ** 2)])
    O = O @ W_final

    recon_loss      = F.mse_loss(O, batch)                     # encoder + decoder (encoder via STE)
    codebook_loss   = F.mse_loss(quantized, latents.detach())  # pulls codebook -> encoder outputs
    commitment_loss = F.mse_loss(quantized.detach(), latents)  # pulls encoder -> codebook
    loss = recon_loss + codebook_loss + 0.8 * commitment_loss
    print("loss:", loss.item())

    # Smoke test that the graph is differentiable end-to-end (an optimizer.step() goes here).
    loss.backward()
    print("grad reached encoder:", W_latent.grad is not None, "| codebook:", vq_codebook.grad is not None)

def train_latent_action_model(data, patch_size):
    B, T, H, W, C = data.shape
    p = patch_size
    data = (
        data.reshape(B, T, H // p, p, W // p, p, C)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, T, (H // p) * (W // p), p * p * C)
    )
    print("Finished loading and shaping data")

    batch_size = 1        # smoke test: activation memory scales linearly with this.
                          # 5 blew past 16 GB RAM during backward; 1 keeps peak ~3-4 GB.
    context_length = 16   # contiguous frames per sample

    dataset = CoinRunWindows(data, context_length)
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * 100)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    batch = next(iter(loader)).to(DEVICE)   # [1, 16, 256, 48]  (for training: `for batch in loader:`)
    print(f"Selected a batch of data: {tuple(batch.shape)} on {batch.device}")

    d_model = 512
    num_heads = 8
    num_blocks = 8   # Genie's ST-transformer stacks several of these
    latent_dim = 32
    codebook_size = 6
    causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=bool, device=DEVICE), diagonal=1)

    # Patch embedding (48 -> d_model) plus additive spatial/temporal position embeddings.
    E = xavier([C * (patch_size ** 2), d_model])
    spatial_embedding = small_normal([1, 1, (H // patch_size) ** 2, d_model])
    temporal_embedding = small_normal([1, context_length, 1, d_model])
    O = batch @ E + spatial_embedding + temporal_embedding

    # Stack of ST-transformer blocks, each with its own independent weights.
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    print("ST-transformer output:", tuple(O.shape))

    # VQ-VAE codebook
    # Rows must be diverse (nonzero) or every patch collapses onto the same code index.
    vq_codebook = torch.randn([codebook_size, latent_dim], device=DEVICE, requires_grad=True)
    W_latent = xavier([O.shape[2] * d_model, latent_dim])   # renamed from F: F is torch.nn.functional
    latents = O.reshape(O.shape[0], O.shape[1], -1) @ W_latent                          # [B, T, N, latent_dim]  project d_model -> 32
    print("latents")
    print(latents.shape)

    # Nearest-neighbour quantization
    flat = latents.reshape(-1, latent_dim)                        # [B*T, N*latent_dim]
    dists = torch.cdist(flat, vq_codebook)                        # [B*T, N*codebook_size]  pairwise L2
    indices = dists.argmin(dim=-1).reshape(latents.shape[:-1])    # [B, T]  chosen code per timestep
    quantized = vq_codebook[indices]                             # [B, T, N, latent_dim]
    print(quantized.shape)

    W_dmodel = xavier([latent_dim, d_model])
    # Straight-through estimator: forward value equals the quantized code, but the gradient
    # is routed to `latents` (the encoder) as if quantization were the identity. argmin is
    # non-differentiable, so without this the reconstruction loss can never reach the encoder.
    # Keep `quantized` (the raw codebook gather) separate for the VQ losses below.
    latent_actions = latents + (quantized - latents).detach()
    O = batch[:, :-1, :, :] @ E + spatial_embedding + temporal_embedding[:, :-1, :, :]
    O = O + (latent_actions[:, 1:, :] @ W_dmodel).unsqueeze(-2)

    # Decoder
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]

    causal_mask = torch.triu(torch.ones(context_length-1, context_length-1, dtype=bool, device=DEVICE), diagonal=1)
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    W_final = xavier([d_model, C * (patch_size ** 2)])
    O = O @ W_final

    recon_loss      = F.mse_loss(O, batch[:, 1:, :, :])                     # encoder + decoder (encoder via STE)
    codebook_loss   = F.mse_loss(quantized, latents.detach())  # pulls codebook -> encoder outputs
    commitment_loss = F.mse_loss(quantized.detach(), latents)  # pulls encoder -> codebook
    loss = recon_loss + codebook_loss + 0.8 * commitment_loss
    print("loss:", loss.item())

    # Smoke test that the graph is differentiable end-to-end (an optimizer.step() goes here).
    loss.backward()
    print("grad reached encoder:", W_latent.grad is not None, "| codebook:", vq_codebook.grad is not None)

def train_dynamics_model(data, tokenizer, lam, patch_size):
    B, T, H, W, C = data.shape
    p = patch_size
    data = (
        data.reshape(B, T, H // p, p, W // p, p, C)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, T, (H // p) * (W // p), p * p * C)
    )
    print("Finished loading and shaping data")

    batch_size = 1        # smoke test: activation memory scales linearly with this.
                          # 5 blew past 16 GB RAM during backward; 1 keeps peak ~3-4 GB.
    context_length = 16   # contiguous frames per sample

    dataset = CoinRunWindows(data, context_length)
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * 100)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    batch = next(iter(loader)).to(DEVICE)   # [1, 16, 256, 48]  (for training: `for batch in loader:`)
    print(f"Selected a batch of data: {tuple(batch.shape)} on {batch.device}")

    d_model = 512
    num_heads = 8
    num_blocks = 8   # Genie's ST-transformer stacks several of these
    latent_dim = 32
    codebook_size = 6
    causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=bool, device=DEVICE), diagonal=1)

    # Patch embedding (48 -> d_model) plus additive spatial/temporal position embeddings.
    E = xavier([C * (patch_size ** 2), d_model])
    spatial_embedding = small_normal([1, 1, (H // patch_size) ** 2, d_model])
    temporal_embedding = small_normal([1, context_length, 1, d_model])
    O = batch @ E + spatial_embedding + temporal_embedding

    # Stack of ST-transformer blocks, each with its own independent weights.
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    print("ST-transformer output:", tuple(O.shape))

    # VQ-VAE codebook
    # Rows must be diverse (nonzero) or every patch collapses onto the same code index.
    vq_codebook = torch.randn([codebook_size, latent_dim], device=DEVICE, requires_grad=True)
    W_latent = xavier([O.shape[2] * d_model, latent_dim])   # renamed from F: F is torch.nn.functional
    latents = O.reshape(O.shape[0], O.shape[1], -1) @ W_latent                          # [B, T, N, latent_dim]  project d_model -> 32
    print("latents")
    print(latents.shape)

    # Nearest-neighbour quantization
    flat = latents.reshape(-1, latent_dim)                        # [B*T, N*latent_dim]
    dists = torch.cdist(flat, vq_codebook)                        # [B*T, N*codebook_size]  pairwise L2
    indices = dists.argmin(dim=-1).reshape(latents.shape[:-1])    # [B, T]  chosen code per timestep
    quantized = vq_codebook[indices]                             # [B, T, N, latent_dim]
    print(quantized.shape)

    W_dmodel = xavier([latent_dim, d_model])
    # Straight-through estimator: forward value equals the quantized code, but the gradient
    # is routed to `latents` (the encoder) as if quantization were the identity. argmin is
    # non-differentiable, so without this the reconstruction loss can never reach the encoder.
    # Keep `quantized` (the raw codebook gather) separate for the VQ losses below.
    latent_actions = latents + (quantized - latents).detach()
    O = batch[:, :-1, :, :] @ E + spatial_embedding + temporal_embedding[:, :-1, :, :]
    O = O + (latent_actions[:, 1:, :] @ W_dmodel).unsqueeze(-2)

    # Decoder
    blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]

    causal_mask = torch.triu(torch.ones(context_length-1, context_length-1, dtype=bool, device=DEVICE), diagonal=1)
    for params in blocks:
        O = st_transformer_block(O, params, causal_mask)

    O = layer_norm(O)
    W_final = xavier([d_model, C * (patch_size ** 2)])
    O = O @ W_final

    recon_loss      = F.mse_loss(O, batch[:, 1:, :, :])                     # encoder + decoder (encoder via STE)
    codebook_loss   = F.mse_loss(quantized, latents.detach())  # pulls codebook -> encoder outputs
    commitment_loss = F.mse_loss(quantized.detach(), latents)  # pulls encoder -> codebook
    loss = recon_loss + codebook_loss + 0.8 * commitment_loss
    print("loss:", loss.item())

    # Smoke test that the graph is differentiable end-to-end (an optimizer.step() goes here).
    loss.backward()
    print("grad reached encoder:", W_latent.grad is not None, "| codebook:", vq_codebook.grad is not None)



def main():
    print("Loading data...")
    level_files = sorted(DATA_DIR.glob("*.mp4"))
    if not level_files:
        raise FileNotFoundError(f"No level_*.mp4 found under {DATA_DIR}")

    # Load the first 10 levels' frames into a numpy array.
    data = torch.from_numpy(np.stack([load_frames(level_files[i]) for i in range(10)]))

    tokenizer = train_video_tokenizer(data, patch_size=4)
    lam = train_latent_action_model(data, patch_size=16)
    train_dynamics_model(data, tokenizer, lam, patch_size=16)

if __name__ == '__main__':
    main()
