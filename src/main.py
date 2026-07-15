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
    `context_length`-frame contiguous window of raw frames -> [context_length, H, W, C]."""
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

def patchify(frames, patch_size):
    """[B, T, H, W, C] -> [B, T, (H/p)*(W/p), p*p*C]. Splits each frame into non-overlapping
    patch_size x patch_size patches and flattens each to a vector."""
    B, T, H, W, C = frames.shape
    p = patch_size
    return (
        frames.reshape(B, T, H // p, p, W // p, p, C)
              .permute(0, 1, 2, 4, 3, 5, 6)
              .reshape(B, T, (H // p) * (W // p), p * p * C)
    )

def vq_quantize(latents, codebook):
    """Nearest-neighbour VQ lookup over the last dim. latents [..., d], codebook [K, d]
    -> (indices [...], quantized [..., d]). Works for any number of leading dims."""
    d = codebook.shape[-1]
    flat = latents.reshape(-1, d)
    dists = torch.cdist(flat, codebook)                          # [prod(leading), K]  pairwise L2
    indices = dists.argmin(dim=-1).reshape(latents.shape[:-1])   # [...]  chosen code per element
    quantized = codebook[indices]                                # [..., d]
    return indices, quantized

def sample_batch(data, context_length=16, batch_size=1):
    """Draw one batch of random `context_length`-frame windows of raw frames, on DEVICE.
    -> [batch_size, context_length, H, W, C]. (For real training, iterate the loader instead.)"""
    dataset = CoinRunWindows(data, context_length)
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * 100)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return next(iter(loader)).to(DEVICE)

def train_video_tokenizer(data, patch_size):
    """Build the video tokenizer, run one reconstruction smoke step, and RETURN its frozen
    encoder as a functional:

        encode(frames [B,T,H,W,C]) -> (token indices [B,T,N], continuous latents [B,T,N,latent_dim])

    The dynamics model consumes the token indices. The closure patchifies internally, so callers
    pass RAW frames and never need to know this model's patch_size."""
    _, _, H, W, C = data.shape
    context_length = 16
    batch_size = 1        # smoke test: activation memory scales linearly with this.
    d_model = 512
    num_heads = 8
    num_blocks = 8        # Genie's ST-transformer stacks several of these
    latent_dim = 32
    codebook_size = 1024
    N = (H // patch_size) * (W // patch_size)
    causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=bool, device=DEVICE), diagonal=1)

    # --- encoder weights: patch embedding (patch_vec -> d_model) + additive position embeddings ---
    E = xavier([C * (patch_size ** 2), d_model])
    spatial_embedding  = small_normal([1, 1, N, d_model])
    temporal_embedding = small_normal([1, context_length, 1, d_model])
    enc_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    W_latent = xavier([d_model, latent_dim])
    # Rows must be diverse (nonzero) or every patch collapses onto the same code index.
    vq_codebook = torch.randn([codebook_size, latent_dim], device=DEVICE, requires_grad=True)

    def encode(frames):
        """Raw frames [B,T,H,W,C] -> (token indices [B,T,N], continuous latents [B,T,N,latent_dim])."""
        O = patchify(frames, patch_size) @ E + spatial_embedding + temporal_embedding
        for params in enc_blocks:
            O = st_transformer_block(O, params, causal_mask)
        O = layer_norm(O)
        latents = O @ W_latent                       # [B, T, N, latent_dim]  project d_model -> 32
        indices, _ = vq_quantize(latents, vq_codebook)
        return indices, latents

    # --- decoder weights: only needed for the reconstruction smoke step below ---
    W_dmodel   = xavier([latent_dim, d_model])
    dec_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    W_final    = xavier([d_model, C * (patch_size ** 2)])

    # --- smoke step: encode a batch, quantize (straight-through), reconstruct, backprop ---
    batch = sample_batch(data, context_length, batch_size)      # raw [B,T,H,W,C]
    indices, latents = encode(batch)
    quantized = vq_codebook[indices]
    # Straight-through estimator: forward value is the quantized code, but gradient is routed
    # to `latents` (the encoder) as if quantization were the identity. argmin is non-differentiable.
    quantized_ste = latents + (quantized - latents).detach()
    O = quantized_ste @ W_dmodel + spatial_embedding + temporal_embedding
    for params in dec_blocks:
        O = st_transformer_block(O, params, causal_mask)
    O = layer_norm(O) @ W_final

    recon_loss      = F.mse_loss(O, patchify(batch, patch_size))
    codebook_loss   = F.mse_loss(quantized, latents.detach())   # pulls codebook -> encoder outputs
    commitment_loss = F.mse_loss(quantized.detach(), latents)   # pulls encoder -> codebook
    loss = recon_loss + codebook_loss + 0.8 * commitment_loss
    loss.backward()
    print(f"[tokenizer] tokens {tuple(indices.shape)} | loss {loss.item():.2f} "
          f"| grad encoder={W_latent.grad is not None} codebook={vq_codebook.grad is not None}")
    return encode

def train_latent_action_model(data, patch_size):
    """Build the latent action model, run one reconstruction smoke step, and RETURN its frozen
    encoder as a functional:

        infer_actions(frames [B,T,H,W,C]) -> (action indices [B,T], action latents [B,T,latent_dim])

    One latent action per timestep. Alignment: index t is the transition INTO frame t (the encoder
    output at position t has causally attended through x_t), so the dynamics model conditions the
    x_t -> x_{t+1} step on a[:, t+1]. The closure patchifies internally; callers pass RAW frames."""
    _, _, H, W, C = data.shape
    context_length = 16
    batch_size = 1
    d_model = 512
    num_heads = 8
    num_blocks = 8
    latent_dim = 32
    codebook_size = 6     # Appendix F.3, Table 16: |A| = 6 for the CoinRun case study
    N = (H // patch_size) * (W // patch_size)
    causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=bool, device=DEVICE), diagonal=1)

    # --- encoder weights ---
    E = xavier([C * (patch_size ** 2), d_model])
    spatial_embedding  = small_normal([1, 1, N, d_model])
    temporal_embedding = small_normal([1, context_length, 1, d_model])
    enc_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    # Collapse all patches of a frame into ONE action per timestep: [N*d_model] -> latent_dim.
    W_latent = xavier([N * d_model, latent_dim])
    vq_codebook = torch.randn([codebook_size, latent_dim], device=DEVICE, requires_grad=True)

    def infer_actions(frames):
        """Raw frames [B,T,H,W,C] -> (action indices [B,T], continuous action latents [B,T,latent_dim])."""
        O = patchify(frames, patch_size) @ E + spatial_embedding + temporal_embedding
        for params in enc_blocks:
            O = st_transformer_block(O, params, causal_mask)
        O = layer_norm(O)
        latents = O.reshape(O.shape[0], O.shape[1], -1) @ W_latent   # [B, T, latent_dim]  (patches collapsed)
        indices, _ = vq_quantize(latents, vq_codebook)               # [B, T]
        return indices, latents

    # --- decoder weights: only needed for the reconstruction smoke step below ---
    W_dmodel   = xavier([latent_dim, d_model])
    dec_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    W_final    = xavier([d_model, C * (patch_size ** 2)])
    dec_mask   = torch.triu(torch.ones(context_length - 1, context_length - 1, dtype=bool, device=DEVICE), diagonal=1)

    # --- smoke step: predict each next frame from the previous frame + the action into it ---
    batch = sample_batch(data, context_length, batch_size)      # raw [B,T,H,W,C]
    indices, latents = infer_actions(batch)
    quantized = vq_codebook[indices]                            # [B, T, latent_dim]
    latent_actions = latents + (quantized - latents).detach()  # straight-through

    patches = patchify(batch, patch_size)                       # [B, T, N, patch_vec]
    # Decoder input: previous frames x_{0..T-2}, each conditioned on the action that "saw" its target.
    O = patches[:, :-1] @ E + spatial_embedding + temporal_embedding[:, :-1]
    O = O + (latent_actions[:, 1:] @ W_dmodel).unsqueeze(-2)    # broadcast one action over all patches
    for params in dec_blocks:
        O = st_transformer_block(O, params, dec_mask)
    O = layer_norm(O) @ W_final

    recon_loss      = F.mse_loss(O, patches[:, 1:])            # predict next frames x_{1..T-1}
    codebook_loss   = F.mse_loss(quantized, latents.detach())
    commitment_loss = F.mse_loss(quantized.detach(), latents)
    loss = recon_loss + codebook_loss + 0.8 * commitment_loss
    loss.backward()
    print(f"[LAM] actions {tuple(indices.shape)} | loss {loss.item():.2f} "
          f"| grad encoder={W_latent.grad is not None} codebook={vq_codebook.grad is not None}")
    return infer_actions

def train_dynamics_model(data, tokenizer, lam, patch_size):
    """Dynamics model (MaskGIT-style, Section 3.3). `tokenizer` and `lam` are the frozen
    functionals returned by the two trainers above:
        tokenizer(frames) -> (z [B,T,N], _)   frame tokens to predict
        lam(frames)       -> (a [B,T],   _)   per-timestep latent actions to condition on
    Both take RAW frames, so no patchifying is needed here (`patch_size` is reserved for the
    dynamics model you're about to build)."""
    context_length = 16
    batch = sample_batch(data, context_length, batch_size=1)    # raw [B,T,H,W,C]

    # Tokenizer and LAM are pre-trained / frozen here: no grad flows back into them.
    with torch.no_grad():
        z, _ = tokenizer(batch)     # frame tokens    [B, T, N]  ints in [0, tokenizer codebook)
        a, _ = lam(batch)           # latent actions  [B, T]     ints in [0, |A|)
    print(f"[dynamics] frame tokens {tuple(z.shape)} | actions {tuple(a.shape)}")

    # TODO(dynamics): MaskGIT transformer over z conditioned on a. Predict z[:, t+1] from
    #   z[:, :t+1] (causal in time) and action a[:, t+1]; randomly mask a fraction of the z
    #   tokens and train cross-entropy over the tokenizer vocabulary to fill them back in.

def main():
    print("Loading data...")
    level_files = sorted(DATA_DIR.glob("*.mp4"))
    if not level_files:
        raise FileNotFoundError(f"No level_*.mp4 found under {DATA_DIR}")

    # Load the first 10 levels' frames into a raw tensor [levels, frames, H, W, C].
    data = torch.from_numpy(np.stack([load_frames(level_files[i]) for i in range(10)]))

    tokenizer = train_video_tokenizer(data, patch_size=4)
    lam = train_latent_action_model(data, patch_size=16)
    train_dynamics_model(data, tokenizer, lam, patch_size=16)

if __name__ == '__main__':
    main()
