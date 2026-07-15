from pathlib import Path
import math
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

def unpatchify(patches, patch_size, H, W):
    """Inverse of patchify. [B, T, (H/p)*(W/p), p*p*C] -> [B, T, H, W, C]."""
    B, T, _, pv = patches.shape
    p = patch_size
    C = pv // (p * p)
    return (
        patches.reshape(B, T, H // p, W // p, p, p, C)
               .permute(0, 1, 2, 4, 3, 5, 6)
               .reshape(B, T, H, W, C)
    )

def causal_time_mask(t):
    """[t, t] boolean upper-triangular mask (True above the diagonal) for causal temporal attention."""
    return torch.triu(torch.ones(t, t, dtype=bool, device=DEVICE), diagonal=1)

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
        """Raw frames [B,T,H,W,C] -> (token indices [B,T,N], continuous latents [B,T,N,latent_dim]).
        Variable-length in T (<= context_length): slices the temporal position embedding to fit."""
        T = frames.shape[1]
        O = patchify(frames, patch_size) @ E + spatial_embedding + temporal_embedding[:, :T]
        for params in enc_blocks:
            O = st_transformer_block(O, params, causal_time_mask(T))
        O = layer_norm(O)
        latents = O @ W_latent                       # [B, T, N, latent_dim]  project d_model -> 32
        indices, _ = vq_quantize(latents, vq_codebook)
        return indices, latents

    # --- decoder weights: used by the reconstruction smoke step AND the inference `decode` closure ---
    W_dmodel   = xavier([latent_dim, d_model])
    dec_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    W_final    = xavier([d_model, C * (patch_size ** 2)])

    def decode(z_idx):
        """Token indices [B,T,N] -> reconstructed pixel frames [B,T,H,W,C] (inference; hard-quantized,
        no straight-through). Variable-length in T (<= context_length)."""
        T = z_idx.shape[1]
        O = vq_codebook[z_idx] @ W_dmodel + spatial_embedding + temporal_embedding[:, :T]
        for params in dec_blocks:
            O = st_transformer_block(O, params, causal_time_mask(T))
        O = layer_norm(O) @ W_final                  # [B, T, N, patch_vec]
        return unpatchify(O, patch_size, H, W)

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
    return encode, decode

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
        """Raw frames [B,T,H,W,C] -> (action indices [B,T], continuous action latents [B,T,latent_dim]).
        Variable-length in T (<= context_length)."""
        T = frames.shape[1]
        O = patchify(frames, patch_size) @ E + spatial_embedding + temporal_embedding[:, :T]
        for params in enc_blocks:
            O = st_transformer_block(O, params, causal_time_mask(T))
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
    """Dynamics model (MaskGIT, Section 3.3): each frame reconstructs its own randomly-masked
    tokens from its visible tokens + past frames + action. `tokenizer` and `lam` are the frozen
    trainers; we consume only their DISCRETE outputs (token / action indices) and embed them with
    this model's OWN learned tables, so the dynamics model is self-contained: at generation time it
    embeds indices it produced itself, with no access to the tokenizer's continuous latents.

        tokenizer(frames) -> (z_idx [B,T,N], _)
        lam(frames)       -> (a_idx [B,T],   _)
    """
    context_length = 16
    num_blocks = 12
    d_model = 512
    num_heads = 8
    codebook_size = 1024    # frame-token vocab — must match the tokenizer's codebook_size
    action_vocab  = 6       # latent-action vocab — must match the LAM's codebook_size (|A|)
    MASK_ID = codebook_size  # reserved last row of token_embed for the [MASK] token (unused until masking)

    batch = sample_batch(data, context_length, batch_size=1)    # raw [B,T,H,W,C]

    # Tokenizer and LAM are frozen: we only read integer indices (argmin is non-differentiable).
    with torch.no_grad():
        z_idx, _ = tokenizer(batch)     # frame tokens    [B, T, N]  ints in [0, codebook_size)
        a_idx, _ = lam(batch)           # latent actions  [B, T]     ints in [0, action_vocab)
    print(f"[dynamics] frame tokens {tuple(z_idx.shape)} | actions {tuple(a_idx.shape)}")

    # --- dynamics model weights ---
    B, T, N = z_idx.shape
    # token_embed has one EXTRA row (index MASK_ID) for the [MASK] token: masked positions look up
    # this row instead of their true token. W_final predicts real tokens only (MASK is never a target).
    token_embed  = small_normal([codebook_size + 1, d_model])
    action_embed = small_normal([action_vocab, d_model])
    # Position embeddings are essential here: every masked slot shares the same [MASK] embedding, so
    # without these the model can't tell which patch (spatial) or which frame (temporal) it predicts.
    spatial_embedding  = small_normal([1, 1, N, d_model])
    temporal_embedding = small_normal([1, T, 1, d_model])
    dec_blocks = [init_st_block(d_model, num_heads) for _ in range(num_blocks)]
    W_final    = xavier([d_model, codebook_size])

    def dynamics(z_seq, a_seq):
        """Token indices [B,t,N] (may contain MASK_ID) + action indices [B,t] -> logits [B,t,N,vocab].
        Variable-length in t (<= context_length). Each frame attends to its own tokens (spatial) and
        past frames (temporal causal); the action a_seq[:, k] conditions frame k."""
        t = z_seq.shape[1]
        O = (token_embed[z_seq] + action_embed[a_seq].unsqueeze(-2)
             + spatial_embedding + temporal_embedding[:, :t])    # [B, t, N, d_model]
        for params in dec_blocks:
            O = st_transformer_block(O, params, causal_time_mask(t))
        return layer_norm(O) @ W_final                           # [B, t, N, codebook_size]

    # --- MaskGIT masking smoke step: hide a random fraction of the tokens, then fill them back in ---
    # Frame 0 is the always-visible seed (nothing precedes it). Frames 1..T-1 get a per-token Bernoulli
    # mask at rate ~ U(0.5, 1.0); rate=1.0 recovers the fully-blind "predict next frame from scratch" case.
    mask_rate = 0.5 + 0.5 * torch.rand(1, device=DEVICE)
    mask = torch.rand(B, T, N, device=DEVICE) < mask_rate    # [B,T,N]  True = masked (must be predicted)
    mask[:, 0] = False                                       # keep the seed frame fully visible
    z_in = z_idx.clone()
    z_in[mask] = MASK_ID                                     # swap masked tokens for [MASK]

    logits = dynamics(z_in, a_idx)                           # [B, T, N, codebook_size]
    # Cross-entropy over ONLY the masked positions (no credit for tokens handed to the model).
    recon_loss = F.cross_entropy(logits[mask], z_idx[mask])
    recon_loss.backward()
    print(f"[dynamics] mask_rate {mask_rate.item():.2f} | masked {int(mask.sum())}/{B*T*N} "
          f"| loss {recon_loss.item():.2f} | grad token_embed={token_embed.grad is not None} "
          f"action_embed={action_embed.grad is not None}")
    return dynamics

# ==============================================================================================
# Inference / generation. Consumes the frozen functionals the trainers now expose:
#   - encode/decode : train_video_tokenizer -> (encode, decode)   frames <-> tokens
#   - dynamics      : train_dynamics_model  -> dynamics           (z_seq, a_seq) -> logits
#   - lam           : train_latent_action_model -> infer_actions  frames -> action indices
# All are variable-length in the time axis (<= context_length). NOTE: these run on UNTRAINED weights
# until the co-training loop exists, so a rollout is structurally valid but visually meaningless.
# ==============================================================================================

def gumbel_like(x):
    """Standard Gumbel(0,1) noise shaped like x. Adding it to log-probs and taking argmax samples
    from the softmax; scaling it on a confidence controls how random the MaskGIT reveal order is."""
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u + 1e-9) + 1e-9)

def maskgit_decode_frame(dynamics, context, action_hist, action, N, vocab, MASK_ID,
                         num_steps=16, temperature=1.0, choice_temperature=1.0):
    """Generate ONE new frame's tokens by MaskGIT iterative parallel decoding.

    Start from an all-[MASK] frame and, over `num_steps` rounds (num_steps << N, that's the whole
    point), progressively commit the highest-confidence predictions until all N tokens are filled.
    NOTE the key difference from training: here the `context` frames are fully known (clean) and only
    the new frame is masked — this is the inference regime we discussed.

        context      already-generated frames' tokens    [B, t, N]   (clean)
        action_hist  actions that produced those frames   [B, t]
        action       latent action INTO the new frame     [B]         (user/policy chosen)
        returns      the new frame's token indices         [B, N]
    """
    B = context.shape[0]
    frame   = torch.full((B, N), MASK_ID, device=DEVICE)         # [B, N]  start fully masked
    unknown = torch.ones((B, N), dtype=bool, device=DEVICE)      # positions still to decide

    for step in range(num_steps):
        # Assemble the running sequence: clean context + the current (partly-masked) new frame.
        z_seq = torch.cat([context, frame.unsqueeze(1)], dim=1)        # [B, t+1, N]
        a_seq = torch.cat([action_hist, action.unsqueeze(1)], dim=1)   # [B, t+1]

        # Run the (frozen) dynamics model; take the logits at the NEW frame's slot.
        with torch.no_grad():
            logits = dynamics(z_seq, a_seq)[:, -1]                     # [B, N, vocab]

        # meat #1: sample a token per position, and score its confidence.
        probs = F.softmax(logits / temperature, dim=-1)               # [B, N, vocab]
        # Gumbel-max sampling (== sampling from `probs`, but MPS-safe; avoids torch.multinomial).
        sampled = (torch.log(probs + 1e-9) + gumbel_like(probs)).argmax(dim=-1)   # [B, N]
        conf = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)         # [B, N] prob of sample
        # Annealed noise on the confidence randomizes which tokens get committed first (more early on).
        conf = conf + choice_temperature * (1 - (step + 1) / num_steps) * gumbel_like(conf)
        # Already-committed positions keep their token and must never be re-masked (+inf confidence).
        sampled = torch.where(unknown, sampled, frame)
        conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))

        # meat #2: cosine schedule — how many tokens to KEEP masked after this round (1 -> 0).
        ratio = math.cos(math.pi / 2 * (step + 1) / num_steps)
        n_keep_masked = 0 if step == num_steps - 1 else int(math.floor(N * ratio))

        # meat #3: keep the n_keep_masked LOWEST-confidence positions masked; commit everything else.
        still_masked = torch.zeros(B, N, dtype=bool, device=DEVICE)
        if n_keep_masked > 0:
            _, low_idx = torch.topk(conf, n_keep_masked, dim=-1, largest=False)   # [B, n_keep_masked]
            still_masked.scatter_(1, low_idx, True)
        frame = torch.where(still_masked, torch.full_like(sampled, MASK_ID), sampled)
        unknown = still_masked

    assert not unknown.any(), "some tokens left undecided after MaskGIT decoding"
    return frame

def generate_rollout(encode, decode, dynamics, lam, prompt_frames, actions,
                     N, vocab, MASK_ID, context_length=16, num_steps=16):
    """Autoregressively roll out a playable video: tokenize a prompt, then generate frames one at a
    time — each conditioned on all prior frames + a chosen latent action — and detokenize to pixels.

        prompt_frames  seed frames (>=1 real frame)          [B, t0, H, W, C]
        actions        chosen action index per new frame     [B, num_new]   (values in [0, |A|))
        returns        the full pixel video                  [B, t0+num_new, H, W, C]
    """
    num_new = actions.shape[1]

    # 1. Tokenize the prompt; seed the action history for those frames by inferring them with the LAM.
    #    (The action into the very first frame is meaningless — nothing precedes it — exactly as in
    #    training, where frame 0 is the always-visible seed.)
    with torch.no_grad():
        history, _ = encode(prompt_frames)        # [B, t0, N]  all frame tokens so far
        action_hist, _ = lam(prompt_frames)       # [B, t0]     inferred seed actions

    # 2. Generate frames one at a time, growing the running token history.
    for i in range(num_new):
        # The dynamics model spans at most context_length frames (incl. the new one), so condition on
        # at most the last context_length-1 frames (a sliding window for rollouts longer than that).
        ctx  = history[:, -(context_length - 1):]
        acts = action_hist[:, -(context_length - 1):]
        new_frame = maskgit_decode_frame(dynamics, ctx, acts, actions[:, i],
                                         N, vocab, MASK_ID, num_steps)            # [B, N]
        history     = torch.cat([history, new_frame.unsqueeze(1)], dim=1)
        action_hist = torch.cat([action_hist, actions[:, i:i + 1]], dim=1)

    # 3. Detokenize the whole rollout back to pixels, in <=context_length chunks (the decoder's
    #    temporal position embedding only spans that many frames).
    with torch.no_grad():
        chunks = [decode(history[:, s:s + context_length])
                  for s in range(0, history.shape[1], context_length)]
    return torch.cat(chunks, dim=1)               # [B, t0+num_new, H, W, C]

def main():
    print("Loading data...")
    level_files = sorted(DATA_DIR.glob("*.mp4"))
    if not level_files:
        raise FileNotFoundError(f"No level_*.mp4 found under {DATA_DIR}")

    # Load the first 10 levels' frames into a raw tensor [levels, frames, H, W, C].
    data = torch.from_numpy(np.stack([load_frames(level_files[i]) for i in range(10)]))

    encode, decode = train_video_tokenizer(data, patch_size=4)
    lam = train_latent_action_model(data, patch_size=16)
    dynamics = train_dynamics_model(data, encode, lam, patch_size=16)

    # Inference demo: seed with a few real frames, generate a couple more from chosen latent actions,
    # detokenize to pixels. Untrained weights -> the video is garbage; this just exercises the path.
    N, vocab, action_vocab, MASK_ID = 256, 1024, 6, 1024   # must match tokenizer/LAM/dynamics configs
    prompt = sample_batch(data, context_length=16, batch_size=1)[:, :4]     # 4-frame seed
    actions = torch.randint(0, action_vocab, (prompt.shape[0], 2), device=DEVICE)  # 2 chosen actions
    video = generate_rollout(encode, decode, dynamics, lam, prompt, actions,
                             N, vocab, MASK_ID, context_length=16, num_steps=6)
    print(f"[generate] rollout video {tuple(video.shape)}")

if __name__ == '__main__':
    main()
