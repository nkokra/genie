# CoinRun data collection (Genie, arXiv:2402.15391, Appendix F)

Reproduces the "Reproducible Case Study" data collection spec exactly:
CoinRun, `distribution_mode="hard"`, random policy, no action repeats,
level seeds 0-10,000, 1,000 timesteps per level, 10M transitions total.

## Apple Silicon (M1/M2/M3) setup

`pip install procgen` won't work here:

- No arm64 macOS wheel exists on PyPI (only cp37-cp310 x86_64/manylinux/win_amd64).
- Running the x86_64 wheel through Rosetta 2 doesn't work either - procgen's
  engine is compiled with AVX2 (`-march=ivybridge`), and Rosetta 2 does not
  emulate AVX/AVX2. It'll crash with an illegal instruction.
- Building upstream `openai/procgen` from source natively on arm64 also fails
  out of the box for the same reason (hardcoded x86 `-march` flag rejected by
  arm64 clang).

`setup_macos_arm64.sh` builds from a fork with that flag patched
(`-march=armv8-a`) so it compiles and runs natively on arm64, no Rosetta,
no AVX dependency:

```bash
# run from this directory; the venv is created at the repo root (../.venv)
./setup_macos_arm64.sh
source ../.venv/bin/activate
```

This installs Python 3.9 via pyenv, Homebrew build deps (cmake/glfw/qt@5),
builds `procgen` from
[M-RR-J/procgen@bugfix/apple-silicon-build](https://github.com/M-RR-J/procgen/tree/bugfix/apple-silicon-build),
installs the shared `../requirements.txt`, and runs a one-step sanity check.

The environment is shared across the whole repo; see the
[top-level README](../README.md) for the overall project layout.

If that fork has drifted or the build fails, an x86_64 Linux box (or Colab)
with `pip install procgen` remains the path of least resistance - this is a
genuinely unmaintained package on macOS arm64.

## Collecting data

```bash
# fast sanity check: 5 levels, 50 steps each
python collect_coinrun_data.py --smoke-test --out-dir ./smoke_test

# full paper spec: 10,000 levels x 1,000 steps = 10M transitions
python collect_coinrun_data.py --out-dir ./coinrun_data --format mp4
```

`--format npz` stores raw uint8 frames (simple, but ~10,000 levels x 1,000 x
64x64x3 bytes is large even compressed). `--format mp4` stores frames as a
per-level video plus a small `.npz` for actions/rewards/dones - much more
disk-efficient for the full 10M-transition run.

## Viewing collected data

`.npz` files store raw frames, not video. Render them to watchable mp4 (or gif)
with `view_npz.py`:

```bash
python view_npz.py ./smoke_test              # every level_*.npz -> mp4 alongside it
python view_npz.py ./smoke_test --gif        # gif instead (no ffmpeg needed)
python view_npz.py ./smoke_test/level_00000.npz --scale 6 --fps 10
```

It also prints per-level stats (frame count, reward sum, episode ends). With the
random policy, expect `reward sum=0` on most short clips — the agent rarely
reaches the coin — so a "coherent but aimless" playthrough is the pass signal.

## Collection details

Runs are parallelized across `--workers` processes (default: CPU count - 1),
each level is an independent `ProcgenGym3Env` pinned to one seed
(`num_levels=1`), and files are skipped if already present, so an
interrupted run can just be re-launched.

Note: "vectorized" in procgen/gym3 (`ProcgenGym3Env(num=...)`) means running
several env instances in one batched call - a software/throughput concern,
unrelated to the CPU's AVX instruction set. It works fine here; nothing about
it requires AVX.
