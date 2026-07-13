# genie

A from-scratch reproduction effort following **Genie: Generative Interactive
Environments** ([arXiv:2402.15391](https://arxiv.org/abs/2402.15391)).

The project is organized into components, built up in stages. The first stage —
generating the training data — lives in [`data_collection/`](data_collection/),
which reproduces the paper's "Reproducible Case Study" (Appendix F): random-policy
CoinRun rollouts, 10M transitions.

## Repository layout

```
genie/
├── README.md                 # you are here
├── requirements.txt          # shared Python dependencies
├── .gitignore
└── data_collection/          # stage 1: CoinRun rollout collection
    ├── setup_macos_arm64.sh  # native Apple Silicon env + procgen build
    ├── collect_coinrun_data.py
    ├── view_npz.py           # render collected .npz levels to mp4/gif
    └── README.md             # component details & the arm64 build story
```

## Setup

This uses a single virtual environment at the repo root, shared across
components. On Apple Silicon (M1/M2/M3), `pip install procgen` does **not** work
(no arm64 wheel, and the x86 wheel needs AVX2 that Rosetta can't emulate) — the
setup script builds procgen from a patched fork instead. See
[`data_collection/README.md`](data_collection/README.md) for the full story.

```bash
# builds .venv/ at the repo root and compiles procgen (a few minutes)
./data_collection/setup_macos_arm64.sh

# then, from the repo root:
source .venv/bin/activate
```

Requirements:
- macOS on Apple Silicon (arm64)
- [Homebrew](https://brew.sh) and [pyenv](https://github.com/pyenv/pyenv)
  (`brew install pyenv`)

On x86_64 Linux (or Colab), `pip install procgen` works directly — the custom
build is only needed for macOS arm64.

## Quickstart

```bash
source .venv/bin/activate

# fast sanity check: 5 levels x 50 steps
python data_collection/collect_coinrun_data.py --smoke-test --out-dir ./data_collection/smoke_test

# eyeball the result as video
python data_collection/view_npz.py ./data_collection/smoke_test
```

See [`data_collection/README.md`](data_collection/README.md) for the full 10M-transition
collection run and all options.

## Notes on what's committed

Following standard practice, the environment and generated data are **not**
checked in (see `.gitignore`) — they're reproducible from the code:

- **`.venv/`** — recreate with the setup script above.
- **Collected data** (`coinrun_data/`, `smoke_test/`, `*.npz`, `*.mp4`) —
  regenerate with `collect_coinrun_data.py`.
