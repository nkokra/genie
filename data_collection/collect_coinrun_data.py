"""CoinRun data collection, per Genie (arXiv:2402.15391) Appendix F.

  "We use the CoinRun environment from the Procgen benchmark ... Using the
  'hard' mode, we collect data using a random policy with no action repeats.
  We sample level seeds between zero and 10,000 and collect 1,000 timesteps
  for each level, for a total of 10M transitions."

Each level is played independently (its own ProcgenGym3Env, num_levels=1
pinned to that seed) for exactly --steps-per-level env steps, regardless of
episode boundaries within that budget - if the agent dies/wins/times out
before the budget is used up, the level simply resets and play continues on
the same level. That is how 10,000 levels x 1,000 timesteps == 10M
transitions.

Output: one file per level under --out-dir, named level_{seed:05d}.npz,
containing:
  observations: (T, 64, 64, 3) uint8   - frame BEFORE the corresponding action
  actions:      (T,) int32
  rewards:      (T,) float32
  dones:        (T,) bool              - True where that step ended an episode
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

ENV_NAME = "coinrun"


def collect_level(seed: int, steps_per_level: int, distribution_mode: str, rng: np.random.Generator):
    from procgen import ProcgenGym3Env  # imported lazily so --help works without procgen installed

    env = ProcgenGym3Env(
        num=1,
        env_name=ENV_NAME,
        distribution_mode=distribution_mode,
        start_level=seed,
        num_levels=1,
        rand_seed=seed,
        use_sequential_levels=False,
    )
    n_actions = env.ac_space.eltype.n

    observations = np.empty((steps_per_level, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((steps_per_level,), dtype=np.int32)
    rewards = np.empty((steps_per_level,), dtype=np.float32)
    dones = np.empty((steps_per_level,), dtype=bool)

    _, obs, _ = env.observe()
    frame = obs["rgb"][0]

    for t in range(steps_per_level):
        action = rng.integers(0, n_actions, size=1, dtype=np.int32)
        observations[t] = frame
        actions[t] = action[0]

        env.act(action)
        rew, obs, first = env.observe()

        rewards[t] = rew[0]
        dones[t] = bool(first[0])  # True iff this action ended the episode (obs is the new episode's first frame)
        frame = obs["rgb"][0]

    return observations, actions, rewards, dones


def _worker(args):
    seed, steps_per_level, distribution_mode, out_dir, fmt = args
    out_path = os.path.join(out_dir, f"level_{seed:05d}.npz")
    if os.path.exists(out_path):
        return seed, "skipped"

    rng = np.random.default_rng(seed)
    observations, actions, rewards, dones = collect_level(seed, steps_per_level, distribution_mode, rng)

    if fmt == "npz":
        np.savez_compressed(
            out_path,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
    elif fmt == "mp4":
        import imageio

        video_path = os.path.join(out_dir, f"level_{seed:05d}.mp4")
        imageio.mimwrite(video_path, observations, fps=15, quality=8)
        np.savez_compressed(
            out_path,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
    else:
        raise ValueError(f"unknown format: {fmt}")

    return seed, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="./coinrun_data", help="output directory")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-levels", type=int, default=10_000, help="10,000 per the paper")
    parser.add_argument("--steps-per-level", type=int, default=1_000, help="1,000 per the paper")
    parser.add_argument("--distribution-mode", default="hard", choices=["easy", "hard", "extreme", "memory", "exploration"])
    parser.add_argument("--format", default="npz", choices=["npz", "mp4"], help="mp4 stores frames as video + a small npz for actions/rewards/dones; npz stores everything raw (bigger)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--smoke-test", action="store_true", help="override to 5 levels / 50 steps to sanity-check the pipeline fast")
    args = parser.parse_args()

    if args.smoke_test:
        args.num_levels = 5
        args.steps_per_level = 50

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.num_levels))
    tasks = [(seed, args.steps_per_level, args.distribution_mode, args.out_dir, args.format) for seed in seeds]

    total_transitions = args.num_levels * args.steps_per_level
    print(f"Collecting {args.num_levels} levels x {args.steps_per_level} steps = {total_transitions:,} transitions")
    print(f"distribution_mode={args.distribution_mode}  format={args.format}  workers={args.workers}  out_dir={args.out_dir}")

    with mp.Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
            pass

    print("Done.")

    if args.smoke_test:
        bytes_written = sum(
            os.path.getsize(os.path.join(args.out_dir, f))
            for f in os.listdir(args.out_dir)
        )
        bytes_per_transition = bytes_written / total_transitions
        paper_transitions = 10_000 * 1_000
        projected_bytes = bytes_per_transition * paper_transitions
        print(
            f"\nSmoke test wrote {bytes_written / 1e6:.1f} MB for {total_transitions:,} transitions "
            f"({bytes_per_transition:.0f} bytes/transition, format={args.format})."
        )
        print(
            f"Projected size for the full paper spec (10,000 levels x 1,000 steps = "
            f"{paper_transitions:,} transitions): ~{projected_bytes / 1e9:.1f} GB.\n"
            "This is a linear extrapolation from a tiny sample - treat it as a ballpark, "
            "not a guarantee (per-level overhead and compression ratios vary)."
        )


if __name__ == "__main__":
    main()
