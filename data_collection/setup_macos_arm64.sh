#!/usr/bin/env bash
# Sets up a native arm64 Python env + procgen build for Apple Silicon.
#
# Why not just `pip install procgen`:
#   - PyPI only ships procgen wheels for cp37-cp310 x86_64/manylinux/win_amd64.
#     There is no arm64 macOS wheel.
#   - Running the x86_64 wheel under Rosetta 2 doesn't work either: procgen's
#     C++ engine is compiled with AVX2 instructions (-march=ivybridge), and
#     Rosetta 2 does not emulate AVX/AVX2 - it crashes with an illegal
#     instruction.
#   - Building upstream openai/procgen from source natively on arm64 also
#     fails, because its CMake config hardcodes that same x86-only -march
#     flag, which arm64 clang rejects.
#
# The fix: build from a fork that patches the CMake flag to -march=armv8-a
# and fixes the Homebrew arm64 Qt5 path. See
# https://github.com/openai/procgen/pull/107 (open, unmerged) and
# https://github.com/M-RR-J/procgen/tree/bugfix/apple-silicon-build
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_VERSION="3.9.18"
VENV_DIR="${REPO_ROOT}/.venv"
PROCGEN_FORK="https://github.com/M-RR-J/procgen.git"
PROCGEN_BRANCH="bugfix/apple-silicon-build"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This script targets native Apple Silicon (arm64). Detected: $(uname -m)" >&2
  exit 1
fi

command -v pyenv >/dev/null || { echo "pyenv not found (brew install pyenv)" >&2; exit 1; }
command -v brew  >/dev/null || { echo "homebrew not found" >&2; exit 1; }

echo "==> Installing native arm64 build deps via Homebrew"
brew install cmake glfw qt@5

echo "==> Ensuring Python ${PYTHON_VERSION} is installed via pyenv"
# -s/--skip-existing: no-op if already installed. Avoids a pipefail+grep -q
# SIGPIPE pitfall that made the old check spuriously re-run `pyenv install`
# (which exits non-zero on an existing version) and abort under `set -e`.
pyenv install -s "${PYTHON_VERSION}"

echo "==> Creating venv at ${VENV_DIR}"
"$(pyenv root)/versions/${PYTHON_VERSION}/bin/python3" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo "==> Building procgen from the Apple Silicon fork (this compiles the game binary, a few minutes)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
git clone --branch "${PROCGEN_BRANCH}" --depth 1 "${PROCGEN_FORK}" "${WORKDIR}/procgen"
# procgen's build imports gym3 (for its libenv.h C header), but PEP 517 build
# isolation hides the venv's packages from the build environment. Install gym3
# first and build with isolation disabled so the build can find it.
pip install "gym3" "numpy<2"
pip install --no-build-isolation "${WORKDIR}/procgen"

echo "==> Installing remaining data-collection deps"
pip install -r "${REPO_ROOT}/requirements.txt"

echo "==> Sanity check: importing procgen and stepping one env"
python - <<'PY'
from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=1, env_name="coinrun", distribution_mode="hard", start_level=0, num_levels=1, rand_seed=0)
_, obs, _ = env.observe()
print("OK - obs shape:", obs["rgb"].shape, "action space:", env.ac_space)
PY

echo
echo "Setup complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
