#!/usr/bin/env bash
# Run anything in this repo with the environment Genesis needs. Use this instead
# of calling python directly.
#
#   ./run.sh scripts/check_arm.py --dof 5
#   ./run.sh scripts/build_lss_urdf.py
#
# The venv: this repo has no .venv of its own yet and borrows the one from
# ~/repos/xarm-sim-construction (Genesis 1.2.3 + torch cu130 + the gs-madrona
# build, ~8.6 GB, already working on this host). Drop a .venv in this repo and
# it wins automatically -- see pyproject.toml for what a standalone sync needs.
#
# Why each pin is here (all of these were real failures in the sibling repo,
# not precaution):
#
#  PATH        torch's cpp_extension shells out to `ninja --version` when it
#              JIT-compiles gsplat's CUDA extension; ninja lives in the venv.
#  SETUPTOOLS_ Genesis imports stdlib distutils before gsplat imports
#  USE_DISTUTILS setuptools; setuptools >=83's _distutils_hack then asserts its
#              vendored copy is active and dies. stdlib skips the override.
#  CXX/CC      gsplat JIT-compiles against the system gcc; a conda-forge gcc on
#              PATH links a libstdc++ newer than the system's.
#  CUDA_HOME   this host has CUDA 13.3.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="$R/.venv"
[ -x "$VENV/bin/python" ] || VENV="$HOME/repos/xarm-sim-construction/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "no venv found: create one here with 'uv sync', or restore ~/repos/xarm-sim-construction/.venv" >&2
  exit 1
fi

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$VENV/bin:$HOME/.pixi/bin:$CUDA_HOME/bin:$PATH"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export CXX=/usr/bin/g++ CC=/usr/bin/gcc

exec "$VENV/bin/python" "$@"
