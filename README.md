# robot_single_arm_testing

Lynxmotion LSS arm in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis):
asset conversion, a reach-to-dot task, and IK datasets in the
`target_xyz -> joint angles` format.

Everything runs through `./run.sh` (see the header of that file for why each
environment pin is there).

```
./run.sh scripts/check_arm.py            # load the arm, print its tree, write stills
./run.sh scripts/reach_task.py           # 5 random dots, drive to each, 5 mp4s + CSV
./run.sh scripts/make_ik_dataset.py --n 10000   # IK training set
./run.sh scripts/build_lss_urdf.py       # regenerate assets/lss_arm/ from upstream
```

## Setup from scratch

Start here on a machine that has none of this. The whole install is four
commands; the notes after them are the things that actually bite.

### 1. System prerequisites

You need a C library stack for OpenGL and, for the GPU backend, an NVIDIA
driver. **You do not need to install the CUDA toolkit** — the PyTorch wheels
bundle their own CUDA runtime, so the driver is enough.

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y git curl libgl1 libglib2.0-0

# RHEL / Fedora
sudo dnf install -y git curl mesa-libGL glib2
```

```bash
nvidia-smi     # driver present? note the CUDA version it reports
```

No NVIDIA GPU is fine — every script takes `--backend cpu` and runs, just
slower. Nothing here needs a display: rendering goes through offscreen cameras,
so a headless server works unchanged.

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # or: source ~/.bashrc
uv --version
```

You do **not** need to install Python yourself. `uv` reads `requires-python`
from `pyproject.toml` and downloads a matching interpreter (3.11 or 3.12;
Genesis does not support 3.13 yet).

### 3. Clone and sync

```bash
git clone https://github.com/grifflee/robot_single_arm_testing.git
cd robot_single_arm_testing
uv sync
```

That is the whole install — it creates `.venv/` and pulls Genesis, PyTorch,
trimesh, tyro and the rest. Two things to expect:

- **it downloads ~8.6 GB and takes a few minutes.** Nearly all of that is the
  CUDA-enabled PyTorch wheel. Timed here: 1m52s on a warm connection.
- **`xacro` comes with `genesis-world`**, so there is nothing extra to install
  for `build_lss_urdf.py`, and no ROS 2 install is needed anywhere.

`run.sh` picks up `.venv/` automatically once it exists.

### 4. Verify

```bash
./run.sh scripts/check_arm.py
```

**The first run takes several minutes and looks like it has hung.** Genesis
compiles its simulation kernels on first use; they are cached afterwards and
the same command then finishes in ~20 s. This is normal — do not kill it.

Success looks like a printed kinematic tree (8 links, 6 DoFs), an AABB spanning
~0.35 m, and three PNGs in `outputs/arm_check/`. The script deliberately fails
loudly if the AABB is under 5 cm, since that is what a silent mesh-loading
failure looks like.

Then run the real thing:

```bash
./run.sh scripts/reach_task.py
```

### Troubleshooting

**`ImportError: libGL.so.1: cannot open shared object file`** — step 1 was
skipped. Install `libgl1` (Debian) or `mesa-libGL` (RHEL).

**`TypeError: Camera.stop_recording() got an unexpected keyword argument`** —
you are on a Genesis version whose recording API differs from the code's.
`lss_common.begin_recording` branches on the signature to support both 1.2 and
1.3, so this should not happen; if it does, that helper is the one place to fix.

**`ModuleNotFoundError: No module named 'xacro.__main__'`** when regenerating
assets — the `xacro` package has no `__main__`, so it must be invoked as the
console script in `.venv/bin/`, which is what `build_lss_urdf.py` does.

**Everything is slow / `torch.cuda.is_available()` is False** — you are silently
on CPU. Check `nvidia-smi`, then
`./run.sh -c "import torch; print(torch.cuda.is_available())"`. If the driver is
older than the wheel's CUDA build, `uv add "torch==2.8.*"` and re-sync.

**Verified here:** Ubuntu-style deps on RHEL 10, RTX 2080, driver CUDA 13.0,
Python 3.11.15, `uv` 0.11.31. A clean `uv sync` resolved to Genesis 1.3.1 +
torch 2.13.0+cu130 and ran both `check_arm.py` and `reach_task.py` to
completion, producing the same numbers as the older pinned environment.

## The asset

`assets/lss_arm/lss_arm_{4,5}dof.urdf` + `assets/lss_arm/meshes/`, converted from
[Lynxmotion/LSS-ROS2-Arms](https://github.com/Lynxmotion/LSS-ROS2-Arms)
(Apache-2.0) by `scripts/build_lss_urdf.py`.

Upstream ships xacro only, and its `$(find lss_arm_description)` needs a ROS 2
install to resolve. The build script fakes the two things xacro asks the ROS
environment for -- an ament index and the `ament_index_python` module -- runs
xacro (which `genesis-world` already depends on, so nothing extra to install),
and rewrites `package://` mesh references to paths relative to the URDF, which
is what Genesis resolves against.

Regeneration is byte-identical to what is committed here: xacro stamps its
absolute input path into a banner comment, and the script rewrites that to the
upstream path so a rebuild from a fresh temp clone still diffs clean.

Two things about the upstream layout that are easy to get wrong:

- there is no `lss_arm_description/meshes/` in the source tree. The package
  *installs* `models/lss_arm_5dof/meshes` to that path, so the 5-DoF mesh set is
  canonical; every STL it shares with the 4-DoF set is byte-identical, so it is
  vendored once and both URDFs point at it.
- the 4-DoF arm's **finger** joints are named `joint_5` and `joint_6`. Only the
  *links* say "finger". Filtering the gripper out by joint name silently hands
  it to the IK solver -- classify on the child link (`lss_common.split_dofs`).

Measured from the URDF, not the datasheet: **4-DoF reach 0.515 m**, 6 DoFs
(4 arm + 2 fingers), 8 links.

## The reach task

`scripts/reach_task.py` measures the reach cone by FK-sampling the joint limits,
samples a dot inside it, and drives the arm there under PD control while logging
every step.

A dot is only accepted if all of these hold:

- its centre is at least `--z-min` (default 6 cm) above the ground plane
- IK solves for it *and* forward kinematics confirms the EE lands within 0.5 mm
- `detect_collision()` is empty at the solved pose -- the plane is in the scene
  during planning, so this rejects ground contact and self-contact alike
- the dot is at least `--body-clearance` from every non-tip link, so it cannot
  spawn buried inside the arm's own structure

The dot is visual-only (`collision=False`); a solid one would get batted away by
the fingers on arrival.

Last run (seed 0, 4-DoF): 5 dots placed in 12 attempts, final EE error
0.6-2.0 mm, **zero contact-steps across all 1500 simulated steps** -- so nothing
touched anything on the way in either, not just at the goal.

Outputs: `outputs/reach/reach_4dof_dot{0..4}.mp4`,
`data/reach_trajectories_4dof.csv` (per step),
`data/reach_solutions_4dof.csv` (final pose per episode).

## The datasets

`data/training_dataset.csv` follows
[Surina-10/RobotArm](https://github.com/Surina-10/RobotArm)'s column layout:

```
target_x,target_y,target_z,q1,q2,q3,q4,q5_claw
```

`--mode ik` samples targets in the measured reachable shell and solves;
`--mode fk` samples joint angles and records where the EE landed. Either way
every row is re-verified by forward kinematics before it is written -- a
non-converged IK result is the main way a set like this fills with wrong labels.
10k rows take ~50 s at a 76.5% accept rate.

`make_ik_dataset.py` is **pure kinematics**: no collision checking, so some rows
put the arm through the floor. If you need physically-valid poses only, take
them from `reach_task.py` instead, which is collision-checked and simulated.

Every CSV has a `.meta.json` sidecar. Read it before training on the columns:
`qN` is a *Genesis DoF index*, not a URDF joint number, and on the 5-DoF arm the
two disagree (its wrist is `joint_6`, its fingers are `joint_5`/`joint_7`).

## Environment

See **Setup from scratch** above for a fresh machine. Two notes about how
`run.sh` resolves its interpreter on the machine this was developed on:

- it prefers a `.venv/` in this repo, and falls back to the one in
  `~/repos/xarm-sim-construction` (Genesis 1.2.3). So the repo works here
  without its own sync, and `uv sync` makes it self-contained.
- its `CUDA_HOME`/`CXX`/`CC` pins are inherited from that sibling repo, which
  JIT-compiles gsplat CUDA extensions. Nothing here does, so those pins are
  harmless but not load-bearing — the NVIDIA driver is the only GPU
  prerequisite.

`genesis-world` is floored at 1.2 and not capped, so a fresh `uv sync` lands on
whatever is current. Both 1.2.3 and 1.3.1 are tested and produce identical
numbers; the one API that moved between them (video recording) is handled in
`lss_common.begin_recording`. This host is headless, so there is no viewer and
all rendering goes through offscreen cameras.
