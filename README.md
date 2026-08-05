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

## The asset

`assets/lss_arm/lss_arm_{4,5}dof.urdf` + `assets/lss_arm/meshes/`, converted from
[Lynxmotion/LSS-ROS2-Arms](https://github.com/Lynxmotion/LSS-ROS2-Arms)
(Apache-2.0) by `scripts/build_lss_urdf.py`.

Upstream ships xacro only, and its `$(find lss_arm_description)` needs a ROS 2
install to resolve. The build script fakes the two things xacro asks the ROS
environment for -- an ament index and the `ament_index_python` module -- runs
xacro from PyPI, and rewrites `package://` mesh references to paths relative to
the URDF, which is what Genesis resolves against.

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

No `.venv` here yet -- `run.sh` borrows the working one from
`~/repos/xarm-sim-construction` (Genesis 1.2.3, torch cu130, CUDA 13.3). Drop a
`.venv` in this repo and it takes precedence; `pyproject.toml` lists what a
standalone `uv sync` needs. This host is headless, so there is no viewer and all
rendering goes through offscreen cameras.
