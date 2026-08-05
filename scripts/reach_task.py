"""Reach-to-dot task: sample a target in the arm's measured reach cone, drive the
arm to it under physics, log the joint data, record an mp4 per episode.

Pipeline, all inside one Genesis scene (the plane has to be present during
planning or the ground-collision test below has nothing to collide with):

  1. measure the reach envelope by FK-sampling the joint limits -- the max
     radius is a property of the URDF, not something to hard-code
  2. sample a dot inside that envelope subject to three rejections:
       - centre at least ``z_min`` above the ground plane
       - IK solves and forward kinematics confirms the EE lands on it
       - at the solved pose ``detect_collision`` is empty, so the arm can hold
         the dot without any link touching the ground, another link, or itself
  3. move there: ease the PD targets from home to the solution, step physics
  4. log every step, and log the final pose as one row in the
     ``make_ik_dataset.py`` column format

The collision test in (2) is the real constraint and it is checked on the
*final* pose only -- a trajectory that grazes something on the way in would
still be accepted, so the per-episode contact count printed at the end is worth
reading rather than trusting the plan. The dot itself is visual-only
(``collision=False``): a solid dot would be batted away by the fingers on
arrival and the video would show the arm knocking its own target off.

    ./run.sh scripts/reach_task.py                    # 5 dots, 5 mp4s
    ./run.sh scripts/reach_task.py --n-dots 20 --no-video
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np
import tyro

from lss_common import REPO, EE_LINK, add_arm, begin_recording, rel as _rel, vis_options, fk as _fk, split_dofs as _split_dofs, to_np as _np


@dataclass
class Config:
    dof: int = 4
    n_dots: int = 5
    seed: int = 0
    backend: str = "gpu"

    dot_radius: float = 0.012
    z_min: float = 0.06          # ground clearance for the dot CENTRE, not its surface
    reach_frac: tuple[float, float] = (0.35, 0.92)  # of measured max reach; 1.0 is the
                                                    # singular fully-locked-out pose
    body_clearance: float = 0.05  # keep the dot this far off the arm's own links

    approach_steps: int = 240     # PD ease-in from home to the solution
    hold_steps: int = 60          # settle at the target so the final log is steady
    kp: float = 120.0
    kv: float = 12.0

    video: bool = True
    res: tuple[int, int] = (640, 480)
    fps: int = 100                # == 1/dt, so the mp4 runs at real time

    out_dir: Path = REPO / "outputs" / "reach"
    data_dir: Path = REPO / "data"
    max_attempts_factor: int = 400


def main(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=getattr(gs, cfg.backend), logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=1 / 100, substeps=4),
        vis_options=vis_options(),
    )
    scene.add_entity(gs.morphs.Plane())
    arm = add_arm(scene, cfg.dof)
    # one dot entity reused across episodes: Genesis takes entities before build,
    # so a dot per episode would mean a scene per episode
    dot = scene.add_entity(
        gs.morphs.Sphere(radius=cfg.dot_radius, pos=(0.0, 0.0, -1.0),
                         fixed=True, collision=False),
        surface=gs.surfaces.Rough(color=(0.95, 0.15, 0.1)),
    )
    # framed on the reach sphere (~0.5 m radius), not on the parked arm, so a dot
    # anywhere in the cone stays in shot for the whole episode
    cam = scene.add_camera(res=cfg.res, pos=(0.62, -0.58, 0.44), lookat=(0.0, 0.0, 0.18),
                           fov=48, GUI=False)
    scene.build()

    ee = arm.get_link(EE_LINK)
    arm_dofs, finger_dofs, joint_names = _split_dofs(arm)
    lower, upper = (_np(x) for x in arm.get_dofs_limit())
    all_dofs = list(range(arm.n_dofs))
    arm.set_dofs_kp(np.full(arm.n_dofs, cfg.kp), all_dofs)
    arm.set_dofs_kv(np.full(arm.n_dofs, cfg.kv), all_dofs)
    home = np.zeros(arm.n_qs)

    reach = _measure_reach(arm, ee, arm_dofs, lower, upper, rng)
    print(f"measured reach: max {reach:.3f} m; sampling "
          f"{cfg.reach_frac[0] * reach:.3f}..{cfg.reach_frac[1] * reach:.3f} m, z >= {cfg.z_min}")

    plans = _plan_dots(cfg, arm, ee, arm_dofs, lower, upper, rng, reach, joint_names)

    n_arm = len(arm_dofs)
    traj_header = (["episode", "step", "t"]
                   + ["target_x", "target_y", "target_z"]
                   + [f"q{i + 1}" for i in range(n_arm)] + [f"q{n_arm + 1}_claw"]
                   + ["ee_x", "ee_y", "ee_z"])
    traj_rows: list[np.ndarray] = []
    sol_rows = np.empty((cfg.n_dots, 3 + n_arm + 1))
    summary = []

    for ep, (target, q_sol) in enumerate(plans):
        arm.set_qpos(home)
        arm.control_dofs_position(home[: arm.n_dofs], all_dofs)
        dot.set_pos(np.asarray(target))
        for _ in range(20):
            scene.step()  # let the reset settle before the clock starts

        path = cfg.out_dir / f"reach_{cfg.dof}dof_dot{ep}.mp4" if cfg.video else None
        stop_recording = begin_recording(cam, path, cfg.fps) if cfg.video else None

        contacts = 0
        target_q = np.zeros(arm.n_dofs)
        target_q[arm_dofs] = q_sol[arm_dofs]
        for step in range(cfg.approach_steps + cfg.hold_steps):
            # raised cosine: zero velocity at both ends, so the arm does not
            # snap off its base at t=0 or overshoot the dot on arrival
            s = min(1.0, step / cfg.approach_steps)
            alpha = 0.5 * (1 - np.cos(np.pi * s))
            arm.control_dofs_position(alpha * target_q, all_dofs)
            scene.step()
            if cfg.video:
                cam.render()

            q = _np(arm.get_dofs_position())
            pos = _np(ee.get_pos()).reshape(3)
            traj_rows.append(np.concatenate([
                [ep, step, step / cfg.fps], target, q[arm_dofs],
                [abs(q[finger_dofs[0]]) if finger_dofs else 0.0], pos,
            ]))
            contacts += len(arm.detect_collision())

        final_q = _np(arm.get_dofs_position())
        final_pos = _np(ee.get_pos()).reshape(3)
        err = float(np.linalg.norm(final_pos - target))
        sol_rows[ep, :3] = target
        sol_rows[ep, 3:-1] = final_q[arm_dofs]
        sol_rows[ep, -1] = abs(final_q[finger_dofs[0]]) if finger_dofs else 0.0

        if stop_recording is not None:
            stop_recording()
        summary.append((ep, target, err, contacts, path))
        print(f"dot {ep}: target {np.round(target, 3)}  final EE error {err * 1000:6.1f} mm  "
              f"contact-steps {contacts:4d}" + (f"  -> {path.name}" if path else ""))

    _write(cfg, traj_header, np.asarray(traj_rows), sol_rows, n_arm, joint_names,
           lower, upper, arm_dofs, summary)
    scene.destroy()


def _measure_reach(arm, ee, arm_dofs, lower, upper, rng, n=4000) -> float:
    """Max EE radius over the joint limits. Sampled, so it is a slight
    underestimate of the true lock-out reach -- which is what we want, since the
    lock-out pose is singular and IK behaves badly right at it."""
    qpos = np.zeros(arm.n_qs)
    best = 0.0
    for _ in range(n):
        qpos[arm_dofs] = rng.uniform(lower[arm_dofs], upper[arm_dofs])
        best = max(best, float(np.linalg.norm(_fk(arm, ee, qpos))))
    return best


def _plan_dots(cfg, arm, ee, arm_dofs, lower, upper, rng, reach, joint_names):
    """Sample dots that are reachable AND collision-free at the solved pose."""
    qpos = np.zeros(arm.n_qs)
    plans, attempts = [], 0
    rejects = {"z": 0, "ik": 0, "limits": 0, "collision": 0, "body": 0}
    budget = cfg.n_dots * cfg.max_attempts_factor

    # links whose proximity to the dot is expected (they are what touches it)
    tip = {"lss_arm_ee", "lss_arm_finger_l", "lss_arm_finger_r",
           f"lss_arm_link_{cfg.dof}", f"lss_arm_link_{cfg.dof + 1}"}
    body_idx = [i for i, link in enumerate(arm.links) if link.name not in tip]

    while len(plans) < cfg.n_dots and attempts < budget:
        attempts += 1
        r = (rng.uniform(cfg.reach_frac[0] ** 3, cfg.reach_frac[1] ** 3)) ** (1 / 3) * reach
        v, phi = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2 * np.pi)
        s = np.sqrt(1 - v * v)
        target = np.array([r * s * np.cos(phi), r * s * np.sin(phi), r * v])
        if target[2] < cfg.z_min:
            rejects["z"] += 1
            continue

        qpos[arm_dofs] = rng.uniform(lower[arm_dofs], upper[arm_dofs])
        arm.set_qpos(qpos)
        sol = _np(arm.inverse_kinematics(
            link=ee, pos=target, rot_mask=[False, False, False],
            pos_tol=5e-4, dofs_idx_local=arm_dofs, respect_joint_limit=True,
        ))
        if np.linalg.norm(_fk(arm, ee, sol) - target) > 5e-4:
            rejects["ik"] += 1
            continue
        if np.any(sol[arm_dofs] < lower[arm_dofs] - 1e-6) or np.any(
            sol[arm_dofs] > upper[arm_dofs] + 1e-6
        ):
            rejects["limits"] += 1
            continue
        # arm.set_qpos already put the solved pose in the scene; the plane is in
        # this scene too, so this catches ground contact as well as self-contact
        if len(arm.detect_collision()):
            rejects["collision"] += 1
            continue
        link_pos = _np(arm.get_links_pos())[body_idx]
        if np.min(np.linalg.norm(link_pos - target, axis=1)) < cfg.body_clearance:
            rejects["body"] += 1
            continue
        plans.append((target, sol))

    if len(plans) < cfg.n_dots:
        raise SystemExit(f"only placed {len(plans)}/{cfg.n_dots} dots in {attempts} "
                         f"attempts; rejects={rejects}")
    print(f"placed {cfg.n_dots} dots in {attempts} attempts; rejects={rejects}")
    return plans


def _write(cfg, traj_header, traj, sol_rows, n_arm, joint_names, lower, upper,
           arm_dofs, summary) -> None:
    traj_path = cfg.data_dir / f"reach_trajectories_{cfg.dof}dof.csv"
    np.savetxt(traj_path, traj, delimiter=",", fmt="%.10g",
               header=",".join(traj_header), comments="")

    sol_header = (["target_x", "target_y", "target_z"]
                  + [f"q{i + 1}" for i in range(n_arm)] + [f"q{n_arm + 1}_claw"])
    sol_path = cfg.data_dir / f"reach_solutions_{cfg.dof}dof.csv"
    np.savetxt(sol_path, sol_rows, delimiter=",", fmt="%.10g",
               header=",".join(sol_header), comments="")

    meta = {
        "urdf": f"assets/lss_arm/lss_arm_{cfg.dof}dof.urdf",
        "ee_link": EE_LINK,
        "source": "reach_task.py (simulated motion, PD position control)",
        "seed": cfg.seed,
        "dt": 0.01,
        "episodes": [
            {"episode": ep, "target": list(np.round(t, 6)),
             "final_ee_error_m": round(err, 6), "contact_steps": c,
             "video": p.name if p else None}
            for ep, t, err, c, p in summary
        ],
        # qN is a Genesis DoF index, not a URDF joint number
        "columns": {**{f"q{i + 1}": joint_names[i] for i in range(n_arm)},
                    f"q{n_arm + 1}_claw": "gripper opening (rad)"},
        "joint_limits": {joint_names[i]: [lower[arm_dofs[i]], upper[arm_dofs[i]]]
                         for i in range(n_arm)},
    }
    meta_path = cfg.data_dir / f"reach_{cfg.dof}dof.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    for p in (traj_path, sol_path, meta_path):
        print(f"wrote {_rel(p)}")
    print(f"      {len(traj)} trajectory rows, {len(sol_rows)} solution rows")


if __name__ == "__main__":
    main(tyro.cli(Config))
