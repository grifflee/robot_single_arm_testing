"""Generate an IK training set from the arm in Genesis: EE target -> joint angles.

Column layout follows Surina-10/RobotArm's training_dataset.py, one row per sample:

    target_x,target_y,target_z,q1,q2,q3,q4,q5_claw

i.e. the Cartesian goal for the end-effector followed by the arm joints in
kinematic order, with the claw opening last. That reference is a 4-joint arm +
claw, so ``--dof 4`` reproduces its header exactly; ``--dof 5`` adds the wrist
roll and the claw column shifts to ``q6_claw``. Genesis's DoF order is not the
URDF's joint-number order (the 5-DoF arm's wrist is ``joint_6``, the fingers are
``joint_5``/``joint_7``), so the column -> joint mapping is written to a sidecar
``.meta.json`` rather than left implicit in the column names.

Two sampling modes, and the difference matters for what the net learns:

  fk  sample joint angles uniformly inside the limits, run forward kinematics,
      record where the EE landed. Every row is exact and reachable by
      construction, but the targets clump wherever the Jacobian is small and a
      given target may appear with several unrelated arm configurations.
  ik  sample targets in the reachable shell and solve. Matches how the
      reference set behaves (its q3 column piles up on a single value, the
      signature of a solver pinned against a joint limit) and yields one
      consistent branch per target, at the cost of throwing away targets the
      solver misses.

Every row is verified by forward kinematics before it is written, whichever
mode produced it -- an IK solver that returns a non-converged qpos is the main
way this dataset silently fills with wrong labels.

    ./run.sh scripts/make_ik_dataset.py --n 20000
    ./run.sh scripts/make_ik_dataset.py --dof 5 --mode fk --out data/fk_5dof.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np
import tyro

from lss_common import (
    REPO, EE_LINK, add_arm, fk as _fk, rel as _rel, split_dofs as _split_dofs, to_np as _np,
)


@dataclass
class Config:
    dof: int = 4                       # 4 reproduces the reference header exactly
    n: int = 10_000                    # rows to write
    mode: str = "ik"                   # ik | fk
    seed: int = 0
    out: Path = REPO / "data" / "training_dataset.csv"
    backend: str = "cpu"               # IK here is a kinematics loop; cpu wins on latency
    claw: str = "zero"                 # zero | random -- reference set is all zeros
    pos_tol: float = 5e-4              # solver tolerance AND the FK acceptance gate (m)
    z_min: float = 0.02                # drop targets at/below the table the arm bolts to
    max_attempts_factor: int = 20      # give up after n * this IK attempts


def _build(cfg: Config):
    gs.init(backend=getattr(gs, cfg.backend), logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    arm = add_arm(scene, cfg.dof)
    scene.build()
    return scene, arm


def _reach_envelope(arm, ee, arm_dofs, lower, upper, rng, n=4000):
    """Radius/height shell the EE actually occupies, measured rather than assumed --
    a hand-guessed box wastes most IK attempts on unreachable targets."""
    qpos = np.zeros(arm.n_qs)
    pts = np.empty((n, 3))
    for i in range(n):
        qpos[arm_dofs] = rng.uniform(lower[arm_dofs], upper[arm_dofs])
        pts[i] = _fk(arm, ee, qpos)
    r = np.linalg.norm(pts, axis=1)
    return float(np.percentile(r, 1)), float(np.percentile(r, 99)), float(pts[:, 2].max())


def main(cfg: Config) -> None:
    if cfg.mode not in ("ik", "fk"):
        raise SystemExit(f"--mode must be ik or fk, got {cfg.mode!r}")
    rng = np.random.default_rng(cfg.seed)

    scene, arm = _build(cfg)
    ee = arm.get_link(EE_LINK)
    arm_dofs, finger_dofs, joint_names = _split_dofs(arm)
    lower, upper = (_np(x) for x in arm.get_dofs_limit())

    rows = np.empty((cfg.n, 3 + len(arm_dofs) + 1))
    qpos = np.zeros(arm.n_qs)
    kept = attempts = 0

    if cfg.mode == "fk":
        while kept < cfg.n:
            qpos[arm_dofs] = rng.uniform(lower[arm_dofs], upper[arm_dofs])
            pos = _fk(arm, ee, qpos)
            attempts += 1
            if pos[2] < cfg.z_min:
                continue
            rows[kept, :3] = pos
            rows[kept, 3:-1] = qpos[arm_dofs]
            kept += 1
    else:
        r_lo, r_hi, z_hi = _reach_envelope(arm, ee, arm_dofs, lower, upper, rng)
        print(f"reachable shell: r {r_lo:.3f}..{r_hi:.3f} m, z up to {z_hi:.3f} m")
        budget = cfg.n * cfg.max_attempts_factor
        while kept < cfg.n and attempts < budget:
            attempts += 1
            target = _sample_shell(rng, r_lo, r_hi, cfg.z_min, z_hi)
            # seed each solve from a fresh random pose: a fixed seed pose makes the
            # solver return one branch everywhere and the set loses its variety
            qpos[arm_dofs] = rng.uniform(lower[arm_dofs], upper[arm_dofs])
            arm.set_qpos(qpos)
            sol = _np(arm.inverse_kinematics(
                link=ee, pos=target, rot_mask=[False, False, False],
                pos_tol=cfg.pos_tol, dofs_idx_local=arm_dofs,
                respect_joint_limit=True,
            ))
            # trust nothing: re-run FK on what came back and measure the miss
            reached = _fk(arm, ee, sol)
            if np.linalg.norm(reached - target) > cfg.pos_tol:
                continue
            if np.any(sol[arm_dofs] < lower[arm_dofs] - 1e-6) or np.any(
                sol[arm_dofs] > upper[arm_dofs] + 1e-6
            ):
                continue
            rows[kept, :3] = target
            rows[kept, 3:-1] = sol[arm_dofs]
            kept += 1
        if kept < cfg.n:
            raise SystemExit(
                f"only {kept}/{cfg.n} targets solved in {attempts} attempts -- "
                "raise --max-attempts-factor or loosen --pos-tol"
            )

    if cfg.claw == "random" and finger_dofs:
        d = finger_dofs[0]
        rows[:, -1] = rng.uniform(max(lower[d], 0.0), upper[d], size=cfg.n)
    else:
        rows[:, -1] = 0.0

    n_arm = len(arm_dofs)
    header = ["target_x", "target_y", "target_z"]
    header += [f"q{i + 1}" for i in range(n_arm)] + [f"q{n_arm + 1}_claw"]

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(cfg.out, rows, delimiter=",", fmt="%.10g", header=",".join(header), comments="")
    meta = {
        "urdf": f"assets/lss_arm/lss_arm_{cfg.dof}dof.urdf",
        "ee_link": EE_LINK,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "rows": int(cfg.n),
        "attempts": int(attempts),
        "pos_tol_m": cfg.pos_tol,
        "claw": cfg.claw,
        # the whole reason this file exists: qN is a Genesis DoF index, not a
        # URDF joint number, and for the 5-DoF arm the two disagree
        "columns": {
            **{f"q{i + 1}": joint_names[i] for i in range(n_arm)},
            f"q{n_arm + 1}_claw": "gripper opening (rad, finger joint)",
        },
        "joint_limits": {
            joint_names[i]: [lower[arm_dofs[i]], upper[arm_dofs[i]]] for i in range(n_arm)
        },
    }
    meta_path = cfg.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    err = np.linalg.norm(rows[:, :3], axis=1)
    print(f"wrote {_rel(cfg.out)}  {cfg.n} rows from {attempts} attempts "
          f"({100 * cfg.n / attempts:.1f}% yield)")
    print(f"      target radius {err.min():.3f}..{err.max():.3f} m, "
          f"z {rows[:, 2].min():.3f}..{rows[:, 2].max():.3f} m")
    print(f"wrote {_rel(meta_path)}")
    scene.destroy()


def _sample_shell(rng, r_lo: float, r_hi: float, z_min: float, z_max: float) -> np.ndarray:
    """Uniform-by-volume in the spherical shell, rejected down to the z band."""
    while True:
        u = rng.uniform(r_lo**3, r_hi**3)
        r = u ** (1 / 3)
        v = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2 * np.pi)
        s = np.sqrt(1 - v * v)
        p = np.array([r * s * np.cos(phi), r * s * np.sin(phi), r * v])
        if z_min <= p[2] <= z_max:
            return p


if __name__ == "__main__":
    main(tyro.cli(Config))
