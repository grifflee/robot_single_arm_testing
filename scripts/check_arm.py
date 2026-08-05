"""Load the LSS arm into an empty Genesis scene and prove it is actually there.

Empty here means a ground plane and the arm, nothing else -- this is the base
scene that task objects get added to later.

Exiting 0 is not evidence the arm loaded right: a URDF whose meshes silently
failed to resolve still builds, and a black frame still writes a PNG. So this
prints the kinematic tree it actually got (links, DoFs, limits) and the
rendered bounding box of the robot, then writes one PNG per camera to eyeball.

    ./run.sh scripts/check_arm.py                    # 4-DoF, three stills
    ./run.sh scripts/check_arm.py --dof 5
    ./run.sh scripts/check_arm.py --video outputs/sweep.mp4   # + joint sweep
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np
import tyro

from lss_common import REPO, add_arm, rel as _rel, split_dofs, vis_options

# The LSS arm is small -- roughly 0.35 m of reach -- so the stock camera
# distances that suit an xArm7 frame it as a speck. These are tuned to it.
CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # name:            (pos,                    lookat)
    "three_quarter": ((0.45, -0.40, 0.35), (0.05, 0.0, 0.10)),
    "front": ((0.55, 0.0, 0.16), (0.0, 0.0, 0.12)),
    "top": ((0.10, 0.0, 0.60), (0.05, 0.0, 0.05)),
}


@dataclass
class Config:
    dof: int = 4                      # 4 or 5; picks assets/lss_arm/lss_arm_{dof}dof.urdf
    backend: str = "gpu"              # gpu | cpu
    res: tuple[int, int] = (640, 480)
    settle_steps: int = 200           # let gravity + the PD hold settle before the still
    out_dir: Path = REPO / "outputs" / "arm_check"
    video: Path | None = None         # also sweep every arm joint and write an mp4
    sweep_steps: int = 60             # frames per half-swing of each joint
    show_fps: bool = False


def _summarize(arm) -> None:
    """Print what Genesis actually built, not what the URDF said."""
    lower, upper = (np.asarray(x.cpu()) for x in arm.get_dofs_limit())
    print(f"\nentity: {arm.n_links} links, {arm.n_dofs} dofs, {arm.n_geoms} geoms")
    print(f"{'joint':28s} {'type':10s} {'dof_idx':>8s}  {'lower':>8s} {'upper':>8s}")
    for joint in arm.joints:
        idx = list(joint.dofs_idx_local)
        rng = ""
        if idx:
            rng = f"  {lower[idx[0]]:8.3f} {upper[idx[0]]:8.3f}"
        print(f"{joint.name:28s} {str(joint.type).split('.')[-1]:10s} {str(idx):>8s}{rng}")

    aabb = np.asarray(arm.get_AABB().cpu()).reshape(2, 3)
    span = aabb[1] - aabb[0]
    print(f"\nAABB min {aabb[0].round(4)}  max {aabb[1].round(4)}  span {span.round(4)}")
    if span.max() < 0.05:
        raise SystemExit("robot AABB is under 5 cm across -- meshes almost certainly did not load")


def main(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=getattr(gs, cfg.backend), logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,  # this host is headless (no DISPLAY); cameras render offscreen
        sim_options=gs.options.SimOptions(dt=1 / 100, substeps=4),
        rigid_options=gs.options.RigidOptions(
            # the fingers are a 2-finger pinch on a light payload; elliptic is
            # what makes such a grasp hold in Genesis, so start as we mean to go on
            constraint_timeconst=0.005,
        ),
        vis_options=vis_options(),
    )

    scene.add_entity(gs.morphs.Plane())
    arm = add_arm(scene, cfg.dof)

    cams = {
        name: scene.add_camera(res=cfg.res, pos=pos, lookat=lookat, fov=40, GUI=False)
        for name, (pos, lookat) in CAMERAS.items()
    }
    scene.build()

    _summarize(arm)

    # hold the zero pose so the arm does not simply fall over under gravity
    dofs = list(range(arm.n_dofs))
    arm.set_dofs_kp(np.full(arm.n_dofs, 50.0), dofs)
    arm.set_dofs_kv(np.full(arm.n_dofs, 5.0), dofs)
    home = np.zeros(arm.n_dofs)
    arm.set_dofs_position(home, dofs)
    arm.control_dofs_position(home, dofs)
    for _ in range(cfg.settle_steps):
        scene.step()

    for name, cam in cams.items():
        rgb = cam.render(rgb=True)[0]
        rgb = np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)[..., :3]
        path = cfg.out_dir / f"{cfg.dof}dof_{name}.png"
        _imwrite(path, rgb)
        print(f"{name:14s} mean brightness {rgb.mean():6.1f}  -> {_rel(path)}")
        if rgb.mean() < 2:
            print(f"  WARNING: {name} is essentially black")

    if cfg.video is not None:
        _sweep(scene, arm, cams["three_quarter"], cfg)


def _sweep(scene, arm, cam, cfg: Config) -> None:
    """Drive each arm joint to both limits in turn -- catches a joint that is
    frozen, inverted, or self-colliding in a way a still frame hides."""
    lower, upper = (np.asarray(x.cpu()) for x in arm.get_dofs_limit())
    dofs = list(range(arm.n_dofs))
    target = np.zeros(arm.n_dofs)

    cfg.video.parent.mkdir(parents=True, exist_ok=True)
    cam.start_recording()
    for d in split_dofs(arm)[0]:  # arm joints only; the fingers are not a reach DoF
        lo, hi = float(lower[d]), float(upper[d])
        # 60% of range keeps the sweep clear of the ground plane and of hard stops
        for frac in np.concatenate([np.linspace(0, 0.6, cfg.sweep_steps),
                                    np.linspace(0.6, -0.6, 2 * cfg.sweep_steps),
                                    np.linspace(-0.6, 0, cfg.sweep_steps)]):
            target[d] = hi * frac if frac > 0 else -lo * frac
            arm.control_dofs_position(target, dofs)
            scene.step()
            cam.render()
        target[d] = 0.0
    cam.stop_recording(save_to_filename=str(cfg.video), fps=60)
    print(f"wrote {cfg.video}")


def _imwrite(path: Path, rgb: np.ndarray) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, rgb.astype(np.uint8))


if __name__ == "__main__":
    main(tyro.cli(Config))
