"""Shared bits for the LSS arm scripts: loading it, and reading its kinematics.

Imported by sibling scripts in this directory (running ``./run.sh scripts/x.py``
puts ``scripts/`` on sys.path, so a plain ``import lss_common`` works).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import genesis as gs
import numpy as np

REPO = Path(__file__).resolve().parent.parent
URDF_DIR = REPO / "assets" / "lss_arm"
EE_LINK = "lss_arm_ee"

# The real LSS arm is black anodised brackets over black servo cases. Plastic
# rather than Rough, and charcoal rather than true black: an unlit 0,0,0 mesh
# renders as a flat silhouette with no readable geometry, and the small
# specular lobe is what puts the link edges back.
ARM_COLOR = (0.055, 0.055, 0.06)


def arm_surface() -> gs.surfaces.Surface:
    return gs.surfaces.Plastic(color=ARM_COLOR)


def vis_options() -> gs.options.VisOptions:
    """Shared look, so stills and episode videos are directly comparable.

    The arm is painted near-black to match the real hardware, which under the
    stock ambient reads as a flat silhouette against the dark floor; lifting
    ambient is what puts the link geometry back.
    """
    return gs.options.VisOptions(
        plane_reflection=False,
        ambient_light=(0.45, 0.45, 0.48),
        background_color=(0.16, 0.17, 0.20),
    )


def begin_recording(cam, path: Path, fps: int) -> Callable[[], None]:
    """Start recording; return the call that finishes the mp4.

    Genesis moved the output filename and fps from ``stop_recording`` to
    ``start_recording`` in 1.3. ``pyproject.toml`` floors genesis-world at 1.2
    without capping it, so a fresh ``uv sync`` lands on the new API while a
    pinned older environment keeps the old one -- and passing the wrong one is a
    hard TypeError, not a warning. Branch on the signature rather than the
    version string, which is what actually determines the call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if "save_to_filename" in inspect.signature(cam.start_recording).parameters:
        cam.start_recording(save_to_filename=str(path), fps=fps)
        return cam.stop_recording
    cam.start_recording()
    return lambda: cam.stop_recording(save_to_filename=str(path), fps=fps)


def urdf_path(dof: int) -> Path:
    p = URDF_DIR / f"lss_arm_{dof}dof.urdf"
    if not p.is_file():
        raise SystemExit(f"{p} not found -- run scripts/build_lss_urdf.py")
    return p


def add_arm(scene, dof: int):
    """Add the arm, painted, with the virtual EE frame preserved.

    ``links_to_keep`` matters: fixed-link merging would otherwise dissolve
    ``lss_arm_ee`` and leave no TCP frame to aim IK at.
    """
    return scene.add_entity(
        gs.morphs.URDF(file=str(urdf_path(dof)), fixed=True, links_to_keep=[EE_LINK]),
        surface=arm_surface(),
    )


def to_np(x) -> np.ndarray:
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=np.float64)


def split_dofs(arm) -> tuple[list[int], list[int], list[str]]:
    """(arm dof indices, finger dof indices, arm joint names) in Genesis order.

    Classify on the child LINK, not the joint name: on the 4-DoF arm the two
    finger joints are ``joint_5``/``joint_6``, so a name-based filter hands the
    gripper to the IK solver and every downstream q column shifts.
    """
    arm_dofs, finger_dofs, names = [], [], []
    for joint in arm.joints:
        idx = list(joint.dofs_idx_local)
        if not idx:
            continue
        if "finger" in joint.link.name:
            finger_dofs += idx
        else:
            arm_dofs += idx
            names += [joint.name] * len(idx)
    return arm_dofs, finger_dofs, names


def fk(arm, ee, qpos: np.ndarray) -> np.ndarray:
    """EE position for a full qpos. set_qpos refreshes link transforms, no step needed."""
    arm.set_qpos(qpos)
    return to_np(ee.get_pos()).reshape(3)


def rel(p: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (--out may point anywhere)."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)
