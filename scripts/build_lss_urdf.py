"""Regenerate assets/lss_arm/ from the upstream Lynxmotion ROS 2 description.

Upstream (github.com/Lynxmotion/LSS-ROS2-Arms, Apache-2.0) ships only xacro, and
its ``$(find lss_arm_description)`` substitutions need a ROS 2 install to
resolve. This script gets a flat, ROS-free URDF out of it on a machine with no
ROS and no sudo:

  1. shallow-clone upstream into a temp dir
  2. fake the two things xacro asks the ROS environment for
       - an ament index (a marker file + a ``share/<pkg>`` dir per package)
       - the ``ament_index_python`` module itself (a ~20-line shim)
  3. run xacro (from PyPI, via ``uv tool run``) for dof=4 and dof=5
  4. rewrite ``package://lss_arm_description/meshes/`` to a plain relative
     ``meshes/``, which is what Genesis resolves against the URDF's own dir
  5. copy the meshes next to the URDFs

Two upstream quirks worth knowing before editing this:

  - there is no ``lss_arm_description/meshes/`` in the source tree. The package
    installs ``models/lss_arm_5dof/meshes`` to that path (see its CMakeLists),
    so the 5-DoF mesh set is canonical and is a superset of the 4-DoF one --
    every shared STL is byte-identical. We vendor it once for both URDFs.
  - ``ros2_control:=false`` is passed on purpose. Genesis ignores
    ``<ros2_control>`` blocks, but leaving them on drags in a
    ``$(find lss_arm_moveit)`` lookup for a controllers yaml we do not have.

    ./run.sh scripts/build_lss_urdf.py            # rewrite assets/lss_arm/
    ./run.sh scripts/build_lss_urdf.py --keep-temp
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "assets" / "lss_arm"
UPSTREAM = "https://github.com/Lynxmotion/LSS-ROS2-Arms.git"
DOFS = (4, 5)

# Minimal stand-in for the ROS 2 package-lookup module xacro imports for $(find).
AMENT_SHIM = '''\
"""Minimal stand-in so xacro's $(find pkg) resolves without a ROS 2 install."""
import os


class PackageNotFoundError(KeyError):
    pass


def _prefixes():
    return [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]


def get_package_prefix(name):
    for prefix in _prefixes():
        if os.path.isdir(os.path.join(prefix, "share", name)):
            return prefix
    raise PackageNotFoundError(name)


def get_package_share_directory(name):
    return os.path.join(get_package_prefix(name), "share", name)
'''


@dataclass
class Config:
    out_dir: Path = OUT_DIR
    keep_temp: bool = False  # leave the clone + staging behind for inspection


def _stage_ament(tmp: Path, desc: Path) -> tuple[Path, Path]:
    """Build the fake ament prefix and the shim package; return (prefix, shim)."""
    prefix = tmp / "ament_prefix"
    (prefix / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True)
    for pkg in ("lss_arm_description", "lss_arm_moveit"):
        (prefix / "share" / "ament_index" / "resource_index" / "packages" / pkg).touch()
    (prefix / "share" / "lss_arm_description").symlink_to(desc)
    (prefix / "share" / "lss_arm_moveit").mkdir()

    shim = tmp / "shim"
    (shim / "ament_index_python").mkdir(parents=True)
    (shim / "ament_index_python" / "packages.py").write_text(AMENT_SHIM)
    (shim / "ament_index_python" / "__init__.py").write_text(
        "from .packages import get_package_share_directory, get_package_prefix  # noqa: F401\n"
    )
    return prefix, shim


def main(cfg: Config) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="lss_urdf_"))
    try:
        subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(tmp / "src")], check=True)
        desc = tmp / "src" / "lss_arm_description"
        meshes = desc / "models" / "lss_arm_5dof" / "meshes"

        prefix, shim = _stage_ament(tmp, desc)
        env = {"AMENT_PREFIX_PATH": str(prefix), "PYTHONPATH": str(shim)}

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        for dof in DOFS:
            raw = tmp / f"lss_arm_{dof}dof.urdf"
            subprocess.run(
                ["uv", "tool", "run", "--from", "xacro", "xacro",
                 str(desc / "urdf" / "lss_arm.urdf.xacro"),
                 f"name:=lss_arm_{dof}dof", f"dof:={dof}",
                 "collision:=true", "ros2_control:=false",
                 "gazebo_preserve_fixed_joint:=false",
                 "-o", str(raw)],
                check=True, env={**_environ(), **env},
            )
            urdf = raw.read_text().replace("package://lss_arm_description/meshes/", "meshes/")
            dst = cfg.out_dir / f"lss_arm_{dof}dof.urdf"
            dst.write_text(urdf)
            print(f"wrote {dst.relative_to(REPO)}")

        shutil.rmtree(cfg.out_dir / "meshes", ignore_errors=True)
        shutil.copytree(meshes, cfg.out_dir / "meshes")
        print(f"wrote {(cfg.out_dir / 'meshes').relative_to(REPO)}")
        _verify(cfg.out_dir)
    finally:
        if cfg.keep_temp:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _verify(out_dir: Path) -> None:
    """Every mesh the URDFs name must exist, or Genesis fails deep inside the loader."""
    import re

    missing = []
    for dof in DOFS:
        text = (out_dir / f"lss_arm_{dof}dof.urdf").read_text()
        for ref in sorted(set(re.findall(r'filename="(meshes/[^"]+)"', text))):
            if not (out_dir / ref).is_file():
                missing.append(f"{dof}dof -> {ref}")
    if missing:
        raise SystemExit("missing meshes:\n  " + "\n  ".join(missing))
    print("all mesh references resolve")


if __name__ == "__main__":
    main(tyro.cli(Config))
