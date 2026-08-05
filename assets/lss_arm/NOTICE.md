# Third-party asset attribution

The URDFs and STL meshes in this directory are derived from the
`lss_arm_description` package of:

    https://github.com/Lynxmotion/LSS-ROS2-Arms

declared **Apache-2.0** in that package's `package.xml`. (The upstream
repository ships no top-level LICENSE file; the package manifest is the licence
statement of record. Full text: https://www.apache.org/licenses/LICENSE-2.0)

Modifications made here, all mechanical and reproducible via
`scripts/build_lss_urdf.py`:

- xacro expanded to flat URDF for `dof=4` and `dof=5`, with
  `ros2_control:=false` and `gazebo_preserve_fixed_joint:=false`
- `package://lss_arm_description/meshes/` mesh references rewritten to paths
  relative to the URDF
- `models/lss_arm_5dof/meshes` vendored as `meshes/` (the path upstream's
  CMakeLists installs it to, and a superset of the 4-DoF set)

The meshes themselves are unmodified.
