"""Helper functions for the animator."""

from pathlib import Path

import mujoco as mj

from mujoco_animator.format import MjAnim


def annotate_clearance(
    model_path: Path,
    anim: MjAnim,
    left_site: str = "left_foot",
    right_site: str = "right_foot",
) -> None:
    """Adds `foot_clearance` to every Frame (in-place)."""
    # load model once
    m = mj.MjModel.from_xml_path(str(model_path))
    d = mj.MjData(m)

    floor = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
    lgeom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "KB_D_501L_L_LEG_FOOT_collision_capsule_0")
    rgeom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "KB_D_501R_R_LEG_FOOT_collision_capsule_0")

    # iterate
    for f in anim.frames:
        d.qpos[:] = f.positions
        mj.mj_forward(m, d)
        left_foot_dist = mj.mj_geomDistance(m, d, lgeom, floor, 1.0, None)
        right_foot_dist = mj.mj_geomDistance(m, d, rgeom, floor, 1.0, None)
        f.foot_clearance = min(left_foot_dist, right_foot_dist)
