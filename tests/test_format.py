"""Tests the Mujoco Animator file format."""

import math
import random
from pathlib import Path

from mujoco_animator.format import MjAnim


def test_save_and_load(tmpdir: Path) -> None:
    num_dofs = 10
    num_steps = 12
    dt = 0.1
    anim = MjAnim(num_dofs)
    total_tsz = 0.0
    for _ in range(num_steps):
        tsz = random.random()
        total_tsz += tsz
        anim.add_frame(tsz, [random.random() for _ in range(num_dofs)])
    anim.save(tmpdir / "test.mjanim")
    anim2 = MjAnim.load(tmpdir / "test.mjanim")
    assert anim == anim2

    nd_array = anim.to_numpy(dt, interp="cubic")
    assert nd_array.shape == (math.ceil(total_tsz / dt), num_dofs)

    nd_array = anim.to_numpy(dt, interp="linear")
    assert nd_array.shape == (math.ceil(total_tsz / dt), num_dofs)
