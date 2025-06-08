"""Tests the Mujoco Animator file format."""

import math
import random
from pathlib import Path

from mujoco_animator import MjAnim


def test_save_and_load(tmpdir: Path) -> None:
    num_dofs = 10
    num_steps = 12
    dt = 0.1
    anim = MjAnim(num_dofs)
    total_tsz = 0.0
    for i in range(num_steps):
        tsz = random.random()
        if i != num_steps - 1:
            total_tsz += tsz
        anim.add_frame(tsz, [random.random() for _ in range(num_dofs)])
    anim.save_binary(tmpdir / "test.mjanim")
    anim2 = MjAnim.load_binary(tmpdir / "test.mjanim")
    assert anim == anim2

    nd_array = anim.to_numpy(dt, interp="cubic", loop=False)
    assert nd_array.shape == (math.ceil(total_tsz / dt), num_dofs)

    nd_array = anim.to_numpy(dt, interp="linear", loop=False)
    assert nd_array.shape == (math.ceil(total_tsz / dt), num_dofs)
