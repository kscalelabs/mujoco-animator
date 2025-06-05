"""Tests the Mujoco Animator file format."""

import random
from pathlib import Path

from mujoco_animator.format import MjAnim


def test_save_and_load(tmpdir: Path) -> None:
    num_dofs = 10
    num_steps = 12
    anim = MjAnim(num_dofs)
    for _ in range(num_steps):
        anim.add_frame(random.random(), [random.random() for _ in range(num_dofs)])
    anim.save(tmpdir / "test.mjanim")
    anim2 = MjAnim.load(tmpdir / "test.mjanim")
    assert anim == anim2
