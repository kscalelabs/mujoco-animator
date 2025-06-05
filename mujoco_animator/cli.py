"""Defines the command-line interface for interacting with Mujoco models."""

import argparse
from pathlib import Path

import mujoco

from mujoco_animator.viewer import GlfwMujocoViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Mujoco Animator")
    parser.add_argument("model", type=str)
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    viewer = GlfwMujocoViewer(model, data)

    while viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()


if __name__ == "__main__":
    main()
