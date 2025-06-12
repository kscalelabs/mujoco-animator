"""Defines the command-line interface for interacting with Mujoco models."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        from PySide6.QtWidgets import QApplication

        from mujoco_animator.animator import MjAnimator

    except ImportError as e:
        raise ImportError(
            "To use the CLI, you should install the additional 'cli' dependencies using the following command: "
            "pip install 'mujoco-animator[cli]'"
        ) from e

    # Create Qt application first
    app = QApplication(sys.argv)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Mujoco Animator")
    parser.add_argument("model", type=str)
    parser.add_argument("--output", type=Path, default=Path("output.mjanim"))
    parser.add_argument("--template", type=Path, default=None)
    args = parser.parse_args()

    # Create and show animator
    animator = MjAnimator(Path(args.model), args.output, args.template)
    animator.show()

    # Run the event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
