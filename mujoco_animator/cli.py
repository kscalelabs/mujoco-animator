"""Defines the command-line interface for interacting with Mujoco models."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from mujoco_animator.animator import MjAnimator


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Create Qt application first
    app = QApplication(sys.argv)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Mujoco Animator")
    parser.add_argument("model", type=str)
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    args = parser.parse_args()

    try:
        # Create and show animator
        animator = MjAnimator(Path(args.model), args.output)
        animator.show()

        # Run the event loop
        return app.exec()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
