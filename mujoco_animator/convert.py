"""Convert animation file from one supported format to another.

Accepts json or mjanim, dependent on file extension of input_path and output_path.
"""

import argparse
from pathlib import Path

from mujoco_animator.format import MjAnim


def convert_anim_file(input_path: Path, output_path: Path) -> None:
    if input_path.suffix == ".json" and output_path.suffix == ".mjanim":
        anim = MjAnim.load_json(input_path)
        anim.save_binary(output_path)
    elif input_path.suffix == ".mjanim" and output_path.suffix == ".json":
        anim = MjAnim.load_binary(input_path)
        anim.save_json(output_path)
    else:
        raise ValueError("Invalid input or output file type")

    print(f"Converted {input_path} to {output_path}")


def main(input_path: Path, output_path: Path) -> None:
    convert_anim_file(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
