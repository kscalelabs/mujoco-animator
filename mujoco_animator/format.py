"""Defines the file format for Mujoco Animator files.

The file format contains a header and a body. The header starts with the
magic number "MJAN" and is followed by a 4-byte integer specifying the
number of degrees of freedom, followed by a 4-byte integer specifying the
number of keyframes. Following the header, the body contains the keyframes,
which are each formatted as follows:

 - 4-byte float specifying the length of the keyframe in seconds
 - N 4-byte floats specifying the joint positions (in radians) of each
   degree of freedom
"""

__all__ = [
    "MjAnim",
]

import copy
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


@dataclass
class Frame:
    length: float
    positions: list[float]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return False
        if not math.isclose(self.length, other.length, rel_tol=1e-6):
            return False
        if len(self.positions) != len(other.positions):
            return False
        for i in range(len(self.positions)):
            if not math.isclose(self.positions[i], other.positions[i], rel_tol=1e-6):
                return False
        return True


class MjAnim:
    def __init__(self, num_dofs: int, frames: list[Frame] | None = None) -> None:
        super().__init__()

        self.num_dofs: int = num_dofs
        self.frames: list[Frame] = [] if frames is None else frames

    def add_frame(self, length: float, positions: list[float], index: int | None = None) -> int:
        if len(positions) != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} positions, got {len(positions)}")
        if index is None:
            self.frames.append(Frame(length, positions))
        else:
            self.frames.insert(index, Frame(length, positions))
        return len(self.frames) - 1 if index is None else index

    # Binary packed save
    def save_binary(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(b"MJAN")
            f.write(struct.pack("<I", self.num_dofs))
            f.write(struct.pack("<I", len(self.frames)))
            for frame in self.frames:
                f.write(struct.pack("<f", frame.length))
                f.write(struct.pack(f"<{len(frame.positions)}f", *frame.positions))

    # JSON save data
    def save_json(self, path: Path) -> None:
        json_path = path.with_suffix(".json")

        json_data: dict[str, int | list[dict[str, int | float | list[float]]]] = {
            "num_dofs": self.num_dofs,
            "num_frames": len(self.frames),
        }

        json_frames: list[dict[str, int | float | list[float]]] = []
        for i, frame in enumerate(self.frames):
            json_frames.append(
                {
                    "index": i,
                    "length": frame.length,
                    "positions": frame.positions,
                }
            )
        json_data["frames"] = json_frames
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    @classmethod
    def load_binary(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            if f.read(4) != b"MJAN":
                raise ValueError("Invalid file format")
            num_dofs = struct.unpack("<I", f.read(4))[0]
            num_frames = struct.unpack("<I", f.read(4))[0]
            frames = []
            for _ in range(num_frames):
                length = struct.unpack("<f", f.read(4))[0]
                positions = struct.unpack(f"<{num_dofs}f", f.read(4 * num_dofs))
                frames.append(Frame(length, list(positions)))
        return cls(num_dofs, frames)

    @classmethod
    def load_json(cls, path: Path) -> Self:
        json_path = path.with_suffix(".json")
        with open(json_path, "r") as f:
            json_data = json.load(f)
            num_dofs = json_data["num_dofs"]
            num_frames = json_data["num_frames"]
            frames = []
            for i in range(num_frames):
                length = json_data["frames"][i]["length"]
                positions = json_data["frames"][i]["positions"]
                frames.append(Frame(length, positions))
        return cls(num_dofs, frames)

    @classmethod
    def load(cls, path: Path) -> Self:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".mjanim":
            return cls.load_binary(path)
        elif suffix == ".json":
            return cls.load_json(path)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}. Supported formats: .mjanim, .json")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MjAnim):
            return False
        return self.num_dofs == other.num_dofs and self.frames == other.frames

    def to_numpy(self, dt: float, interp: Literal["linear", "cubic"] = "cubic", loop: bool = False) -> np.ndarray:
        """Convert animation frames to a numpy array with evenly spaced time steps.

        Args:
            dt: Time step in seconds between each frame in the output array.
            interp: Interpolation method to use ("cubic" or "linear").
            loop: Whether to loop the animation, by having the last frame return
                to the first frame.

        Returns:
            A numpy array of shape (num_steps, num_dofs) containing the joint
            positions at each time step. The positions are interpolated using
            the specified method.
        """
        if not self.frames:
            return np.zeros((0, self.num_dofs))

        frames = copy.deepcopy(self.frames)
        if len(frames) == 0:
            return np.zeros((0, self.num_dofs))
        if len(frames) == 1:
            return np.array([frames[0].positions])

        if loop:
            frames.append(Frame(0.0, frames[0].positions))
        else:
            frames[-1].length = 0.0

        # Calculate cumulative times for each frame.
        times = np.cumsum([frame.length for frame in frames])
        times = np.concatenate([[0], times])

        # Calculate total duration and number of steps
        total_duration = times[-1]
        num_steps = int(np.ceil(total_duration / dt))

        # Create time points for output, ensuring we don't exceed total_duration.
        output_times = np.linspace(0, total_duration, num_steps, endpoint=True)

        # Create output array
        positions = np.zeros((num_steps, self.num_dofs))

        for dof in range(self.num_dofs):
            dof_positions = np.array([frame.positions[dof] for frame in frames])
            if interp == "cubic":
                spline = CubicSpline(times[:-1], dof_positions, bc_type="natural")
            elif interp == "linear":
                spline = interp1d(
                    times[:-1],
                    dof_positions,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                raise ValueError(f"Invalid interpolation type: {interp}")
            positions[:, dof] = spline(output_times)

        return positions
