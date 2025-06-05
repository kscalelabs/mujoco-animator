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

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from scipy.interpolate import CubicSpline


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

    def add_frame(self, length: float, positions: list[float]) -> None:
        if len(positions) != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} positions, got {len(positions)}")
        self.frames.append(Frame(length, positions))

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(b"MJAN")
            f.write(struct.pack("<I", self.num_dofs))
            f.write(struct.pack("<I", len(self.frames)))
            for frame in self.frames:
                f.write(struct.pack("<f", frame.length))
                f.write(struct.pack(f"<{len(frame.positions)}f", *frame.positions))

    @classmethod
    def load(cls, path: Path) -> Self:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MjAnim):
            return False
        return self.num_dofs == other.num_dofs and self.frames == other.frames

    def to_numpy(self, dt: float) -> np.ndarray:
        """Convert animation frames to a numpy array with evenly spaced time steps.

        Args:
            dt: Time step in seconds between each frame in the output array.

        Returns:
            A numpy array of shape (num_steps, num_dofs) containing the joint positions
            at each time step. The positions are interpolated using cubic splines.
        """
        if not self.frames:
            return np.zeros((0, self.num_dofs))

        # Calculate total duration and number of steps
        total_duration = sum(frame.length for frame in self.frames)
        num_steps = int(np.ceil(total_duration / dt))

        # Create output array
        positions = np.zeros((num_steps, self.num_dofs))

        # Calculate cumulative times for each frame
        times = np.zeros(len(self.frames) + 1)
        for i, frame in enumerate(self.frames):
            times[i + 1] = times[i] + frame.length

        output_times = np.arange(0, total_duration, dt)

        for dof in range(self.num_dofs):
            dof_positions = np.array([frame.positions[dof] for frame in self.frames])
            spline = CubicSpline(times[:-1], dof_positions, bc_type="natural")
            positions[:, dof] = spline(output_times)

        return positions
