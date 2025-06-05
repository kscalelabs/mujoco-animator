# ruff: noqa: N802
"""Core animator functionality for creating and editing animations."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mujoco_animator.format import Frame, MjAnim
from mujoco_animator.viewer import QtMujocoViewer


@dataclass
class AnimationState:
    """Current state of the animation being edited."""

    anim: MjAnim
    current_frame: int = 0
    selected_dof: int = 0


class MjAnimator(QMainWindow):
    """Main window for the Mujoco Animator tool."""

    def __init__(self, model_path: Path, output_path: Optional[Path] = None) -> None:
        """Initialize the animator.

        Args:
            model_path: Path to the Mujoco model file
            output_path: Path to save the animation (optional)
        """
        super().__init__()

        # Set window title first
        self.setWindowTitle("Mujoco Animator")

        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Initialize animation
        self.state = AnimationState(MjAnim(self.model.nq))
        self.output_path = output_path

        # Add first frame.
        self.add_frame()

        # Create central widget and layout first
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._main_layout = QVBoxLayout(central_widget)

        # Set up viewer with proper parent
        self.viewer = QtMujocoViewer(self.model, self.data, central_widget)
        self.viewer.set_camera(-1)  # Free camera
        self.viewer.set_key_callback(self.handle_key)

        # Add viewer to layout
        self._main_layout.addWidget(self.viewer)

        # Set up remaining UI elements
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Set minimum size to ensure OpenGL context has space
        self.setMinimumSize(800, 600)

    def showEvent(self, event: QShowEvent) -> None:
        """Handle show event to ensure proper OpenGL initialization."""
        super().showEvent(event)

        # Let Qt handle OpenGL initialization automatically
        if hasattr(self, "viewer"):
            self.viewer.update()
            # Ensure the viewer has focus to receive key events
            self.viewer.setFocus()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        if hasattr(self, "viewer"):
            self.viewer.closeEvent(event)
        super().closeEvent(event)

    def setup_ui(self) -> None:
        """Set up the user interface."""
        # Create controls
        controls = QHBoxLayout()

        # Buttons
        self.add_frame_btn = QPushButton("Add Frame (Space)")
        self.add_frame_btn.clicked.connect(self.add_frame)
        controls.addWidget(self.add_frame_btn)

        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.save_btn.clicked.connect(self.save_animation)
        controls.addWidget(self.save_btn)

        self._main_layout.addLayout(controls)

    def connect_signals(self) -> None:
        """Connect keyboard shortcuts."""
        pass  # Already connected in __init__

    def handle_key(self, key: int, scancode: int, action: int, mods: Qt.KeyboardModifier) -> None:
        """Handle keyboard input."""
        print("action:", action)

        if action != 1:  # Only handle key press events
            return

        match key:
            case Qt.Key.Key_Space:
                self.add_frame()
            case Qt.Key.Key_S if mods & Qt.KeyboardModifier.ControlModifier:
                self.save_animation()
            case Qt.Key.Key_Plus | Qt.Key.Key_Equal:
                self.adjust_dof(0.1)
            case Qt.Key.Key_Minus:
                self.adjust_dof(-0.1)
            case Qt.Key.Key_Left:
                self.on_dof_changed(self.state.selected_dof - 1)
            case Qt.Key.Key_Right:
                self.on_dof_changed(self.state.selected_dof + 1)
            case Qt.Key.Key_Escape:
                self.close()
            case _:
                pass

    def on_dof_changed(self, value: int) -> None:
        """Handle DOF selection change."""
        if 0 <= value < self.model.nq:
            self.state.selected_dof = value
            self.data.qpos[:] = self.state.anim.frames[self.state.current_frame].positions
            self.viewer.update()

    def on_frame_changed(self, value: int) -> None:
        """Handle frame selection change."""
        if 0 <= value < len(self.state.anim.frames):
            self.state.current_frame = value
            self.data.qpos[:] = self.state.anim.frames[value].positions
            self.viewer.update()

    def adjust_dof(self, delta: float) -> None:
        """Adjust the current DOF value."""
        if not self.state.anim.frames:
            return

        frame = self.state.anim.frames[self.state.current_frame]
        frame.positions[self.state.selected_dof] += delta
        self.data.qpos[:] = frame.positions
        self.viewer.update()

    def add_frame(self) -> None:
        """Add a new frame to the animation."""
        # Create a new frame with current positions
        frame = Frame(0.1, list(self.data.qpos))  # Default 0.1s duration
        self.state.anim.add_frame(frame.length, frame.positions)

    def save_animation(self) -> None:
        """Save the animation to file."""
        if self.output_path is None:
            return
        self.state.anim.save(self.output_path)


def main() -> None:
    app = QApplication(sys.argv)

    if len(sys.argv) < 2:
        print("Usage: python -m mujoco_animator.animator <model_path> [output_path]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    animator = MjAnimator(model_path, output_path)
    animator.show()
    sys.exit(app.exec())
