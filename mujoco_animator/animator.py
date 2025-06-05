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
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mujoco_animator.format import Frame, MjAnim
from mujoco_animator.viewer import QtMujocoViewer


@dataclass
class AnimationState:
    """Current state of the animation being edited."""

    anim: MjAnim
    current_frame: int = -1
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

        # Create central widget and layout first
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._main_layout = QHBoxLayout(central_widget)  # Changed to horizontal

        # Set up viewer with proper parent
        self.viewer = QtMujocoViewer(self.model, self.data, central_widget)
        self.viewer.set_camera(-1)  # Free camera
        self.viewer.set_key_callback(self.handle_key)

        # Add viewer to layout (takes up most space)
        self._main_layout.addWidget(self.viewer, stretch=3)

        # Create side panel
        self.setup_side_panel()

        # Set up remaining UI elements
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Set minimum size to ensure OpenGL context has space
        self.setMinimumSize(1200, 600)  # Increased width for side panel

        # Adds the first frame.
        self.add_frame()

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

    def setup_side_panel(self) -> None:
        """Set up the side panel with DOF controls."""
        # Create side panel widget
        side_panel = QWidget()
        side_panel.setMaximumWidth(300)
        side_panel.setMinimumWidth(250)
        side_panel_layout = QVBoxLayout(side_panel)

        # Title
        title = QLabel("DOF Values")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        side_panel_layout.addWidget(title)

        # Frame info
        self.frame_info = QLabel("Frame: 0 / 1")
        side_panel_layout.addWidget(self.frame_info)

        # Selected DOF info
        self.selected_dof_label = QLabel(f"Selected DOF: {self.state.selected_dof}")
        self.selected_dof_label.setStyleSheet("color: blue; font-weight: bold;")
        side_panel_layout.addWidget(self.selected_dof_label)

        # Create scroll area for DOF controls
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create DOF spinboxes
        self.dof_spinboxes = []
        for i in range(self.model.nq):
            dof_layout = QHBoxLayout()

            # Get DOF/joint name
            dof_name = self.get_dof_name(i)

            # DOF label
            dof_label = QLabel(f"{dof_name}:")
            dof_label.setMinimumWidth(80)  # Increased width for longer names
            dof_layout.addWidget(dof_label)

            # DOF value spinbox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-10.0, 10.0)  # Reasonable range for joint angles
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0.0)
            spinbox.valueChanged.connect(lambda value, dof=i: self.on_dof_value_changed(dof, value))
            self.dof_spinboxes.append(spinbox)
            dof_layout.addWidget(spinbox)

            scroll_layout.addLayout(dof_layout)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        side_panel_layout.addWidget(scroll_area)

        # Add side panel to main layout
        self._main_layout.addWidget(side_panel, stretch=1)

        # Initialize the side panel with current values
        self.update_side_panel()

    def get_dof_name(self, dof_index: int) -> str:
        """Get a meaningful name for a DOF."""
        try:
            # Look through all joints to find which one this DOF belongs to
            for joint_id in range(self.model.njnt):
                joint_qposadr = self.model.jnt_qposadr[joint_id]
                joint_type = self.model.jnt_type[joint_id]

                # Calculate how many DOFs this joint has
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    joint_dofs = 7  # free joint has 7 DOFs (3 pos + 4 quat)
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    joint_dofs = 4  # ball joint has 4 DOFs (quaternion)
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    joint_dofs = 1  # hinge joint has 1 DOF
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    joint_dofs = 1  # slide joint has 1 DOF
                else:
                    joint_dofs = 1  # default to 1 DOF

                # Check if this DOF belongs to this joint
                if joint_qposadr <= dof_index < joint_qposadr + joint_dofs:
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    if not joint_name:
                        joint_name = f"joint_{joint_id}"

                    if joint_dofs == 1:
                        return joint_name
                    else:
                        # For multi-DOF joints, add suffix
                        dof_offset = dof_index - joint_qposadr
                        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                            suffixes = ["_x", "_y", "_z", "_qw", "_qx", "_qy", "_qz"]
                            if dof_offset < len(suffixes):
                                return f"{joint_name}{suffixes[dof_offset]}"
                        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                            suffixes = ["_qw", "_qx", "_qy", "_qz"]
                            if dof_offset < len(suffixes):
                                return f"{joint_name}{suffixes[dof_offset]}"

                        return f"{joint_name}_{dof_offset}"

            # Fallback to index if no joint found
            return f"DOF_{dof_index}"

        except Exception:
            # Fallback to index if any error occurs
            return f"DOF_{dof_index}"

    def setup_ui(self) -> None:
        """Set up the user interface."""
        # Create bottom controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        # Buttons
        self.add_frame_btn = QPushButton("Add Frame (F)")
        self.add_frame_btn.clicked.connect(self.add_frame)
        controls_layout.addWidget(self.add_frame_btn)

        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.save_btn.clicked.connect(self.save_animation)
        controls_layout.addWidget(self.save_btn)

        # Add controls to the viewer side (not the side panel)
        viewer_and_controls = QWidget()
        viewer_controls_layout = QVBoxLayout(viewer_and_controls)
        viewer_controls_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        viewer_controls_layout.addWidget(self.viewer, stretch=1)  # Viewer takes all available space
        viewer_controls_layout.addWidget(controls, stretch=0)  # Controls stay at fixed size

        # Replace the viewer in the main layout with the viewer+controls widget
        self._main_layout.removeWidget(self.viewer)
        self._main_layout.insertWidget(0, viewer_and_controls, stretch=3)

    def connect_signals(self) -> None:
        """Connect keyboard shortcuts."""
        pass  # Already connected in __init__

    def handle_key(self, key: int, scancode: int, action: int, mods: Qt.KeyboardModifier) -> None:
        """Handle keyboard input."""
        if action != 1:  # Only handle key press events
            return

        match key:
            case Qt.Key.Key_F:
                self.add_frame()
            case Qt.Key.Key_Backspace:
                self.delete_frame()
            case Qt.Key.Key_S if mods & Qt.KeyboardModifier.ControlModifier:
                self.save_animation()
            case Qt.Key.Key_W:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.adjust_dof(0.01)
                else:
                    self.adjust_dof(0.1)
            case Qt.Key.Key_S:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.adjust_dof(-0.01)
                else:
                    self.adjust_dof(-0.1)
            case Qt.Key.Key_Q:
                self.on_frame_changed(self.state.current_frame - 1)
            case Qt.Key.Key_E:
                self.on_frame_changed(self.state.current_frame + 1)
            case Qt.Key.Key_A:
                self.on_dof_changed(self.state.selected_dof - 1)
            case Qt.Key.Key_D:
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
            self.update_side_panel()

    def on_dof_value_changed(self, dof: int, value: float) -> None:
        """Handle DOF value change from spinbox."""
        if self.state.anim.frames:
            self.state.anim.frames[self.state.current_frame].positions[dof] = value
            self.data.qpos[dof] = value
            self.viewer.update()

    def on_frame_changed(self, value: int) -> None:
        """Handle frame selection change."""
        if 0 <= value < len(self.state.anim.frames):
            self.state.current_frame = value
            self.data.qpos[:] = self.state.anim.frames[value].positions
            self.viewer.update()
            self.update_side_panel()

    def update_side_panel(self) -> None:
        """Update the side panel with current values."""
        # Update frame info
        total_frames = len(self.state.anim.frames)
        self.frame_info.setText(f"Frame: {self.state.current_frame + 1} / {total_frames}")

        # Update selected DOF label with name
        selected_dof_name = self.get_dof_name(self.state.selected_dof)
        self.selected_dof_label.setText(f"Selected: {selected_dof_name}")

        # Update DOF values
        if self.state.anim.frames:
            current_positions = self.state.anim.frames[self.state.current_frame].positions
            for i, spinbox in enumerate(self.dof_spinboxes):
                # Temporarily disconnect to avoid triggering value change
                spinbox.blockSignals(True)
                spinbox.setValue(current_positions[i])

                # Highlight selected DOF
                if i == self.state.selected_dof:
                    spinbox.setStyleSheet("""
                        QDoubleSpinBox {
                            background-color: #4A90E2;
                            color: white;
                            font-weight: bold;
                            border: 2px solid #2E5C9A;
                        }
                    """)
                else:
                    spinbox.setStyleSheet("")

                spinbox.blockSignals(False)

    def adjust_dof(self, delta: float) -> None:
        """Adjust the current DOF value."""
        if not self.state.anim.frames:
            return

        frame = self.state.anim.frames[self.state.current_frame]
        frame.positions[self.state.selected_dof] += delta
        self.data.qpos[:] = frame.positions
        self.viewer.update()
        self.update_side_panel()

    def add_frame(self) -> None:
        """Add a new frame to the animation."""
        # Create a new frame with current positions
        frame = Frame(0.1, list(self.data.qpos))  # Default 0.1s duration
        index = self.state.anim.add_frame(frame.length, frame.positions, index=self.state.current_frame + 1)

        # Update current frame to the newly added one
        self.state.current_frame = index
        self.update_side_panel()

    def delete_frame(self) -> None:
        """Delete the current frame."""
        if self.state.current_frame == -1 or len(self.state.anim.frames) == 1:
            return
        self.state.anim.frames.pop(self.state.current_frame)
        if self.state.current_frame == len(self.state.anim.frames):
            self.state.current_frame -= 1
        self.data.qpos[:] = self.state.anim.frames[self.state.current_frame].positions
        self.update_side_panel()

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
