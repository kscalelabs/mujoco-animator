# ruff: noqa: N802
"""Core animator functionality for creating and editing animations."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mujoco_scenes.mjcf import load_mjmodel
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
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

    def __init__(self, model_path: Path, output_path: Path, template_path: Path | None = None) -> None:
        """Initialize the animator.

        Args:
            model_path: Path to the Mujoco model file
            output_path: Path to save the animation
            template_path: Path to a template animation to use for initialization (optional)
        """
        super().__init__()

        # Set window title first
        self.setWindowTitle("Mujoco Animator")

        # Load model
        self.model = load_mjmodel(str(model_path), scene="smooth")
        self.data = mujoco.MjData(self.model)

        # Store output path
        self.output_path = output_path

        self.template_path = template_path

        # If a template animation is provided, copy it to the output path
        if self.template_path:
            if not self.template_path.exists():
                raise FileNotFoundError(f"Template path '{self.template_path}' does not exist")

            if self.output_path and self.output_path.exists():
                raise FileExistsError(f"Output path '{self.output_path}' already exists")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(self.template_path, self.output_path)

        # Initialize cubic interpolation setting
        self.use_cubic_interp = False

        # Initialize loop animation setting
        self.loop_animation = False

        # Create central widget and layout first
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._main_layout = QHBoxLayout(central_widget)  # Changed to horizontal

        # Set up viewer with proper parent
        self.viewer = QtMujocoViewer(self.model, self.data, central_widget)
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

        # Initialize animation - load existing or create new
        if self.output_path and self.output_path.exists():
            if self.output_path.suffix == ".json":
                self.state = AnimationState(MjAnim.load_json(self.output_path))
            else:
                self.state = AnimationState(MjAnim.load_binary(self.output_path))

            if not self.state.anim.frames:
                raise ValueError("Animation file is empty")
            if self.state.anim.num_dofs != self.model.nq:
                raise ValueError("Animation file does not match model")
            self.state.current_frame = 0
            self.data.qpos[:] = self.state.anim.frames[0].positions
            self.update_side_panel()

        else:
            self.state = AnimationState(MjAnim(self.model.nq))
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

        # Frame information group
        frame_group = QGroupBox("Frame Information")
        frame_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        frame_layout = QVBoxLayout()

        self.frame_info = QLabel("Frame: 0 / 1\nTime: 0.00s")
        self.frame_info.setStyleSheet("margin: 5px;")
        frame_layout.addWidget(self.frame_info)

        frame_group.setLayout(frame_layout)
        side_panel_layout.addWidget(frame_group)

        # Controls information group
        controls_group = QGroupBox("Keyboard Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        controls_layout = QVBoxLayout()

        self.user_control_keys = QLabel(
            "<b>G:</b> Add frame<br/>"
            "<b>Backspace:</b> Delete frame<br/>"
            "<b>Space:</b> Play/stop animation<br/>"
            "<b>Q/A:</b> Adjust DOF<br/>"
            "<b>W/S:</b> Select DOF<br/>"
            "<b>D/E:</b> Select frame<br/>"
            "<b>R/F:</b> Adjust time"
        )
        self.user_control_keys.setStyleSheet("margin: 5px; font-size: 11px;")
        controls_layout.addWidget(self.user_control_keys)

        controls_group.setLayout(controls_layout)
        side_panel_layout.addWidget(controls_group)

        # Animation settings group
        animation_group = QGroupBox("Animation Settings")
        animation_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        animation_layout = QVBoxLayout()

        # Cubic interpolation checkbox
        self.cubic_interp_checkbox = QCheckBox("Use Cubic Interpolation")
        self.cubic_interp_checkbox.setChecked(self.use_cubic_interp)
        self.cubic_interp_checkbox.stateChanged.connect(self.on_cubic_interp_changed)
        self.cubic_interp_checkbox.setStyleSheet("margin: 5px; color: palette(windowText);")
        animation_layout.addWidget(self.cubic_interp_checkbox)

        # Loop animation checkbox.
        self.loop_animation_checkbox = QCheckBox("Loop Animation")
        self.loop_animation_checkbox.setChecked(self.loop_animation)
        self.loop_animation_checkbox.stateChanged.connect(self.on_loop_animation_changed)
        self.loop_animation_checkbox.setStyleSheet("margin: 5px; color: palette(windowText);")
        animation_layout.addWidget(self.loop_animation_checkbox)

        animation_group.setLayout(animation_layout)
        side_panel_layout.addWidget(animation_group)

        # DOF values group
        dof_group = QGroupBox("DOF Values")
        dof_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        dof_layout = QVBoxLayout()

        # Create scroll area for DOF controls
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create DOF spinboxes
        self.dof_spinboxes = []
        self.dof_widgets = []  # Store references to the container widgets
        for i in range(self.model.nq):
            # Create container widget for this DOF
            dof_container = QWidget()
            dof_container_layout = QHBoxLayout(dof_container)
            dof_container_layout.setContentsMargins(2, 2, 2, 2)  # Small margins

            # Get DOF/joint name
            dof_name = self.get_dof_name(i)

            # DOF label
            dof_label = QLabel(f"{dof_name}:")
            dof_label.setMinimumWidth(80)  # Increased width for longer names
            dof_label.setStyleSheet("""
                QLabel {
                    color: palette(windowText);
                    background-color: transparent;
                }
            """)
            dof_container_layout.addWidget(dof_label)

            # DOF value spinbox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-10.0, 10.0)  # Reasonable range for joint angles
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0.0)
            spinbox.valueChanged.connect(lambda value, dof=i: self.on_dof_value_changed(dof, value))
            self.dof_spinboxes.append(spinbox)
            dof_container_layout.addWidget(spinbox)

            # Store reference to container widget and add to scroll layout
            self.dof_widgets.append(dof_container)
            scroll_layout.addWidget(dof_container)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        dof_layout.addWidget(scroll_area)

        dof_group.setLayout(dof_layout)
        side_panel_layout.addWidget(dof_group)

        # Store reference to scroll area for auto-scrolling
        self.scroll_area = scroll_area

        # Add side panel to main layout
        self._main_layout.addWidget(side_panel, stretch=1)

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
                        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
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

        self.playing_animation = False
        self.playing_animation_btn = QPushButton("Play Animation (Space)")
        self.playing_animation_btn.clicked.connect(self.toggle_playing_animation)
        controls_layout.addWidget(self.playing_animation_btn)

        # Add controls to the viewer side (not the side panel)
        viewer_and_controls = QWidget()
        viewer_controls_layout = QVBoxLayout(viewer_and_controls)
        viewer_controls_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        viewer_controls_layout.addWidget(self.viewer, stretch=1)  # Viewer takes all available space
        viewer_controls_layout.addWidget(controls, stretch=0)  # Controls stay at fixed size

        # Replace the viewer in the main layout with the viewer+controls widget
        self._main_layout.removeWidget(self.viewer)
        self._main_layout.insertWidget(0, viewer_and_controls, stretch=3)

    def toggle_playing_animation(self) -> None:
        """Toggle the playing animation."""
        self.playing_animation = not self.playing_animation
        if self.playing_animation:
            self.playing_animation_btn.setText("Stop Animation (Space)")
            self.viewer.animation = self.state.anim.to_numpy(
                self.viewer.animation_dt,
                loop=self.loop_animation,
                interp="cubic" if self.use_cubic_interp else "linear",
            )
        else:
            self.playing_animation_btn.setText("Play Animation (Space)")
            self.viewer.animation = None
            self.viewer.animation_time = 0
            self.on_frame_changed(self.state.current_frame)

    def connect_signals(self) -> None:
        """Connect keyboard shortcuts."""
        pass  # Already connected in __init__

    def handle_key(self, key: int, scancode: int, action: int, mods: Qt.KeyboardModifier) -> None:
        """Handle keyboard input."""
        if action != 1:  # Only handle key press events
            return

        match key:
            case Qt.Key.Key_G:
                self.add_frame()
            case Qt.Key.Key_Backspace:
                self.delete_frame()
            case Qt.Key.Key_Q:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.adjust_dof(0.01)
                else:
                    self.adjust_dof(0.1)
            case Qt.Key.Key_A:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.adjust_dof(-0.01)
                else:
                    self.adjust_dof(-0.1)
            case Qt.Key.Key_D:
                self.on_frame_changed(self.state.current_frame - 1)
            case Qt.Key.Key_E:
                self.on_frame_changed(self.state.current_frame + 1)
            case Qt.Key.Key_W:
                self.on_dof_changed(self.state.selected_dof - 1)
            case Qt.Key.Key_S:
                self.on_dof_changed(self.state.selected_dof + 1)
            case Qt.Key.Key_R:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.on_time_change(0.1)
                else:
                    self.on_time_change(0.01)
            case Qt.Key.Key_F:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.on_time_change(-0.1)
                else:
                    self.on_time_change(-0.01)
            case Qt.Key.Key_Space:
                self.toggle_playing_animation()
            case Qt.Key.Key_Escape:
                self.close()
            case _:
                pass

    def on_time_change(self, value_delta: float) -> None:
        """Handle time change."""
        if self.state.anim.frames:
            self.state.anim.frames[self.state.current_frame].length += value_delta
            self.viewer.update()
            self.update_side_panel()
            self.auto_save()

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
            self.auto_save()

    def on_cubic_interp_changed(self, state: int) -> None:
        """Handle cubic interpolation checkbox change."""
        self.use_cubic_interp = state == 2  # Qt.CheckState.Checked is 2
        # If animation is currently playing, restart it with new interpolation
        if self.playing_animation:
            self.viewer.animation = self.state.anim.to_numpy(
                self.viewer.animation_dt,
                loop=self.loop_animation,
                interp="cubic" if self.use_cubic_interp else "linear",
            )

    def on_loop_animation_changed(self, state: int) -> None:
        """Handle loop animation checkbox change."""
        self.loop_animation = state == 2  # Qt.CheckState.Checked is 2
        if self.playing_animation:
            self.viewer.loop_animation = self.loop_animation
            self.viewer.animation = self.state.anim.to_numpy(
                self.viewer.animation_dt,
                loop=self.loop_animation,
                interp="cubic" if self.use_cubic_interp else "linear",
            )
            self.viewer.animation_time = 0

    def on_frame_changed(self, value: int) -> None:
        """Handle frame selection change."""
        if value >= len(self.state.anim.frames):
            value = 0
        elif value < 0:
            value = len(self.state.anim.frames) - 1
        self.state.current_frame = value
        self.data.qpos[:] = self.state.anim.frames[value].positions
        self.viewer.update()
        self.update_side_panel()

    def update_side_panel(self) -> None:
        """Update the side panel with current values."""
        # Update frame info
        total_frames = len(self.state.anim.frames)
        if total_frames == 0:
            return
        self.frame_info.setText(
            f"Frame: {self.state.current_frame + 1} / {total_frames}\n"
            f"Time: {self.state.anim.frames[self.state.current_frame].length:.2f}s"
        )

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

        # Auto-scroll to keep selected DOF visible
        if hasattr(self, "scroll_area") and 0 <= self.state.selected_dof < len(self.dof_widgets):
            selected_widget = self.dof_widgets[self.state.selected_dof]
            # Use ensureWidgetVisible to scroll the selected DOF into view
            self.scroll_area.ensureWidgetVisible(selected_widget, 50, 50)  # 50px margin

    def adjust_dof(self, delta: float) -> None:
        """Adjust the current DOF value."""
        if not self.state.anim.frames:
            return

        frame = self.state.anim.frames[self.state.current_frame]
        frame.positions[self.state.selected_dof] += delta
        self.data.qpos[:] = frame.positions
        self.viewer.update()
        self.update_side_panel()
        self.auto_save()

    def add_frame(self) -> None:
        """Add a new frame to the animation."""
        # Create a new frame with current positions
        frame = Frame(1.0, list(self.data.qpos))  # Default 1.0s duration
        index = self.state.anim.add_frame(frame.length, frame.positions, index=self.state.current_frame + 1)

        # Update current frame to the newly added one
        self.state.current_frame = index
        self.update_side_panel()
        self.auto_save()

    def delete_frame(self) -> None:
        """Delete the current frame."""
        if self.state.current_frame == -1 or len(self.state.anim.frames) == 1:
            return
        self.state.anim.frames.pop(self.state.current_frame)
        if self.state.current_frame == len(self.state.anim.frames):
            self.state.current_frame -= 1
        self.data.qpos[:] = self.state.anim.frames[self.state.current_frame].positions
        self.update_side_panel()
        self.auto_save()

    def save_animation(self) -> None:
        """Save the animation to file."""
        if self.output_path is None:
            return
        if self.output_path.suffix == ".json":
            self.state.anim.save_json(self.output_path)
        else:
            self.state.anim.save_binary(self.output_path)

    def auto_save(self) -> None:
        """Automatically save the animation if an output path is set."""
        if self.output_path is not None and self.state.anim.frames:
            if self.output_path.suffix == ".json":
                self.state.anim.save_json(self.output_path)
            else:
                self.state.anim.save_binary(self.output_path)
