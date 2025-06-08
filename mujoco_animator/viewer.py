# ruff: noqa: N802
"""MuJoCo viewer implementation with interactive visualization capabilities."""

__all__ = [
    "QtMujocoViewer",
]

from threading import Lock
from typing import Callable, Literal, Optional

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

RenderMode = Literal["window", "offscreen"]
Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


def configure_scene(
    scene: mujoco.MjvScene,
    vopt: mujoco.MjvOption,
    shadow: bool = False,
    reflection: bool = False,
    contact_force: bool = False,
    contact_point: bool = False,
    inertia: bool = False,
) -> mujoco.MjvScene:
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contact_point
    vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = inertia
    return scene


class QtMujocoViewer(QOpenGLWidget):
    """Qt-based viewer for MuJoCo environments."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        parent: QWidget | None = None,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        max_geom: int = 10000,
    ) -> None:
        """Initialize the MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            parent: Parent widget
            shadow: Whether to show shadow
            reflection: Whether to show reflection
            contact_force: Whether to show contact force
            contact_point: Whether to show contact point
            inertia: Whether to show inertia
            max_geom: Maximum number of geoms to render
        """
        super().__init__(parent)

        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self._gui_lock = Lock()
        self._render_every_frame = True
        self._time_per_render = 1 / 60.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._key_callback: Optional[Callable[[int, int, int, Qt.KeyboardModifier], None]] = None

        if data is None:
            data = mujoco.MjData(model)

        self.model = model
        self.data = data
        self.is_alive = True

        # Initialize MuJoCo visualization objects
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=max_geom)
        self.pert = mujoco.MjvPerturb()

        # Set up default camera
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        self.cam.distance = 3.5
        self.cam.lookat = [0.0, 0.0, 0.5]
        self.cam.elevation = 10.0
        self.cam.azimuth = 90.0

        # Defer OpenGL context initialization to initializeGL()
        self._rect = mujoco.MjrRect(0, 0, 0, 0)  # Will be set in resizeGL

        configure_scene(
            self.scn,
            self.vopt,
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )

        # Mouse interaction variables
        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Set up timer for rendering
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(int(1000 / 60))  # 60 FPS

        # Play animation
        self.animation_time = 0
        self.loop_animation = False
        self.animation_dt = 1 / 30.0
        self.animation: np.ndarray | None = None

    def set_key_callback(self, callback: Callable[[int, int, int, Qt.KeyboardModifier], None]) -> None:
        """Set the key callback function."""
        self._key_callback = callback

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        # Ensure the viewer has focus when clicked
        self.setFocus()

        self._button_left = bool(event.buttons() & Qt.MouseButton.LeftButton)
        self._button_right = bool(event.buttons() & Qt.MouseButton.RightButton)
        self._button_middle = bool(event.buttons() & Qt.MouseButton.MiddleButton)
        self._last_mouse_x = event.position().x()
        self._last_mouse_y = event.position().y()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        self._button_left = bool(event.buttons() & Qt.MouseButton.LeftButton)
        self._button_right = bool(event.buttons() & Qt.MouseButton.RightButton)
        self._button_middle = bool(event.buttons() & Qt.MouseButton.MiddleButton)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        xpos = event.position().x()
        ypos = event.position().y()
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y

        # Update mouse position
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

        # If already applying a perturbation force
        if self.pert.active:
            # Get window size to normalize mouse movement
            height = self.height()
            mod_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if self._button_right:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self._button_left:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            with self._gui_lock:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert,
                )
            return

        # Left button: rotate camera
        if self._button_left:
            self.cam.azimuth -= dx * 0.5
            self.cam.elevation -= dy * 0.5

        # Right button: pan camera
        elif self._button_right:
            forward = np.array(
                [
                    np.cos(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                    np.sin(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                    np.sin(np.deg2rad(self.cam.elevation)),
                ]
            )
            right = np.array([-np.sin(np.deg2rad(self.cam.azimuth)), np.cos(np.deg2rad(self.cam.azimuth)), 0])
            up = np.cross(right, forward)

            # Scale pan speed with distance
            scale = self.cam.distance * 0.001

            self.cam.lookat[0] += right[0] * dx * scale - up[0] * dy * scale
            self.cam.lookat[1] += right[1] * dx * scale - up[1] * dy * scale
            self.cam.lookat[2] += right[2] * dx * scale - up[2] * dy * scale

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel events."""
        self.cam.distance *= 0.99 if event.angleDelta().y() > 0 else 1.01

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events."""
        if self._key_callback is not None:
            self._key_callback(event.key(), 0, 1, event.modifiers())

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key release events."""
        if self._key_callback is not None:
            self._key_callback(event.key(), 0, 0, event.modifiers())

    def initializeGL(self) -> None:
        """Initialize OpenGL."""
        # Initialize OpenGL context
        self.makeCurrent()

        # Create MuJoCo rendering context
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # Initialize the default free camera with better positioning
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

    def resizeGL(self, width: int, height: int) -> None:
        """Handle window resize."""
        # Update viewport
        self._rect.left = 0
        self._rect.bottom = 0
        self._rect.width = width * 2
        self._rect.height = height * 2

    def paintGL(self) -> None:
        """Render the scene."""
        with self._gui_lock:
            if self.animation is not None:
                if self.loop_animation:
                    self.animation_time += 1
                    if self.animation_time >= self.animation.shape[0]:
                        self.animation_time = 0
                else:
                    self.animation_time = min(self.animation_time + 1, self.animation.shape[0] - 1)
                self.data.qpos[:] = self.animation[self.animation_time]

            # Update physics
            mujoco.mj_forward(self.model, self.data)

            # Update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # Render scene - simplified approach
            mujoco.mjr_render(self._rect, self.scn, self.ctx)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close."""
        self.is_alive = False
        self.ctx.free()
        super().closeEvent(event)

    def apply_perturbations(self) -> None:
        """Apply user perturbations (via Ctrl+click and drag) to the simulation."""
        self.data.xfrc_applied[:] = 0
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)
