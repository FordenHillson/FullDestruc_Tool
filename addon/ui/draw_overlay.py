"""PySide6 Transparent Drawing, Vertex-Adjust, Radial Wheel & Extrude Overlay for Apple Bite.

Phases:
- DRAW:    Frameless transparent PySide6 glass over 3D Viewport (Polygon click + Freehand drag).
- READY:   After closing the shape, locks the 3D polygon and turns click-through so you can orbit
           Blender freely. Hold Ctrl+Scroll to increase/decrease vertices live, or click the
           floating center button to open the Radial Donut Wheel Menu (Through Cut, Extrude,
           Move Camera, Redraw, Cancel).
- EXTRUDE: Optional custom-depth mode with a draggable round handle dot at the extruded face center.
- CAMERA:  Click-through camera adjust mode before drawing from a new angle.
"""

from __future__ import annotations

import copy
import ctypes
import math
import os
import sys
from typing import Callable

import bpy
import numpy as np
from mathutils import Vector

from ..core import apple_bite
from ..core.draw_engine import (
    CameraSnapshot,
    adjust_polygon_vertex_count,
    ensure_ccw_2d,
    extrude_polygon_2d_frustum_to_mesh,
    extrude_polygon_2d_to_mesh,
    extrude_polygon_3d_to_mesh,
    polygon_signed_area_2d,
    project_world_to_qt_2d,
    simplify_polyline,
    triangulate_polygon_2d,
    unproject_qt_to_world_plane,
)
from .qt_bridge import PYSIDE_AVAILABLE, _ensure_qapp
from .qt_window import _tag_all_view3d_redraw, blender_view3d_override

if PYSIDE_AVAILABLE:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
else:
    QtCore = None  # type: ignore
    QtGui = None  # type: ignore
    QtWidgets = None  # type: ignore

_active_overlay = None
_active_rv3d_ref = None
_active_win_ref = None
_active_region_ref = None
_overlay_timer_running = False

_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020

# Windows Virtual-Key codes (layout-independent: works identically on Thai & English keyboards)
_VK_BACK = 0x08
_VK_TAB = 0x09
_VK_RETURN = 0x0D
_VK_CONTROL = 0x11
_VK_ESCAPE = 0x1B
_VK_SPACE = 0x20
_VK_M = 0x4D
_TRACKED_VKS = (_VK_BACK, _VK_TAB, _VK_RETURN, _VK_ESCAPE, _VK_SPACE, _VK_M)
_prev_vk_down: dict[int, bool] = {}
_blender_hwnd: int = 0


def _reset_vk_state() -> None:
    """Snapshot currently held keys so opening/switching never fires stale presses."""
    _prev_vk_down.clear()
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        for vk in _TRACKED_VKS:
            _prev_vk_down[vk] = bool(user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        pass


def _is_ctrl_held() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_CONTROL) & 0x8000)
    except Exception:
        return False


def _poll_global_shortcuts(overlay) -> None:
    """Edge-detect layout-independent hotkeys while Blender or our overlay is the active window."""
    if sys.platform != "win32" or overlay is None:
        return
    try:
        user32 = ctypes.windll.user32
        fg = int(user32.GetForegroundWindow() or 0)
        allowed_hwnds = {_blender_hwnd} if _blender_hwnd else set()
        for w in overlay.get_all_hwnds():
            allowed_hwnds.add(w)
        if allowed_hwnds and fg not in allowed_hwnds:
            return

        for vk in _TRACKED_VKS:
            is_down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
            was_down = _prev_vk_down.get(vk, False)
            _prev_vk_down[vk] = is_down
            if is_down and not was_down:
                overlay.handle_vk_shortcut(vk)
                if not overlay.isVisible():
                    return
    except Exception:
        pass


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _set_win_click_through(widget, enabled: bool) -> None:
    """Toggle OS-level click-through on a Qt window so Blender receives camera orbit/pan/zoom."""
    if not PYSIDE_AVAILABLE or widget is None:
        return
    cur_flag = bool(widget.windowFlags() & QtCore.Qt.WindowTransparentForInput)
    if cur_flag != enabled:
        widget.setWindowFlag(QtCore.Qt.WindowTransparentForInput, enabled)
        widget.show()
    if sys.platform == "win32":
        try:
            hwnd = int(widget.winId())
            user32 = ctypes.windll.user32
            ex_style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if enabled:
                ex_style |= _WS_EX_LAYERED | _WS_EX_TRANSPARENT
            else:
                ex_style &= ~_WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, ex_style)
        except Exception:
            pass


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def _find_blender_hwnd(window: bpy.types.Window | None = None) -> int:
    """Return the native Blender GHOST_WindowClass HWND (never a PySide6 child/tool window)."""
    if sys.platform != "win32":
        return 0
    try:
        user32 = ctypes.windll.user32
        pid = os.getpid()
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        target_w = int(window.width) if window is not None else 0
        target_h = int(window.height) if window is not None else 0

        best_hwnd = 0
        best_diff = 10**9

        def _enum_cb(hwnd, _lparam):
            nonlocal best_hwnd, best_diff
            wpid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value != pid or not user32.IsWindowVisible(hwnd):
                return True
            cls_buf = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls_buf, 64)
            if cls_buf.value != "GHOST_WindowClass":
                return True
            rc = _RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rc))
            cw = int(rc.right - rc.left)
            ch = int(rc.bottom - rc.top)
            diff = abs(cw - target_w) + abs(ch - target_h) if (target_w > 0 and target_h > 0) else 0
            if diff < best_diff:
                best_diff = diff
                best_hwnd = int(hwnd)
            return True

        user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
        if best_hwnd:
            return best_hwnd
        return int(user32.GetForegroundWindow() or user32.GetActiveWindow() or 0)
    except Exception:
        return 0


def _get_viewport_qt_rect(
    window: bpy.types.Window,
    region: bpy.types.Region,
    blender_hwnd: int = 0,
) -> tuple[int, int, int, int]:
    """Compute Qt logical (x, y, w, h) matching Blender's 3D Viewport WINDOW region on Windows."""
    phys_x = int(region.x)
    phys_y = int(window.height - (region.y + region.height))
    phys_w = max(int(region.width), 200)
    phys_h = max(int(region.height), 200)

    screen_x = phys_x
    screen_y = phys_y

    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            hwnd = blender_hwnd or _find_blender_hwnd(window)
            if hwnd:
                pt = _POINT(phys_x, phys_y)
                if user32.ClientToScreen(hwnd, ctypes.byref(pt)):
                    screen_x = int(pt.x)
                    screen_y = int(pt.y)
        except Exception:
            pass

    if PYSIDE_AVAILABLE:
        app = _ensure_qapp()
        if app is not None:
            for scr in app.screens():
                dpr = float(scr.devicePixelRatio() or 1.0)
                g = scr.geometry()
                sx0 = int(round(g.x() * dpr))
                sy0 = int(round(g.y() * dpr))
                sx1 = sx0 + int(round(g.width() * dpr))
                sy1 = sy0 + int(round(g.height() * dpr))
                if sx0 <= screen_x < sx1 and sy0 <= screen_y < sy1:
                    lx = g.x() + int(round((screen_x - sx0) / dpr))
                    ly = g.y() + int(round((screen_y - sy0) / dpr))
                    lw = max(int(round(phys_w / dpr)), 200)
                    lh = max(int(round(phys_h / dpr)), 200)
                    return lx, ly, lw, lh
            primary = app.primaryScreen()
            if primary is not None:
                dpr = float(primary.devicePixelRatio() or 1.0)
                return (
                    int(round(screen_x / dpr)),
                    int(round(screen_y / dpr)),
                    max(int(round(phys_w / dpr)), 200),
                    max(int(round(phys_h / dpr)), 200),
                )

    return screen_x, screen_y, phys_w, phys_h


def _update_camera_matrices(
    cam: CameraSnapshot,
    rv3d: bpy.types.RegionView3D,
    target_corners: np.ndarray | None = None,
) -> None:
    """Refresh live camera matrices (and target depth range) on an existing CameraSnapshot in place."""
    view_mat = np.array(rv3d.view_matrix, dtype=np.float64)
    persp_mat = np.array(rv3d.perspective_matrix, dtype=np.float64)
    cam.view_matrix = view_mat
    cam.perspective_matrix = persp_mat
    cam.inv_perspective_matrix = np.linalg.inv(persp_mat)
    cam.is_perspective = bool(rv3d.is_perspective)
    inv_view = np.linalg.inv(view_mat)
    cam.camera_pos = inv_view[:3, 3].copy()
    fwd = -inv_view[:3, 2].copy()
    fwd_len = float(np.linalg.norm(fwd))
    if fwd_len > 1e-12:
        cam.view_forward = fwd / fwd_len
    if target_corners is not None and len(target_corners) > 0:
        depths = target_corners @ cam.view_forward
        cam.target_depth_min = float(np.min(depths))
        cam.target_depth_max = float(np.max(depths))


def collect_target_bbox_corners(target_objects: list[bpy.types.Object]) -> np.ndarray:
    """Return world-space bounding-box corners (K, 3) across all target mesh objects."""
    pts: list[list[float]] = []
    for obj in target_objects:
        if obj is None or obj.type != "MESH":
            continue
        mw = apple_bite._compute_true_matrix_world(obj)
        for corner in obj.bound_box:
            wc = mw @ Vector(corner)
            pts.append([wc.x, wc.y, wc.z])
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def collect_cutter_faces_world(cutter_obj: bpy.types.Object) -> list[dict]:
    """Extract world-space polygons (index, verts, normal, center) from a cutter mesh for Face Mask picking."""
    mesh = cutter_obj.data if (cutter_obj is not None and cutter_obj.type == "MESH") else None
    if mesh is None or len(mesh.polygons) == 0:
        return []
    mw = apple_bite._compute_true_matrix_world(cutter_obj)
    faces: list[dict] = []
    for poly in mesh.polygons:
        if len(poly.vertices) < 3:
            continue
        w_pts = [mw @ mesh.vertices[vi].co for vi in poly.vertices]
        arr = np.asarray([[p.x, p.y, p.z] for p in w_pts], dtype=np.float64)
        center = np.mean(arr, axis=0)
        # Newell's method for accurate world-space normal under arbitrary scale/rotation
        n_vec = np.zeros(3, dtype=np.float64)
        k = len(arr)
        for i in range(k):
            p0 = arr[i]
            p1 = arr[(i + 1) % k]
            n_vec[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
            n_vec[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
            n_vec[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
        n_len = float(np.linalg.norm(n_vec))
        if n_len <= 1e-9:
            continue
        n_vec /= n_len
        faces.append({
            "index": int(poly.index),
            "verts": arr,
            "normal": n_vec,
            "center": center,
        })
    return faces


def _point_in_polygon_2d_ray(px: float, py: float, poly_2d: np.ndarray) -> bool:
    """Even-odd ray casting point-in-polygon test (works for arbitrary convex & concave polygons)."""
    inside = False
    n = len(poly_2d)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly_2d[i, 0]), float(poly_2d[i, 1])
        xj, yj = float(poly_2d[j, 0]), float(poly_2d[j, 1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _ray_hit_cutter_faces(
    qx: float,
    qy: float,
    cam: CameraSnapshot,
    cutter_faces: list[dict],
) -> int | None:
    """Raycast from Qt screen pixel (qx, qy) against cutter faces; prefer front-facing nearest hit."""
    if not cutter_faces:
        return None
    w = max(float(cam.width), 1.0)
    h = max(float(cam.height), 1.0)
    ndc_x = (qx / w) * 2.0 - 1.0
    ndc_y = 1.0 - (qy / h) * 2.0

    near_h = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64) @ cam.inv_perspective_matrix.T
    far_h = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64) @ cam.inv_perspective_matrix.T
    p_near = near_h[:3] / (near_h[3] if abs(near_h[3]) > 1e-12 else 1.0)
    p_far = far_h[:3] / (far_h[3] if abs(far_h[3]) > 1e-12 else 1.0)
    ray_dir = p_far - p_near
    r_len = float(np.linalg.norm(ray_dir))
    if r_len <= 1e-12:
        return None
    ray_dir /= r_len

    best_front_idx: int | None = None
    best_front_t = float("inf")
    best_back_idx: int | None = None
    best_back_t = float("inf")

    for f in cutter_faces:
        normal = f["normal"]
        verts = f["verts"]
        denom = float(np.dot(ray_dir, normal))
        if abs(denom) < 1e-6:
            continue
        t = float(np.dot(verts[0] - p_near, normal) / denom)
        if cam.is_perspective and t <= 0.0:
            continue
        p_hit = p_near + ray_dir * t

        # Build local 2D basis (u, v) on the face plane
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u_axis = np.cross(normal, ref)
        u_len = float(np.linalg.norm(u_axis))
        if u_len <= 1e-9:
            continue
        u_axis /= u_len
        v_axis = np.cross(normal, u_axis)

        rel_verts = verts - verts[0]
        poly_2d = np.column_stack([rel_verts @ u_axis, rel_verts @ v_axis])
        rel_hit = p_hit - verts[0]
        hx = float(np.dot(rel_hit, u_axis))
        hy = float(np.dot(rel_hit, v_axis))

        if _point_in_polygon_2d_ray(hx, hy, poly_2d):
            if denom < 0.0:
                if t < best_front_t:
                    best_front_t = t
                    best_front_idx = f["index"]
            else:
                if t < best_back_t:
                    best_back_t = t
                    best_back_idx = f["index"]

    return best_front_idx if best_front_idx is not None else best_back_idx


def capture_camera_snapshot(
    rv3d: bpy.types.RegionView3D,
    target_objects: list[bpy.types.Object],
    qt_width: float,
    qt_height: float,
) -> CameraSnapshot:
    """Capture a pure-NumPy CameraSnapshot from Blender's active 3D Viewport."""
    view_mat = np.array(rv3d.view_matrix, dtype=np.float64)
    persp_mat = np.array(rv3d.perspective_matrix, dtype=np.float64)
    inv_persp = np.linalg.inv(persp_mat)
    inv_view = np.linalg.inv(view_mat)

    camera_pos = inv_view[:3, 3].copy()
    fwd = -inv_view[:3, 2].copy()
    fwd_len = float(np.linalg.norm(fwd))
    if fwd_len > 1e-12:
        fwd /= fwd_len
    else:
        fwd = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    depths: list[float] = []
    min_v = np.array([float("inf")] * 3, dtype=np.float64)
    max_v = np.array([float("-inf")] * 3, dtype=np.float64)
    for obj in target_objects:
        if obj is None or obj.type != "MESH":
            continue
        mw = apple_bite._compute_true_matrix_world(obj)
        for corner in obj.bound_box:
            wc = mw @ Vector(corner)
            p = np.array([wc.x, wc.y, wc.z], dtype=np.float64)
            depths.append(float(np.dot(p, fwd)))
            min_v = np.minimum(min_v, p)
            max_v = np.maximum(max_v, p)

    if depths:
        d_min = min(depths)
        d_max = max(depths)
        center = (min_v + max_v) * 0.5
    else:
        center = np.zeros(3, dtype=np.float64)
        c_depth = float(np.dot(center, fwd))
        d_min = c_depth - 1.5
        d_max = c_depth + 1.5

    return CameraSnapshot(
        width=float(qt_width),
        height=float(qt_height),
        view_matrix=view_mat,
        perspective_matrix=persp_mat,
        inv_perspective_matrix=inv_persp,
        is_perspective=bool(rv3d.is_perspective),
        camera_pos=camera_pos,
        view_forward=fwd,
        target_center=center,
        target_depth_min=d_min,
        target_depth_max=d_max,
    )


if PYSIDE_AVAILABLE:

    _WHEEL_SLICES_READY: tuple[tuple[str, str, str, tuple[int, int, int]], ...] = (
        ("through", "Through Cut", "Cut all the way through", (46, 198, 112)),
        ("extrude", "Extrude", "Drag custom depth", (245, 134, 36)),
        ("camera", "Move Camera", "Adjust angle first", (56, 168, 255)),
        ("redraw", "Redraw", "Draw new shape", (240, 196, 42)),
        ("cancel", "Cancel", "Close draw mode", (224, 66, 66)),
    )

    _WHEEL_SLICES_EXTRUDE: tuple[tuple[str, str, str, tuple[int, int, int]], ...] = (
        ("confirm", "Confirm Cut", "Use current depth", (46, 198, 112)),
        ("through", "Through Cut", "Cut all the way through", (245, 134, 36)),
        ("camera", "Move Camera", "Adjust angle first", (56, 168, 255)),
        ("redraw", "Redraw", "Draw new shape", (240, 196, 42)),
        ("cancel", "Cancel", "Close draw mode", (224, 66, 66)),
    )

    _WHEEL_SLICES_EDIT: tuple[tuple[str, str, str, tuple[int, int, int]], ...] = (
        ("carve", "Carve Out", "Subtract from cutter", (245, 118, 42)),
        ("keep_inside", "Keep Inside", "Intersect with cutter", (46, 198, 112)),
        ("add_shape", "Add Shape", "Merge into cutter", (36, 196, 192)),
        ("camera", "Move Camera", "Adjust angle first", (56, 168, 255)),
        ("redraw", "Redraw", "Draw new shape", (240, 196, 42)),
        ("cancel", "Cancel", "Close draw mode", (224, 66, 66)),
    )

    class CenterMenuButton(QtWidgets.QWidget):
        """Floating circular menu button at the center of the closed 3D shape (READY phase)."""

        BTN_SIZE = 44

        def __init__(self, owner: "DrawCutterOverlay"):
            super().__init__(None)
            self._owner = owner
            self._hovered = False

            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setFixedSize(self.BTN_SIZE, self.BTN_SIZE)
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.setToolTip("Click for Radial Menu | Ctrl+Scroll to adjust vertices")

        def enterEvent(self, event) -> None:
            self._hovered = True
            self.update()

        def leaveEvent(self, event) -> None:
            self._hovered = False
            self.update()

        def mousePressEvent(self, event) -> None:
            if event.button() == QtCore.Qt.LeftButton:
                self._owner.open_radial_wheel()
            elif event.button() == QtCore.Qt.RightButton:
                self._owner.cancel_all()

        def wheelEvent(self, event) -> None:
            delta = 1 if event.angleDelta().y() > 0 else -1
            self._owner.adjust_vertices(delta)
            event.accept()

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            cx = self.width() * 0.5
            cy = self.height() * 0.5
            r = 18.0 if self._hovered else 16.0

            bg = QtGui.QColor(34, 38, 48, 238) if self._hovered else QtGui.QColor(22, 25, 32, 224)
            rim = QtGui.QColor(255, 200, 75, 255) if self._hovered else QtGui.QColor(255, 175, 60, 210)
            painter.setBrush(bg)
            painter.setPen(QtGui.QPen(rim, 2.4))
            painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)

            # Vertex count label inside center button
            font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(255, 245, 215, 255))
            v_text = f"{self._owner.vertex_count}v"
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, v_text)
            painter.end()

    class DrawRadialWheel(QtWidgets.QWidget):
        """PySide6 Radial Donut Wheel Menu matching FullDestruc's wheel_menu.py style."""

        WHEEL_SIZE = 320
        INNER_R = 44.0
        OUTER_R = 132.0
        HOVER_POP = 9.0
        GAP_DEG = 3.2

        def __init__(self, owner: "DrawCutterOverlay"):
            super().__init__(None)
            self._owner = owner
            self._hovered: int | None = None

            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setFixedSize(self.WHEEL_SIZE, self.WHEEL_SIZE)
            self.setMouseTracking(True)
            self.setCursor(QtCore.Qt.PointingHandCursor)

        def _slices(self):
            if self._owner.is_edit_mode:
                return _WHEEL_SLICES_EDIT
            return _WHEEL_SLICES_EXTRUDE if self._owner.phase == "EXTRUDE" else _WHEEL_SLICES_READY

        def _hit_test(self, mx: float, my: float) -> int | None:
            slices = self._slices()
            cx = self.width() * 0.5
            cy = self.height() * 0.5
            dx = mx - cx
            dy = cy - my  # flip Y so +Y is UP (standard math angle)
            dist = math.hypot(dx, dy)
            if dist < self.INNER_R or dist > (self.OUTER_R + 14.0):
                return None
            ang_deg = math.degrees(math.atan2(dy, dx))
            cw_from_top = (90.0 - ang_deg) % 360.0
            span = 360.0 / len(slices)
            return int(cw_from_top // span) % len(slices)

        def mouseMoveEvent(self, event) -> None:
            pos = event.position()
            new_h = self._hit_test(float(pos.x()), float(pos.y()))
            if new_h != self._hovered:
                self._hovered = new_h
                self.update()

        def mousePressEvent(self, event) -> None:
            if event.button() == QtCore.Qt.RightButton:
                self._owner.close_radial_wheel()
                return
            if event.button() == QtCore.Qt.LeftButton:
                pos = event.position()
                slices = self._slices()
                picked = self._hit_test(float(pos.x()), float(pos.y()))
                self._owner.close_radial_wheel()
                if picked is not None:
                    action_id = slices[picked][0]
                    self._owner.execute_wheel_action(action_id)

        def wheelEvent(self, event) -> None:
            delta = 1 if event.angleDelta().y() > 0 else -1
            self._owner.adjust_vertices(delta)
            self.update()
            event.accept()

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            slices = self._slices()
            cx = self.width() * 0.5
            cy = self.height() * 0.5
            count = len(slices)
            span = 360.0 / count

            for idx, (_aid, label, _hint, rgb) in enumerate(slices):
                is_hover = idx == self._hovered
                r_in = self.INNER_R + (2.0 if is_hover else 0.0)
                r_out = self.OUTER_R + (self.HOVER_POP if is_hover else 0.0)

                # Qt arc angles: 0 deg = 3 o'clock, positive = CCW
                a_end_deg = 90.0 - idx * span - self.GAP_DEG * 0.5
                a_start_deg = 90.0 - (idx + 1) * span + self.GAP_DEG * 0.5
                sweep = a_end_deg - a_start_deg

                path = QtGui.QPainterPath()
                outer_rect = QtCore.QRectF(cx - r_out, cy - r_out, r_out * 2.0, r_out * 2.0)
                inner_rect = QtCore.QRectF(cx - r_in, cy - r_in, r_in * 2.0, r_in * 2.0)
                path.arcMoveTo(outer_rect, a_start_deg)
                path.arcTo(outer_rect, a_start_deg, sweep)
                path.arcTo(inner_rect, a_end_deg, -sweep)
                path.closeSubpath()

                accent = QtGui.QColor(rgb[0], rgb[1], rgb[2], 242)
                fill = accent if is_hover else QtGui.QColor(26, 30, 38, 228)
                rim = QtGui.QColor(255, 255, 255, 245) if is_hover else QtGui.QColor(rgb[0], rgb[1], rgb[2], 150)

                painter.setBrush(fill)
                painter.setPen(QtGui.QPen(rim, 2.2 if is_hover else 1.2))
                painter.drawPath(path)

                # Label at slice midpoint
                a_mid_rad = math.radians((a_start_deg + a_end_deg) * 0.5)
                r_mid = (r_in + r_out) * 0.5
                tx = cx + r_mid * math.cos(a_mid_rad)
                ty = cy - r_mid * math.sin(a_mid_rad)

                font = QtGui.QFont("Segoe UI", 9 if is_hover else 8, QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(18, 20, 24, 255) if is_hover else QtGui.QColor(240, 244, 250, 245))
                txt_rect = QtCore.QRectF(tx - 46.0, ty - 14.0, 92.0, 28.0)
                painter.drawText(txt_rect, QtCore.Qt.AlignCenter, label)

            # Center hub disk
            hub_r = self.INNER_R - 4.0
            painter.setBrush(QtGui.QColor(18, 21, 28, 242))
            if self._hovered is not None:
                rgb = slices[self._hovered][3]
                hub_rim = QtGui.QColor(rgb[0], rgb[1], rgb[2], 245)
            else:
                hub_rim = QtGui.QColor(130, 145, 170, 190)
            painter.setPen(QtGui.QPen(hub_rim, 2.0))
            painter.drawEllipse(QtCore.QPointF(cx, cy), hub_r, hub_r)

            hub_font = QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold)
            painter.setFont(hub_font)
            painter.setPen(QtGui.QColor(255, 240, 200, 255))
            if self._owner.phase == "EXTRUDE":
                hub_txt = f"{self._owner.extrude_length:.2f} m"
            else:
                hub_txt = f"{self._owner.vertex_count} verts"
            painter.drawText(
                QtCore.QRectF(cx - hub_r, cy - hub_r, hub_r * 2.0, hub_r * 2.0),
                QtCore.Qt.AlignCenter,
                hub_txt,
            )
            painter.end()

    class ReadyStatusPill(QtWidgets.QFrame):
        """Compact click-through top status badge showing vertex count & Ctrl+Scroll hint."""

        def __init__(self):
            super().__init__(None)
            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.WindowTransparentForInput
            )
            self.setStyleSheet(
                """
                QFrame {
                    background-color: rgba(22, 25, 32, 215);
                    border: 1px solid rgba(255, 200, 90, 175);
                    border-radius: 14px;
                }
                QLabel {
                    color: #f4f6fa;
                    font-family: "Segoe UI", sans-serif;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                }
                """
            )
            hl = QtWidgets.QHBoxLayout(self)
            hl.setContentsMargins(14, 4, 14, 4)
            self.label = QtWidgets.QLabel("")
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            hl.addWidget(self.label)

    class ExtrudeHandleDot(QtWidgets.QWidget):
        """Draggable glowing round handle positioned at the center of the extruded face."""

        HANDLE_SIZE = 32

        def __init__(self, owner: "DrawCutterOverlay"):
            super().__init__(None)
            self._owner = owner
            self._hovered = False
            self._pressed = False
            self._dragging = False
            self._drag_start_global = QtCore.QPointF()
            self._drag_start_len = 0.0

            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setFixedSize(self.HANDLE_SIZE, self.HANDLE_SIZE)
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.setToolTip("Click: Wheel Menu | Drag: Extrude")

        def enterEvent(self, event) -> None:
            self._hovered = True
            self.update()

        def leaveEvent(self, event) -> None:
            self._hovered = False
            self.update()

        def mousePressEvent(self, event) -> None:
            if event.button() == QtCore.Qt.LeftButton:
                self._pressed = True
                self._dragging = False
                self._drag_start_global = event.globalPosition()
                self._drag_start_len = self._owner.extrude_length
                self.grabMouse()
                self.update()

        def mouseMoveEvent(self, event) -> None:
            if self._pressed:
                cur = event.globalPosition()
                dx = float(cur.x() - self._drag_start_global.x())
                dy = float(cur.y() - self._drag_start_global.y())
                if not self._dragging and (dx * dx + dy * dy) >= 25.0:
                    self._dragging = True
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                    self.update()
                if self._dragging:
                    self._owner.on_handle_drag(dx, dy, self._drag_start_len)

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == QtCore.Qt.LeftButton and self._pressed:
                was_drag = self._dragging
                self._pressed = False
                self._dragging = False
                self.releaseMouse()
                self.setCursor(QtCore.Qt.OpenHandCursor)
                self.update()
                if not was_drag:
                    self._owner.open_radial_wheel()

        def keyPressEvent(self, event) -> None:
            if sys.platform == "win32":
                event.accept()
                return
            self._owner.handle_extrude_key(event.key())

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            cx = self.width() * 0.5
            cy = self.height() * 0.5

            glow_col = QtGui.QColor(255, 215, 70, 170 if (self._hovered or self._dragging) else 100)
            painter.setPen(QtGui.QPen(glow_col, 3.0))
            painter.setBrush(QtGui.QColor(25, 28, 36, 200))
            r_outer = 12.0 if (self._hovered or self._dragging) else 10.0
            painter.drawEllipse(QtCore.QPointF(cx, cy), r_outer, r_outer)

            core_col = QtGui.QColor(255, 225, 90, 255) if self._dragging else QtGui.QColor(255, 165, 40, 255)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(core_col)
            painter.drawEllipse(QtCore.QPointF(cx, cy), 5.5, 5.5)
            painter.end()

    class ExtrudeHudBar(QtWidgets.QFrame):
        """Compact floating interactive HUD bar shown while dragging custom Extrude depth."""

        def __init__(self, owner: "DrawCutterOverlay"):
            super().__init__(None)
            self._owner = owner
            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setStyleSheet(
                """
                QFrame {
                    background-color: rgba(24, 26, 32, 232);
                    border: 1px solid rgba(255, 200, 90, 210);
                    border-radius: 8px;
                }
                QLabel {
                    color: #f4f6fa;
                    font-family: "Segoe UI", sans-serif;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                }
                QPushButton {
                    background-color: #343946;
                    color: #f4f6fa;
                    border: 1px solid #52596b;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #464d5e; }
                QPushButton#ConfirmBtn {
                    background-color: #2c6e49;
                    border-color: #3d9966;
                    font-weight: bold;
                }
                QPushButton#ConfirmBtn:hover { background-color: #37875a; }
                QPushButton#CancelBtn {
                    background-color: #7c2d2d;
                    border-color: #a33d3d;
                }
                """
            )
            hl = QtWidgets.QHBoxLayout(self)
            hl.setContentsMargins(12, 6, 12, 6)
            hl.setSpacing(8)

            self.label = QtWidgets.QLabel("")
            hl.addWidget(self.label, 1)

            btn_ok = QtWidgets.QPushButton("Confirm (Enter)")
            btn_ok.setObjectName("ConfirmBtn")
            btn_ok.clicked.connect(owner.confirm_extrude)
            hl.addWidget(btn_ok)

            btn_thru = QtWidgets.QPushButton("Through (Space)")
            btn_thru.clicked.connect(owner.confirm_through_building)
            hl.addWidget(btn_thru)

            btn_menu = QtWidgets.QPushButton("Back to Wheel")
            btn_menu.clicked.connect(owner.restore_previous_ready)
            hl.addWidget(btn_menu)

            btn_cancel = QtWidgets.QPushButton("Cancel (ESC)")
            btn_cancel.setObjectName("CancelBtn")
            btn_cancel.clicked.connect(owner.cancel_all)
            hl.addWidget(btn_cancel)

        def keyPressEvent(self, event) -> None:
            if sys.platform == "win32":
                event.accept()
                return
            self._owner.handle_extrude_key(event.key())

    class CameraHudBar(QtWidgets.QFrame):
        """Compact floating interactive HUD bar shown while orbiting camera before drawing/redrawing."""

        def __init__(self, owner: "DrawCutterOverlay"):
            super().__init__(None)
            self._owner = owner
            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setStyleSheet(
                """
                QFrame {
                    background-color: rgba(24, 26, 32, 232);
                    border: 1px solid rgba(90, 195, 255, 210);
                    border-radius: 8px;
                }
                QLabel {
                    color: #f4f6fa;
                    font-family: "Segoe UI", sans-serif;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                }
                QPushButton {
                    background-color: #343946;
                    color: #f4f6fa;
                    border: 1px solid #52596b;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #464d5e; }
                QPushButton#ConfirmBtn {
                    background-color: #2c6e49;
                    border-color: #3d9966;
                    font-weight: bold;
                }
                QPushButton#ConfirmBtn:hover { background-color: #37875a; }
                QPushButton#CancelBtn {
                    background-color: #7c2d2d;
                    border-color: #a33d3d;
                }
                """
            )
            hl = QtWidgets.QHBoxLayout(self)
            hl.setContentsMargins(12, 6, 12, 6)
            hl.setSpacing(8)

            self.label = QtWidgets.QLabel("")
            hl.addWidget(self.label, 1)

            btn_draw_here = QtWidgets.QPushButton("Draw Here (Tab)")
            btn_draw_here.setObjectName("ConfirmBtn")
            btn_draw_here.clicked.connect(owner.return_from_camera_phase)
            hl.addWidget(btn_draw_here)

            self.btn_restore_prev = QtWidgets.QPushButton("Back to Prev Shape")
            self.btn_restore_prev.clicked.connect(owner.restore_previous_ready)
            hl.addWidget(self.btn_restore_prev)

            btn_cancel = QtWidgets.QPushButton("Cancel (ESC)")
            btn_cancel.setObjectName("CancelBtn")
            btn_cancel.clicked.connect(owner.cancel_all)
            hl.addWidget(btn_cancel)

        def keyPressEvent(self, event) -> None:
            if sys.platform == "win32":
                event.accept()
                return
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                self._owner.cancel_all()
            elif key in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self._owner.enter_draw_phase()

    class DrawCutterOverlay(QtWidgets.QWidget):
        """Frameless transparent PySide6 canvas over 3D Viewport (DRAW -> READY + Wheel -> EXTRUDE)."""

        def __init__(
            self,
            cam: CameraSnapshot,
            target_corners: np.ndarray,
            zone: str,
            zone_rgba: tuple[float, float, float, float],
            on_confirm: Callable[[np.ndarray, list[tuple[int, ...]], str], None],
            on_cancel: Callable[[], None],
            edit_cutter_name: str | None = None,
            on_sculpt_confirm: Callable[[np.ndarray, list[tuple[int, ...]], str, list[int], float], None] | None = None,
            cutter_faces: list[dict] | None = None,
            default_mask_depth: float = 1.5,
        ):
            super().__init__(None)
            self._cam = cam
            self._target_corners = target_corners
            self._zone = str(zone or "A").strip() or "A"
            self._zone_color = QtGui.QColor(
                int(zone_rgba[0] * 255),
                int(zone_rgba[1] * 255),
                int(zone_rgba[2] * 255),
            )
            self._on_confirm = on_confirm
            self._on_cancel = on_cancel
            self._edit_cutter_name: str | None = edit_cutter_name or None
            self._on_sculpt_confirm = on_sculpt_confirm

            # Face Mask state (for EDIT_CUTTER mode: click cutter faces to mask sculpt area)
            self._cutter_faces: list[dict] = list(cutter_faces) if cutter_faces else []
            self._mask_face_indices: set[int] = set()
            self._mask_depth: float = max(0.1, float(default_mask_depth))
            self._hovered_mask_face: int | None = None
            self._return_phase_after_camera: str = "DRAW"

            # Mode state: "DRAW" | "READY" | "EXTRUDE" | "CAMERA"
            self._phase: str = "DRAW"
            self._points: list[tuple[float, float]] = []
            self._mouse_pos: tuple[float, float] = (cam.width * 0.5, cam.height * 0.5)
            self._lmb_down: bool = False
            self._lmb_press_pos: tuple[float, float] = (0.0, 0.0)
            self._is_freehand: bool = False

            # Locked 2D source + camera snapshot (for live Ctrl+Scroll vertex count adjustment!)
            self._raw_closed_2d: np.ndarray = np.empty((0, 2), dtype=np.float64)
            self._pts_ccw_locked: np.ndarray = np.empty((0, 2), dtype=np.float64)
            self._draw_cam_snapshot: CameraSnapshot | None = None
            self._front_depth_locked: float = 0.0
            self._vertex_count: int = 0

            # Locked 3D base polygon (persists across camera orbit!)
            self._front_3d: np.ndarray = np.empty((0, 3), dtype=np.float64)
            self._cap_tris: list[tuple[int, int, int]] = []
            self._front_center_3d: np.ndarray = np.zeros(3, dtype=np.float64)
            self._extrude_dir: np.ndarray = cam.view_forward.copy()
            self._extrude_length: float = max(cam.target_thickness * 0.65, 1.0)
            self._through_length: float = cam.through_building_range(margin=0.5)[1]

            self._ctrl_wheel_capture: bool = False
            self._center_btn: CenterMenuButton | None = None
            self._radial_wheel: DrawRadialWheel | None = None
            self._ready_pill: ReadyStatusPill | None = None
            self._handle_dot: ExtrudeHandleDot | None = None
            self._extrude_hud: ExtrudeHudBar | None = None
            self._camera_hud: CameraHudBar | None = None

            self.setWindowFlags(
                QtCore.Qt.Window
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setCursor(QtCore.Qt.CrossCursor)

            self._build_draw_hud()

        @property
        def phase(self) -> str:
            return self._phase

        @property
        def extrude_length(self) -> float:
            return self._extrude_length

        @property
        def vertex_count(self) -> int:
            return self._vertex_count

        @property
        def is_wheel_open(self) -> bool:
            return self._radial_wheel is not None and self._radial_wheel.isVisible()

        @property
        def is_edit_mode(self) -> bool:
            return bool(self._edit_cutter_name)

        def _build_draw_hud(self) -> None:
            self._hud = QtWidgets.QFrame(self)
            self._hud.setStyleSheet(
                """
                QFrame {
                    background-color: rgba(24, 26, 32, 220);
                    border: 1px solid rgba(255, 200, 90, 180);
                    border-radius: 8px;
                }
                QLabel {
                    color: #f4f6fa;
                    font-family: "Segoe UI", sans-serif;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                }
                QPushButton {
                    background-color: #343946;
                    color: #f4f6fa;
                    border: 1px solid #52596b;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #464d5e; }
                QPushButton#ConfirmBtn {
                    background-color: #2c6e49;
                    border-color: #3d9966;
                    font-weight: bold;
                }
                QPushButton#CancelBtn {
                    background-color: #7c2d2d;
                    border-color: #a33d3d;
                }
                QPushButton#MaskBtn {
                    background-color: #1f4f63;
                    border-color: #38a8d8;
                    font-weight: bold;
                }
                QPushButton#MaskBtn:hover { background-color: #28657f; }
                QPushButton#MaskActiveBtn {
                    background-color: #1c859e;
                    border-color: #6ce5ff;
                    font-weight: bold;
                }
                """
            )
            hl = QtWidgets.QHBoxLayout(self._hud)
            hl.setContentsMargins(12, 6, 12, 6)
            hl.setSpacing(8)

            self._hud_label = QtWidgets.QLabel("")
            hl.addWidget(self._hud_label, 1)

            self._btn_mask_face = QtWidgets.QPushButton("Pick Face Mask (M)")
            self._btn_mask_face.setObjectName("MaskBtn")
            self._btn_mask_face.setCursor(QtCore.Qt.ArrowCursor)
            self._btn_mask_face.clicked.connect(self.toggle_mask_pick_phase)
            self._btn_mask_face.setVisible(self.is_edit_mode and bool(self._cutter_faces))
            hl.addWidget(self._btn_mask_face)

            self._btn_clear_mask = QtWidgets.QPushButton("Clear Mask")
            self._btn_clear_mask.setCursor(QtCore.Qt.ArrowCursor)
            self._btn_clear_mask.clicked.connect(self.clear_face_mask)
            self._btn_clear_mask.setVisible(False)
            hl.addWidget(self._btn_clear_mask)

            btn_cam = QtWidgets.QPushButton("Move Camera (Tab)")
            btn_cam.setCursor(QtCore.Qt.ArrowCursor)
            btn_cam.clicked.connect(self.enter_camera_phase)
            hl.addWidget(btn_cam)

            self._btn_restore_prev = QtWidgets.QPushButton("Back to Prev Shape")
            self._btn_restore_prev.setCursor(QtCore.Qt.ArrowCursor)
            self._btn_restore_prev.clicked.connect(self.restore_previous_ready)
            self._btn_restore_prev.setVisible(False)
            hl.addWidget(self._btn_restore_prev)

            self._btn_undo = QtWidgets.QPushButton("Undo (Bksp)")
            self._btn_undo.setCursor(QtCore.Qt.ArrowCursor)
            self._btn_undo.clicked.connect(self._undo_last_point)
            hl.addWidget(self._btn_undo)

            btn_cancel = QtWidgets.QPushButton("Cancel (ESC)")
            btn_cancel.setObjectName("CancelBtn")
            btn_cancel.setCursor(QtCore.Qt.ArrowCursor)
            btn_cancel.clicked.connect(self.cancel_all)
            hl.addWidget(btn_cancel)

            self._update_hud_text()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._cam.width = float(max(self.width(), 1))
            self._cam.height = float(max(self.height(), 1))
            hud_w = min(max(self.width() - 32, 460), 1020)
            self._hud.setGeometry((self.width() - hud_w) // 2, 12, hud_w, 38)
            self._position_floating_widgets()

        def _refresh_mask_buttons(self) -> None:
            if not hasattr(self, "_btn_mask_face"):
                return
            can_mask = self.is_edit_mode and bool(self._cutter_faces)
            self._btn_mask_face.setVisible(can_mask)
            m_cnt = len(self._mask_face_indices)
            self._btn_clear_mask.setVisible(can_mask and m_cnt > 0)
            if hasattr(self, "_btn_undo"):
                self._btn_undo.setVisible(self._phase == "DRAW")
            if self._phase == "MASK_PICK":
                self._btn_mask_face.setText("Done Mask -> Draw (M)")
                self._btn_mask_face.setObjectName("MaskActiveBtn")
            elif m_cnt > 0:
                self._btn_mask_face.setText(f"Mask: {m_cnt}f / {self._mask_depth:.1f}m (M)")
                self._btn_mask_face.setObjectName("MaskActiveBtn")
            else:
                self._btn_mask_face.setText("Pick Face Mask (M)")
                self._btn_mask_face.setObjectName("MaskBtn")
            self._btn_mask_face.style().unpolish(self._btn_mask_face)
            self._btn_mask_face.style().polish(self._btn_mask_face)

        def _update_hud_text(self) -> None:
            self._refresh_mask_buttons()
            m_cnt = len(self._mask_face_indices)
            mask_note = f" [Mask {m_cnt}f, {self._mask_depth:.2f}m]" if m_cnt > 0 else ""
            if self._phase == "MASK_PICK":
                self._hud_label.setText(
                    f"[Pick Face Mask | {m_cnt} faces | Depth: {self._mask_depth:.2f}m]  "
                    f"LMB: Toggle Face | Scroll: Depth | Tab: Camera | RMB / M: Draw"
                )
            elif self._phase == "DRAW":
                n = len(self._points)
                if self.is_edit_mode:
                    self._hud_label.setText(
                        f"[Draw Edit{mask_note} | Zone {self._zone}]  "
                        f"LMB: Corner/Drag | M: Face Mask | Tab: Camera | RMB: Finish ({n} pts)"
                    )
                else:
                    self._hud_label.setText(
                        f"[Draw Cutter | Zone {self._zone}]  "
                        f"LMB Click: Corner | Drag: Freehand | Close 1st pt / RMB: Finish ({n} pts)"
                    )
            elif self._phase == "READY" and self._ready_pill is not None:
                if self.is_edit_mode:
                    self._ready_pill.label.setText(
                        f"[Draw Edit{mask_note} | Zone {self._zone}]  {self._vertex_count} verts  |  "
                        f"Ctrl+Scroll: Vertices  |  Click center dot: Carve / Keep / Add"
                    )
                else:
                    self._ready_pill.label.setText(
                        f"[Zone {self._zone}]  {self._vertex_count} verts  |  "
                        f"Ctrl+Scroll: Vertices  |  Click center dot: Menu"
                    )
            elif self._phase == "CAMERA" and self._camera_hud is not None:
                if self.is_edit_mode:
                    back_lbl = "Pick Mask (Tab)" if self._return_phase_after_camera == "MASK_PICK" else "Draw Here (Tab)"
                    self._camera_hud.label.setText(
                        f"[Move Camera{mask_note} | Zone {self._zone}]  "
                        f"Orbit / Pan / Zoom freely, then click '{back_lbl}'"
                    )
                else:
                    self._camera_hud.label.setText(
                        f"[Move Camera | Zone {self._zone}]  "
                        f"Orbit / Pan / Zoom freely, then click 'Draw Here (Tab)'"
                    )
            elif self._phase == "EXTRUDE" and self._extrude_hud is not None:
                self._extrude_hud.label.setText(
                    f"[Extrude | Zone {self._zone}]  Depth: {self._extrude_length:.2f} m  |  "
                    f"Orbit camera freely & drag center dot to Extrude"
                )

        def _dist_sq(self, a: tuple[float, float], b: tuple[float, float]) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return dx * dx + dy * dy

        def _rebuild_locked_3d_from_vertex_count(self) -> bool:
            """Recompute locked 3D polygon & triangulation from raw 2D points at self._vertex_count."""
            if len(self._raw_closed_2d) < 3 or self._draw_cam_snapshot is None:
                return False
            pts_adj = adjust_polygon_vertex_count(self._raw_closed_2d, self._vertex_count)
            if len(pts_adj) < 3 or abs(polygon_signed_area_2d(pts_adj)) < 1.0:
                return False
            pts_ccw = ensure_ccw_2d(pts_adj)
            cap_tris = triangulate_polygon_2d(pts_ccw)
            if not cap_tris:
                return False
            self._vertex_count = len(pts_ccw)
            self._pts_ccw_locked = pts_ccw
            self._front_3d = unproject_qt_to_world_plane(
                pts_ccw, self._draw_cam_snapshot, self._front_depth_locked
            )
            self._cap_tris = cap_tris
            self._front_center_3d = np.mean(self._front_3d, axis=0)
            return True

        def _enter_ready_phase(self) -> None:
            """Lock the closed shape and enter READY phase (center menu button + Ctrl+Scroll vertices)."""
            if len(self._points) < 3:
                return
            raw_clean = simplify_polyline(self._points, min_dist_px=2.0, rdp_epsilon=0.0, closed=True)
            if len(raw_clean) < 3 or abs(polygon_signed_area_2d(raw_clean)) < 1.0:
                return
            initial_simp = simplify_polyline(self._points, min_dist_px=3.0, rdp_epsilon=2.0, closed=True)
            init_count = max(3, len(initial_simp))

            self._raw_closed_2d = raw_clean
            self._draw_cam_snapshot = copy.deepcopy(self._cam)
            through_margin = 1.0 if self.is_edit_mode else 0.45
            front_depth, thru_len = self._cam.through_building_range(margin=through_margin)
            self._front_depth_locked = front_depth
            self._extrude_dir = self._cam.view_forward.copy()
            self._through_length = thru_len
            self._extrude_length = thru_len
            self._vertex_count = init_count

            if not self._rebuild_locked_3d_from_vertex_count():
                return

            self._show_ready_phase()

        def _show_ready_phase(self) -> None:
            self._phase = "READY"
            self._ctrl_wheel_capture = False
            self._hud.setVisible(False)
            self._hide_all_floating()
            _set_win_click_through(self, True)

            if self._center_btn is None:
                self._center_btn = CenterMenuButton(self)
            if self._ready_pill is None:
                self._ready_pill = ReadyStatusPill()

            self._update_hud_text()
            self._position_floating_widgets()
            self._ready_pill.show()
            self._center_btn.show()
            self._center_btn.raise_()
            self.update()

        def restore_previous_ready(self) -> None:
            """Return to READY phase with the previously drawn 3D shape intact."""
            if len(self._front_3d) < 3 or not self._cap_tris:
                return
            self._show_ready_phase()

        def adjust_vertices(self, step_dir: int) -> None:
            """Increase or decrease polygon vertex count (called by Ctrl+Scroll in READY phase)."""
            if self._phase != "READY" or len(self._raw_closed_2d) < 3:
                return
            step = 1
            if self._vertex_count >= 60:
                step = 4
            elif self._vertex_count >= 24:
                step = 2
            new_count = max(3, min(256, self._vertex_count + (step if step_dir > 0 else -step)))
            if new_count == self._vertex_count:
                return
            self._vertex_count = new_count
            if self._rebuild_locked_3d_from_vertex_count():
                self._update_hud_text()
                self._position_floating_widgets()
                if self._center_btn is not None:
                    self._center_btn.update()
                self.update()

        def open_radial_wheel(self) -> None:
            """Open the Radial Donut Wheel Menu centered on the active center handle button."""
            anchor = self._center_btn if self._phase == "READY" else self._handle_dot
            if self._phase not in ("READY", "EXTRUDE") or anchor is None:
                return
            if self._radial_wheel is None:
                self._radial_wheel = DrawRadialWheel(self)
            bg = anchor.geometry()
            cx = bg.x() + bg.width() // 2
            cy = bg.y() + bg.height() // 2
            w_half = self._radial_wheel.WHEEL_SIZE // 2
            self._radial_wheel.move(cx - w_half, cy - w_half)
            self._radial_wheel._hovered = None
            self._radial_wheel.update()
            self._radial_wheel.show()
            self._radial_wheel.raise_()
            self._radial_wheel.activateWindow()

        def close_radial_wheel(self) -> None:
            if self._radial_wheel is not None:
                self._radial_wheel.hide()

        def execute_wheel_action(self, action_id: str) -> None:
            if action_id == "carve":
                self.confirm_sculpt("DIFFERENCE")
            elif action_id == "keep_inside":
                self.confirm_sculpt("INTERSECT")
            elif action_id == "add_shape":
                self.confirm_sculpt("UNION")
            elif action_id == "confirm":
                self.confirm_extrude()
            elif action_id == "through":
                self.confirm_through_building()
            elif action_id == "extrude":
                self.enter_extrude_phase()
            elif action_id == "camera":
                self.enter_camera_phase()
            elif action_id == "redraw":
                self.enter_draw_phase()
            elif action_id == "cancel":
                self.cancel_all()

        def enter_extrude_phase(self) -> None:
            """Enter interactive 3D Extrude drag mode from the locked 3D shape."""
            if len(self._front_3d) < 3 or not self._cap_tris:
                return
            self._phase = "EXTRUDE"
            self._ctrl_wheel_capture = False
            self._hud.setVisible(False)
            self._hide_all_floating()
            _set_win_click_through(self, True)

            if self._handle_dot is None:
                self._handle_dot = ExtrudeHandleDot(self)
            if self._extrude_hud is None:
                self._extrude_hud = ExtrudeHudBar(self)

            self._update_hud_text()
            self._position_floating_widgets()
            self._extrude_hud.show()
            self._extrude_hud.raise_()
            self._handle_dot.show()
            self._handle_dot.raise_()
            self.update()

        def enter_mask_pick_phase(self) -> None:
            """Enter interactive Cutter Face Mask picking mode (LMB toggles faces, Scroll adjusts depth)."""
            if not self.is_edit_mode or not self._cutter_faces:
                return
            self._phase = "MASK_PICK"
            self._ctrl_wheel_capture = False
            self._hovered_mask_face = None
            self._hide_all_floating()
            _set_win_click_through(self, False)
            self._btn_restore_prev.setVisible(False)
            self._hud.setVisible(True)
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self._update_hud_text()
            self.raise_()
            self.activateWindow()
            self.update()

        def toggle_mask_pick_phase(self) -> None:
            if not self.is_edit_mode or not self._cutter_faces:
                return
            if self._phase == "MASK_PICK":
                self.enter_draw_phase()
            else:
                self.enter_mask_pick_phase()

        def clear_face_mask(self) -> None:
            self._mask_face_indices.clear()
            self._hovered_mask_face = None
            self._update_hud_text()
            self.update()

        def return_from_camera_phase(self) -> None:
            if self._return_phase_after_camera == "MASK_PICK" and self.is_edit_mode and self._cutter_faces:
                self.enter_mask_pick_phase()
            else:
                self.enter_draw_phase()

        def enter_camera_phase(self) -> None:
            """Unlock camera (click-through) so user can orbit/pan/zoom freely before drawing."""
            self._return_phase_after_camera = "MASK_PICK" if self._phase == "MASK_PICK" else "DRAW"
            self._phase = "CAMERA"
            self._ctrl_wheel_capture = False
            self._points = []
            self._hovered_mask_face = None
            self._hud.setVisible(False)
            self._hide_all_floating()
            _set_win_click_through(self, True)

            if self._camera_hud is None:
                self._camera_hud = CameraHudBar(self)
            self._camera_hud.btn_restore_prev.setVisible(len(self._front_3d) >= 3)
            self._update_hud_text()
            self._position_floating_widgets()
            self._camera_hud.show()
            self._camera_hud.raise_()
            self.update()

        def enter_draw_phase(self) -> None:
            """Lock current camera angle and start drawing a fresh polygon/freehand stroke."""
            self._phase = "DRAW"
            self._ctrl_wheel_capture = False
            self._points = []
            self._hovered_mask_face = None
            self._hide_all_floating()
            _set_win_click_through(self, False)
            self.setCursor(QtCore.Qt.CrossCursor)
            self._btn_restore_prev.setVisible(len(self._front_3d) >= 3)
            self._hud.setVisible(True)
            self._update_hud_text()
            self.raise_()
            self.activateWindow()
            self.update()

        def _hide_all_floating(self) -> None:
            for w in (
                self._center_btn,
                self._radial_wheel,
                self._ready_pill,
                self._handle_dot,
                self._extrude_hud,
                self._camera_hud,
            ):
                if w is not None:
                    w.hide()

        def _position_floating_widgets(self) -> None:
            g = self.geometry()
            if self._phase == "CAMERA":
                if self._camera_hud is not None:
                    hud_w = min(max(g.width() - 32, 480), 860)
                    self._camera_hud.setGeometry(g.x() + (g.width() - hud_w) // 2, g.y() + 12, hud_w, 38)
                return

            if self._phase == "READY":
                if self._ready_pill is not None:
                    pill_w = min(max(g.width() - 80, 360), 540)
                    self._ready_pill.setGeometry(g.x() + (g.width() - pill_w) // 2, g.y() + 12, pill_w, 30)
                if self._center_btn is not None and len(self._front_3d) >= 3:
                    proj = project_world_to_qt_2d(self._front_center_3d[np.newaxis, :], self._cam)[0]
                    hx, hy = float(proj[0]), float(proj[1])
                    if -20.0 <= hx <= g.width() + 20.0 and -20.0 <= hy <= g.height() + 20.0:
                        half = self._center_btn.BTN_SIZE // 2
                        self._center_btn.move(int(round(g.x() + hx - half)), int(round(g.y() + hy - half)))
                        if not self._center_btn.isVisible():
                            self._center_btn.show()
                    else:
                        self._center_btn.hide()
                return

            if self._phase == "EXTRUDE":
                if self._extrude_hud is not None:
                    hud_w = min(max(g.width() - 32, 480), 860)
                    self._extrude_hud.setGeometry(g.x() + (g.width() - hud_w) // 2, g.y() + 12, hud_w, 38)
                if self._handle_dot is not None and len(self._front_3d) >= 3:
                    handle_3d = self._front_center_3d + self._extrude_dir * self._extrude_length
                    proj = project_world_to_qt_2d(handle_3d[np.newaxis, :], self._cam)[0]
                    hx, hy = float(proj[0]), float(proj[1])
                    if -20.0 <= hx <= g.width() + 20.0 and -20.0 <= hy <= g.height() + 20.0:
                        half = self._handle_dot.HANDLE_SIZE // 2
                        self._handle_dot.move(int(round(g.x() + hx - half)), int(round(g.y() + hy - half)))
                        if not self._handle_dot.isVisible():
                            self._handle_dot.show()
                    else:
                        self._handle_dot.hide()

        def sync_live_camera(self, rv3d: bpy.types.RegionView3D) -> None:
            """Called at ~60fps while in READY, CAMERA, or EXTRUDE phase."""
            _update_camera_matrices(self._cam, rv3d, self._target_corners)

            # In READY phase, temporarily capture wheel events while Ctrl is held so Ctrl+Scroll
            # adjusts vertex count without zooming Blender!
            if self._phase == "READY" and not self.is_wheel_open:
                want_capture = _is_ctrl_held()
                if want_capture != self._ctrl_wheel_capture:
                    self._ctrl_wheel_capture = want_capture
                    _set_win_click_through(self, not want_capture)

            if self._phase in ("READY", "EXTRUDE"):
                self._position_floating_widgets()
                self.update()
            elif self._phase == "CAMERA" and self._mask_face_indices:
                self.update()

        def on_handle_drag(self, dx_px: float, dy_px: float, start_len: float) -> None:
            """Project 2D mouse drag onto the 3D extrude axis so dragging sticks 1:1 to the cursor."""
            p0_3d = self._front_center_3d + self._extrude_dir * start_len
            p1_3d = p0_3d + self._extrude_dir
            proj = project_world_to_qt_2d(np.vstack([p0_3d, p1_3d]), self._cam)
            axis_2d = proj[1] - proj[0]
            axis_len_sq = float(axis_2d[0] * axis_2d[0] + axis_2d[1] * axis_2d[1])

            if axis_len_sq >= 16.0:
                delta_m = float((dx_px * axis_2d[0] + dy_px * axis_2d[1]) / axis_len_sq)
            else:
                scale = max(self._cam.target_thickness / 180.0, 0.015)
                delta_m = (dx_px - dy_px * 0.5) * scale

            new_len = start_len + delta_m
            if abs(new_len) < 0.05:
                new_len = 0.05 if new_len >= 0.0 else -0.05
            self._extrude_length = max(-200.0, min(200.0, new_len))
            self._update_hud_text()
            self._position_floating_widgets()
            self.update()

        def _close_subwidgets(self) -> None:
            for attr in ("_center_btn", "_radial_wheel", "_ready_pill", "_handle_dot", "_extrude_hud", "_camera_hud"):
                w = getattr(self, attr, None)
                if w is not None:
                    w.close()
                    setattr(self, attr, None)

        def closeEvent(self, event) -> None:
            self._close_subwidgets()
            super().closeEvent(event)

        def _undo_last_point(self) -> None:
            if self._points:
                self._points.pop()
                self._update_hud_text()
                self.update()

        def confirm_extrude(self) -> None:
            if self._phase in ("READY", "EXTRUDE") and len(self._front_3d) >= 3:
                try:
                    verts, faces = extrude_polygon_3d_to_mesh(
                        self._front_3d,
                        self._cap_tris,
                        self._extrude_dir,
                        self._extrude_length,
                    )
                except ValueError:
                    return
            elif len(self._points) >= 3:
                front, length = self._cam.through_building_range(margin=0.45)
                try:
                    verts, faces = extrude_polygon_2d_to_mesh(
                        self._points,
                        self._cam,
                        front_depth=front,
                        extrude_length=length,
                        simplify=True,
                    )
                except ValueError:
                    return
            else:
                return

            self._close_subwidgets()
            self.close()
            self._on_confirm(verts, faces, self._zone)

        def confirm_through_building(self) -> None:
            if self._phase in ("READY", "EXTRUDE") and len(self._front_3d) >= 3:
                self._extrude_length = self._through_length
                self.confirm_extrude()
            elif len(self._points) >= 3:
                front, length = self._cam.through_building_range(margin=0.5)
                try:
                    verts, faces = extrude_polygon_2d_to_mesh(
                        self._points,
                        self._cam,
                        front_depth=front,
                        extrude_length=length,
                        simplify=True,
                    )
                except ValueError:
                    return
                self._close_subwidgets()
                self.close()
                self._on_confirm(verts, faces, self._zone)

        def confirm_sculpt(self, operation: str) -> None:
            """Extrude locked 2D shape along exact camera rays (Perspective/Ortho) through the target cutter."""
            if not self.is_edit_mode or self._on_sculpt_confirm is None:
                return
            if (
                self._phase == "READY"
                and len(self._pts_ccw_locked) >= 3
                and self._draw_cam_snapshot is not None
                and self._cap_tris
            ):
                try:
                    verts, faces = extrude_polygon_2d_frustum_to_mesh(
                        self._pts_ccw_locked,
                        self._cap_tris,
                        self._draw_cam_snapshot,
                        front_depth=self._front_depth_locked,
                        back_depth=self._front_depth_locked + self._through_length,
                    )
                except ValueError:
                    return
            elif len(self._points) >= 3:
                pts_2d = simplify_polyline(self._points, min_dist_px=3.0, rdp_epsilon=2.0, closed=True)
                if len(pts_2d) < 3 or abs(polygon_signed_area_2d(pts_2d)) < 1.0:
                    return
                pts_ccw = ensure_ccw_2d(pts_2d)
                cap_tris = triangulate_polygon_2d(pts_ccw)
                if not cap_tris:
                    return
                front, length = self._cam.through_building_range(margin=1.0)
                try:
                    verts, faces = extrude_polygon_2d_frustum_to_mesh(
                        pts_ccw,
                        cap_tris,
                        self._cam,
                        front_depth=front,
                        back_depth=front + length,
                    )
                except ValueError:
                    return
            else:
                return

            self._close_subwidgets()
            self.close()
            self._on_sculpt_confirm(
                verts,
                faces,
                operation,
                sorted(self._mask_face_indices),
                float(self._mask_depth),
            )

        def cancel_all(self) -> None:
            self._close_subwidgets()
            self.close()
            self._on_cancel()

        def handle_extrude_key(self, key: int) -> None:
            if key == QtCore.Qt.Key_Escape:
                self.cancel_all()
            elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.confirm_extrude()
            elif key == QtCore.Qt.Key_Space:
                self.confirm_through_building()

        def get_all_hwnds(self) -> list[int]:
            hwnds: list[int] = []
            for w in (
                self,
                self._center_btn,
                self._radial_wheel,
                self._ready_pill,
                self._handle_dot,
                self._extrude_hud,
                self._camera_hud,
            ):
                if w is not None:
                    try:
                        hwnds.append(int(w.winId()))
                    except Exception:
                        pass
            return hwnds

        def handle_vk_shortcut(self, vk: int) -> None:
            if vk == _VK_ESCAPE:
                if self.is_wheel_open:
                    self.close_radial_wheel()
                elif self._phase == "MASK_PICK":
                    self.enter_draw_phase()
                else:
                    self.cancel_all()
            elif vk == _VK_M:
                if self.is_edit_mode and self._phase in ("DRAW", "MASK_PICK", "READY"):
                    self.toggle_mask_pick_phase()
            elif vk == _VK_TAB:
                if self._phase in ("DRAW", "MASK_PICK"):
                    self.enter_camera_phase()
                elif self._phase == "CAMERA":
                    self.return_from_camera_phase()
            elif vk == _VK_RETURN:
                if self._phase == "MASK_PICK":
                    self.enter_draw_phase()
                elif self._phase == "DRAW":
                    self._enter_ready_phase()
                elif self._phase == "CAMERA":
                    self.return_from_camera_phase()
                elif self._phase in ("READY", "EXTRUDE"):
                    if self.is_edit_mode:
                        self.open_radial_wheel()
                    else:
                        self.confirm_extrude()
            elif vk == _VK_SPACE:
                if self.is_edit_mode:
                    if self._phase == "MASK_PICK":
                        self.enter_draw_phase()
                    elif self._phase == "DRAW":
                        self._enter_ready_phase()
                    elif self._phase == "READY":
                        self.open_radial_wheel()
                elif self._phase in ("DRAW", "READY", "EXTRUDE"):
                    self.confirm_through_building()
            elif vk == _VK_BACK and self._phase == "DRAW":
                self._undo_last_point()

        def event(self, ev) -> bool:
            if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_Tab:
                if sys.platform != "win32":
                    self.handle_vk_shortcut(_VK_TAB)
                return True
            return super().event(ev)

        def wheelEvent(self, event) -> None:
            if self._phase == "READY":
                delta = 1 if event.angleDelta().y() > 0 else -1
                self.adjust_vertices(delta)
                event.accept()
            elif self._phase == "MASK_PICK":
                step = 0.15 if event.angleDelta().y() > 0 else -0.15
                self._mask_depth = max(0.10, min(20.0, round(self._mask_depth + step, 2)))
                self._update_hud_text()
                self.update()
                event.accept()

        def mousePressEvent(self, event) -> None:
            pos = event.position()
            pt = (float(pos.x()), float(pos.y()))
            self._mouse_pos = pt

            if self._phase == "MASK_PICK":
                if event.button() == QtCore.Qt.LeftButton:
                    hit_idx = _ray_hit_cutter_faces(pt[0], pt[1], self._cam, self._cutter_faces)
                    if hit_idx is not None:
                        if hit_idx in self._mask_face_indices:
                            self._mask_face_indices.remove(hit_idx)
                        else:
                            self._mask_face_indices.add(hit_idx)
                        self._hovered_mask_face = hit_idx
                        self._update_hud_text()
                        self.update()
                elif event.button() == QtCore.Qt.RightButton:
                    self.enter_draw_phase()
                return

            if self._phase != "DRAW":
                return

            if event.button() == QtCore.Qt.LeftButton:
                self._lmb_down = True
                self._lmb_press_pos = pt
                self._is_freehand = False
                # Snap-close when clicking near first vertex -> enter READY phase (center menu button!)
                if len(self._points) >= 3 and self._dist_sq(pt, self._points[0]) <= 15.0 * 15.0:
                    self._lmb_down = False
                    self._enter_ready_phase()
                    return
                self._points.append(pt)
                self._update_hud_text()
                self.update()

            elif event.button() == QtCore.Qt.RightButton:
                if len(self._points) >= 3:
                    self._enter_ready_phase()
                elif len(self._points) == 0 and len(self._front_3d) >= 3:
                    self.restore_previous_ready()
                else:
                    self.cancel_all()

        def mouseMoveEvent(self, event) -> None:
            pos = event.position()
            pt = (float(pos.x()), float(pos.y()))
            self._mouse_pos = pt

            if self._phase == "MASK_PICK":
                new_h = _ray_hit_cutter_faces(pt[0], pt[1], self._cam, self._cutter_faces)
                if new_h != self._hovered_mask_face:
                    self._hovered_mask_face = new_h
                    self.update()
                return

            if self._phase != "DRAW":
                return

            if self._lmb_down:
                if not self._is_freehand and self._dist_sq(pt, self._lmb_press_pos) >= 8.0 * 8.0:
                    self._is_freehand = True
                if self._is_freehand:
                    if not self._points or self._dist_sq(pt, self._points[-1]) >= 6.0 * 6.0:
                        self._points.append(pt)
                        self._update_hud_text()
            self.update()

        def mouseReleaseEvent(self, event) -> None:
            if self._phase == "DRAW" and event.button() == QtCore.Qt.LeftButton:
                was_freehand = self._is_freehand
                self._lmb_down = False
                self._is_freehand = False
                if was_freehand and len(self._points) >= 5:
                    if self._dist_sq(self._points[-1], self._points[0]) <= 22.0 * 22.0:
                        self._enter_ready_phase()

        def keyPressEvent(self, event) -> None:
            if sys.platform == "win32":
                if event.key() in (
                    QtCore.Qt.Key_Escape,
                    QtCore.Qt.Key_Tab,
                    QtCore.Qt.Key_Return,
                    QtCore.Qt.Key_Enter,
                    QtCore.Qt.Key_Space,
                    QtCore.Qt.Key_Backspace,
                ):
                    event.accept()
                    return
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                self.cancel_all()
            elif key == QtCore.Qt.Key_Tab:
                if self._phase == "DRAW":
                    self.enter_camera_phase()
                elif self._phase == "CAMERA":
                    self.enter_draw_phase()
            elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if self._phase == "DRAW":
                    self._enter_ready_phase()
                elif self._phase == "CAMERA":
                    self.enter_draw_phase()
                elif self.is_edit_mode:
                    self.open_radial_wheel()
                else:
                    self.confirm_extrude()
            elif key == QtCore.Qt.Key_Space:
                if self.is_edit_mode:
                    if self._phase == "DRAW":
                        self._enter_ready_phase()
                    elif self._phase == "READY":
                        self.open_radial_wheel()
                else:
                    self.confirm_through_building()
            elif key == QtCore.Qt.Key_Backspace and self._phase == "DRAW":
                self._undo_last_point()

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            if self._phase == "MASK_PICK":
                painter.fillRect(self.rect(), QtGui.QColor(8, 18, 26, 22))
                painter.setPen(QtGui.QPen(QtGui.QColor(70, 225, 255, 220), 3))
                painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
                self._paint_mask_overlay(painter, interactive=True)
            elif self._phase == "DRAW":
                painter.fillRect(self.rect(), QtGui.QColor(12, 16, 24, 24))
                painter.setPen(QtGui.QPen(self._zone_color, 3))
                painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
                if self._mask_face_indices:
                    self._paint_mask_overlay(painter, interactive=False)
                self._paint_draw_phase(painter)
            elif self._phase == "READY":
                border_col = QtGui.QColor(self._zone_color)
                border_col.setAlpha(150)
                painter.setPen(QtGui.QPen(border_col, 2))
                painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
                if self._mask_face_indices:
                    self._paint_mask_overlay(painter, interactive=False)
                self._paint_ready_phase(painter)
            elif self._phase == "CAMERA":
                cam_pen = QtGui.QPen(QtGui.QColor(90, 195, 255, 190), 2.5, QtCore.Qt.DashLine)
                painter.setPen(cam_pen)
                painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
                if self._mask_face_indices:
                    self._paint_mask_overlay(painter, interactive=False)
            else:
                border_col = QtGui.QColor(self._zone_color)
                border_col.setAlpha(150)
                painter.setPen(QtGui.QPen(border_col, 2))
                painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
                self._paint_extrude_phase(painter)

            painter.end()

        def _paint_mask_overlay(self, painter: QtGui.QPainter, interactive: bool = False) -> None:
            """Draw selected Cutter Face Mask(s), their inward 3D depth slab preview, and hover highlight."""
            if not self._cutter_faces:
                return

            # 1. In MASK_PICK phase, lightly outline front-facing cutter faces so clickable faces are obvious
            if interactive:
                faint_pen = QtGui.QPen(QtGui.QColor(130, 220, 255, 75), 1.1)
                painter.setBrush(QtGui.QColor(60, 180, 230, 14))
                painter.setPen(faint_pen)
                for f in self._cutter_faces:
                    if float(np.dot(f["normal"], self._cam.view_forward)) >= 0.05:
                        continue
                    p2d = project_world_to_qt_2d(f["verts"], self._cam)
                    poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in p2d])
                    painter.drawPolygon(poly)

            # 2. Draw selected mask faces + inward depth slab preview
            for f in self._cutter_faces:
                idx = f["index"]
                if idx not in self._mask_face_indices:
                    continue
                verts = f["verts"]
                normal = f["normal"]
                back_verts = verts - normal[np.newaxis, :] * float(self._mask_depth)

                front_2d = project_world_to_qt_2d(verts, self._cam)
                back_2d = project_world_to_qt_2d(back_verts, self._cam)

                front_poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in front_2d])
                back_poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in back_2d])

                # Inward depth slab wireframe
                painter.setBrush(QtGui.QColor(40, 215, 255, 18))
                painter.setPen(QtGui.QPen(QtGui.QColor(70, 235, 255, 130), 1.4, QtCore.Qt.DashLine))
                painter.drawPolygon(back_poly)
                for i in range(len(front_2d)):
                    painter.drawLine(
                        QtCore.QPointF(float(front_2d[i, 0]), float(front_2d[i, 1])),
                        QtCore.QPointF(float(back_2d[i, 0]), float(back_2d[i, 1])),
                    )

                # Selected cutter face highlight
                fill_a = 72 if interactive else 42
                painter.setBrush(QtGui.QColor(40, 220, 255, fill_a))
                painter.setPen(QtGui.QPen(QtGui.QColor(90, 245, 255, 235), 2.4))
                painter.drawPolygon(front_poly)

            # 3. In MASK_PICK phase, draw hovered face in bright gold/cyan
            if interactive and self._hovered_mask_face is not None:
                for f in self._cutter_faces:
                    if f["index"] == self._hovered_mask_face:
                        p2d = project_world_to_qt_2d(f["verts"], self._cam)
                        poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in p2d])
                        is_sel = f["index"] in self._mask_face_indices
                        fill = QtGui.QColor(255, 120, 90, 85) if is_sel else QtGui.QColor(255, 225, 85, 85)
                        rim = QtGui.QColor(255, 150, 120, 245) if is_sel else QtGui.QColor(255, 235, 110, 245)
                        painter.setBrush(fill)
                        painter.setPen(QtGui.QPen(rim, 2.8))
                        painter.drawPolygon(poly)
                        break

        def _paint_draw_phase(self, painter: QtGui.QPainter) -> None:
            if not self._points:
                return

            fill_col = QtGui.QColor(self._zone_color)
            fill_col.setAlpha(45)
            line_pen = QtGui.QPen(self._zone_color, 2.5)
            dash_pen = QtGui.QPen(QtGui.QColor(255, 230, 120, 210), 1.8, QtCore.Qt.DashLine)

            poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self._points])
            if len(self._points) >= 3:
                painter.setBrush(fill_col)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawPolygon(poly)

            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(line_pen)
            painter.drawPolyline(poly)

            last = self._points[-1]
            painter.setPen(dash_pen)
            painter.drawLine(
                QtCore.QPointF(last[0], last[1]),
                QtCore.QPointF(self._mouse_pos[0], self._mouse_pos[1]),
            )

            painter.setBrush(QtGui.QColor(255, 255, 255, 230))
            painter.setPen(QtGui.QPen(self._zone_color, 1.5))
            for idx, (x, y) in enumerate(self._points):
                if idx == 0 or not self._is_freehand or idx % 4 == 0:
                    painter.drawEllipse(QtCore.QPointF(x, y), 3.5, 3.5)

            if len(self._points) >= 3:
                fx, fy = self._points[0]
                near_start = self._dist_sq(self._mouse_pos, self._points[0]) <= 15.0 * 15.0
                ring_col = QtGui.QColor(80, 240, 130, 240) if near_start else QtGui.QColor(80, 240, 130, 130)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(ring_col, 2.2))
                painter.drawEllipse(QtCore.QPointF(fx, fy), 11.0, 11.0)

        def _paint_ready_phase(self, painter: QtGui.QPainter) -> None:
            """Draw the closed 3D polygon + crisp vertex dots (live preview while Ctrl+Scrolling)."""
            if len(self._front_3d) < 3:
                return
            pts_2d = project_world_to_qt_2d(self._front_3d, self._cam)
            poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in pts_2d])

            fill_col = QtGui.QColor(self._zone_color)
            fill_col.setAlpha(55)
            painter.setBrush(fill_col)
            painter.setPen(QtGui.QPen(self._zone_color, 2.6))
            painter.drawPolygon(poly)

            # Draw every vertex dot clearly so Ctrl+Scroll vertex changes are unmistakable
            painter.setBrush(QtGui.QColor(255, 240, 140, 245))
            painter.setPen(QtGui.QPen(QtGui.QColor(24, 26, 32, 240), 1.4))
            dot_r = 4.2 if len(pts_2d) <= 40 else 3.0
            for p in pts_2d:
                painter.drawEllipse(QtCore.QPointF(float(p[0]), float(p[1])), dot_r, dot_r)

        def _paint_extrude_phase(self, painter: QtGui.QPainter) -> None:
            if len(self._front_3d) < 3:
                return
            try:
                verts, _faces = extrude_polygon_3d_to_mesh(
                    self._front_3d,
                    self._cap_tris,
                    self._extrude_dir,
                    self._extrude_length,
                )
            except ValueError:
                return

            n = len(self._front_3d)
            proj = project_world_to_qt_2d(verts, self._cam)
            front_2d = proj[:n]
            back_2d = proj[n:]

            front_poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in front_2d])
            back_poly = QtGui.QPolygonF([QtCore.QPointF(float(p[0]), float(p[1])) for p in back_2d])

            front_fill = QtGui.QColor(self._zone_color)
            front_fill.setAlpha(45)
            painter.setBrush(front_fill)
            painter.setPen(QtGui.QPen(self._zone_color, 2.4))
            painter.drawPolygon(front_poly)

            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 220, 120, 210), 1.6))
            for i in range(n):
                painter.drawLine(
                    QtCore.QPointF(float(front_2d[i, 0]), float(front_2d[i, 1])),
                    QtCore.QPointF(float(back_2d[i, 0]), float(back_2d[i, 1])),
                )

            back_fill = QtGui.QColor(self._zone_color)
            back_fill.setAlpha(60)
            painter.setBrush(back_fill)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 215, 70, 230), 2.2))
            painter.drawPolygon(back_poly)

            handle_3d = self._front_center_3d + self._extrude_dir * self._extrude_length
            centers_2d = project_world_to_qt_2d(
                np.vstack([self._front_center_3d, handle_3d]),
                self._cam,
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 225, 90, 220), 2.0, QtCore.Qt.DashLine))
            painter.drawLine(
                QtCore.QPointF(float(centers_2d[0, 0]), float(centers_2d[0, 1])),
                QtCore.QPointF(float(centers_2d[1, 0]), float(centers_2d[1, 1])),
            )


def _overlay_event_pump():
    global _overlay_timer_running, _active_overlay, _active_rv3d_ref
    if not PYSIDE_AVAILABLE or _active_overlay is None or not _active_overlay.isVisible():
        _overlay_timer_running = False
        _active_overlay = None
        _active_rv3d_ref = None
        return None
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()
    try:
        if _active_overlay is not None and _active_overlay.isVisible():
            _poll_global_shortcuts(_active_overlay)
        if (
            _active_overlay is not None
            and _active_overlay.isVisible()
            and _active_overlay.phase in ("READY", "CAMERA", "EXTRUDE")
            and _active_rv3d_ref is not None
        ):
            _active_overlay.sync_live_camera(_active_rv3d_ref)
    except ReferenceError:
        close_draw_overlay()
        return None
    except Exception:
        pass
    return 0.016


def close_draw_overlay() -> None:
    global _active_overlay, _active_rv3d_ref, _active_win_ref, _active_region_ref, _overlay_timer_running
    _overlay_timer_running = False
    if _active_overlay is not None:
        try:
            _active_overlay.close()
        except Exception:
            pass
        _active_overlay = None
    _active_rv3d_ref = None
    _active_win_ref = None
    _active_region_ref = None


def open_draw_cutter_overlay(context: bpy.types.Context, zone: str | None = None) -> tuple[bool, str]:
    """Launch the standalone PySide6 Draw, Vertex-Adjust & Radial Wheel overlay over the active 3D Viewport."""
    global _active_overlay, _active_rv3d_ref, _active_win_ref, _active_region_ref, _overlay_timer_running, _blender_hwnd

    if not PYSIDE_AVAILABLE:
        return False, "PySide6 is required for Draw Cutter overlay."

    from ..core.session import load_session

    session = load_session(context.scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite"

    target_win = context.window
    target_area = context.area if (context.area and context.area.type == "VIEW_3D") else None
    target_region = None
    if target_area is None and context.window_manager is not None:
        for win in context.window_manager.windows:
            if win.screen is None:
                continue
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    target_win = win
                    target_area = area
                    break
            if target_area is not None:
                break

    if target_win is None or target_area is None:
        return False, "ไม่พบหน้าต่าง 3D Viewport"

    for reg in target_area.regions:
        if reg.type == "WINDOW":
            target_region = reg
            break
    rv3d = target_area.spaces.active.region_3d if target_area.spaces.active else None
    if target_region is None or rv3d is None:
        return False, "ไม่สามารถดึงมุมกล้องจาก 3D Viewport ได้"

    close_draw_overlay()
    _ensure_qapp()
    _blender_hwnd = _find_blender_hwnd(target_win)
    _reset_vk_state()

    lx, ly, lw, lh = _get_viewport_qt_rect(target_win, target_region, blender_hwnd=_blender_hwnd)
    targets = session.get_target_objects()
    cam = capture_camera_snapshot(rv3d, targets, lw, lh)
    target_corners = collect_target_bbox_corners(targets)

    settings = getattr(context.scene, "fdt_settings", None)
    active_zone = str(zone or (settings.active_zone if settings else "A") or "A").strip() or "A"
    zone_rgba = apple_bite.get_zone_color(active_zone)

    def _on_confirm(verts: np.ndarray, faces: list[tuple[int, ...]], z_id: str) -> None:
        global _active_overlay, _active_rv3d_ref
        _active_overlay = None
        _active_rv3d_ref = None
        with blender_view3d_override():
            ok, msg, _cutter = apple_bite.add_mesh_data_cutter(
                bpy.context,
                vertices=verts,
                faces=faces,
                zone=z_id,
            )
            st = getattr(bpy.context.scene, "fdt_settings", None)
            if st is not None:
                st.last_status = msg
            if ok:
                from ..ops import apple_bite_ops

                apple_bite_ops._ensure_wire_overlay()
        _tag_all_view3d_redraw()

    def _on_cancel() -> None:
        global _active_overlay, _active_rv3d_ref
        _active_overlay = None
        _active_rv3d_ref = None
        _tag_all_view3d_redraw()

    overlay = DrawCutterOverlay(
        cam=cam,
        target_corners=target_corners,
        zone=active_zone,
        zone_rgba=zone_rgba,
        on_confirm=_on_confirm,
        on_cancel=_on_cancel,
    )
    overlay.setGeometry(lx, ly, lw, lh)
    overlay.show()
    overlay.raise_()
    overlay.activateWindow()
    _active_overlay = overlay
    _active_rv3d_ref = rv3d
    _active_win_ref = target_win
    _active_region_ref = target_region

    if not bpy.app.timers.is_registered(_overlay_event_pump):
        _overlay_timer_running = True
        bpy.app.timers.register(_overlay_event_pump, first_interval=0.016, persistent=False)

    return True, f"เปิดโหมดวาดตัวตัด (โซน {active_zone}) แล้ว — วาดปิดรูป แล้วกด Ctrl+ลูกกลิ้ง ปรับจุด หรือคลิกปุ่มกลางเปิดวงล้อได้เลย"


def open_draw_sculpt_overlay(context: bpy.types.Context, cutter_name: str) -> tuple[bool, str]:
    """Launch PySide6 Draw Overlay in Edit mode to Carve/Keep/Add onto an existing cutter."""
    global _active_overlay, _active_rv3d_ref, _active_win_ref, _active_region_ref, _overlay_timer_running, _blender_hwnd

    if not PYSIDE_AVAILABLE:
        return False, "PySide6 is required for Draw Cut overlay."

    from ..core.session import load_session

    session = load_session(context.scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite"

    cutter_obj = bpy.data.objects.get(cutter_name)
    if cutter_obj is None or cutter_obj.type != "MESH" or not bool(cutter_obj.get(apple_bite.CUTTER_PROP, False)):
        return False, f"ไม่พบตัวตัด '{cutter_name}'"

    target_win = context.window
    target_area = context.area if (context.area and context.area.type == "VIEW_3D") else None
    target_region = None
    if target_area is None and context.window_manager is not None:
        for win in context.window_manager.windows:
            if win.screen is None:
                continue
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    target_win = win
                    target_area = area
                    break
            if target_area is not None:
                break

    if target_win is None or target_area is None:
        return False, "ไม่พบหน้าต่าง 3D Viewport"

    for reg in target_area.regions:
        if reg.type == "WINDOW":
            target_region = reg
            break
    rv3d = target_area.spaces.active.region_3d if target_area.spaces.active else None
    if target_region is None or rv3d is None:
        return False, "ไม่สามารถดึงมุมกล้องจาก 3D Viewport ได้"

    close_draw_overlay()
    _ensure_qapp()
    _blender_hwnd = _find_blender_hwnd(target_win)
    _reset_vk_state()

    lx, ly, lw, lh = _get_viewport_qt_rect(target_win, target_region, blender_hwnd=_blender_hwnd)
    bbox_objs = list(session.get_target_objects())
    if cutter_obj not in bbox_objs:
        bbox_objs.append(cutter_obj)
    cam = capture_camera_snapshot(rv3d, bbox_objs, lw, lh)
    target_corners = collect_target_bbox_corners(bbox_objs)

    cutter_zone = str(cutter_obj.get(apple_bite.CUTTER_ZONE_PROP, "A") or "A").strip() or "A"
    zone_rgba = apple_bite.get_zone_color(cutter_zone)
    cutter_faces = collect_cutter_faces_world(cutter_obj)
    dims = cutter_obj.dimensions
    min_dim = min(float(dims.x), float(dims.y), float(dims.z)) if (dims.x > 0 and dims.y > 0 and dims.z > 0) else 2.0
    default_mask_depth = round(min(max(min_dim * 0.45, 0.4), 1.8), 2)

    def _noop_confirm(_v, _f, _z) -> None:
        pass

    def _on_sculpt_confirm(
        verts: np.ndarray,
        faces: list[tuple[int, ...]],
        op_type: str,
        mask_face_indices: list[int],
        mask_depth: float,
    ) -> None:
        global _active_overlay, _active_rv3d_ref
        _active_overlay = None
        _active_rv3d_ref = None
        from ..ops import apple_bite_ops

        apple_bite_ops.stage_sculpt_payload(
            cutter_name,
            verts,
            faces,
            op_type,
            mask_face_indices=mask_face_indices,
            mask_depth=mask_depth,
        )
        try:
            with blender_view3d_override():
                bpy.ops.fdt.applebite_apply_sculpt_cutter()
        except Exception as exc:
            print(f"[FullDestruc_Tool] Draw Cut sculpt failed: {exc}")
            st = getattr(bpy.context.scene, "fdt_settings", None)
            if st is not None:
                st.last_status = f"แต่งตัวตัดไม่สำเร็จ: {exc}"
        _tag_all_view3d_redraw()

    def _on_cancel() -> None:
        global _active_overlay, _active_rv3d_ref
        _active_overlay = None
        _active_rv3d_ref = None
        _tag_all_view3d_redraw()

    overlay = DrawCutterOverlay(
        cam=cam,
        target_corners=target_corners,
        zone=cutter_zone,
        zone_rgba=zone_rgba,
        on_confirm=_noop_confirm,
        on_cancel=_on_cancel,
        edit_cutter_name=cutter_name,
        on_sculpt_confirm=_on_sculpt_confirm,
        cutter_faces=cutter_faces,
        default_mask_depth=default_mask_depth,
    )
    overlay.setGeometry(lx, ly, lw, lh)
    overlay.show()
    overlay.raise_()
    overlay.activateWindow()
    _active_overlay = overlay
    _active_rv3d_ref = rv3d
    _active_win_ref = target_win
    _active_region_ref = target_region

    if not bpy.app.timers.is_registered(_overlay_event_pump):
        _overlay_timer_running = True
        bpy.app.timers.register(_overlay_event_pump, first_interval=0.016, persistent=False)

    return True, f"เปิดโหมดวาดแต่งตัวตัด '{cutter_name}' (โซน {cutter_zone}) แล้ว — กด Tab ขยับกล้อง หรือวาดปิดรูปแล้วเลือก ปาดออก / เก็บข้างใน / ต่อเติม ได้เลย"
