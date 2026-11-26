"""GPU line overlay used while dragging rebar sliders.

This is intentionally a **module-local** state object, not a singleton hung off
``bpy.types.WindowManager`` (the Red Panda bug: state survives module reload in
ways nobody can reset).  ``register()`` installs one View3D draw handler;
``unregister()`` always removes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from mathutils import Vector


@dataclass
class _OverlayState:
    lines: list[list[Vector]] = field(default_factory=list)
    alpha: float = 0.0
    handler: object | None = None


_STATE = _OverlayState()


def set_lines(lines, alpha: float = 1.0) -> None:
    """Store world-space polylines and make the overlay visible."""
    _STATE.lines = [
        [Vector(p) for p in polyline]
        for polyline in lines
        if polyline is not None and len(polyline) >= 2
    ]
    _STATE.alpha = float(alpha) if _STATE.lines else 0.0
    tag_redraw()


def clear() -> None:
    _STATE.lines = []
    _STATE.alpha = 0.0
    tag_redraw()


def is_active() -> bool:
    return bool(_STATE.lines) and _STATE.alpha > 0.0


def get_line_count() -> int:
    return len(_STATE.lines)


def tag_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _draw() -> None:
    if not is_active():
        return
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "fdt_rebar_settings", None)
    # The overlay is only for fast dragging; once the user hides the generator
    # preview there is nothing to draw.
    if settings is not None and not bool(settings.ui_show_rebar):
        return

    coords: list[tuple[float, float, float]] = []
    for polyline in _STATE.lines:
        for a, b in zip(polyline[:-1], polyline[1:]):
            coords.append((a.x, a.y, a.z))
            coords.append((b.x, b.y, b.z))
    if not coords:
        return

    try:
        gpu.state.blend_set("ALPHA")
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        shader.uniform_float("color", (0.0, 0.8, 1.0, _STATE.alpha * 0.75))
        batch = batch_for_shader(shader, "LINES", {"pos": coords})
        batch.draw(shader)
    except Exception:
        pass
    finally:
        try:
            gpu.state.blend_set("NONE")
        except Exception:
            pass


def register() -> None:
    if _STATE.handler is not None:
        return
    try:
        _STATE.handler = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
    except Exception as exc:
        print(f"[FullDestruc_Tool] Rebar overlay registration failed: {exc}")
        _STATE.handler = None


def unregister() -> None:
    if _STATE.handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_STATE.handler, "WINDOW")
        except Exception:
            pass
        _STATE.handler = None
    clear()
