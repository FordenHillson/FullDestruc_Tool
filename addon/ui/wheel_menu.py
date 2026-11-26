"""Custom GPU Radial Donut Wheel Menu with vector icons and a searchable Zone popup."""

from __future__ import annotations

import math
from dataclasses import dataclass

import blf
import bpy
import gpu
from bpy.props import FloatProperty
from gpu_extras.batch import batch_for_shader

from ..core import apple_bite, zone_registry
from ..core.apple_bite import CUTTER_PROP
from ..core.session import load_session

INNER_RADIUS = 44.0
OUTER_RADIUS = 134.0
HOVER_POP = 8.0
GAP_DEG = 3.0

SUB_INNER_RADIUS = 144.0
SUB_MID_INNER = 144.0
SUB_MID_OUTER = 192.0
SUB_FAR_INNER = 198.0
SUB_FAR_OUTER = 246.0
SUB_OUTER_RADIUS = SUB_FAR_OUTER

SUB_ZONES_ROW1 = tuple(chr(c) for c in range(ord("A"), ord("N")))  # A - M (13)
SUB_ZONES_ROW2 = tuple(chr(c) for c in range(ord("N"), ord("Z") + 1))  # N - Z (13)
SUB_ZONES = SUB_ZONES_ROW1 + SUB_ZONES_ROW2  # All 26 zones A - Z

# Two-tier outer sub-ring spanning 220 degrees centered on the "Zone" slice (0 deg = right)
SUB_CENTER_DEG = 0.0
SUB_TOTAL_SPAN_DEG = 220.0
SUB_GAP_DEG = 1.6

# Right-side outer curved wing for Layer Up (top half) / Layer Down (bottom half)
LAYER_WING_INNER = 144.0
LAYER_WING_OUTER = 198.0
LAYER_WING_SPAN_DEG = 56.0
LAYER_WING_GAP_DEG = 4.0
LAYER_WING_ITEMS: tuple[tuple[str, str, str, str, tuple[float, float, float, float], float, float], ...] = (
    (
        "ABOVE",
        "Up",
        "Layer Above...",
        "arrow_up",
        (0.22, 0.86, 0.62, 0.96),
        LAYER_WING_GAP_DEG * 0.5,
        LAYER_WING_SPAN_DEG,
    ),
    (
        "BELOW",
        "Down",
        "Layer Below...",
        "arrow_down",
        (0.98, 0.62, 0.16, 0.96),
        -LAYER_WING_SPAN_DEG,
        -LAYER_WING_GAP_DEG * 0.5,
    ),
)

# Dedicated View-mode wheel: 3 top-level wedges; hovering the Hole wedge splits it into 2.
VIEW_WHEEL_INNER = 48.0
VIEW_WHEEL_OUTER = 150.0
VIEW_WHEEL_GAP_DEG = 3.0
VIEW_WHEEL_ITEMS: tuple[tuple[str, str, str, str, tuple[float, float, float, float], float, float], ...] = (
    ("ALL", "All", "Building + Pieces", "eye", (0.30, 0.78, 0.96, 0.96), 30.0, 150.0),
    ("PIECES", "Piece Only", "Cut Pieces Only", "cube", (0.98, 0.68, 0.20, 0.96), 150.0, 270.0),
    ("MAIN", "Hole Only", "Hole / Wire", "hole", (0.28, 0.84, 0.52, 0.96), 270.0, 390.0),
)
HOLE_SPLIT_ITEMS: tuple[tuple[str, str, str, tuple[float, float, float, float], float, float], ...] = (
    ("MAIN", "Solid", "Hole: Solid", (0.28, 0.84, 0.52, 0.96), 270.0, 330.0),
    ("MAIN_WIRE", "Wire", "Hole + Wire", (0.96, 0.42, 0.52, 0.96), 330.0, 390.0),
)
VIEW_MODE_SHORT = {
    "ALL": "All",
    "PIECES": "Piece",
    "MAIN": "Hole",
    "MAIN_WIRE": "Hole+W",
}
VIEW_MODE_ACCENT = {
    "ALL": VIEW_WHEEL_ITEMS[0][4],
    "PIECES": VIEW_WHEEL_ITEMS[1][4],
    "MAIN": VIEW_WHEEL_ITEMS[2][4],
    "MAIN_WIRE": HOLE_SPLIT_ITEMS[1][3],
}


@dataclass(frozen=True)
class WheelItem:
    id: str
    label: str
    hint: str
    icon: str
    accent: tuple[float, float, float, float]


WHEEL_ITEMS: tuple[WheelItem, ...] = (
    WheelItem("pick_axis", "Next Box", "Pick 3D Axis", "axis", (0.96, 0.52, 0.14, 0.95)),
    WheelItem("zone_menu", "Zone", "Search / manage zones", "zone", (0.20, 0.65, 1.00, 0.95)),
    WheelItem("toggle_rough", "Rough", "Rough / Smooth", "rough", (0.76, 0.36, 1.00, 0.95)),
    WheelItem("apply", "Apply", "Bake Cuts", "check", (0.18, 0.78, 0.44, 0.95)),
    WheelItem("delete", "Delete", "Remove Box", "cross", (0.88, 0.25, 0.25, 0.95)),
    WheelItem("toggle_mute", "Mute", "Pause / Resume", "pause", (0.94, 0.76, 0.16, 0.95)),
    WheelItem("add_box", "New Box", "Add Cutter", "plus_box", (0.14, 0.76, 0.72, 0.95)),
    WheelItem("view_mode", "View", "All / Piece / Hole", "eye", (0.28, 0.84, 0.96, 0.95)),
    WheelItem("draw_edit", "Draw Cut", "Carve / Keep / Add", "pen", (0.98, 0.38, 0.62, 0.95)),
)

ZONE_SLICE_INDEX = 1
VIEW_SLICE_INDEX = 7
_active_wheel: "FDT_OT_cutter_wheel_menu | None" = None
_active_view_wheel: "FDT_OT_view_mode_wheel_menu | None" = None


def _slice_angles(index: int, count: int) -> tuple[float, float]:
    """Return (start_rad, end_rad) for main slice `index`, starting clockwise from +90 deg."""
    span = 360.0 / count
    a_end_deg = 90.0 - index * span - (GAP_DEG * 0.5)
    a_start_deg = 90.0 - (index + 1) * span + (GAP_DEG * 0.5)
    return math.radians(a_start_deg), math.radians(a_end_deg)


def _sub_zone_angles(index: int, count: int = len(SUB_ZONES_ROW1)) -> tuple[float, float]:
    """Return (start_rad, end_rad) for an outer Zone sub-ring wedge (13 wedges per tier)."""
    span = SUB_TOTAL_SPAN_DEG / count
    top_deg = SUB_CENTER_DEG + SUB_TOTAL_SPAN_DEG * 0.5
    a_end_deg = top_deg - index * span - (SUB_GAP_DEG * 0.5)
    a_start_deg = top_deg - (index + 1) * span + (SUB_GAP_DEG * 0.5)
    return math.radians(a_start_deg), math.radians(a_end_deg)


def _iter_sub_zone_cells():
    """Yield (zone_letter, r_in, r_out, a0, a1) for all 26 zones (A-M inner tier, N-Z outer tier)."""
    for idx, letter in enumerate(SUB_ZONES_ROW1):
        a0, a1 = _sub_zone_angles(idx, len(SUB_ZONES_ROW1))
        yield letter, SUB_MID_INNER, SUB_MID_OUTER, a0, a1
    for idx, letter in enumerate(SUB_ZONES_ROW2):
        a0, a1 = _sub_zone_angles(idx, len(SUB_ZONES_ROW2))
        yield letter, SUB_FAR_INNER, SUB_FAR_OUTER, a0, a1


def build_wedge_geometry(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    a0: float,
    a1: float,
    steps: int = 20,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Build triangle vertices and outline loop for a curved radial donut slice."""
    tris: list[tuple[float, float]] = []
    outer_pts: list[tuple[float, float]] = []
    inner_pts: list[tuple[float, float]] = []

    for i in range(steps + 1):
        t = i / steps
        ang = a0 + (a1 - a0) * t
        ca, sa = math.cos(ang), math.sin(ang)
        inner_pts.append((cx + r_inner * ca, cy + r_inner * sa))
        outer_pts.append((cx + r_outer * ca, cy + r_outer * sa))

    for i in range(steps):
        i0, i1 = inner_pts[i], inner_pts[i + 1]
        o0, o1 = outer_pts[i], outer_pts[i + 1]
        tris.extend([i0, o0, o1, i0, o1, i1])

    outline = outer_pts + list(reversed(inner_pts))
    return tris, outline


def build_disk_geometry(
    cx: float,
    cy: float,
    radius: float,
    steps: int = 36,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Build triangle fan (as TRIS) and rim loop for the center hub disk."""
    tris: list[tuple[float, float]] = []
    ring: list[tuple[float, float]] = []
    for i in range(steps):
        ang = (2.0 * math.pi * i) / steps
        ring.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    for i in range(steps):
        p0 = ring[i]
        p1 = ring[(i + 1) % steps]
        tris.extend([(cx, cy), p0, p1])
    return tris, ring


def hit_test_wheel(
    mouse_x: float,
    mouse_y: float,
    cx: float,
    cy: float,
    count: int = len(WHEEL_ITEMS),
    r_inner: float = INNER_RADIUS,
    r_outer: float = OUTER_RADIUS,
) -> int | None:
    """Return hovered main slice index (0..count-1) or None if outside main ring."""
    dx = mouse_x - cx
    dy = mouse_y - cy
    dist = math.hypot(dx, dy)
    if dist < r_inner or dist > (r_outer + 10.0):
        return None

    ang_deg = math.degrees(math.atan2(dy, dx))
    cw_from_top = (90.0 - ang_deg) % 360.0
    span = 360.0 / count
    return int(cw_from_top // span) % count


def hit_test_zone_subring(
    mouse_x: float,
    mouse_y: float,
    cx: float,
    cy: float,
) -> str | None:
    """Return Zone letter ('A'..'Z') if mouse is inside either tier of the Zone sub-ring, else None."""
    dx = mouse_x - cx
    dy = mouse_y - cy
    dist = math.hypot(dx, dy)
    if dist < (SUB_MID_INNER - 6.0) or dist > (SUB_FAR_OUTER + 18.0):
        return None

    ang_deg = math.degrees(math.atan2(dy, dx))
    top_deg = SUB_CENTER_DEG + SUB_TOTAL_SPAN_DEG * 0.5
    bot_deg = SUB_CENTER_DEG - SUB_TOTAL_SPAN_DEG * 0.5
    if ang_deg > top_deg or ang_deg < bot_deg:
        return None

    row = SUB_ZONES_ROW1 if dist <= (SUB_MID_OUTER + 3.0) else SUB_ZONES_ROW2
    span = SUB_TOTAL_SPAN_DEG / len(row)
    idx = int((top_deg - ang_deg) // span)
    if 0 <= idx < len(row):
        return row[idx]
    return None


def hit_test_layer_wing(
    mouse_x: float,
    mouse_y: float,
    cx: float,
    cy: float,
) -> str | None:
    """Return 'ABOVE' (right top half), 'BELOW' (right bottom half), or None."""
    dx = mouse_x - cx
    dy = mouse_y - cy
    dist = math.hypot(dx, dy)
    if dist < (LAYER_WING_INNER - 4.0) or dist > (LAYER_WING_OUTER + 12.0):
        return None
    ang_deg = math.degrees(math.atan2(dy, dx))
    for wid, _lbl, _hint, _icon, _acc, deg0, deg1 in LAYER_WING_ITEMS:
        if deg0 <= ang_deg <= deg1:
            return wid
    return None


def _in_view_wedge(ang_deg: float, deg0: float, deg1: float, gap: float = VIEW_WHEEL_GAP_DEG) -> bool:
    """Angle test supporting wedges that wrap past 360 degrees (e.g. 270..390)."""
    lo = deg0 + gap * 0.5
    hi = deg1 - gap * 0.5
    if hi <= 360.0:
        return lo <= ang_deg <= hi
    return ang_deg >= lo or ang_deg <= (hi - 360.0)


def _view_wheel_angle(
    mouse_x: float,
    mouse_y: float,
    cx: float,
    cy: float,
) -> float | None:
    dx = mouse_x - cx
    dy = mouse_y - cy
    dist = math.hypot(dx, dy)
    if dist < VIEW_WHEEL_INNER or dist > (VIEW_WHEEL_OUTER + 10.0):
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def hit_test_view_wheel(mouse_x: float, mouse_y: float, cx: float, cy: float) -> str | None:
    """Return the top-level View mode ('ALL' / 'PIECES' / 'MAIN') under the cursor, else None."""
    ang = _view_wheel_angle(mouse_x, mouse_y, cx, cy)
    if ang is None:
        return None
    for mid, _lbl, _hint, _icon, _acc, deg0, deg1 in VIEW_WHEEL_ITEMS:
        if _in_view_wedge(ang, deg0, deg1):
            return mid
    return None


def hit_test_hole_split(mouse_x: float, mouse_y: float, cx: float, cy: float) -> str | None:
    """Return 'MAIN' (Solid) or 'MAIN_WIRE' (Wire) only inside the Hole wedge."""
    ang = _view_wheel_angle(mouse_x, mouse_y, cx, cy)
    if ang is None:
        return None
    for mid, _lbl, _hint, _acc, deg0, deg1 in HOLE_SPLIT_ITEMS:
        if _in_view_wedge(ang, deg0, deg1):
            return mid
    return None


def _draw_vector_icon(
    shader,
    icon_name: str,
    x: float,
    y: float,
    size: float,
    color: tuple[float, float, float, float],
) -> None:
    """Draw a crisp GPU line icon centered at (x, y)."""
    s = size
    lines: list[tuple[float, float]] = []

    if icon_name == "axis":
        # Small box + arrow pointing right
        lines.extend([
            (x - s, y - s * 0.6), (x - s * 0.1, y - s * 0.6),
            (x - s * 0.1, y - s * 0.6), (x - s * 0.1, y + s * 0.6),
            (x - s * 0.1, y + s * 0.6), (x - s, y + s * 0.6),
            (x - s, y + s * 0.6), (x - s, y - s * 0.6),
            # Arrow
            (x + s * 0.1, y), (x + s, y),
            (x + s, y), (x + s * 0.55, y + s * 0.45),
            (x + s, y), (x + s * 0.55, y - s * 0.45),
        ])
    elif icon_name == "zone":
        # Layered diamond / tag icon
        lines.extend([
            (x, y + s * 0.85), (x + s * 0.85, y + s * 0.2),
            (x + s * 0.85, y + s * 0.2), (x, y - s * 0.45),
            (x, y - s * 0.45), (x - s * 0.85, y + s * 0.2),
            (x - s * 0.85, y + s * 0.2), (x, y + s * 0.85),
            # Lower layer chevron
            (x - s * 0.75, y - s * 0.35), (x, y - s * 0.9),
            (x, y - s * 0.9), (x + s * 0.75, y - s * 0.35),
        ])
    elif icon_name == "rough":
        # Jagged rough-edge zigzag wave icon
        lines.extend([
            (x - s * 0.90, y - s * 0.15), (x - s * 0.45, y + s * 0.55),
            (x - s * 0.45, y + s * 0.55), (x, y - s * 0.45),
            (x, y - s * 0.45), (x + s * 0.45, y + s * 0.55),
            (x + s * 0.45, y + s * 0.55), (x + s * 0.90, y - s * 0.15),
        ])
    elif icon_name == "check":
        lines.extend([
            (x - s * 0.75, y + s * 0.05), (x - s * 0.15, y - s * 0.65),
            (x - s * 0.15, y - s * 0.65), (x + s * 0.85, y + s * 0.65),
        ])
    elif icon_name == "cross":
        lines.extend([
            (x - s * 0.65, y - s * 0.65), (x + s * 0.65, y + s * 0.65),
            (x - s * 0.65, y + s * 0.65), (x + s * 0.65, y - s * 0.65),
        ])
    elif icon_name == "pause":
        lines.extend([
            (x - s * 0.4, y - s * 0.7), (x - s * 0.4, y + s * 0.7),
            (x + s * 0.4, y - s * 0.7), (x + s * 0.4, y + s * 0.7),
        ])
    elif icon_name == "plus_box":
        lines.extend([
            (x - s * 0.75, y - s * 0.75), (x + s * 0.75, y - s * 0.75),
            (x + s * 0.75, y - s * 0.75), (x + s * 0.75, y + s * 0.75),
            (x + s * 0.75, y + s * 0.75), (x - s * 0.75, y + s * 0.75),
            (x - s * 0.75, y + s * 0.75), (x - s * 0.75, y - s * 0.75),
            # Plus sign
            (x - s * 0.38, y), (x + s * 0.38, y),
            (x, y - s * 0.38), (x, y + s * 0.38),
        ])
    elif icon_name == "arrow_up":
        # Top bar + bold upward arrow (Layer Above)
        lines.extend([
            (x - s * 0.75, y + s * 0.78), (x + s * 0.75, y + s * 0.78),
            (x, y - s * 0.78), (x, y + s * 0.45),
            (x, y + s * 0.45), (x - s * 0.55, y - s * 0.10),
            (x, y + s * 0.45), (x + s * 0.55, y - s * 0.10),
        ])
    elif icon_name == "arrow_down":
        # Bottom bar + bold downward arrow (Layer Below)
        lines.extend([
            (x - s * 0.75, y - s * 0.78), (x + s * 0.75, y - s * 0.78),
            (x, y + s * 0.78), (x, y - s * 0.45),
            (x, y - s * 0.45), (x - s * 0.55, y + s * 0.10),
            (x, y - s * 0.45), (x + s * 0.55, y + s * 0.10),
        ])
    elif icon_name == "eye":
        # Crisp eye icon (almond outline + pupil diamond)
        lines.extend([
            (x - s * 0.90, y), (x - s * 0.45, y + s * 0.50),
            (x - s * 0.45, y + s * 0.50), (x + s * 0.45, y + s * 0.50),
            (x + s * 0.45, y + s * 0.50), (x + s * 0.90, y),
            (x + s * 0.90, y), (x + s * 0.45, y - s * 0.50),
            (x + s * 0.45, y - s * 0.50), (x - s * 0.45, y - s * 0.50),
            (x - s * 0.45, y - s * 0.50), (x - s * 0.90, y),
            # Pupil
            (x, y + s * 0.28), (x + s * 0.28, y),
            (x + s * 0.28, y), (x, y - s * 0.28),
            (x, y - s * 0.28), (x - s * 0.28, y),
            (x - s * 0.28, y), (x, y + s * 0.28),
        ])
    elif icon_name == "cube":
        # Isometric cube outline (cut piece)
        lines.extend([
            (x - s * 0.70, y - s * 0.35), (x, y - s * 0.75),
            (x, y - s * 0.75), (x + s * 0.70, y - s * 0.35),
            (x + s * 0.70, y - s * 0.35), (x + s * 0.70, y + s * 0.40),
            (x + s * 0.70, y + s * 0.40), (x, y + s * 0.82),
            (x, y + s * 0.82), (x - s * 0.70, y + s * 0.40),
            (x - s * 0.70, y + s * 0.40), (x - s * 0.70, y - s * 0.35),
            (x - s * 0.70, y - s * 0.35), (x, y + s * 0.05),
            (x, y + s * 0.05), (x + s * 0.70, y - s * 0.35),
            (x, y + s * 0.05), (x, y + s * 0.82),
        ])
    elif icon_name == "wire":
        # Wireframe cube (edges only, open feel)
        lines.extend([
            (x - s * 0.72, y - s * 0.40), (x, y - s * 0.80),
            (x, y - s * 0.80), (x + s * 0.72, y - s * 0.40),
            (x + s * 0.72, y - s * 0.40), (x + s * 0.72, y + s * 0.42),
            (x + s * 0.72, y + s * 0.42), (x, y + s * 0.84),
            (x, y + s * 0.84), (x - s * 0.72, y + s * 0.42),
            (x - s * 0.72, y + s * 0.42), (x - s * 0.72, y - s * 0.40),
            (x - s * 0.72, y + s * 0.42), (x + s * 0.72, y - s * 0.40),
            (x - s * 0.72, y - s * 0.40), (x + s * 0.72, y + s * 0.42),
        ])
    elif icon_name == "hole":
        # Solid block with a bite taken out of one corner
        lines.extend([
            (x - s * 0.78, y - s * 0.55), (x - s * 0.78, y + s * 0.70),
            (x - s * 0.78, y + s * 0.70), (x + s * 0.78, y + s * 0.70),
            (x + s * 0.78, y + s * 0.70), (x + s * 0.78, y + s * 0.05),
            (x + s * 0.78, y + s * 0.05), (x + s * 0.10, y + s * 0.05),
            (x + s * 0.10, y + s * 0.05), (x + s * 0.10, y - s * 0.55),
            (x + s * 0.10, y - s * 0.55), (x - s * 0.78, y - s * 0.55),
            (x + s * 0.10, y + s * 0.05), (x + s * 0.78, y - s * 0.55),
        ])
    elif icon_name == "pen":
        # Crisp diagonal pencil / carve pen icon
        lines.extend([
            (x - s * 0.80, y - s * 0.80), (x - s * 0.35, y - s * 0.70),
            (x - s * 0.80, y - s * 0.80), (x - s * 0.70, y - s * 0.35),
            (x - s * 0.35, y - s * 0.70), (x + s * 0.75, y + s * 0.40),
            (x - s * 0.70, y - s * 0.35), (x + s * 0.40, y + s * 0.75),
            (x + s * 0.75, y + s * 0.40), (x + s * 0.40, y + s * 0.75),
            # Tip divider
            (x - s * 0.35, y - s * 0.70), (x - s * 0.70, y - s * 0.35),
        ])

    if not lines:
        return
    gpu.state.line_width_set(2.2)
    batch = batch_for_shader(shader, "LINES", {"pos": lines})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_centered_text(
    font_id: int,
    text: str,
    x: float,
    y: float,
    size: int,
    color: tuple[float, float, float, float],
) -> None:
    blf.size(font_id, size)
    blf.color(font_id, *color)
    w, h = blf.dimensions(font_id, text)
    blf.position(font_id, x - w * 0.5, y - h * 0.4, 0)
    blf.draw(font_id, text)


class FDT_OT_cutter_wheel_menu(bpy.types.Operator):
    """Custom GPU Radial Donut Wheel Menu with icons; Zone opens the shared registry picker."""

    bl_idname = "fdt.cutter_wheel_menu"
    bl_label = "FullDestruc Radial Wheel"
    bl_description = "Open custom radial wheel menu centered at mouse cursor"

    _handle = None
    _cx: float = 0.0
    _cy: float = 0.0
    _hovered: int | None = None
    _zone_sub_open: bool = False
    _hovered_zone: str | None = None
    _hovered_layer: str | None = None
    _cutter_name: str = ""

    def _get_layer_info(self, scene: bpy.types.Scene | None) -> tuple[int, int]:
        """Return `(1-based layer number of active cutter, total cutter count)`."""
        if scene is None:
            return 1, 1
        session = load_session(scene)
        if session is None or not session.cutters:
            return 1, 1
        idx = next((i for i, c in enumerate(session.cutters) if c.name == self._cutter_name), 0)
        return idx + 1, len(session.cutters)

    def _cleanup(self, context: bpy.types.Context | None = None) -> None:
        global _active_wheel
        if self._handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            except Exception:
                pass
            self._handle = None
        if _active_wheel is self:
            _active_wheel = None
        if context is not None and context.area is not None:
            context.area.tag_redraw()

    def _draw_wheel(self) -> None:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")

        count = len(WHEEL_ITEMS)
        cx, cy = self._cx, self._cy
        obj = bpy.data.objects.get(self._cutter_name)
        current_zone = str(obj.get("fdt_cutter_zone", "A") or "A") if obj is not None else "A"
        scene = getattr(bpy.context, "scene", None)
        current_token = zone_registry.zone_token(current_zone, scene)
        settings = getattr(scene, "fdt_settings", None) if scene is not None else None
        cur_view = str(getattr(settings, "preview_display_mode", "ALL") or "ALL") if settings is not None else "ALL"
        cur_layer_num, total_cutters = self._get_layer_info(scene)
        can_reorder = total_cutters >= 2

        # 1. Draw main curved donut slices + vector icons
        for idx, item in enumerate(WHEEL_ITEMS):
            is_hover = (idx == self._hovered) or (idx == ZONE_SLICE_INDEX and self._zone_sub_open)
            is_active_view = item.id == "view_mode" and cur_view != "ALL"
            a0, a1 = _slice_angles(idx, count)
            r_in = INNER_RADIUS + (2.0 if is_hover else 0.0)
            r_out = OUTER_RADIUS + (HOVER_POP if is_hover else 0.0)

            tris, outline = build_wedge_geometry(cx, cy, r_in, r_out, a0, a1, steps=22)

            if is_hover:
                fill_col = item.accent
            elif is_active_view:
                fill_col = (item.accent[0] * 0.42, item.accent[1] * 0.42, item.accent[2] * 0.42, 0.92)
            else:
                fill_col = (0.12, 0.14, 0.18, 0.88)
            batch_fill = batch_for_shader(shader, "TRIS", {"pos": tris})
            shader.bind()
            shader.uniform_float("color", fill_col)
            batch_fill.draw(shader)

            gpu.state.line_width_set(2.2 if is_hover else 1.2)
            rim_col = (1.0, 1.0, 1.0, 0.95) if is_hover else (item.accent[0], item.accent[1], item.accent[2], 0.55)
            batch_line = batch_for_shader(shader, "LINE_LOOP", {"pos": outline})
            shader.bind()
            shader.uniform_float("color", rim_col)
            batch_line.draw(shader)

            # Vector icon above label
            a_mid = (a0 + a1) * 0.5
            r_mid = (INNER_RADIUS + OUTER_RADIUS) * 0.5 + (HOVER_POP * 0.5 if is_hover else 0.0)
            ix = cx + r_mid * math.cos(a_mid)
            iy = cy + r_mid * math.sin(a_mid) + 10.0
            icon_col = (0.08, 0.08, 0.08, 1.0) if is_hover else (item.accent[0], item.accent[1], item.accent[2], 1.0)
            _draw_vector_icon(shader, item.icon, ix, iy, 9.0, icon_col)

        # 2. Zone is now a searchable popup; the legacy sub-ring is never drawn.
        #    (Geometry/hit-test helpers are kept only for test compatibility.)
        if False:
            for z_letter, r_in, r_out_base, a0, a1 in _iter_sub_zone_cells():
                is_z_hover = z_letter == self._hovered_zone
                is_z_cur = z_letter == current_zone
                zr, zg, zb, _ = apple_bite.get_zone_color(z_letter)

                r_out = r_out_base + (5.0 if is_z_hover else 0.0)
                tris, outline = build_wedge_geometry(cx, cy, r_in, r_out, a0, a1, steps=8)

                if is_z_hover:
                    fill_col = (zr, zg, zb, 0.98)
                elif is_z_cur:
                    fill_col = (zr * 0.55, zg * 0.55, zb * 0.55, 0.90)
                else:
                    fill_col = (0.14, 0.16, 0.21, 0.88)

                batch_z = batch_for_shader(shader, "TRIS", {"pos": tris})
                shader.bind()
                shader.uniform_float("color", fill_col)
                batch_z.draw(shader)

                gpu.state.line_width_set(2.0 if (is_z_hover or is_z_cur) else 1.1)
                rim_col = (1.0, 1.0, 1.0, 0.95) if is_z_hover else (zr, zg, zb, 0.78)
                batch_zl = batch_for_shader(shader, "LINE_LOOP", {"pos": outline})
                shader.bind()
                shader.uniform_float("color", rim_col)
                batch_zl.draw(shader)
        else:
            # Right-side outer curved wing: Top half = Layer Up (ABOVE), Bottom half = Layer Down (BELOW)
            for wid, _lbl, _hint, icon_name, acc, deg0, deg1 in LAYER_WING_ITEMS:
                is_w_hover = (self._hovered_layer == wid) and can_reorder
                a0 = math.radians(deg0)
                a1 = math.radians(deg1)
                r_in = LAYER_WING_INNER + (2.0 if is_w_hover else 0.0)
                r_out = LAYER_WING_OUTER + (6.0 if is_w_hover else 0.0)
                tris, outline = build_wedge_geometry(cx, cy, r_in, r_out, a0, a1, steps=18)

                if not can_reorder:
                    fill_col = (0.12, 0.13, 0.16, 0.52)
                    rim_col = (0.45, 0.48, 0.55, 0.35)
                    icon_col = (0.55, 0.58, 0.65, 0.50)
                elif is_w_hover:
                    fill_col = acc
                    rim_col = (1.0, 1.0, 1.0, 0.98)
                    icon_col = (0.06, 0.06, 0.08, 1.0)
                else:
                    fill_col = (0.12, 0.14, 0.19, 0.90)
                    rim_col = (acc[0], acc[1], acc[2], 0.78)
                    icon_col = (acc[0], acc[1], acc[2], 1.0)

                b_wf = batch_for_shader(shader, "TRIS", {"pos": tris})
                shader.bind()
                shader.uniform_float("color", fill_col)
                b_wf.draw(shader)

                gpu.state.line_width_set(2.2 if is_w_hover else 1.4)
                b_wl = batch_for_shader(shader, "LINE_LOOP", {"pos": outline})
                shader.bind()
                shader.uniform_float("color", rim_col)
                b_wl.draw(shader)

                a_mid = (a0 + a1) * 0.5
                r_mid = (LAYER_WING_INNER + LAYER_WING_OUTER) * 0.5 + (3.0 if is_w_hover else 0.0)
                ix = cx + r_mid * math.cos(a_mid)
                iy = cy + r_mid * math.sin(a_mid) + 9.0
                _draw_vector_icon(shader, icon_name, ix, iy, 8.0, icon_col)

        # 3. Draw center hub disk
        hub_tris, hub_ring = build_disk_geometry(cx, cy, INNER_RADIUS - 4.0, steps=36)
        batch_hub = batch_for_shader(shader, "TRIS", {"pos": hub_tris})
        shader.bind()
        shader.uniform_float("color", (0.09, 0.10, 0.13, 0.94))
        batch_hub.draw(shader)

        gpu.state.line_width_set(1.8)
        if self._hovered_zone is not None:
            hub_rim_col = apple_bite.get_zone_color(self._hovered_zone)
        elif self._hovered_layer is not None:
            w_item = next((w for w in LAYER_WING_ITEMS if w[0] == self._hovered_layer), None)
            hub_rim_col = w_item[4] if (w_item is not None and can_reorder) else (0.50, 0.52, 0.58, 0.65)
        elif self._hovered is not None:
            hub_rim_col = WHEEL_ITEMS[self._hovered].accent
        else:
            hub_rim_col = (0.45, 0.50, 0.60, 0.70)

        batch_hub_ring = batch_for_shader(shader, "LINE_LOOP", {"pos": hub_ring})
        shader.bind()
        shader.uniform_float("color", hub_rim_col)
        batch_hub_ring.draw(shader)

        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")

        # 4. Labels on main slices (below icon)
        font_id = 0
        for idx, item in enumerate(WHEEL_ITEMS):
            is_hover = (idx == self._hovered) or (idx == ZONE_SLICE_INDEX and self._zone_sub_open)
            a0, a1 = _slice_angles(idx, count)
            a_mid = (a0 + a1) * 0.5
            r_mid = (INNER_RADIUS + OUTER_RADIUS) * 0.5 + (HOVER_POP * 0.5 if is_hover else 0.0)
            tx = cx + r_mid * math.cos(a_mid)
            ty = cy + r_mid * math.sin(a_mid) - 11.0
            txt_col = (0.08, 0.08, 0.08, 1.0) if is_hover else (0.94, 0.96, 0.99, 0.96)
            lbl = f"View: {VIEW_MODE_SHORT.get(cur_view, 'All')}" if item.id == "view_mode" else item.label
            _draw_centered_text(font_id, lbl, tx, ty, 13 if is_hover else 12, txt_col)

        # 5. Labels on the Layer Wing (zone labels now live in the popup)
        if False:
            for z_letter, r_in, r_out_base, a0, a1 in _iter_sub_zone_cells():
                a_mid = (a0 + a1) * 0.5
                is_z_hover = z_letter == self._hovered_zone
                r_mid = (r_in + r_out_base) * 0.5 + (2.5 if is_z_hover else 0.0)
                tx = cx + r_mid * math.cos(a_mid)
                ty = cy + r_mid * math.sin(a_mid)
                txt_col = (0.08, 0.08, 0.08, 1.0) if is_z_hover else (0.96, 0.97, 1.0, 0.98)
                _draw_centered_text(font_id, z_letter, tx, ty, 14 if is_z_hover else 12, txt_col)
        else:
            for wid, lbl, _hint, _icon, _acc, deg0, deg1 in LAYER_WING_ITEMS:
                is_w_hover = (self._hovered_layer == wid) and can_reorder
                a_mid = math.radians((deg0 + deg1) * 0.5)
                r_mid = (LAYER_WING_INNER + LAYER_WING_OUTER) * 0.5 + (3.0 if is_w_hover else 0.0)
                tx = cx + r_mid * math.cos(a_mid)
                ty = cy + r_mid * math.sin(a_mid) - 10.0
                if not can_reorder:
                    txt_col = (0.55, 0.58, 0.65, 0.55)
                elif is_w_hover:
                    txt_col = (0.06, 0.06, 0.08, 1.0)
                else:
                    txt_col = (0.94, 0.96, 0.99, 0.96)
                _draw_centered_text(font_id, lbl, tx, ty, 12 if is_w_hover else 11, txt_col)

        # 6. Center hub text (shows active Zone + current Layer # when idle)
        if self._hovered_zone is not None:
            _draw_centered_text(font_id, f"Zone {self._hovered_zone}", cx, cy, 13, (1.0, 0.95, 0.80, 1.0))
        elif self._hovered_layer is not None:
            if not can_reorder:
                _draw_centered_text(font_id, "Need 2+", cx, cy, 11, (0.72, 0.75, 0.82, 0.90))
            else:
                w_item = next((w for w in LAYER_WING_ITEMS if w[0] == self._hovered_layer), None)
                hint_txt = w_item[2] if w_item is not None else "Layer"
                _draw_centered_text(font_id, hint_txt, cx, cy, 10, (1.0, 0.95, 0.80, 1.0))
        elif self._hovered is not None:
            _draw_centered_text(font_id, WHEEL_ITEMS[self._hovered].hint, cx, cy, 11, (1.0, 0.95, 0.80, 1.0))
        else:
            _draw_centered_text(font_id, f"Zone {current_token}", cx, cy + 7.0, 12, (0.92, 0.94, 0.98, 0.98))
            _draw_centered_text(font_id, f"Layer #{cur_layer_num}", cx, cy - 9.0, 10, (0.70, 0.78, 0.92, 0.92))

    def _assign_zone(self, context: bpy.types.Context, zone_letter: str) -> None:
        obj = bpy.data.objects.get(self._cutter_name)
        settings = getattr(context.scene, "fdt_settings", None)
        if settings is not None:
            settings.active_zone = zone_letter
        if obj is not None:
            apple_bite.set_cutters_zone(context, [obj], zone_letter)
            if settings is not None:
                settings.last_status = f"Set '{obj.name}' to Zone {zone_letter}"

    def _execute_item(self, context: bpy.types.Context, item_id: str) -> None:
        cname = self._cutter_name
        if item_id == "pick_axis":
            bpy.ops.fdt.applebite_pick_axis("INVOKE_DEFAULT")
        elif item_id == "zone_menu":
            bpy.ops.fdt.zone_picker("INVOKE_DEFAULT", cutter_name=cname)
        elif item_id == "toggle_rough":
            bpy.ops.fdt.applebite_toggle_rough(cutter_name=cname)
        elif item_id == "apply":
            bpy.ops.fdt.applebite_apply()
        elif item_id == "delete":
            bpy.ops.fdt.applebite_delete_cutter(cutter_name=cname)
        elif item_id == "toggle_mute":
            bpy.ops.fdt.applebite_toggle_mute(cutter_name=cname)
        elif item_id == "add_box":
            bpy.ops.fdt.applebite_add_box()
        elif item_id == "draw_edit":
            bpy.ops.fdt.applebite_draw_sculpt_cutter(cutter_name=cname)

    def modal(self, context, event):
        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
            new_layer = hit_test_layer_wing(mx, my, self._cx, self._cy)
            new_main = None if new_layer is not None else hit_test_wheel(mx, my, self._cx, self._cy)
            changed = new_main != self._hovered or new_layer != self._hovered_layer
            if changed:
                self._hovered = new_main
                self._hovered_zone = None
                self._hovered_layer = new_layer
                if context.area:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
            picked_layer = hit_test_layer_wing(mx, my, self._cx, self._cy)
            if picked_layer is not None:
                _cur_num, total_cutters = self._get_layer_info(context.scene)
                if total_cutters < 2:
                    self.report({"WARNING"}, "ต้องมีกล่องตัดอย่างน้อย 2 กล่องเพื่อสลับลำดับ Layer")
                    return {"RUNNING_MODAL"}
                cname = self._cutter_name
                self._cleanup(context)
                bpy.ops.fdt.applebite_pick_layer_target(
                    "INVOKE_DEFAULT",
                    source_cutter=cname,
                    position=picked_layer,
                )
                return {"FINISHED"}

            picked = hit_test_wheel(mx, my, self._cx, self._cy)
            if picked == ZONE_SLICE_INDEX:
                # Close the radial wheel, then open the shared searchable zone popup.
                cname = self._cutter_name
                self._cleanup(context)
                bpy.ops.fdt.zone_picker("INVOKE_DEFAULT", cutter_name=cname)
                return {"FINISHED"}

            if picked is not None and WHEEL_ITEMS[picked].id == "view_mode":
                # Close the cutter wheel and open the dedicated View-mode wheel at the cursor.
                self._cleanup(context)
                bpy.ops.fdt.view_mode_wheel_menu("INVOKE_DEFAULT", cx=mx, cy=my)
                return {"FINISHED"}

            self._cleanup(context)
            if picked is not None:
                self._execute_item(context, WHEEL_ITEMS[picked].id)
                return {"FINISHED"}
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        global _active_wheel
        if _active_wheel is not None:
            _active_wheel._cleanup(context)

        obj = context.active_object
        if obj is None or not bool(obj.get(CUTTER_PROP, False)):
            self.report({"WARNING"}, "Select a cutter object first.")
            return {"CANCELLED"}

        self._cutter_name = obj.name
        self._cx = float(event.mouse_region_x)
        self._cy = float(event.mouse_region_y)
        self._hovered = None
        self._zone_sub_open = False
        self._hovered_zone = None
        self._hovered_layer = None
        _active_wheel = self

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_wheel, (), "WINDOW", "POST_PIXEL"
        )
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


class FDT_OT_view_mode_wheel_menu(bpy.types.Operator):
    """Dedicated 3-mode View wheel: All / Piece Only / Hole Only (Solid or Wire)."""

    bl_idname = "fdt.view_mode_wheel_menu"
    bl_label = "FullDestruc View Mode Wheel"
    bl_description = "Switch between All / Piece Only / Hole Only (Solid or Wire) at the cursor"
    bl_options = {"REGISTER", "INTERNAL"}

    cx: FloatProperty(name="Center X", default=0.0, options={"SKIP_SAVE"})
    cy: FloatProperty(name="Center Y", default=0.0, options={"SKIP_SAVE"})

    _handle = None
    _cx: float = 0.0
    _cy: float = 0.0
    _hovered_main: str | None = None
    _hovered_hole: str | None = None

    @staticmethod
    def _current_mode(context: bpy.types.Context | None) -> str:
        settings = getattr(getattr(context, "scene", None), "fdt_settings", None)
        return str(getattr(settings, "preview_display_mode", "ALL") or "ALL") if settings is not None else "ALL"

    def _cleanup(self, context: bpy.types.Context | None = None) -> None:
        global _active_view_wheel
        if self._handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            except Exception:
                pass
            self._handle = None
        if _active_view_wheel is self:
            _active_view_wheel = None
        if context is not None and context.area is not None:
            context.area.tag_redraw()

    def _draw_wheel(self) -> None:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        cx, cy = self._cx, self._cy
        cur_mode = self._current_mode(bpy.context)
        is_split = self._hovered_main == "MAIN"

        wedges: list[tuple[str, str, str, tuple[float, float, float, float], float, float]] = []
        for mid, lbl, hint, icon, acc, d0, d1 in VIEW_WHEEL_ITEMS:
            if mid == "MAIN" and is_split:
                for sid, slbl, shint, sacc, sd0, sd1 in HOLE_SPLIT_ITEMS:
                    wedges.append((sid, slbl, shint, sacc, sd0, sd1))
            else:
                wedges.append((mid, lbl, hint, acc, d0, d1))

        mode_icon = {
            "ALL": "eye",
            "PIECES": "cube",
            "MAIN": "hole",
            "MAIN_WIRE": "wire",
        }

        for mid, lbl, hint, acc, d0, d1 in wedges:
            is_hover = mid == (self._hovered_hole if is_split else self._hovered_main)
            is_cur = mid == cur_mode
            a0, a1 = math.radians(d0), math.radians(d1)
            r_in = VIEW_WHEEL_INNER + (2.0 if is_hover else 0.0)
            r_out = VIEW_WHEEL_OUTER + (8.0 if is_hover else 0.0)
            tris, outline = build_wedge_geometry(cx, cy, r_in, r_out, a0, a1, steps=18)

            if is_hover:
                fill_col = acc
            elif is_cur:
                fill_col = (acc[0] * 0.45, acc[1] * 0.45, acc[2] * 0.45, 0.94)
            else:
                fill_col = (0.12, 0.14, 0.18, 0.90)
            batch_fill = batch_for_shader(shader, "TRIS", {"pos": tris})
            shader.bind()
            shader.uniform_float("color", fill_col)
            batch_fill.draw(shader)

            gpu.state.line_width_set(2.4 if is_hover else 1.4)
            rim_col = (1.0, 1.0, 1.0, 0.96) if is_hover else (acc[0], acc[1], acc[2], 0.72)
            batch_line = batch_for_shader(shader, "LINE_LOOP", {"pos": outline})
            shader.bind()
            shader.uniform_float("color", rim_col)
            batch_line.draw(shader)

            a_mid = (a0 + a1) * 0.5
            r_mid = (VIEW_WHEEL_INNER + VIEW_WHEEL_OUTER) * 0.5 + (4.0 if is_hover else 0.0)
            ix = cx + r_mid * math.cos(a_mid)
            iy = cy + r_mid * math.sin(a_mid) + 12.0
            icon_col = (0.07, 0.07, 0.09, 1.0) if is_hover else (acc[0], acc[1], acc[2], 1.0)
            _draw_vector_icon(shader, mode_icon.get(mid, "eye"), ix, iy, 9.0, icon_col)

        hub_tris, hub_ring = build_disk_geometry(cx, cy, VIEW_WHEEL_INNER - 4.0, steps=36)
        batch_hub = batch_for_shader(shader, "TRIS", {"pos": hub_tris})
        shader.bind()
        shader.uniform_float("color", (0.09, 0.10, 0.13, 0.95))
        batch_hub.draw(shader)

        gpu.state.line_width_set(1.8)
        acc = VIEW_MODE_ACCENT.get(cur_mode, VIEW_MODE_ACCENT["ALL"])
        batch_hub_ring = batch_for_shader(shader, "LINE_LOOP", {"pos": hub_ring})
        shader.bind()
        shader.uniform_float("color", acc)
        batch_hub_ring.draw(shader)

        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")

        font_id = 0
        for mid, lbl, hint, acc, d0, d1 in wedges:
            is_hover = mid == (self._hovered_hole if is_split else self._hovered_main)
            a_mid = math.radians((d0 + d1) * 0.5)
            r_mid = (VIEW_WHEEL_INNER + VIEW_WHEEL_OUTER) * 0.5 + (4.0 if is_hover else 0.0)
            tx = cx + r_mid * math.cos(a_mid)
            ty = cy + r_mid * math.sin(a_mid) - 14.0
            txt_col = (0.07, 0.07, 0.09, 1.0) if is_hover else (0.95, 0.96, 0.99, 0.97)
            _draw_centered_text(font_id, lbl, tx, ty, 13 if is_hover else 12, txt_col)

        _draw_centered_text(font_id, "View Mode", cx, cy + 9.0, 11, (0.92, 0.94, 0.98, 0.95))
        _draw_centered_text(font_id, VIEW_MODE_SHORT.get(cur_mode, "All"), cx, cy - 10.0, 12, acc)

    def modal(self, context, event):
        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
            main = hit_test_view_wheel(mx, my, self._cx, self._cy)
            hole = hit_test_hole_split(mx, my, self._cx, self._cy) if main == "MAIN" else None
            if main != self._hovered_main or hole != self._hovered_hole:
                self._hovered_main = main
                self._hovered_hole = hole
                if context.area:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
            main = hit_test_view_wheel(mx, my, self._cx, self._cy)
            hole = hit_test_hole_split(mx, my, self._cx, self._cy) if main == "MAIN" else None
            mode = hole or main
            self._cleanup(context)
            if mode is not None:
                bpy.ops.fdt.applebite_set_view_mode(mode=mode)
                return {"FINISHED"}
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        global _active_view_wheel
        if _active_view_wheel is not None:
            _active_view_wheel._cleanup(context)

        self._cx = float(self.cx)
        self._cy = float(self.cy)
        self._hovered_main = None
        self._hovered_hole = None
        _active_view_wheel = self

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_wheel, (), "WINDOW", "POST_PIXEL"
        )
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


CLASSES = (FDT_OT_cutter_wheel_menu, FDT_OT_view_mode_wheel_menu)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    global _active_wheel, _active_view_wheel
    if _active_wheel is not None:
        _active_wheel._cleanup(None)
    if _active_view_wheel is not None:
        _active_view_wheel._cleanup(None)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
