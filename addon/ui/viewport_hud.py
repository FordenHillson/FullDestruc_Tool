"""Minimal 3D Viewport Floating Action Dock (Active Session Only, Custom 2D Pill Gizmos).

Renders a sleek translucent dark glass capsule at the bottom-center of the 3D
Viewport ONLY while an Apple Bite session or live/stale Rebar workflow is active.
Leaves the viewport 100% clean when idle and never intercepts G/R/S, MMB orbit,
selection, or Undo outside the capsule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from ..core import apple_bite, rebar_state, zone_registry
from ..core.apple_bite import CUTTER_PROP
from ..core.session import load_session

# Unified Semantic Palette (RGBA)
COL_GLASS_BG = (0.075, 0.086, 0.114, 0.88)
COL_GLASS_RIM = (1.0, 1.0, 1.0, 0.14)
COL_DIVIDER = (1.0, 1.0, 1.0, 0.16)

COL_EMERALD = (0.14, 0.72, 0.42, 0.96)
COL_EMERALD_HOVER = (0.20, 0.84, 0.50, 1.00)
COL_AMBER = (0.96, 0.62, 0.08, 0.96)
COL_ROSE = (0.96, 0.25, 0.37, 0.96)
COL_CYAN = (0.22, 0.74, 0.97, 0.96)
COL_RED = (0.92, 0.26, 0.28, 0.96)
COL_SLATE = (0.64, 0.70, 0.80, 0.92)
COL_PILL_BG = (0.14, 0.16, 0.21, 0.84)
COL_PILL_HOVER = (0.22, 0.25, 0.33, 0.96)
COL_TEXT = (0.95, 0.97, 1.00, 0.98)
COL_TEXT_DARK = (0.06, 0.08, 0.10, 1.00)
COL_TEXT_DIM = (0.60, 0.64, 0.72, 0.75)

VIEW_SHORT = {
    "ALL": "All",
    "PIECES": "Piece",
    "MAIN": "Hole",
    "MAIN_WIRE": "Wire",
}


@dataclass(frozen=True)
class DockButtonSpec:
    """Layout & visual descriptor for one clickable pill in the floating dock."""

    slot_id: str
    label: str
    icon: str
    op_id: str
    accent: tuple[float, float, float, float]
    style: str  # 'zone' | 'normal' | 'hero_amber' | 'hero_green' | 'alert_red' | 'subtle' | 'muted' | 'status'
    rect: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    divider_before: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class DockLayout:
    """Computed 2D pixel geometry for the entire bottom-center floating dock."""

    mode: str  # 'APPLE_PHASE_A' | 'APPLE_PHASE_B' | 'REBAR_LIVE'
    capsule_rect: tuple[float, float, float, float]
    divider_xs: tuple[float, ...]
    buttons: tuple[DockButtonSpec, ...]
    compact: bool
    ui_scale: float


def compute_dock_layout(
    region_width: float,
    mode: str,
    *,
    zone_token: str = "A",
    zone_color: tuple[float, float, float, float] = COL_AMBER,
    active_cutters: int = 0,
    total_cutters: int = 0,
    view_mode: str = "ALL",
    has_cut_material: bool = True,
    has_output_enabled: bool = True,
    rebar_estimate: int = 0,
    rebar_stale: bool = False,
    ui_scale: float = 1.0,
) -> DockLayout:
    """Compute centered, non-overlapping 2D pixel rectangles for the Viewport Dock.

    Pure geometry helper (no GPU/Blender calls) so narrow viewports, DPI scaling,
    and every progressive workflow phase can be unit-tested deterministically.
    """
    s = max(0.75, min(float(ui_scale or 1.0), 2.5))
    compact = float(region_width) < (680.0 * s)

    pad_x = 10.0 * s
    pad_y = 6.0 * s
    pill_h = 30.0 * s
    gap = (5.0 if compact else 7.0) * s
    div_gap = (10.0 if compact else 14.0) * s
    bottom_y = 20.0 * s

    def _est_w(text: str, *, has_icon: bool = True, extra: float = 0.0) -> float:
        char_w = (6.6 if compact else 7.2) * s
        icon_w = (18.0 * s) if has_icon else 0.0
        inner_pad = (14.0 if compact else 20.0) * s
        if not text:
            return max(30.0 * s, icon_w + 12.0 * s)
        return max(36.0 * s, inner_pad + icon_w + len(text) * char_w + extra * s)

    raw_items: list[tuple[str, str, str, str, tuple[float, float, float, float], str, bool, bool, float]] = []
    # tuple: (slot_id, label, icon, op_id, accent, style, divider_before, enabled, extra_w)

    z_short = (zone_token or "A")[: (4 if compact else 8)]
    if mode == "APPLE_PHASE_A":
        z_lbl = z_short if compact else f"Zone {z_short}"
        raw_items.append(("zone", z_lbl, "zone_dot", "fdt.viewport_zone_quick", zone_color, "zone", False, True, 2.0))
        raw_items.append(
            ("add_box", "+Box" if compact else "+ Add Box", "plus_box", "fdt.applebite_add_box", COL_AMBER, "hero_amber", True, True, 4.0)
        )
        raw_items.append(("draw", "Draw", "pen", "fdt.applebite_draw_cutter", COL_ROSE, "normal", False, True, 2.0))
        raw_items.append(("cancel_zero", "", "cross", "fdt.applebite_cancel", COL_RED, "subtle", True, True, 0.0))

    elif mode == "APPLE_PHASE_B":
        z_lbl = z_short if compact else f"Zone {z_short}"
        raw_items.append(("zone", z_lbl, "zone_dot", "fdt.viewport_zone_quick", zone_color, "zone", False, True, 2.0))
        raw_items.append(("add_box", "+Box", "plus_box", "fdt.applebite_add_box", COL_AMBER, "normal", True, True, 0.0))
        raw_items.append(("draw", "Draw", "pen", "fdt.applebite_draw_cutter", COL_ROSE, "normal", False, True, 0.0))
        v_short = VIEW_SHORT.get(view_mode, "All")
        v_lbl = v_short if compact else f"View: {v_short}"
        raw_items.append(("view", v_lbl, "eye", "fdt.view_mode_wheel_menu", COL_CYAN, "normal", False, True, 0.0))

        if not has_cut_material:
            raw_items.append(
                ("mat_alert", "Mat!", "drop_alert", "fdt.pick_surface_material", COL_RED, "alert_red", False, True, 2.0)
            )

        # Primary APPLY button with Round-3 All-Muted & Output-None safety guards
        if not has_output_enabled:
            raw_items.append(
                ("apply", "No Out", "check", "fdt.applebite_apply", COL_RED, "muted", True, False, 4.0)
            )
        elif active_cutters <= 0 and total_cutters > 0:
            raw_items.append(
                ("apply", "Muted" if compact else "All Muted", "pause", "fdt.applebite_apply", COL_SLATE, "muted", True, False, 4.0)
            )
        else:
            apply_lbl = f"{active_cutters}" if compact else f"APPLY · {active_cutters}"
            raw_items.append(
                ("apply", apply_lbl, "check", "fdt.applebite_apply", COL_EMERALD, "hero_green", True, True, 8.0)
            )

        # Safe Pause only on Phase B dock (Cancel & Restore stays in N-Panel / Qt with confirm guard)
        raw_items.append(("pause", "", "pause", "fdt.applebite_exit", COL_SLATE, "subtle", True, True, 0.0))

    elif mode == "REBAR_LIVE":
        est_txt = f"~{rebar_estimate}" if rebar_estimate > 0 else "Live"
        status_lbl = est_txt if compact else f"Rebar · {est_txt}"
        raw_items.append(("rebar_status", status_lbl, "rebar", "", COL_CYAN, "status", False, False, 2.0))
        if rebar_stale:
            raw_items.append(
                ("rebar_rebuild", "Re-apply", "refresh", "fdt.rebar_rebuild", COL_AMBER, "hero_amber", True, True, 4.0)
            )
        raw_items.append(
            ("rebar_bake", "BAKE", "check", "fdt.rebar_bake", COL_EMERALD, "hero_green", not rebar_stale, True, 8.0)
        )
        raw_items.append(("rebar_edit", "" if compact else "Edit", "pen", "fdt.rebar_reedit", COL_CYAN, "subtle", True, True, 0.0))
        raw_items.append(("rebar_remove", "", "trash", "fdt.rebar_remove", COL_RED, "subtle", False, True, 0.0))

    # Measure total width and center horizontally
    widths = [_est_w(lbl, has_icon=bool(ico), extra=ew) for _sid, lbl, ico, _op, _acc, _st, _div, _en, ew in raw_items]
    total_w = pad_x * 2.0
    for i, w in enumerate(widths):
        if i > 0:
            total_w += div_gap if raw_items[i][6] else gap
        total_w += w

    cx = float(region_width) * 0.5
    cap_x0 = max(12.0 * s, cx - total_w * 0.5)
    cap_x1 = cap_x0 + total_w
    cap_y0 = bottom_y
    cap_y1 = bottom_y + pill_h + pad_y * 2.0

    buttons: list[DockButtonSpec] = []
    divider_xs: list[float] = []
    cur_x = cap_x0 + pad_x
    py0 = cap_y0 + pad_y
    py1 = py0 + pill_h

    for i, (sid, lbl, ico, op_id, acc, style, div_before, enabled, _ew) in enumerate(raw_items):
        if i > 0:
            if div_before:
                divider_xs.append(cur_x + div_gap * 0.5)
                cur_x += div_gap
            else:
                cur_x += gap
        w = widths[i]
        buttons.append(
            DockButtonSpec(
                slot_id=sid,
                label=lbl,
                icon=ico,
                op_id=op_id,
                accent=acc,
                style=style,
                rect=(cur_x, py0, cur_x + w, py1),
                divider_before=div_before,
                enabled=enabled,
            )
        )
        cur_x += w

    return DockLayout(
        mode=mode,
        capsule_rect=(cap_x0, cap_y0, cap_x1, cap_y1),
        divider_xs=tuple(divider_xs),
        buttons=tuple(buttons),
        compact=compact,
        ui_scale=s,
    )


# ---------------------------------------------------------------------------
# GPU 2D Rounded-Rect & Vector Icon Drawing Helpers
# ---------------------------------------------------------------------------


def _rounded_rect_poly(x0: float, y0: float, x1: float, y1: float, radius: float, segs: int = 6) -> list[tuple[float, float]]:
    r = max(0.0, min(radius, (x1 - x0) * 0.5, (y1 - y0) * 0.5))
    if r <= 0.5:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    corners = (
        (x1 - r, y0 + r, -math.pi * 0.5, 0.0),
        (x1 - r, y1 - r, 0.0, math.pi * 0.5),
        (x0 + r, y1 - r, math.pi * 0.5, math.pi),
        (x0 + r, y0 + r, math.pi, math.pi * 1.5),
    )
    pts: list[tuple[float, float]] = []
    for cx, cy, a0, a1 in corners:
        for k in range(segs + 1):
            ang = a0 + (a1 - a0) * (k / segs)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _draw_rounded_rect(
    shader,
    rect: tuple[float, float, float, float],
    radius: float,
    fill_col: tuple[float, float, float, float],
    rim_col: tuple[float, float, float, float] | None = None,
    rim_width: float = 1.2,
) -> None:
    x0, y0, x1, y1 = rect
    ring = _rounded_rect_poly(x0, y0, x1, y1, radius)
    if len(ring) < 3:
        return
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    tris: list[tuple[float, float]] = []
    n = len(ring)
    for i in range(n):
        tris.extend([(cx, cy), ring[i], ring[(i + 1) % n]])

    batch = batch_for_shader(shader, "TRIS", {"pos": tris})
    shader.bind()
    shader.uniform_float("color", fill_col)
    batch.draw(shader)

    if rim_col is not None and rim_col[3] > 0.01:
        gpu.state.line_width_set(rim_width)
        b_rim = batch_for_shader(shader, "LINE_LOOP", {"pos": ring})
        shader.bind()
        shader.uniform_float("color", rim_col)
        b_rim.draw(shader)


def _draw_dock_icon(
    shader,
    icon_name: str,
    cx: float,
    cy: float,
    s: float,
    color: tuple[float, float, float, float],
) -> None:
    if icon_name == "zone_dot":
        pts = _rounded_rect_poly(cx - s, cy - s, cx + s, cy + s, s, segs=5)
        tris = []
        for i in range(len(pts)):
            tris.extend([(cx, cy), pts[i], pts[(i + 1) % len(pts)]])
        b = batch_for_shader(shader, "TRIS", {"pos": tris})
        shader.bind()
        shader.uniform_float("color", color)
        b.draw(shader)
        return

    lines: list[tuple[float, float]] = []
    if icon_name == "check":
        lines = [
            (cx - s * 0.80, cy + s * 0.02), (cx - s * 0.18, cy - s * 0.65),
            (cx - s * 0.18, cy - s * 0.65), (cx + s * 0.85, cy + s * 0.68),
        ]
    elif icon_name == "plus_box":
        lines = [
            (cx - s * 0.78, cy - s * 0.78), (cx + s * 0.78, cy - s * 0.78),
            (cx + s * 0.78, cy - s * 0.78), (cx + s * 0.78, cy + s * 0.78),
            (cx + s * 0.78, cy + s * 0.78), (cx - s * 0.78, cy + s * 0.78),
            (cx - s * 0.78, cy + s * 0.78), (cx - s * 0.78, cy - s * 0.78),
            (cx - s * 0.40, cy), (cx + s * 0.40, cy),
            (cx, cy - s * 0.40), (cx, cy + s * 0.40),
        ]
    elif icon_name == "pen":
        lines = [
            (cx - s * 0.78, cy - s * 0.78), (cx - s * 0.30, cy - s * 0.68),
            (cx - s * 0.78, cy - s * 0.78), (cx - s * 0.68, cy - s * 0.30),
            (cx - s * 0.30, cy - s * 0.68), (cx + s * 0.74, cy + s * 0.36),
            (cx - s * 0.68, cy - s * 0.30), (cx + s * 0.36, cy + s * 0.74),
            (cx + s * 0.74, cy + s * 0.36), (cx + s * 0.36, cy + s * 0.74),
        ]
    elif icon_name == "eye":
        lines = [
            (cx - s * 0.90, cy), (cx - s * 0.42, cy + s * 0.50),
            (cx - s * 0.42, cy + s * 0.50), (cx + s * 0.42, cy + s * 0.50),
            (cx + s * 0.42, cy + s * 0.50), (cx + s * 0.90, cy),
            (cx + s * 0.90, cy), (cx + s * 0.42, cy - s * 0.50),
            (cx + s * 0.42, cy - s * 0.50), (cx - s * 0.42, cy - s * 0.50),
            (cx - s * 0.42, cy - s * 0.50), (cx - s * 0.90, cy),
            (cx, cy + s * 0.28), (cx + s * 0.28, cy),
            (cx + s * 0.28, cy), (cx, cy - s * 0.28),
            (cx, cy - s * 0.28), (cx - s * 0.28, cy),
            (cx - s * 0.28, cy), (cx, cy + s * 0.28),
        ]
    elif icon_name == "drop_alert":
        lines = [
            (cx, cy + s * 0.85), (cx + s * 0.85, cy - s * 0.70),
            (cx + s * 0.85, cy - s * 0.70), (cx - s * 0.85, cy - s * 0.70),
            (cx - s * 0.85, cy - s * 0.70), (cx, cy + s * 0.85),
            (cx, cy + s * 0.25), (cx, cy - s * 0.20),
            (cx, cy - s * 0.42), (cx, cy - s * 0.52),
        ]
    elif icon_name == "pause":
        lines = [
            (cx - s * 0.36, cy - s * 0.65), (cx - s * 0.36, cy + s * 0.65),
            (cx + s * 0.36, cy - s * 0.65), (cx + s * 0.36, cy + s * 0.65),
        ]
    elif icon_name == "cross":
        lines = [
            (cx - s * 0.60, cy - s * 0.60), (cx + s * 0.60, cy + s * 0.60),
            (cx - s * 0.60, cy + s * 0.60), (cx + s * 0.60, cy - s * 0.60),
        ]
    elif icon_name == "refresh":
        lines = [
            (cx - s * 0.65, cy + s * 0.20), (cx - s * 0.15, cy + s * 0.72),
            (cx - s * 0.15, cy + s * 0.72), (cx + s * 0.55, cy + s * 0.50),
            (cx + s * 0.65, cy - s * 0.20), (cx + s * 0.15, cy - s * 0.72),
            (cx + s * 0.15, cy - s * 0.72), (cx - s * 0.55, cy - s * 0.50),
            (cx + s * 0.55, cy + s * 0.50), (cx + s * 0.65, cy + s * 0.05),
            (cx + s * 0.55, cy + s * 0.50), (cx + s * 0.12, cy + s * 0.48),
        ]
    elif icon_name == "rebar":
        lines = [
            (cx - s * 0.75, cy - s * 0.65), (cx - s * 0.75, cy + s * 0.65),
            (cx - s * 0.75, cy + s * 0.42), (cx + s * 0.75, cy + s * 0.55),
            (cx - s * 0.75, cy), (cx + s * 0.80, cy - s * 0.12),
            (cx - s * 0.75, cy - s * 0.42), (cx + s * 0.65, cy - s * 0.62),
        ]
    elif icon_name == "trash":
        lines = [
            (cx - s * 0.72, cy + s * 0.55), (cx + s * 0.72, cy + s * 0.55),
            (cx - s * 0.30, cy + s * 0.55), (cx - s * 0.25, cy + s * 0.80),
            (cx - s * 0.25, cy + s * 0.80), (cx + s * 0.25, cy + s * 0.80),
            (cx + s * 0.25, cy + s * 0.80), (cx + s * 0.30, cy + s * 0.55),
            (cx - s * 0.55, cy + s * 0.55), (cx - s * 0.42, cy - s * 0.72),
            (cx - s * 0.42, cy - s * 0.72), (cx + s * 0.42, cy - s * 0.72),
            (cx + s * 0.42, cy - s * 0.72), (cx + s * 0.55, cy + s * 0.55),
        ]

    if not lines:
        return
    gpu.state.line_width_set(2.2)
    b = batch_for_shader(shader, "LINES", {"pos": lines})
    shader.bind()
    shader.uniform_float("color", color)
    b.draw(shader)


# ---------------------------------------------------------------------------
# Active State & Modal Auto-Hide Checks
# ---------------------------------------------------------------------------


def _any_modal_overlay_active() -> bool:
    """Return True if a modal viewport HUD/wheel/overlay is active (auto-hide dock)."""
    try:
        from ..ops import apple_bite_ops

        if bool(getattr(apple_bite_ops, "_interactive_active", False)):
            return True
    except Exception:
        pass
    try:
        from . import wheel_menu

        if getattr(wheel_menu, "_active_wheel", None) is not None:
            return True
        if getattr(wheel_menu, "_active_view_wheel", None) is not None:
            return True
    except Exception:
        pass
    try:
        from . import cutter_gizmo

        if getattr(cutter_gizmo, "_active_axis_picker", None) is not None:
            return True
        if getattr(cutter_gizmo, "_active_layer_picker", None) is not None:
            return True
    except Exception:
        pass
    try:
        from . import draw_overlay

        ov = getattr(draw_overlay, "_active_overlay", None)
        if ov is not None and hasattr(ov, "isVisible") and ov.isVisible():
            return True
    except Exception:
        pass
    return False


def _resolve_active_rebar(scene) -> tuple[object | None, int]:
    """Return ``(live_or_stale_entry, estimate)`` if a Rebar dock should show, else ``(None, 0)``."""
    rset = getattr(scene, "fdt_rebar_settings", None)
    if rset is None:
        return None, 0
    active_sid = str(getattr(rset, "rebar_active_sid", "") or "")
    entry = rebar_state.find_entry(active_sid, scene) if active_sid else None
    if entry is not None and (not entry.baked or entry.stale):
        return entry, int(getattr(rset, "rebar_estimate", 0) or 0)
    # Also surface any stale rebar set if the user is on the Rebar tab
    settings = getattr(scene, "fdt_settings", None)
    if settings is not None and getattr(settings, "ui_active_tool", "") == "REBAR":
        for reg_entry in rebar_state.load_registry(scene):
            if reg_entry.stale:
                return reg_entry, 0
    return None, 0


def _resolve_dock_mode(context) -> tuple[str | None, dict]:
    """Resolve active dock mode & state using Dual-Active Priority Rule."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return None, {}
    settings = getattr(scene, "fdt_settings", None)
    if settings is None or not bool(getattr(settings, "show_viewport_dock", True)):
        return None, {}

    session = load_session(scene)
    rebar_entry, rebar_est = _resolve_active_rebar(scene)

    active_tool = str(getattr(settings, "ui_active_tool", "APPLE_BITE") or "APPLE_BITE")
    pick_rebar = False
    if session is not None and rebar_entry is not None:
        pick_rebar = active_tool == "REBAR"
    elif rebar_entry is not None:
        pick_rebar = True
    elif session is None:
        return None, {}

    if pick_rebar and rebar_entry is not None:
        return "REBAR_LIVE", {
            "rebar_entry": rebar_entry,
            "rebar_estimate": rebar_est,
            "rebar_stale": bool(rebar_entry.stale),
        }

    # Apple Bite active session (counts ONLY valid existing cutters — handles manual X delete!)
    valid_all = session.get_valid_cutters(include_muted=True)
    valid_active = session.get_valid_cutters(include_muted=False)
    total_c = len(valid_all)
    active_c = len(valid_active)

    zone_key = str(getattr(settings, "active_zone", "A") or "A")
    active_obj = getattr(context, "active_object", None)
    if active_obj is not None and bool(active_obj.get(CUTTER_PROP, False)):
        zone_key = str(active_obj.get(apple_bite.CUTTER_ZONE_PROP, zone_key) or zone_key)
    zdef = zone_registry.resolve_zone(zone_key, scene)
    z_col = (float(zdef.color[0]), float(zdef.color[1]), float(zdef.color[2]), 1.0)

    mode = "APPLE_PHASE_A" if total_c == 0 else "APPLE_PHASE_B"
    return mode, {
        "zone_token": zdef.token,
        "zone_color": z_col,
        "active_cutters": active_c,
        "total_cutters": total_c,
        "view_mode": str(getattr(settings, "preview_display_mode", "ALL") or "ALL"),
        "has_cut_material": getattr(settings, "cut_material", None) is not None,
        "has_output_enabled": bool(
            getattr(settings, "apply_keep_result", True) or getattr(settings, "apply_keep_pieces", True)
        ),
    }


# ---------------------------------------------------------------------------
# Operators supporting the Viewport Dock
# ---------------------------------------------------------------------------


class FDT_OT_viewport_zone_quick(bpy.types.Operator):
    """Open searchable Zone Picker for selected cutter, or quick-switch Active Zone."""

    bl_idname = "fdt.viewport_zone_quick"
    bl_label = "Quick Zone"
    bl_description = "เลือกหรือเปลี่ยน Zone ของกล่องตัดที่เลือกอยู่ (หรือ Active Zone สำหรับกล่องถัดไป)"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        obj = getattr(context, "active_object", None)
        if obj is not None and bool(obj.get(CUTTER_PROP, False)):
            return bpy.ops.fdt.zone_picker("INVOKE_DEFAULT", cutter_name=obj.name)
        return context.window_manager.invoke_popup(self, width=300)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.fdt_settings
        layout.label(text="Active Zone (New Cutters)", icon="COLOR")
        layout.prop(settings, "active_zone", text="")
        grid = layout.grid_flow(row_major=True, columns=4, even_columns=True, align=True)
        for zdef in zone_registry.load_zones(scene)[:8]:
            op = grid.operator(
                "fdt.applebite_set_zone",
                text=zdef.token[:8],
                depress=(settings.active_zone == zdef.key),
            )
            op.zone = zdef.key
        layout.separator()
        row = layout.row(align=True)
        op_new = row.operator("fdt.zone_manage", text="New Zone", icon="ADD")
        op_new.action = "NEW"
        op_new.zone_key = settings.active_zone

    def execute(self, context):
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Custom 2D Gizmos (Glass Capsule Backdrop + Clickable Rounded Pills)
# ---------------------------------------------------------------------------


class FDT_GT_dock_backdrop(bpy.types.Gizmo):
    """Non-interactive translucent dark glass capsule drawn behind dock pills."""

    bl_idname = "FDT_GT_dock_backdrop"

    def setup(self):
        self._capsule_rect = (0.0, 0.0, 0.0, 0.0)
        self._divider_xs: tuple[float, ...] = ()
        self._ui_scale = 1.0

    def test_select(self, context, location):
        return -1

    def draw(self, context):
        x0, y0, x1, y1 = self._capsule_rect
        if x1 <= x0 or y1 <= y0:
            return
        s = self._ui_scale
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        try:
            _draw_rounded_rect(shader, self._capsule_rect, 14.0 * s, COL_GLASS_BG, COL_GLASS_RIM, rim_width=1.2)
            if self._divider_xs:
                pad = 9.0 * s
                lines: list[tuple[float, float]] = []
                for dx in self._divider_xs:
                    lines.extend([(dx, y0 + pad), (dx, y1 - pad)])
                gpu.state.line_width_set(1.0)
                b = batch_for_shader(shader, "LINES", {"pos": lines})
                shader.bind()
                shader.uniform_float("color", COL_DIVIDER)
                b.draw(shader)
        finally:
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set("NONE")


class FDT_GT_dock_pill(bpy.types.Gizmo):
    """Interactive 2D rounded-pill gizmo with exact rect hit-test and native tooltip."""

    bl_idname = "FDT_GT_dock_pill"

    def setup(self):
        self._rect = (0.0, 0.0, 0.0, 0.0)
        self._label = ""
        self._icon = ""
        self._accent = COL_SLATE
        self._style = "normal"
        self._enabled_flag = True
        self._ui_scale = 1.0

    def test_select(self, context, location):
        if not self._enabled_flag:
            return -1
        x0, y0, x1, y1 = self._rect
        mx, my = float(location[0]), float(location[1])
        if x0 <= mx <= x1 and y0 <= my <= y1:
            return 0
        return -1

    def draw(self, context):
        x0, y0, x1, y1 = self._rect
        if x1 <= x0 or y1 <= y0:
            return
        s = self._ui_scale
        hover = bool(self.is_highlight) and self._enabled_flag
        acc = self._accent
        style = self._style

        if style == "hero_green":
            fill = COL_EMERALD_HOVER if hover else COL_EMERALD
            rim = (0.75, 1.0, 0.86, 0.95) if hover else (0.35, 0.92, 0.62, 0.70)
            icon_col = COL_TEXT
            txt_col = COL_TEXT
        elif style == "hero_amber":
            fill = (1.0, 0.70, 0.16, 0.98) if hover else (0.90, 0.56, 0.08, 0.94)
            rim = (1.0, 0.94, 0.72, 0.95) if hover else (1.0, 0.76, 0.30, 0.65)
            icon_col = COL_TEXT_DARK
            txt_col = COL_TEXT_DARK
        elif style == "alert_red":
            fill = (0.96, 0.30, 0.32, 0.98) if hover else (0.78, 0.18, 0.20, 0.90)
            rim = (1.0, 0.78, 0.78, 0.95) if hover else (0.98, 0.42, 0.44, 0.75)
            icon_col = COL_TEXT
            txt_col = COL_TEXT
        elif style == "zone":
            fill = (acc[0] * 0.34, acc[1] * 0.34, acc[2] * 0.34, 0.94) if hover else (0.14, 0.16, 0.22, 0.86)
            rim = (acc[0], acc[1], acc[2], 0.98 if hover else 0.72)
            icon_col = acc
            txt_col = COL_TEXT
        elif style == "muted":
            fill = (0.13, 0.14, 0.18, 0.65)
            rim = (acc[0], acc[1], acc[2], 0.45)
            icon_col = (acc[0], acc[1], acc[2], 0.65)
            txt_col = COL_TEXT_DIM
        elif style == "status":
            fill = (0.12, 0.15, 0.20, 0.75)
            rim = (acc[0], acc[1], acc[2], 0.42)
            icon_col = acc
            txt_col = COL_TEXT
        elif style == "subtle":
            fill = (acc[0] * 0.28, acc[1] * 0.28, acc[2] * 0.28, 0.90) if hover else (0.12, 0.14, 0.18, 0.55)
            rim = (acc[0], acc[1], acc[2], 0.85) if hover else (1.0, 1.0, 1.0, 0.08)
            icon_col = acc if hover else (acc[0], acc[1], acc[2], 0.85)
            txt_col = COL_TEXT
        else:  # 'normal'
            fill = COL_PILL_HOVER if hover else COL_PILL_BG
            rim = (acc[0], acc[1], acc[2], 0.92 if hover else 0.42)
            icon_col = acc
            txt_col = COL_TEXT

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        try:
            _draw_rounded_rect(shader, self._rect, 9.0 * s, fill, rim, rim_width=1.6 if hover else 1.1)
            cy = (y0 + y1) * 0.5
            icon_r = 5.5 * s
            if self._label:
                ix = x0 + 13.0 * s
                if self._icon:
                    _draw_dock_icon(shader, self._icon, ix, cy, icon_r, icon_col)
            else:
                if self._icon:
                    _draw_dock_icon(shader, self._icon, (x0 + x1) * 0.5, cy, icon_r * 1.05, icon_col)
        finally:
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set("NONE")

        if self._label:
            font_id = 0
            fsize = max(10, int(round(12.0 * s)))
            blf.size(font_id, fsize)
            blf.color(font_id, *txt_col)
            tw, th = blf.dimensions(font_id, self._label)
            left_pad = (23.0 * s) if self._icon else 0.0
            avail_x0 = x0 + left_pad
            tx = avail_x0 + max(0.0, (x1 - avail_x0 - tw) * 0.5) - (2.0 * s if self._icon else 0.0)
            ty = cy - th * 0.36
            blf.position(font_id, tx, ty, 0)
            blf.draw(font_id, self._label)


# Fixed slot registry -> (slot_id, operator_id)
_DOCK_SLOTS: tuple[tuple[str, str], ...] = (
    ("zone", "fdt.viewport_zone_quick"),
    ("add_box", "fdt.applebite_add_box"),
    ("draw", "fdt.applebite_draw_cutter"),
    ("view", "fdt.view_mode_wheel_menu"),
    ("mat_alert", "fdt.pick_surface_material"),
    ("apply", "fdt.applebite_apply"),
    ("pause", "fdt.applebite_exit"),
    ("cancel_zero", "fdt.applebite_cancel"),
    ("rebar_status", ""),
    ("rebar_rebuild", "fdt.rebar_rebuild"),
    ("rebar_bake", "fdt.rebar_bake"),
    ("rebar_edit", "fdt.rebar_reedit"),
    ("rebar_remove", "fdt.rebar_remove"),
)


class FDT_GGT_viewport_dock(bpy.types.GizmoGroup):
    """Active-session-only 2D floating action dock centered at bottom of 3D Viewport."""

    bl_idname = "FDT_GGT_viewport_dock"
    bl_label = "FullDestruc Viewport Action Dock"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"PERSISTENT"}

    @classmethod
    def poll(cls, context):
        if _any_modal_overlay_active():
            return False
        mode, _state = _resolve_dock_mode(context)
        return mode is not None

    def setup(self, context):
        self._backdrop = self.gizmos.new(FDT_GT_dock_backdrop.bl_idname)
        self._pills: dict[str, FDT_GT_dock_pill] = {}
        self._op_props: dict[str, object] = {}

        for slot_id, op_id in _DOCK_SLOTS:
            gz = self.gizmos.new(FDT_GT_dock_pill.bl_idname)
            gz.hide = True
            if op_id:
                try:
                    self._op_props[slot_id] = gz.target_set_operator(op_id)
                except Exception:
                    pass
            self._pills[slot_id] = gz

    def draw_prepare(self, context):
        mode, state = _resolve_dock_mode(context)
        if mode is None or context.region is None:
            self._backdrop.hide = True
            for gz in self._pills.values():
                gz.hide = True
            return

        prefs = getattr(context, "preferences", None)
        sys_prefs = getattr(prefs, "system", None) if prefs is not None else None
        ui_scale = float(getattr(sys_prefs, "ui_scale", 1.0) or 1.0)

        layout = compute_dock_layout(
            float(context.region.width),
            mode,
            zone_token=str(state.get("zone_token", "A")),
            zone_color=state.get("zone_color", COL_AMBER),
            active_cutters=int(state.get("active_cutters", 0)),
            total_cutters=int(state.get("total_cutters", 0)),
            view_mode=str(state.get("view_mode", "ALL")),
            has_cut_material=bool(state.get("has_cut_material", True)),
            has_output_enabled=bool(state.get("has_output_enabled", True)),
            rebar_estimate=int(state.get("rebar_estimate", 0)),
            rebar_stale=bool(state.get("rebar_stale", False)),
            ui_scale=ui_scale,
        )

        self._backdrop.hide = False
        self._backdrop._capsule_rect = layout.capsule_rect
        self._backdrop._divider_xs = layout.divider_xs
        self._backdrop._ui_scale = layout.ui_scale

        # Sync rebar sid onto Edit operator properties when active
        rebar_entry = state.get("rebar_entry")
        if rebar_entry is not None:
            edit_props = self._op_props.get("rebar_edit")
            if edit_props is not None and hasattr(edit_props, "sid"):
                try:
                    edit_props.sid = str(rebar_entry.sid)
                except Exception:
                    pass

        active_specs = {b.slot_id: b for b in layout.buttons}
        for slot_id, gz in self._pills.items():
            spec = active_specs.get(slot_id)
            if spec is None:
                gz.hide = True
                continue
            gz.hide = False
            gz._rect = spec.rect
            gz._label = spec.label
            gz._icon = spec.icon
            gz._accent = spec.accent
            gz._style = spec.style
            gz._enabled_flag = spec.enabled
            gz._ui_scale = layout.ui_scale


CLASSES = (
    FDT_OT_viewport_zone_quick,
    FDT_GT_dock_backdrop,
    FDT_GT_dock_pill,
    FDT_GGT_viewport_dock,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
