"""UI Redesign automated test suite: Custom Colored Icons, 3D Viewport Dock & Popovers."""

from __future__ import annotations

import struct
from pathlib import Path

import bpy

from FullDestruc_Tool.ui import icons, viewport_hud


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        header = fh.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"Not a valid PNG: {path}"
    w, h = struct.unpack(">II", header[16:24])
    return int(w), int(h)


def test_u01_custom_icons_exist_and_are_64x64_png() -> None:
    required = (
        "apply_green",
        "start_orange",
        "reedit_cyan",
        "pause_slate",
        "cancel_red",
        "trash_red",
        "tool_apple_bite",
        "tool_rebar",
        "tool_cleanup",
        "tool_batch",
        "add_box",
        "use_mesh",
        "draw_pen",
        "click_place",
        "rough_edge",
        "view_eye",
        "zone_grid",
        "layers",
        "settings_gear",
        "window_qt",
        "eyedropper",
        "mat_alert_red",
        "fill_cap",
        "rename_pen",
        "plus_add",
        "zones_grid",
        "output_box",
        "debris",
        "stale_warn",
        "rebuild_amber",
        "zone_c0",
        "zone_c7",
    )
    idir = icons.icons_dir()
    assert idir.is_dir(), f"Missing icons dir: {idir}"
    for name in required:
        p = Path(icons.get_icon_path(name))
        assert p.is_file(), f"Missing icon PNG: {name}"
        w, h = _read_png_dimensions(p)
        assert (w, h) == (64, 64), f"Expected 64x64 for {name}, got {(w, h)}"


def test_u02_icons_helpers_and_zone_color_matching() -> None:
    kw = icons.icon_kw("apply_green", fallback="CHECKMARK")
    assert "icon_value" in kw or kw.get("icon") == "CHECKMARK", kw

    missing_kw = icons.icon_kw("nonexistent_icon_xyz", fallback="INFO")
    assert missing_kw == {"icon": "INFO"}, missing_kw

    assert icons.closest_zone_dot_name((0.96, 0.51, 0.12)) == "zone_c0"
    assert icons.closest_zone_dot_name((0.22, 0.69, 1.00)) == "zone_c1"
    assert icons.closest_zone_dot_name((0.18, 0.83, 0.48)) == "zone_c2"


def _assert_non_overlapping(layout: viewport_hud.DockLayout) -> None:
    cx0, cy0, cx1, cy1 = layout.capsule_rect
    assert cx1 > cx0 and cy1 > cy0
    prev_x1 = cx0
    for btn in layout.buttons:
        x0, y0, x1, y1 = btn.rect
        assert x1 > x0 and y1 > y0, f"Empty rect for {btn.slot_id}"
        assert x0 >= prev_x1 - 1e-4, f"Overlap before {btn.slot_id}: {x0} < {prev_x1}"
        assert x1 <= cx1 + 1e-4, f"Overflow right for {btn.slot_id}: {x1} > {cx1}"
        assert y0 >= cy0 and y1 <= cy1
        prev_x1 = x1


def test_u03_viewport_dock_progressive_phases_and_rebar() -> None:
    # Phase A (0 cutters): only Zone, +Add Box, Draw, Cancel — NO View or APPLY!
    la = viewport_hud.compute_dock_layout(1280.0, "APPLE_PHASE_A", zone_token="B")
    ids_a = [b.slot_id for b in la.buttons]
    assert ids_a == ["zone", "add_box", "draw", "cancel_zero"], ids_a
    assert "apply" not in ids_a and "view" not in ids_a
    _assert_non_overlapping(la)

    # Phase B (>= 1 cutter, Material set): Zone, +Box, Draw, View, APPLY, Pause (NO Cancel next to APPLY!)
    lb = viewport_hud.compute_dock_layout(
        1280.0,
        "APPLE_PHASE_B",
        zone_token="B",
        active_cutters=12,
        total_cutters=12,
        has_cut_material=True,
    )
    ids_b = [b.slot_id for b in lb.buttons]
    assert ids_b == ["zone", "add_box", "draw", "view", "apply", "pause"], ids_b
    apply_btn = next(b for b in lb.buttons if b.slot_id == "apply")
    assert apply_btn.style == "hero_green" and apply_btn.enabled is True
    assert "12" in apply_btn.label
    _assert_non_overlapping(lb)

    # Phase B with UNSET Cut Material: red Mat! alert pill appears!
    lb_nomat = viewport_hud.compute_dock_layout(
        1280.0,
        "APPLE_PHASE_B",
        active_cutters=3,
        total_cutters=3,
        has_cut_material=False,
    )
    assert any(b.slot_id == "mat_alert" and b.style == "alert_red" for b in lb_nomat.buttons)
    _assert_non_overlapping(lb_nomat)

    # Rebar Live + Stale
    lr = viewport_hud.compute_dock_layout(
        1280.0,
        "REBAR_LIVE",
        rebar_estimate=42,
        rebar_stale=True,
    )
    ids_r = [b.slot_id for b in lr.buttons]
    assert "rebar_rebuild" in ids_r and "rebar_bake" in ids_r
    _assert_non_overlapping(lr)


def test_u04_viewport_dock_responsive_narrow_and_dpi() -> None:
    narrow = viewport_hud.compute_dock_layout(
        500.0,
        "APPLE_PHASE_B",
        zone_token="LongZoneToken",
        active_cutters=7,
        total_cutters=7,
        ui_scale=1.0,
    )
    assert narrow.compact is True
    _assert_non_overlapping(narrow)

    hidpi = viewport_hud.compute_dock_layout(
        1920.0,
        "APPLE_PHASE_B",
        active_cutters=7,
        total_cutters=7,
        ui_scale=1.5,
    )
    assert hidpi.ui_scale == 1.5
    assert (hidpi.capsule_rect[3] - hidpi.capsule_rect[1]) > (narrow.capsule_rect[3] - narrow.capsule_rect[1])
    _assert_non_overlapping(hidpi)


def test_u05_viewport_dock_all_muted_and_no_output_guards() -> None:
    muted = viewport_hud.compute_dock_layout(
        1280.0,
        "APPLE_PHASE_B",
        active_cutters=0,
        total_cutters=5,
        has_output_enabled=True,
    )
    apply_m = next(b for b in muted.buttons if b.slot_id == "apply")
    assert apply_m.enabled is False and apply_m.style == "muted"

    no_out = viewport_hud.compute_dock_layout(
        1280.0,
        "APPLE_PHASE_B",
        active_cutters=5,
        total_cutters=5,
        has_output_enabled=False,
    )
    apply_n = next(b for b in no_out.buttons if b.slot_id == "apply")
    assert apply_n.enabled is False and apply_n.style == "muted"


def test_u06_popovers_draw_and_idle_dock_poll() -> None:
    from FullDestruc_Tool.ui.panel_n import (
        FDT_PT_popover_output,
        FDT_PT_popover_rough,
        FDT_PT_popover_settings,
        FDT_PT_popover_zones,
    )

    class _MockPopLayout:
        def __init__(self):
            self.props = []
            self.ops = []
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.enabled = True
            self.alert = False

        def row(self, align=False):
            return self

        def column(self, align=False):
            return self

        def box(self):
            return self

        def grid_flow(self, **kwargs):
            return self

        def separator(self, **kwargs):
            return None

        def label(self, text="", **kwargs):
            return None

        def prop(self, data, prop_name, **kwargs):
            assert hasattr(data, prop_name), prop_name
            self.props.append(prop_name)

        def operator(self, op_id, text="", **kwargs):
            self.ops.append(op_id)

            class _P:
                pass

            return _P()

    for cls in (
        FDT_PT_popover_settings,
        FDT_PT_popover_zones,
        FDT_PT_popover_rough,
        FDT_PT_popover_output,
    ):
        dummy = type("_Dummy", (), {})()
        lay = _MockPopLayout()
        dummy.layout = lay
        cls.draw(dummy, bpy.context)

    # Idle scene must keep 3D Viewport 100% clean (dock poll returns False)
    from FullDestruc_Tool.core import rebar_state, session as sess_mod

    if sess_mod.ACTIVE_SESSION_PROP in bpy.context.scene:
        del bpy.context.scene[sess_mod.ACTIVE_SESSION_PROP]
    bpy.context.scene.fdt_rebar_settings.rebar_active_sid = ""
    if rebar_state.REGISTRY_PROP in bpy.context.scene:
        del bpy.context.scene[rebar_state.REGISTRY_PROP]
    assert viewport_hud.FDT_GGT_viewport_dock.poll(bpy.context) is False


ALL_TESTS = [
    test_u01_custom_icons_exist_and_are_64x64_png,
    test_u02_icons_helpers_and_zone_color_matching,
    test_u03_viewport_dock_progressive_phases_and_rebar,
    test_u04_viewport_dock_responsive_narrow_and_dpi,
    test_u05_viewport_dock_all_muted_and_no_output_guards,
    test_u06_popovers_draw_and_idle_dock_poll,
]
