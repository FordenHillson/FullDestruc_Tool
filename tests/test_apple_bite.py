"""Automated Blender 5.2 headless tests for FullDestruc_Tool Phase 1 (Apple Bite v2)."""

from __future__ import annotations

import bmesh
import bpy
from mathutils import Vector

import FullDestruc_Tool as fdt
from FullDestruc_Tool.core import apple_bite, session, vault, zone_shield
from FullDestruc_Tool.ui import qt_bridge


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Re-ensure addon properties are available on fresh scene
    if not hasattr(bpy.context.scene, "fdt_settings"):
        fdt.register()


def _make_box(name: str, location=(0.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0)) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.scale = Vector(scale)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_1_enter_exit_cancel_lifecycle() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Wall_Test", location=(0, 0, 0), scale=(4, 0.5, 3))

    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, f"Enter failed: {msg}"
    assert session.has_active_session(ctx.scene)

    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=(0.5, 0, 0), size=1.5, zone="A")
    assert ok and cutter is not None, f"Add box cutter failed: {msg}"

    ok, msg = apple_bite.exit_session(ctx)
    assert ok, f"Exit failed: {msg}"
    assert not session.has_active_session(ctx.scene), "Session flag stayed active after exit!"

    # Start another session on a new object and cancel it to verify clean restore
    pillar = _make_box("Pillar_Test", location=(5, 0, 0), scale=(1, 1, 3))
    ok, msg, sess2 = apple_bite.enter_session(ctx, target_objects=[pillar])
    assert ok and sess2 is not None, f"Second enter failed: {msg}"

    ok, msg = apple_bite.cancel_session(ctx)
    assert ok, f"Cancel failed: {msg}"
    assert not session.has_active_session(ctx.scene)
    assert pillar.name in ctx.scene.collection.objects, "Original object was not restored on cancel!"
    assert not zone_shield.is_protected_object(pillar), "Protected flag should be removed on restore!"


def test_2_multi_object_cutting_and_zone_tagging() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Building_Wall", location=(0, 0, 0), scale=(4, 0.4, 3))
    pillar = _make_box("Building_Pillar", location=(0.5, 0, 0), scale=(0.6, 0.6, 3))
    floor = _make_box("Building_Floor", location=(0, 0, -1.2), scale=(4, 4, 0.4))

    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall, pillar, floor])
    assert ok and sess is not None, f"Multi-object enter failed: {msg}"
    assert len(sess.target_names) == 3

    # Box cutter tagged Zone A intersecting Wall & Pillar
    ok, msg, c_box = apple_bite.add_box_cutter(ctx, location=(0.5, 0, 0), size=1.6, zone="A")
    assert ok and c_box is not None, f"Box cutter failed: {msg}"

    # Custom mesh cutter tagged Zone B intersecting Floor
    custom_mesh = _make_box("MyCustomCutter", location=(0, 0, -1.2), scale=(1.5, 1.5, 1.5))
    ok, msg, added = apple_bite.add_custom_mesh_cutters(ctx, [custom_mesh], zone="B")
    assert ok and len(added) == 1, f"Custom mesh cutter failed: {msg}"

    ok, msg, baked = apple_bite.apply_session(ctx, keep_result=True, keep_pieces=True)
    assert ok, f"Apply failed: {msg}"
    names = [o.name for o in baked]
    assert any("_FDST_ID-A" in n for n in names), f"Missing Zone A pieces in {names}"
    assert any("_FDST_ID-B" in n for n in names), f"Missing Zone B pieces in {names}"
    assert any(n.endswith("_Cut") for n in names), f"Missing Cut main meshes in {names}"


def test_3_apply_with_excluded_vault_and_blender52_attributes() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Concrete_Wall_FDST_ID-Z", location=(0, 0, 0), scale=(4, 0.5, 3))

    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, f"Enter failed: {msg}"

    # Enable rough edge to test FDT_GN_RoughCutter in Blender 5.2
    ctx.scene.fdt_settings.enable_rough = True
    ctx.scene.fdt_settings.rough_strength = 0.05
    ctx.scene.fdt_settings.rough_voxel_size = 0.08

    ok, msg, _cutter = apple_bite.add_box_cutter(ctx, location=(0, 0, 0), size=1.8, zone="C")
    assert ok, f"Add cutter failed: {msg}"

    # Explicitly exclude AND hide Original Vault (reproducing the exact Red_Panda Apply bug!)
    cols = sess.get_collections()
    vault_col = cols["vault"]
    vault_lc = vault.find_layer_collection(ctx.view_layer, vault_col)
    assert vault_col is not None and vault_lc is not None
    vault_col.hide_viewport = True
    vault_lc.hide_viewport = True
    vault_lc.exclude = True
    ctx.view_layer.update()

    ok, msg, baked = apple_bite.apply_session(ctx, keep_result=True, keep_pieces=True)
    assert ok, f"Apply with excluded vault failed: {msg}"
    assert len(baked) >= 2, f"Expected both Cut main mesh and Zone C piece, got {len(baked)}"

    # Verify vault stays hidden after apply
    assert vault_col.hide_viewport is True, "Vault should remain hidden after Apply!"

    # Verify Blender 5.2 attributes: UVMap0, cut material, and freestyle_edge
    piece = next((o for o in baked if "_FDST_ID-C" in o.name), None)
    assert piece is not None, f"Zone C piece missing from {[o.name for o in baked]}"
    mesh = piece.data
    assert len(mesh.polygons) > 0, "Baked piece has 0 polygons!"
    assert "UVMap0" in mesh.attributes or "UVMap0" in mesh.uv_layers, "UVMap0 missing on baked piece!"
    fs_attr = mesh.attributes.get("freestyle_edge")
    assert fs_attr is not None and fs_attr.domain == "EDGE", "Blender 5.2 freestyle_edge attribute missing!"
    fs_vals = [False] * len(mesh.edges)
    fs_attr.data.foreach_get("value", fs_vals)
    assert any(fs_vals), "No cut border edges were marked as freestyle_edge!"


def test_4_protected_vault_shield() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("House_Wall_FDST_ID-A", location=(0, 0, 0), scale=(3, 0.4, 3))

    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, f"Enter failed: {msg}"

    cols = sess.get_collections()
    vault_col = cols["vault"]
    assert zone_shield.is_protected_collection(vault_col)
    assert zone_shield.is_protected_object(wall)

    # Simulate a Zone Visibility pass checking all objects in view_layer
    controllable = zone_shield.filter_unprotected_objects(ctx.view_layer.objects)
    assert wall not in controllable, "Protected original leaked into Zone Visibility!"
    assert vault_col.hide_viewport is True, "Protected vault should remain hidden!"


def test_5_registration_reload_and_pyside_check() -> None:
    # Ensure clean unregister -> register cycle (no modal/handler leaks)
    fdt.unregister()
    fdt.register()
    assert hasattr(bpy.types.Scene, "fdt_settings")
    # Note: --factory-startup skips user extension site-packages; verify bridge flag is boolean
    assert isinstance(qt_bridge.PYSIDE_AVAILABLE, bool)


def test_6_smart_step1_detection() -> None:
    _reset_scene()
    ctx = bpy.context

    # Nothing selected -> NONE
    info = apple_bite.detect_step1_action(ctx)
    assert info["mode"] == "NONE", f"Expected NONE, got {info}"

    # Normal mesh selected -> START
    wall = _make_box("Smart_Wall", location=(0, 0, 0), scale=(4, 0.5, 3))
    apple_bite.safe_select_objects(ctx, [wall], active=wall)
    info = apple_bite.detect_step1_action(ctx)
    assert info["mode"] == "START" and len(info["targets"]) == 1, f"Expected START, got {info}"

    # Enter, cut, apply -> baked mesh selected -> REEDIT!
    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, msg
    sid = sess.session_id
    ok, msg, _ = apple_bite.add_box_cutter(ctx, size=1.5, zone="A")
    assert ok, msg
    ok, msg, baked = apple_bite.apply_session(ctx)
    assert ok and len(baked) >= 2, msg

    apple_bite.safe_select_objects(ctx, [baked[0]], active=baked[0])
    info = apple_bite.detect_step1_action(ctx)
    assert info["mode"] == "BOTH" and info["session_id"] == sid, f"Expected BOTH for {sid}, got {info}"


def test_11_cut_further_on_already_applied_mesh() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("MultiStage_Wall", location=(0, 0, 0), scale=(6, 0.5, 3))

    # Stage 1: Cut Zone A and Apply
    ok, msg, sess1 = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess1 is not None, msg
    ok, msg, _ = apple_bite.add_box_cutter(ctx, location=(-1.5, 0, 0), size=1.4, zone="A")
    assert ok, msg
    ok, msg, baked1 = apple_bite.apply_session(ctx)
    assert ok and len(baked1) == 2, msg

    main_cut = next(o for o in baked1 if o.name.endswith("_Cut"))
    apple_bite.safe_select_objects(ctx, [main_cut], active=main_cut)
    info = apple_bite.detect_step1_action(ctx)
    assert info["mode"] == "BOTH" and info["session_id"] == sess1.session_id

    # Stage 2: Cut further on the already-applied main_cut into Zone B!
    ok, msg, sess2 = apple_bite.enter_session(ctx, target_objects=info["targets"])
    assert ok and sess2 is not None and sess2.session_id != sess1.session_id, msg
    ok, msg, _ = apple_bite.add_box_cutter(ctx, location=(1.5, 0, 0), size=1.4, zone="B")
    assert ok, msg
    ok, msg, baked2 = apple_bite.apply_session(ctx)
    assert ok and any("_FDST_ID-B" in o.name for o in baked2), f"Missing Zone B in {[o.name for o in baked2]}"


def test_7_cutter_spawns_at_building_center_with_zone_color() -> None:
    _reset_scene()
    ctx = bpy.context
    ctx.scene.cursor.location = Vector((0.0, 0.0, 0.0))
    wall = _make_box("Far_Wall", location=(10.0, 5.0, 2.0), scale=(4, 0.5, 3))

    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=None, size=1.6, zone="B")
    assert ok and cutter is not None, msg
    dist = (cutter.location - Vector((10.0, 5.0, 2.0))).length
    assert dist < 0.01, f"Expected cutter at building center (10,5,2), got {cutter.location}"
    expected_b = apple_bite.get_zone_color("B")
    assert abs(cutter.color[2] - expected_b[2]) < 0.01, f"Expected Zone B blue color, got {tuple(cutter.color)}"


def test_8_cutter_list_mute_cycle_delete() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Manage_Wall", location=(0, 0, 0), scale=(4, 0.5, 3))
    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    ok, msg, c1 = apple_bite.add_box_cutter(ctx, size=1.5, zone="A")
    assert ok and c1 is not None, msg
    c1_name = c1.name

    # Cycle zone A -> B — cutter is renamed to carry the new zone letter
    result = apple_bite.cycle_cutter_zone(ctx, c1_name)
    assert result is not None and result[0] == "B", f"Expected zone B, got {result}"
    assert "_ID-B_" in result[1], f"Expected cutter renamed with zone B, got {result[1]}"
    c1_name = result[1]

    # Mute cutter -> summary shows 0 active cutters, 1 muted
    ok, is_muted = apple_bite.toggle_cutter_mute(ctx, c1_name)
    assert ok and is_muted is True
    summary = apple_bite.get_apply_preview_summary(ctx.scene)
    assert summary["active_cutters"] == 0 and summary["muted_cutters"] == 1, f"Bad summary: {summary}"

    # Unmute -> 1 active
    ok, is_muted = apple_bite.toggle_cutter_mute(ctx, c1_name)
    assert ok and is_muted is False
    summary = apple_bite.get_apply_preview_summary(ctx.scene)
    assert summary["active_cutters"] == 1 and summary["zone_counts"].get("B") == 1, f"Bad summary: {summary}"

    # Delete cutter -> 0 cutters left
    ok, msg = apple_bite.delete_cutter(ctx, c1_name)
    assert ok, msg
    assert c1_name not in bpy.data.objects
    assert len(session.load_session(ctx.scene).cutters) == 0


def test_9_reedit_after_apply_overwrites_final_cleanly() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Reedit_Wall", location=(0, 0, 0), scale=(6, 0.5, 3))
    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, msg
    sid = sess.session_id

    ok, msg, _c1 = apple_bite.add_box_cutter(ctx, location=(-1.2, 0, 0), size=1.4, zone="A")
    assert ok, msg
    ok, msg, baked1 = apple_bite.apply_session(ctx)
    assert ok and len(baked1) == 2, f"Expected 2 objects on first Apply, got {len(baked1)}"

    # Re-edit session: Final collection must be hidden, Vault MUST stay hidden!
    ok, msg, resumed = apple_bite.resume_session(ctx, sid)
    assert ok and resumed is not None, msg
    cols = resumed.get_collections()
    assert cols["final"] is not None and cols["final"].hide_viewport is True, "Final col should hide during re-edit!"
    assert cols["vault"] is not None and cols["vault"].hide_viewport is True, "Vault col MUST stay hidden!"

    # Add second cutter Zone B and re-Apply -> cleanly overwrites Final (3 objects, no duplicates!)
    ok, msg, _c2 = apple_bite.add_box_cutter(ctx, location=(1.2, 0, 0), size=1.4, zone="B")
    assert ok, msg
    ok, msg, baked2 = apple_bite.apply_session(ctx)
    assert ok and len(baked2) == 3, f"Expected 3 objects (Cut + Zone A + Zone B), got {len(baked2)}"
    final_col = bpy.data.collections.get(f"FDT_Final_{sid}")
    assert final_col is not None and final_col.hide_viewport is False
    assert len(final_col.objects) == 3, f"Old baked objects were not overwritten! Count={len(final_col.objects)}"


class _MockUILayout:
    """Minimal mock of bpy.types.UILayout to catch NameError/AttributeError in Panel.draw()."""

    def __init__(self):
        self.operators: list[str] = []
        self.scale_y = 1.0
        self.enabled = True

    def box(self):
        return self

    def row(self, align=False):
        return self

    def column(self, align=False):
        return self

    def split(self, factor=0.0, align=False):
        return self

    def grid_flow(self, **kwargs):
        return self

    def label(self, text="", icon="NONE"):
        return None

    def prop(self, data, prop_name, **kwargs):
        assert hasattr(data, prop_name), f"Missing property '{prop_name}' on {data}"
        return None

    def operator(self, op_id, text="", **kwargs):
        self.operators.append(op_id)

        class _OpProps:
            pass

        return _OpProps()


def test_10_panel_draw_renders_apply_button_without_errors() -> None:
    from FullDestruc_Tool.ui.panel_n import FDT_PT_main_panel

    class _DummyPanel:
        _draw_step1_idle = FDT_PT_main_panel._draw_step1_idle
        _draw_step2_and_step3_active = FDT_PT_main_panel._draw_step2_and_step3_active

    _reset_scene()
    ctx = bpy.context

    # 1. Idle state draw
    panel = _DummyPanel()
    mock_idle = _MockUILayout()
    panel.layout = mock_idle
    FDT_PT_main_panel.draw(panel, ctx)
    assert "fdt.applebite_smart_start" in mock_idle.operators

    # 2. Active session with cutter -> MUST render Apply, Exit, Cancel without NameError!
    wall = _make_box("Panel_Wall", location=(0, 0, 0), scale=(4, 0.5, 3))
    ok, msg, _ = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg
    ok, msg, _ = apple_bite.add_box_cutter(ctx, size=1.5, zone="A")
    assert ok, msg

    mock_active = _MockUILayout()
    panel.layout = mock_active
    FDT_PT_main_panel.draw(panel, ctx)
    assert "fdt.applebite_apply" in mock_active.operators, "Apply button was NOT drawn in N-Panel!"
    assert "fdt.applebite_exit" in mock_active.operators, "Exit button was NOT drawn in N-Panel!"
    assert "fdt.applebite_cancel" in mock_active.operators, "Cancel button was NOT drawn in N-Panel!"


def test_12_adjacent_box_cutter_overlap_no_sliver() -> None:
    _reset_scene()
    ctx = bpy.context
    # 4m x 2m x 2m block split completely into 2 flush zones (Zone A + Zone B) -> 0 leftover main mesh!
    wall = _make_box("Flush_Wall", location=(0.0, 0.0, 0.0), scale=(4.0, 2.0, 2.0))
    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    # Place Cutter A covering left half (-1.0, 0, 0), width 2.2m (slightly overhanging outer wall)
    ok, msg, c_a = apple_bite.add_box_cutter(ctx, location=(-1.05, 0.0, 0.0), size=2.2, zone="A")
    assert ok and c_a is not None, msg

    # Add adjacent Cutter B along +X -> auto-overlaps by 0.05m and auto-advances to Zone B!
    ok, msg, c_b = apple_bite.add_adjacent_box_cutter(ctx, direction="+X", overlap=0.05, auto_next_zone=True)
    assert ok and c_b is not None, msg
    assert c_b.get("fdt_cutter_zone") == "B", f"Expected auto-next Zone B, got {c_b.get('fdt_cutter_zone')}"
    dx = c_b.matrix_world.translation.x - c_a.matrix_world.translation.x
    assert abs(dx - (2.2 - 0.05)) < 0.01, f"Expected step 2.15m, got {dx}"

    # Apply: because A and B overlap by 5cm and cover the full 4m wall, ZERO leftover _Cut sliver!
    ok, msg, baked = apple_bite.apply_session(ctx, keep_result=True, keep_pieces=True)
    assert ok, msg
    names = [o.name for o in baked]
    assert not any(n.endswith("_Cut") for n in names), f"Unexpected leftover sliver in {names}"
    assert any("_FDST_ID-A" in n for n in names) and any("_FDST_ID-B" in n for n in names), f"Missing zones in {names}"


def test_13_cutter_gear_gizmo_and_wheel_menu() -> None:
    from FullDestruc_Tool.ui.cutter_gizmo import (
        FDT_GGT_cutter_gear,
        FDT_MT_cutter_wheel,
        _compute_axis_segments,
    )

    _reset_scene()
    ctx = bpy.context
    wall = _make_box("Gear_Wall", location=(0, 0, 0), scale=(4, 2, 2))
    assert FDT_GGT_cutter_gear.poll(ctx) is False, "Gear gizmo should NOT show before entering session"

    ok, msg, _ = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg
    ok, msg, cutter = apple_bite.add_box_cutter(ctx, size=1.5, zone="A")
    assert ok and cutter is not None, msg

    # Active object is now cutter -> Gear gizmo MUST poll True!
    assert FDT_GGT_cutter_gear.poll(ctx) is True, "Gear gizmo should show when cutter is active!"

    # 3D axis segments should compute all 6 directions cleanly
    segs = _compute_axis_segments(cutter)
    assert len(segs) == 6 and {s[0] for s in segs} == {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}

    # Wheel menu draw smoke test
    class _MockPieLayout(_MockUILayout):
        def menu_pie(self):
            return self

    class _DummyMenu:
        pass

    m = _DummyMenu()
    mock_pie = _MockPieLayout()
    m.layout = mock_pie
    FDT_MT_cutter_wheel.draw(m, ctx)
    assert "fdt.applebite_pick_axis" in mock_pie.operators, "Wheel menu missing 3D axis picker!"


def test_14_custom_radial_wheel_hit_test_and_geometry() -> None:
    import math
    from FullDestruc_Tool.ui import wheel_menu

    cx, cy = 500.0, 400.0
    # Center deadzone -> None
    assert wheel_menu.hit_test_wheel(cx, cy, cx, cy) is None
    assert wheel_menu.hit_test_wheel(cx + 10.0, cy - 5.0, cx, cy) is None

    # Far outside -> None
    assert wheel_menu.hit_test_wheel(cx + 400.0, cy, cx, cy) is None

    # Slice 0 (top-right, ~60 deg) -> index 0 ("pick_axis")
    r_mid = (wheel_menu.INNER_RADIUS + wheel_menu.OUTER_RADIUS) * 0.5
    x0 = cx + r_mid * math.cos(math.radians(60.0))
    y0 = cy + r_mid * math.sin(math.radians(60.0))
    assert wheel_menu.hit_test_wheel(x0, y0, cx, cy) == 0
    assert wheel_menu.WHEEL_ITEMS[0].id == "pick_axis"

    # Geometry builders produce non-empty triangles & outlines
    tris, outline = wheel_menu.build_wedge_geometry(cx, cy, 44.0, 132.0, 0.0, 1.0, steps=16)
    assert len(tris) == 16 * 6 and len(outline) == 34
    hub_tris, hub_ring = wheel_menu.build_disk_geometry(cx, cy, 40.0, steps=24)
    assert len(hub_tris) == 24 * 3 and len(hub_ring) == 24


def test_15_wheel_zone_subring_hit_test() -> None:
    import math
    from FullDestruc_Tool.ui import wheel_menu

    cx, cy = 500.0, 400.0

    # Verify ALL 26 Zones (A-M inner tier, N-Z outer tier) hit cleanly at their midpoint
    tested = set()
    for expected_zone, r_in, r_out, a0, a1 in wheel_menu._iter_sub_zone_cells():
        r_mid = (r_in + r_out) * 0.5
        amid = (a0 + a1) * 0.5
        mx = cx + r_mid * math.cos(amid)
        my = cy + r_mid * math.sin(amid)
        hit = wheel_menu.hit_test_zone_subring(mx, my, cx, cy)
        assert hit == expected_zone, f"Expected sub-ring zone {expected_zone}, got {hit}"
        tested.add(hit)

    assert len(tested) == 26, f"Expected all 26 A-Z zones, got {len(tested)}"

    # Left side (180 deg) should NOT hit Zone sub-ring
    assert wheel_menu.hit_test_zone_subring(cx - 170.0, cy, cx, cy) is None


def test_16_view_mode_wheel_hit_tests() -> None:
    import math

    from FullDestruc_Tool.ui import wheel_menu

    cx, cy = 500.0, 400.0
    r_mid = (wheel_menu.VIEW_WHEEL_INNER + wheel_menu.VIEW_WHEEL_OUTER) * 0.5

    def at(deg: float, radius: float = r_mid) -> tuple[float, float]:
        a = math.radians(deg)
        return cx + radius * math.cos(a), cy + radius * math.sin(a)

    # Center deadzone and far outside -> nothing
    assert wheel_menu.hit_test_view_wheel(cx, cy, cx, cy) is None
    assert wheel_menu.hit_test_view_wheel(*at(90.0, radius=10.0), cx, cy) is None
    assert wheel_menu.hit_test_view_wheel(*at(90.0, radius=400.0), cx, cy) is None

    # Three dedicated wedges: All (top), Piece Only (left), Hole Only (right/bottom)
    for deg, expected in ((90.0, "ALL"), (210.0, "PIECES"), (330.0, "MAIN")):
        mx, my = at(deg)
        assert wheel_menu.hit_test_view_wheel(mx, my, cx, cy) == expected, f"At {deg} deg"

    # Hole splits into Solid / Wire; other wedges are not sub-modes
    for deg, expected in ((300.0, "MAIN"), (0.0, "MAIN_WIRE")):
        mx, my = at(deg)
        assert wheel_menu.hit_test_hole_split(mx, my, cx, cy) == expected, f"Hole at {deg} deg"
    assert wheel_menu.hit_test_hole_split(*at(90.0), cx, cy) is None
    assert wheel_menu.hit_test_hole_split(*at(210.0), cx, cy) is None


def test_17_per_cutter_rough_cut_independence() -> None:
    _reset_scene()
    ctx = bpy.context
    wall = _make_box("SplitRough_Wall", location=(0, 0, 0), scale=(6, 0.5, 3))
    ok, msg, _ = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    # Add two cutters while global rough is OFF -> both start smooth
    ok, msg, c1 = apple_bite.add_box_cutter(ctx, location=(-1.2, 0, 0), size=1.4, zone="A")
    assert ok and c1 is not None, msg
    ok, msg, c2 = apple_bite.add_box_cutter(ctx, location=(1.2, 0, 0), size=1.4, zone="B")
    assert ok and c2 is not None, msg

    assert c1.modifiers.get("FDT_RoughCutter") is None
    assert c2.modifiers.get("FDT_RoughCutter") is None

    # Turn ON Rough ONLY for c1 -> c1 gets FDT_RoughCutter, c2 stays flat/smooth!
    ok, is_rough = apple_bite.toggle_cutter_rough(ctx, c1.name)
    assert ok and is_rough is True
    assert c1.modifiers.get("FDT_RoughCutter") is not None, "c1 should have FDT_RoughCutter!"
    assert c2.modifiers.get("FDT_RoughCutter") is None, "c2 MUST stay smooth!"

    # Toggle c1 OFF again -> both smooth
    ok, is_rough = apple_bite.toggle_cutter_rough(ctx, c1.name)
    assert ok and is_rough is False
    assert c1.modifiers.get("FDT_RoughCutter") is None


def test_18_scene_zone_registry_custom_tokens() -> None:
    from FullDestruc_Tool.core import zone_registry

    _reset_scene()
    ctx = bpy.context
    scene = ctx.scene

    zones = zone_registry.load_zones(scene)
    assert len(zones) == 26 and zones[0].key == "A"

    ok, msg, key = zone_registry.create_zone("ID-A#B", "A", "ID", scene=scene)
    assert ok, msg
    zdef = zone_registry.find_zone(key, scene)
    assert zdef is not None and zdef.token == "ID-A#B"
    assert zdef.base == "A" and zdef.kind == "ID"
    assert tuple(zdef.color) == apple_bite.ZONE_COLORS["A"]

    labels = {k: token for k, token, _desc in zone_registry.iter_zone_items(scene)}
    assert labels[key] == "ID-A#B"

    ok_dup, _msg_dup, _ = zone_registry.create_zone("id-a#b", "B", "ID", scene=scene)
    assert not ok_dup, "Token uniqueness must be case-insensitive"

    ok_long, msg_long, _ = zone_registry.create_zone("X" * 41, "A", "ID", scene=scene)
    assert not ok_long and "40" in msg_long

    custom_color = (0.10, 0.20, 0.30, 1.0)
    ok, msg = zone_registry.update_zone(key, token="ID-B#7", base="B", color=custom_color, scene=scene)
    assert ok, msg
    zdef = zone_registry.find_zone(key, scene)
    assert zdef is not None and zdef.token == "ID-B#7" and zdef.base == "B"
    assert tuple(zdef.color) == custom_color

    ok, msg = zone_registry.update_zone(key, base="C", scene=scene)
    assert ok, msg
    zdef = zone_registry.find_zone(key, scene)
    assert zdef is not None and tuple(zdef.color) == custom_color, "Changing base must not jump the color"

    wall = _make_box("ZoneRegistryWall", location=(0, 0, 0), scale=(4, 0.5, 3))
    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg
    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=(0, 0, 1), size=1.2, zone=key)
    assert ok and cutter is not None, msg
    assert cutter.get(apple_bite.CUTTER_ZONE_PROP) == key
    assert "ID-B#7" in cutter.name, cutter.name

    ok_del, msg_del = zone_registry.delete_zone(key, scene)
    assert not ok_del and "cutter" in msg_del

    assert zone_registry.resolve_zone("Q", scene).key == "Q"
    assert zone_registry.resolve_zone("missing-key", scene).key == "A"
    assert zone_registry.next_zone_key("Z", scene) == key

    baked = apple_bite._format_zone_name("Wall", "ID-A#B")
    assert baked.endswith("_FDST_ID-ID-A#B"), baked

    assert apple_bite.set_cutters_zone(ctx, [cutter], "Z") == 1
    cycled = apple_bite.cycle_cutter_zone(ctx, cutter.name)
    assert cycled is not None and cycled[0] == key and "ID-B#7" in cycled[1]


def test_19_builtin_zone_rename_survives_reload() -> None:
    from FullDestruc_Tool.core import zone_registry

    _reset_scene()
    scene = bpy.context.scene
    custom_color = (0.11, 0.22, 0.33, 1.0)

    ok, msg = zone_registry.update_zone("A", token="ID-A#B", color=custom_color, scene=scene)
    assert ok, msg
    reloaded = zone_registry.find_zone("A", scene)
    assert reloaded is not None and reloaded.token == "ID-A#B"
    assert reloaded.builtin is True and tuple(reloaded.color) == custom_color
    assert zone_registry.resolve_zone("ID-A#B", scene).key == "A"
    assert zone_registry.resolve_zone("not-a-zone", scene).key == "A"

    ok_del, msg_del = zone_registry.delete_zone("A", scene)
    assert not ok_del and "fallback" in msg_del

    ok_dup, _msg_dup = zone_registry.update_zone("B", token="id-a#b", scene=scene)
    assert not ok_dup, "Case-insensitive uniqueness must protect renamed built-ins"


def test_20_edge_band_inward_rough_cut() -> None:
    from FullDestruc_Tool.core import geonodes

    _reset_scene()
    ctx = bpy.context
    wall = _make_box("EdgeRough_Wall", location=(0, 0, 0), scale=(6, 0.5, 3))
    ok, msg, _ = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    # 2x2x2 cube centered at origin -> local bounds [-1, +1] on every axis
    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=(0, 0, 0), size=2.0, zone="A")
    assert ok and cutter is not None, msg

    settings = ctx.scene.fdt_settings
    settings.rough_inward_ratio = 0.35
    settings.rough_strength = 0.15
    settings.rough_scale = 3.0
    settings.rough_detail = 0.5
    settings.rough_subdiv = 3
    settings.rough_seed = 1.0

    ok, is_rough = apple_bite.toggle_cutter_rough(ctx, cutter.name)
    assert ok and is_rough is True

    mod = cutter.modifiers.get("FDT_RoughCutter")
    assert mod is not None and mod.node_group is not None
    assert int(mod.node_group.get("fdt_gn_version", 1)) == geonodes.ROUGH_CUTTER_VERSION

    input_names = {
        item.name
        for item in mod.node_group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }
    for expected in ("Geometry", "Inward Ratio", "Strength", "Noise Scale", "Noise Detail", "Subdiv Level", "Seed"):
        assert expected in input_names, f"Missing socket: {expected}"

    def _eval_local_verts():
        deps = ctx.evaluated_depsgraph_get()
        eval_obj = cutter.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        try:
            return [tuple(v.co) for v in mesh.vertices]
        finally:
            eval_obj.to_mesh_clear()

    # 1. Inward-only guarantee: no vertex may bulge outside [-1.0, +1.0]
    verts = _eval_local_verts()
    assert len(verts) > 200, f"Expected subdivided cutter verts, got {len(verts)}"
    max_abs = max(max(abs(x), abs(y), abs(z)) for x, y, z in verts)
    assert max_abs <= 1.0005, f"Rough cutter bulged outward beyond box bounds: max_abs={max_abs:.5f}"

    # 2. Edge band: center of top face (|x|, |y| < 0.2, z > 0.5) must remain flat (z ~ 1.0),
    # while edge vertices must carve noticeably inward
    center_top = [z for x, y, z in verts if abs(x) < 0.2 and abs(y) < 0.2 and z > 0.5]
    assert center_top, "Expected center vertices on top face"
    assert max(abs(1.0 - z) for z in center_top) < 1e-3, (
        f"Center of face should stay smooth at Inward Ratio=0.35, got min z={min(center_top):.4f}"
    )

    edge_inward = max(1.0 - max(abs(x), abs(y), abs(z)) for x, y, z in verts)
    assert edge_inward > 0.01, f"Expected noticeable inward carving near edges, got {edge_inward:.5f}"

    # 3. Inward Ratio = 0.0 -> zero carving everywhere
    settings.rough_inward_ratio = 0.0
    apple_bite.rebuild_session(ctx)
    verts_zero = _eval_local_verts()
    max_dev_zero = max(abs(1.0 - max(abs(x), abs(y), abs(z))) for x, y, z in verts_zero)
    assert max_dev_zero < 1e-4, f"Expected zero carving at Inward Ratio=0.0, got {max_dev_zero:.5f}"

    # 4. Inward Ratio = 1.0 -> carving reaches inner area of the face
    settings.rough_inward_ratio = 1.0
    apple_bite.rebuild_session(ctx)
    verts_full = _eval_local_verts()
    inner_top = [z for x, y, z in verts_full if abs(x) < 0.35 and abs(y) < 0.35 and z > 0.5]
    assert inner_top and max(1.0 - z for z in inner_top) > 0.005, (
        "Expected inward carving to reach inner face region at Inward Ratio=1.0"
    )


ALL_TESTS = [
    test_1_enter_exit_cancel_lifecycle,
    test_2_multi_object_cutting_and_zone_tagging,
    test_3_apply_with_excluded_vault_and_blender52_attributes,
    test_4_protected_vault_shield,
    test_5_registration_reload_and_pyside_check,
    test_6_smart_step1_detection,
    test_7_cutter_spawns_at_building_center_with_zone_color,
    test_8_cutter_list_mute_cycle_delete,
    test_9_reedit_after_apply_overwrites_final_cleanly,
    test_10_panel_draw_renders_apply_button_without_errors,
    test_11_cut_further_on_already_applied_mesh,
    test_12_adjacent_box_cutter_overlap_no_sliver,
    test_13_cutter_gear_gizmo_and_wheel_menu,
    test_14_custom_radial_wheel_hit_test_and_geometry,
    test_15_wheel_zone_subring_hit_test,
    test_16_view_mode_wheel_hit_tests,
    test_17_per_cutter_rough_cut_independence,
    test_18_scene_zone_registry_custom_tokens,
    test_19_builtin_zone_rename_survives_reload,
    test_20_edge_band_inward_rough_cut,
]
