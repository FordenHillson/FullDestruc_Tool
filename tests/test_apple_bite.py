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

    def template_list(self, listtype_name, list_id, dataptr, propname, active_dataptr, active_propname, **kwargs):
        assert hasattr(dataptr, propname), f"Missing list property '{propname}' on {dataptr}"
        assert hasattr(active_dataptr, active_propname), f"Missing index property '{active_propname}' on {active_dataptr}"
        self.lists = getattr(self, "lists", [])
        self.lists.append((listtype_name, kwargs.get("rows"), kwargs.get("maxrows")))
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

    # 2. Active session with cutter -> MUST render scrollable UIList, Apply, Exit, Cancel without NameError!
    wall = _make_box("Panel_Wall", location=(0, 0, 0), scale=(4, 0.5, 3))
    ok, msg, _ = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg
    ok, msg, c1 = apple_bite.add_box_cutter(ctx, size=1.5, zone="A")
    assert ok and c1 is not None, msg
    assert len(ctx.scene.fdt_settings.cutter_layer_items) == 1
    assert ctx.scene.fdt_settings.cutter_layer_items[0].name == c1.name

    mock_active = _MockUILayout()
    panel.layout = mock_active
    FDT_PT_main_panel.draw(panel, ctx)
    assert any(lt == "FDT_UL_cutter_layers" for lt, _, _ in getattr(mock_active, "lists", [])), (
        "Scrollable FDT_UL_cutter_layers UIList was NOT drawn in N-Panel!"
    )
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
    assert "_ID-Z_" in cutter.name, f"Expected cutter renamed to Zone Z, got {cutter.name}"
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

    # Two independent 2x2x2 box cutters
    ok, msg, c1 = apple_bite.add_box_cutter(ctx, location=(-1.5, 0, 0), size=2.0, zone="A")
    assert ok and c1 is not None, msg
    ok, msg, c2 = apple_bite.add_box_cutter(ctx, location=(1.5, 0, 0), size=2.0, zone="B")
    assert ok and c2 is not None, msg

    ok, r1 = apple_bite.toggle_cutter_rough(ctx, c1.name)
    assert ok and r1 is True
    ok, r2 = apple_bite.toggle_cutter_rough(ctx, c2.name)
    assert ok and r2 is True

    # Configure c1 and c2 with different per-cutter rough settings
    c1.fdt_rough.inward_depth = 0.0
    c1.fdt_rough.jaggedness = 0.45
    c1.fdt_rough.noise_scale = 1.5
    c1.fdt_rough.detail = 2
    c1.fdt_rough.seed = 1.0

    c2.fdt_rough.inward_depth = 0.5
    c2.fdt_rough.jaggedness = 0.65
    c2.fdt_rough.noise_scale = 2.5
    c2.fdt_rough.detail = 1
    c2.fdt_rough.seed = 7.0

    apple_bite.rebuild_session(ctx)

    mod1 = c1.modifiers.get("FDT_RoughCutter")
    assert mod1 is not None and mod1.node_group is not None
    assert int(mod1.node_group.get("fdt_gn_version", 1)) == geonodes.ROUGH_CUTTER_VERSION

    input_names = {
        item.name
        for item in mod1.node_group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }
    for expected in ("Geometry", "Inward Depth", "Jaggedness", "Noise Scale", "Detail", "Seed"):
        assert expected in input_names, f"Missing socket: {expected}"

    def _eval_local_mesh(obj):
        deps = ctx.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        try:
            return [tuple(v.co) for v in mesh.vertices], len(mesh.polygons)
        finally:
            eval_obj.to_mesh_clear()

    # 1. Per-cutter independence: c1 (Detail=2, full size) vs c2 (Detail=1, scaled inward)
    v1_0, p1 = _eval_local_mesh(c1)
    v2, p2 = _eval_local_mesh(c2)
    assert 8 <= p1 <= 260, f"Expected low-poly planar cleavage facets on c1, got {p1}"
    assert 6 <= p2 <= 160, f"Expected ultra low-poly planar cleavage facets on c2, got {p2}"
    max_abs_1 = max(max(abs(x), abs(y), abs(z)) for x, y, z in v1_0)
    max_abs_2 = max(max(abs(x), abs(y), abs(z)) for x, y, z in v2)
    assert max_abs_1 <= 1.0005 and max_abs_1 > 0.85, f"c1 at Inward Depth=0 should span full box: {max_abs_1:.4f}"
    assert max_abs_2 < 0.65, f"c2 at Inward Depth=0.5 should scale inward (~0.575): {max_abs_2:.4f}"

    # 2. Inward Depth scales c1 uniformly inward while preserving jagged shape (ratio ~ 1 - 0.5*0.85 = 0.575)
    c1.fdt_rough.inward_depth = 0.5
    apple_bite.rebuild_session(ctx)
    v1_half, _ = _eval_local_mesh(c1)
    max_abs_1_half = max(max(abs(x), abs(y), abs(z)) for x, y, z in v1_half)
    expected_scale = 1.0 - 0.5 * 0.85
    assert abs((max_abs_1_half / max_abs_1) - expected_scale) < 0.03, (
        f"Expected uniform inward scale {expected_scale:.3f}, got {max_abs_1_half / max_abs_1:.3f}"
    )

    # 3. Rough wireframe visibility: active tweak shows WIRE; idle hides black rough wire as BOUNDS
    assert c1.display_type == "WIRE", f"Expected WIRE while tweaking rough settings, got {c1.display_type}"
    apple_bite._ROUGH_WIRE_PREVIEW_UNTIL.clear()
    apple_bite.rebuild_session(ctx)
    assert c1.display_type == "BOUNDS", f"Expected BOUNDS when rough cutter is idle, got {c1.display_type}"


def test_21_set_zone_renames_cutter_and_active_zone_dropdown() -> None:
    from FullDestruc_Tool.core import zone_registry
    from FullDestruc_Tool.props import set_active_zone_silent

    _reset_scene()
    ctx = bpy.context
    scene = ctx.scene

    # Target already carries _FDST_ID-C_LOD0 so we verify _FDST_ID- is never mangled
    wall = _make_box("PubVillage_E_2L01_V2_dst_FDST_ID-C_LOD0", location=(0, 0, 0), scale=(4, 0.5, 3))
    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg

    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=(0, 0, 0), size=1.5, zone="B")
    assert ok and cutter is not None, msg
    assert "_ID-B_" in cutter.name, f"Expected initial _ID-B_ name, got {cutter.name}"

    # 1. set_cutters_zone (used by N-panel buttons, wheel, zone picker, Qt) renames cutter & working pieces
    assert apple_bite.set_cutters_zone(ctx, [cutter], "A") == 1
    assert cutter.get(apple_bite.CUTTER_ZONE_PROP) == "A"
    assert "_ID-A_" in cutter.name and "_ID-B_" not in cutter.name, f"Cutter was not renamed: {cutter.name}"
    assert "_ID-A_" in cutter.data.name, f"Cutter mesh data was not renamed: {cutter.data.name}"
    sess_now = session.load_session(scene)
    assert sess_now is not None and sess_now.cutters[0].name == cutter.name and sess_now.cutters[0].zone == "A"

    pieces_col = sess_now.get_collections()["pieces"]
    if pieces_col is not None:
        for p in pieces_col.objects:
            assert p.get(apple_bite.PIECE_ZONE_PROP) == "A"
            assert "_FDST_ID-C_LOD0_ID-A_" in p.name, f"Piece name out of sync: {p.name}"

    # 2. Out-of-sync name healing: cutter already tagged A but still named _ID-B_01
    cutter.name = cutter.name.replace("_ID-A_", "_ID-B_")
    sess_now.cutters[0].name = cutter.name
    session.save_session(scene, sess_now)
    assert apple_bite.set_cutters_zone(ctx, [cutter], "A") == 1
    assert "_ID-A_" in cutter.name and "_ID-B_" not in cutter.name, f"Failed to heal stale name: {cutter.name}"

    # 3. active_zone dropdown updates selected cutter; set_active_zone_silent does NOT
    apple_bite.safe_select_objects(ctx, [cutter], active=cutter)
    scene.fdt_settings.active_zone = "D"
    assert cutter.get(apple_bite.CUTTER_ZONE_PROP) == "D"
    assert "_ID-D_" in cutter.name, f"active_zone dropdown did not rename selected cutter: {cutter.name}"

    set_active_zone_silent(scene.fdt_settings, "E")
    assert scene.fdt_settings.active_zone == "E"
    assert cutter.get(apple_bite.CUTTER_ZONE_PROP) == "D" and "_ID-D_" in cutter.name

    # 4. Renaming a zone token updates existing session cutters using that zone key
    ok_z, msg_z, custom_key = zone_registry.create_zone("ID-A#B", "A", "ID", scene=scene)
    assert ok_z, msg_z
    assert apple_bite.set_cutters_zone(ctx, [cutter], custom_key) == 1
    assert "_ID-ID-A#B_" in cutter.name, cutter.name

    ok_u, msg_u = zone_registry.update_zone(custom_key, token="ID-A_VIS-A#B", scene=scene)
    assert ok_u, msg_u
    assert apple_bite.sync_zone_visuals_and_names(ctx, custom_key) == 1
    assert "_ID-ID-A_VIS-A#B_" in cutter.name, f"Zone rename did not update cutter name: {cutter.name}"

    # 5. Custom mesh cutters without _ID- keep their user-given name
    custom_mesh = _make_box("MyCustomCutter", location=(1.2, 0, 0), scale=(1.0, 1.0, 1.0))
    ok_c, msg_c, added = apple_bite.add_custom_mesh_cutters(ctx, [custom_mesh], zone="B")
    assert ok_c and len(added) == 1, msg_c
    assert apple_bite.set_cutters_zone(ctx, [custom_mesh], "A") == 1
    assert custom_mesh.name == "MyCustomCutter" and custom_mesh.get(apple_bite.CUTTER_ZONE_PROP) == "A"


def test_22_user_driven_rename_tool() -> None:
    from FullDestruc_Tool.core import rename_tool
    from FullDestruc_Tool.ui.panel_n import FDT_PT_main_panel

    _reset_scene()
    ctx = bpy.context
    scene = ctx.scene

    # 1. FULL mode: single object direct rename + mesh sync
    m1 = _make_box("TempCube", location=(0, 0, 0))
    ok, msg, pairs = rename_tool.rename_objects(ctx, [m1], mode="FULL", text="CustomWall_Main", sync_mesh_data=True)
    assert ok and m1.name == "CustomWall_Main" and m1.data.name == "CustomWall_Main_Mesh", (msg, pairs)

    # 2. FULL mode: multiple objects auto-numbered _01, _02
    m2 = _make_box("AlphaBox", location=(2, 0, 0))
    m3 = _make_box("BetaBox", location=(4, 0, 0))
    ok, msg, _ = rename_tool.rename_objects(ctx, [m3, m2], mode="FULL", text="PillarPart", auto_number=True, start_index=1)
    assert ok and m2.name == "PillarPart_01" and m3.name == "PillarPart_02", (msg, m2.name, m3.name)

    # 3. TAG mode: replaces ID freely without touching Zone Registry
    baked_like = _make_box("PubVillage_E_2L01_V2_dst_FDST_ID-C_LOD0", location=(6, 0, 0))
    ok, msg, _ = rename_tool.rename_objects(ctx, [baked_like], mode="TAG", text="A#Custom99")
    assert ok and baked_like.name == "PubVillage_E_2L01_V2_dst_FDST_ID-A#Custom99_LOD0", (msg, baked_like.name)

    # 4. Cutter inside active Apple Bite session: rename keeps session & pieces synced without changing zone
    wall = _make_box("RenameSessionWall", location=(0, 5, 0), scale=(4, 0.5, 3))
    ok, msg, _sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok, msg
    ok, msg, cutter = apple_bite.add_box_cutter(ctx, location=(0, 5, 0), size=1.5, zone="B")
    assert ok and cutter is not None, msg

    ok, msg, _ = rename_tool.rename_objects(ctx, [cutter], mode="TAG", text="FreeID")
    assert ok and "_ID-FreeID_" in cutter.name, (msg, cutter.name)
    assert cutter.get(apple_bite.CUTTER_ZONE_PROP) == "B", "Manual rename must NOT change fdt_cutter_zone"
    sess_after = session.load_session(scene)
    assert sess_after is not None and sess_after.cutters[0].name == cutter.name and sess_after.cutters[0].zone == "B"

    # Also test full free-form rename on cutter
    ok, msg, _ = rename_tool.rename_objects(ctx, [cutter], mode="FULL", text="MyManualCutter")
    assert ok and cutter.name == "MyManualCutter", (msg, cutter.name)
    sess_after2 = session.load_session(scene)
    assert sess_after2 is not None and sess_after2.cutters[0].name == "MyManualCutter"

    # 5. Panel draw renders fdt.rename_objects cleanly
    class _DummyPanel:
        _draw_step1_idle = FDT_PT_main_panel._draw_step1_idle
        _draw_step2_and_step3_active = FDT_PT_main_panel._draw_step2_and_step3_active

    panel = _DummyPanel()
    mock_layout = _MockUILayout()
    panel.layout = mock_layout
    FDT_PT_main_panel.draw(panel, ctx)
    assert "fdt.rename_popup" in mock_layout.operators, "Rename popup button was not drawn in N-Panel!"


def test_23_next_box_spawns_cube_and_duplicate_cutter() -> None:
    _reset_scene()
    ctx = bpy.context
    scene = ctx.scene

    wall = _make_box("DupNext_Wall", location=(0, 0, 0), scale=(8, 0.5, 3))
    ok, msg, sess = apple_bite.enter_session(ctx, target_objects=[wall])
    assert ok and sess is not None, msg

    # 1. Create a non-box custom cutter (16-segment cylinder -> 32 verts, 18 faces)
    cyl_mesh = bpy.data.meshes.new("CylCutter_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=16, radius1=0.8, radius2=0.8, depth=2.0)
    bm.to_mesh(cyl_mesh)
    bm.free()
    cyl_obj = bpy.data.objects.new("CylCutter", cyl_mesh)
    cyl_obj.location = Vector((-1.5, 0.0, 0.0))
    scene.collection.objects.link(cyl_obj)
    ok, msg, added = apple_bite.add_custom_mesh_cutters(ctx, [cyl_obj], zone="A")
    assert ok and len(added) == 1, msg
    assert len(cyl_obj.data.vertices) > 8

    # Next Box from a custom/drawn cutter MUST spawn a clean 8-vert, 6-face Box!
    ok, msg, next_box = apple_bite.add_adjacent_box_cutter(ctx, direction="+X", overlap=0.05, auto_next_zone=True)
    assert ok and next_box is not None, msg
    assert len(next_box.data.vertices) == 8 and len(next_box.data.polygons) == 6, (
        f"Next Box should be a pure 8v/6f cube, got {len(next_box.data.vertices)}v/{len(next_box.data.polygons)}f"
    )
    assert next_box.get(apple_bite.CUTTER_ZONE_PROP) == "B"

    # 2. Ctrl+D operator core (duplicate_selected_cutters): independent mesh, sequential name, registered in session
    apple_bite.safe_select_objects(ctx, [next_box], active=next_box)
    ok, msg, dups = apple_bite.duplicate_selected_cutters(ctx)
    assert ok and len(dups) == 1, msg
    dup = dups[0]
    assert dup.data is not next_box.data, "Duplicated cutter must have its own independent mesh data!"
    assert dup.get(apple_bite.CUTTER_ZONE_PROP) == "B"
    dup.location.x += 1.8

    sess_now = session.load_session(scene)
    assert sess_now is not None and len(sess_now.cutters) == 3
    assert any(c.name == dup.name for c in sess_now.cutters)

    # 3. Shift+D / Alt+D simulation: duplicate object directly inside FDT_Cutters_* and call adopt_duplicated_cutters
    cutters_col = sess_now.get_collections()["cutters"]
    raw_copy = bpy.data.objects.new(f"{next_box.name}.001", next_box.data)  # linked mesh like Alt+D
    raw_copy.matrix_world = next_box.matrix_world.copy()
    raw_copy.location.x -= 3.5
    raw_copy[apple_bite.CUTTER_PROP] = True
    raw_copy[apple_bite.CUTTER_ZONE_PROP] = "C"
    raw_copy[apple_bite.OBJ_SESSION_ID_PROP] = sess_now.session_id
    cutters_col.objects.link(raw_copy)

    adopted = apple_bite.adopt_duplicated_cutters(scene)
    assert len(adopted) == 1 and adopted[0] == raw_copy, f"Expected raw_copy adopted, got {adopted}"
    assert raw_copy.data is not next_box.data, "Linked duplicate should be made single-user on adopt!"
    assert not raw_copy.name.endswith(".001"), f"Expected clean sequential name, got {raw_copy.name}"
    sess_after = session.load_session(scene)
    assert sess_after is not None and len(sess_after.cutters) == 4
    assert len(scene.fdt_settings.cutter_layer_items) == 4


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
    test_21_set_zone_renames_cutter_and_active_zone_dropdown,
    test_22_user_driven_rename_tool,
    test_23_next_box_spawns_cube_and_duplicate_cutter,
]
