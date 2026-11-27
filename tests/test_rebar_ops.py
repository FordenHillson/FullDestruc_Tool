"""Phase D tests: operators, material asset, overlay and debounced preview."""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector

from FullDestruc_Tool.core import rebar_generate as rg
from FullDestruc_Tool.core import rebar_overlay, rebar_state as st
from FullDestruc_Tool.core import rebar_spline as rs
from FullDestruc_Tool.ops import rebar_ops

TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reset() -> None:
    scene = bpy.context.scene
    if st.REGISTRY_PROP in scene:
        del scene[st.REGISTRY_PROP]
    settings = scene.fdt_rebar_settings
    rebar_ops.params_to_settings(st.default_params(), settings)
    settings.rebar_active_sid = ""
    settings.rebar_estimate = 0
    settings.rebar_status = ""
    settings.target_material = None
    settings.rebar_material = None
    settings.rebar_label = ""
    settings.ui_show_rebar = False
    settings.ui_show_basics = True
    settings.ui_show_look = False
    settings.ui_show_advanced = False
    settings.ui_show_system = False
    rebar_overlay.clear()


def _make_host(name: str = "FDT_Rebar_Host") -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    mat = bpy.data.materials.get("FDT_Test_Target") or bpy.data.materials.new("FDT_Test_Target")
    mesh.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.material_index = 0
        polygon.select = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _cleanup(*objects) -> None:
    for obj in objects:
        if obj is None:
            continue
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for col in list(bpy.data.collections):
        if col.name.startswith(st.COLLECTION_PREFIX):
            bpy.data.collections.remove(col)
    for mat in list(bpy.data.materials):
        if mat.name in {"FDT_Test_Target", st.REBAR_MATERIAL_NAME}:
            bpy.data.materials.remove(mat)


def _create(host, *, selected=(0, 1)) -> tuple[str, bpy.types.Object]:
    """Run the Create operator and return ``(sid, rebar_object)``."""
    settings = bpy.context.scene.fdt_rebar_settings
    settings.target_material = host.data.materials[0]
    for polygon in host.data.polygons:
        polygon.select = polygon.index in selected
    bpy.context.view_layer.objects.active = host
    result = bpy.ops.fdt.rebar_create()
    assert result == {"FINISHED"}, result
    sid = settings.rebar_active_sid
    entry = st.find_entry(sid)
    assert entry is not None
    return sid, st.resolve_object(entry)


# ---------------------------------------------------------------------------
# Settings <-> params
# ---------------------------------------------------------------------------


def test_d01_settings_to_params_converts_mm_to_m() -> None:
    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    settings.spacing_x = 250.0
    settings.sink_depth = 30.0
    settings.diameter_mm = 20.0
    params = rebar_ops.settings_to_params(settings)
    assert math.isclose(params["spacing_x"], 0.25, rel_tol=TOL)
    assert math.isclose(params["sink_depth"], 0.03, rel_tol=TOL)
    assert params["diameter_mm"] == 20.0
    assert params["bar_size"] == "DB12" or params["bar_size"] == "CUSTOM"


def test_d02_params_to_settings_roundtrip() -> None:
    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    params = st.normalize_params(
        {
            "spacing_x": 0.4,
            "sink_depth": 0.05,
            "min_length": 0.2,
            "max_length": 0.9,
            "diameter_mm": 16.0,
            "bar_size": "DB16",
            "bend_scale": 0.25,
            "fast_drag": False,
        }
    )
    rebar_ops.params_to_settings(params, settings)
    assert settings.spacing_x == 400.0
    assert math.isclose(settings.sink_depth, 50.0, abs_tol=TOL)
    assert settings.min_length == 200.0
    assert settings.max_length == 900.0
    assert settings.diameter_mm == 16.0
    assert math.isclose(settings.bend_scale_mm, 250.0, abs_tol=TOL)
    assert settings.fast_drag is False
    again = rebar_ops.settings_to_params(settings)
    assert math.isclose(again["spacing_x"], 0.4, rel_tol=TOL)
    assert math.isclose(again["bend_scale"], 0.25, rel_tol=TOL)


def test_d03_resolve_faces_prefers_selected_matching_faces() -> None:
    _reset()
    host = _make_host("FaceSelect")
    try:
        settings = bpy.context.scene.fdt_rebar_settings
        settings.target_material = host.data.materials[0]
        for polygon in host.data.polygons:
            polygon.select = polygon.index in (2, 4)
        faces = rebar_ops.resolve_faces(bpy.context, host, settings)
        assert sorted(faces) == [2, 4]
    finally:
        _cleanup(host)


def test_d04_resolve_faces_falls_back_to_all_matching_faces() -> None:
    _reset()
    host = _make_host("FaceFallback")
    try:
        settings = bpy.context.scene.fdt_rebar_settings
        settings.target_material = host.data.materials[0]
        faces = rebar_ops.resolve_faces(bpy.context, host, settings)
        assert len(faces) == len(host.data.polygons)
    finally:
        _cleanup(host)


def test_d05_resolve_faces_rejects_unassigned_material() -> None:
    _reset()
    host = _make_host("FaceWrongMat")
    other = bpy.data.materials.new("FDT_Test_Other")
    try:
        settings = bpy.context.scene.fdt_rebar_settings
        settings.target_material = other
        assert rebar_ops.resolve_faces(bpy.context, host, settings) == []
    finally:
        bpy.data.materials.remove(other)
        _cleanup(host)


def test_d06_estimate_for_counts_grid_rays() -> None:
    _reset()
    host = _make_host("Estimate")
    try:
        faces = list(range(len(host.data.polygons)))
        estimate = rebar_ops.estimate_for(host.data, faces, st.default_params())
        assert estimate > 0
    finally:
        _cleanup(host)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_d07_create_builds_tagged_live_generator() -> None:
    _reset()
    host = _make_host("CreateHost")
    try:
        sid, obj = _create(host)
        entry = st.find_entry(sid)
        assert entry is not None
        assert entry.host == host.name
        assert entry.group >= 0
        assert entry.baked is False
        assert st.object_sid(obj) == sid
        assert math.isclose(st.load_params(obj)["spacing_x"], 0.4, rel_tol=TOL)
        assert len(rg.tagged_face_indices(host.data, entry.group)) == 2
        assert len(obj.data.vertices) > 0, "create produced no rebar geometry"
        assert len(obj.data.polygons) > 0
        assert obj.data.uv_layers.get("UVMap0") is not None
        assert bpy.context.scene.fdt_rebar_settings.rebar_active_sid == sid
    finally:
        _reset()
        _cleanup(host)


def test_d08_create_rejects_missing_target_material() -> None:
    _reset()
    host = _make_host("CreateNoMat")
    try:
        bpy.context.view_layer.objects.active = host
        result = bpy.ops.fdt.rebar_create()
        assert result == {"CANCELLED"}, result
        assert st.load_registry() == []
    finally:
        _cleanup(host)


def test_d09_rebuild_is_deterministic_and_seed_sensitive() -> None:
    _reset()
    host = _make_host("RebuildHost")
    try:
        sid, obj = _create(host)
        settings = bpy.context.scene.fdt_rebar_settings
        first = [v.co.copy() for v in obj.data.vertices]
        assert bpy.ops.fdt.rebar_rebuild() == {"FINISHED"}
        second = [v.co.copy() for v in obj.data.vertices]
        assert len(first) == len(second)
        assert all((a - b).length < TOL for a, b in zip(first, second))

        settings.random_seed = 99
        assert bpy.ops.fdt.rebar_rebuild() == {"FINISHED"}
        third = [v.co.copy() for v in obj.data.vertices]
        assert any((a - b).length > TOL for a, b in zip(second, third))
    finally:
        _reset()
        _cleanup(host)


def test_d10_bake_freezes_params_and_moves_to_collection() -> None:
    _reset()
    host = _make_host("BakeHost")
    try:
        sid, obj = _create(host)
        settings = bpy.context.scene.fdt_rebar_settings
        settings.spacing_x = 250.0
        settings.rebar_label = "Wall_A"
        settings.ui_show_look = True
        settings.ui_show_advanced = True
        assert bpy.ops.fdt.rebar_bake() == {"FINISHED"}
        assert settings.ui_show_look is False
        assert settings.ui_show_advanced is False
        entry = st.find_entry(sid)
        assert entry.baked is True
        assert entry.label == "Wall_A"
        assert math.isclose(entry.params["spacing_x"], 0.25, rel_tol=TOL)
        assert math.isclose(st.load_params(obj)["spacing_x"], 0.25, rel_tol=TOL)
        col = bpy.data.collections.get(st.collection_name(sid))
        assert col is not None and obj.name in col.objects
    finally:
        _reset()
        _cleanup(host)


def test_d11_reedit_loads_params_into_buffer() -> None:
    _reset()
    host = _make_host("ReeditHost")
    try:
        sid, _obj = _create(host)
        settings = bpy.context.scene.fdt_rebar_settings
        settings.spacing_x = 333.0
        bpy.ops.fdt.rebar_bake()
        settings.spacing_x = 900.0
        assert bpy.ops.fdt.rebar_reedit(sid=sid) == {"FINISHED"}
        assert settings.spacing_x == 333.0
        assert st.find_entry(sid).baked is False
        assert settings.rebar_active_sid == sid
        # Edit must visibly open the detail sections, not act silently.
        assert settings.ui_show_rebar is True
        assert settings.ui_show_basics is True
        assert settings.ui_show_look is True
        assert settings.ui_show_advanced is True
    finally:
        _reset()
        _cleanup(host)


def test_d12_select_faces_selects_only_tagged_faces() -> None:
    _reset()
    host = _make_host("SelectFacesHost")
    try:
        sid, _obj = _create(host, selected=(0, 2))
        assert bpy.ops.fdt.rebar_select_faces(sid=sid) == {"FINISHED"}
        assert bpy.context.view_layer.objects.active is host
        bm = bmesh.from_edit_mesh(host.data)
        selected = sorted(f.index for f in bm.faces if f.select)
        assert selected == [0, 2], selected
        bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        _reset()
        _cleanup(host)


def test_d13_delete_guard_then_clear_then_delete() -> None:
    _reset()
    host = _make_host("DeleteHost")
    try:
        sid, _obj = _create(host)
        assert bpy.ops.fdt.rebar_delete(sid=sid) == {"CANCELLED"}
        assert st.find_entry(sid) is not None
        assert bpy.ops.fdt.rebar_clear_faces(sid=sid) == {"FINISHED"}
        assert bpy.ops.fdt.rebar_delete(sid=sid) == {"FINISHED"}
        assert st.find_entry(sid) is None
        assert bpy.data.objects.get(st.object_name(sid)) is None
    finally:
        _reset()
        _cleanup(host)


def test_d14_no_auto_bake_on_setting_change() -> None:
    _reset()
    host = _make_host("NoAutoBake")
    try:
        sid, _obj = _create(host)
        settings = bpy.context.scene.fdt_rebar_settings
        settings.spacing_x = 200.0
        # A property change only schedules a debounced preview; it must never
        # commit the generator by itself.
        assert st.find_entry(sid).baked is False
        rebar_ops._cancel("rebuild")
        rebar_ops._cancel("commit")
    finally:
        _reset()
        _cleanup(host)


def test_d15_fast_drag_uses_overlay_then_commits_mesh() -> None:
    _reset()
    host = _make_host("FastDrag")
    try:
        sid, obj = _create(host)
        settings = bpy.context.scene.fdt_rebar_settings
        settings.fast_drag = True
        settings.spacing_x = 120.0
        rebar_ops._rebuild_timer()
        assert rebar_overlay.is_active(), "fast drag should draw GPU lines"
        assert rebar_overlay.get_line_count() > 0
        before = [v.co.copy() for v in obj.data.vertices]
        rebar_ops._commit_timer()
        assert not rebar_overlay.is_active()
        after = [v.co.copy() for v in obj.data.vertices]
        assert len(after) != len(before) or any((a - b).length > TOL for a, b in zip(before, after))
        rebar_ops._cancel("rebuild")
        rebar_ops._cancel("commit")
    finally:
        _reset()
        _cleanup(host)


# ---------------------------------------------------------------------------
# Material + overlay
# ---------------------------------------------------------------------------


def test_d16_material_appends_from_bundled_blend() -> None:
    _reset()
    try:
        for mat in list(bpy.data.materials):
            if mat.name == st.REBAR_MATERIAL_NAME:
                bpy.data.materials.remove(mat)
        mat, source = st.resolve_rebar_material()
        assert mat is not None, "bundled rebar material could not be appended"
        assert mat.name == st.REBAR_MATERIAL_NAME
        assert source == "appended", source
        assert st._ebt_resource_name(mat) == st.REBAR_RESOURCE_NAME
        again, again_source = st.resolve_rebar_material()
        assert again is mat and again_source == "existing"
    finally:
        _reset()
        _cleanup()


def test_d17_material_does_not_fall_back_to_grey_stub() -> None:
    _reset()
    original = st.REBAR_ASSET_PATH
    try:
        for mat in list(bpy.data.materials):
            if mat.name == st.REBAR_MATERIAL_NAME:
                bpy.data.materials.remove(mat)
        st.REBAR_ASSET_PATH = original.parent / "missing_asset.blend"
        mat, source = st.resolve_rebar_material()
        assert mat is None
        assert source == ""
    finally:
        st.REBAR_ASSET_PATH = original
        _cleanup()


def test_d18_overlay_state_is_module_local() -> None:
    _reset()
    try:
        rebar_overlay.set_lines([[Vector((0, 0, 0)), Vector((1, 0, 0))]])
        assert rebar_overlay.is_active()
        assert not hasattr(bpy.types.WindowManager, "fdt_rebar_state")
        rebar_overlay.clear()
        assert not rebar_overlay.is_active()
    finally:
        _reset()


class _MockLayout:
    """Minimal UILayout mock that records operators and validates properties."""

    def __init__(self):
        self.operators: list[str] = []
        self.operator_texts: list[str] = []
        self.op_instances: list[tuple[str, object]] = []
        self.labels: list[str] = []
        self.scale_y = 1.0
        self.enabled = True
        self.alert = False

    def box(self):
        return self

    def row(self, align=False):
        return self

    def column(self, align=False):
        return self

    def split(self, factor=0.0, align=False):
        return self

    def separator(self, **kwargs):
        return None

    def label(self, text="", **kwargs):
        self.labels.append(text)
        return None

    def popover(self, panel="", **kwargs):
        return None

    def prop(self, data, prop_name, **kwargs):
        assert hasattr(data, prop_name), f"Missing property '{prop_name}'"
        return None

    def operator(self, op_id, text="", **kwargs):
        self.operators.append(op_id)
        self.operator_texts.append(text)

        class _OpProps:
            pass

        instance = _OpProps()
        self.op_instances.append((op_id, instance))
        return instance


class _StrictOpProps:
    """Like real bpy operator RNA: only declared properties may be assigned."""

    _allowed = {"slot", "sid"}

    def __setattr__(self, name, value):
        if name not in self._allowed:
            raise AttributeError(f"unknown operator property: {name}")
        super().__setattr__(name, value)


class _StrictLayout(_MockLayout):
    def operator(self, op_id, text="", **kwargs):
        self.operators.append(op_id)
        return _StrictOpProps()


def test_d20_label_is_cosmetic_identity_is_sid() -> None:
    _reset()
    host = _make_host("LabelHost")
    try:
        settings = bpy.context.scene.fdt_rebar_settings
        settings.rebar_label = "ID-A_VIS-!B#A"
        sid, obj = _create(host, selected=(0,))
        entry = st.find_entry(sid)
        assert obj.name == "ID-A_VIS-!B#A"
        assert entry.label == "ID-A_VIS-!B#A"

        # Outliner rename must not detach the generator.
        obj.name = "Totally_Renamed"
        assert st.find_entry(sid).sid == sid
        assert st.resolve_object(entry) is obj

        # Bake keeps the renamed cosmetic label and still resolves by sid.
        assert bpy.ops.fdt.rebar_bake() == {"FINISHED"}
        entry = st.find_entry(sid)
        assert entry.baked is True
        assert obj.name == "Totally_Renamed"
        assert entry.label == "Totally_Renamed"
        assert st.resolve_object(entry) is obj
    finally:
        _reset()
        _cleanup(host)


def test_d19_expanded_panel_draws_all_operators() -> None:
    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    _reset()
    host = _make_host("PanelDrawHost")
    try:
        sid, _obj = _create(host, selected=(0,))
        settings = bpy.context.scene.fdt_rebar_settings
        settings.ui_show_rebar = True
        settings.ui_show_basics = True
        settings.ui_show_look = True
        settings.ui_show_advanced = True
        settings.ui_show_system = True
        layout = _MockLayout()
        draw_rebar(layout, bpy.context)
        for op_id in (
            "fdt.rebar_pick_material",
            "fdt.rebar_load_material",
            "fdt.rebar_create",
            "fdt.rebar_bake",
            "fdt.rebar_reedit",
            "fdt.rebar_select_faces",
            "fdt.rebar_clear_faces",
            "fdt.rebar_delete",
            "fdt.rebar_remove",
        ):
            assert op_id in layout.operators, f"missing {op_id}"
        # Create must stay a full-text, prominent button even with an active set.
        assert "Create Another Rebar" in layout.operator_texts
        assert layout.operators.index("fdt.rebar_create") < layout.operators.index("fdt.rebar_bake")
        # Live settings already preview in realtime: only Bake is needed, and
        # there must be no ambiguous "Re-bake" wording anywhere.
        assert "Bake" in layout.operator_texts
        assert "fdt.rebar_rebuild" not in layout.operators
        assert any("Live preview" in text for text in layout.labels)
        # Identity/status text must not be inserted between the button rows.
        assert not [t for t in layout.operator_texts if "Re-bake" in t]

        # After a re-cut the extra button appears, but only then.
        assert st.set_stale(sid, True)
        stale_layout = _MockLayout()
        draw_rebar(stale_layout, bpy.context)
        assert "fdt.rebar_rebuild" in stale_layout.operators
        assert "Re-apply" in stale_layout.operator_texts
        assert any("Host changed" in text for text in stale_layout.labels)

        settings.ui_show_rebar = False
        settings.ui_show_basics = True
        settings.ui_show_look = False
        settings.ui_show_advanced = False
        settings.ui_show_system = False
    finally:
        _reset()
        _cleanup(host)


def test_d23_density_preset_drives_spacing_and_returns_to_custom() -> None:
    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    settings.density_preset = "FINE"
    assert settings.spacing_x == 75.0
    assert settings.spacing_y == 75.0
    assert settings.spacing_z == 75.0

    settings.spacing_x = 123.0
    assert settings.density_preset == "CUSTOM"

    rebar_ops.params_to_settings(
        st.normalize_params({"spacing_x": 0.25, "spacing_y": 0.25, "spacing_z": 0.25}),
        settings,
    )
    assert settings.density_preset == "COARSE"
    assert settings.spacing_x == 250.0


def test_d25_remove_set_clears_tags_and_deletes_in_one_step() -> None:
    _reset()
    host = _make_host("RemoveHost")
    try:
        sid, obj = _create(host)
        obj_name = obj.name  # RNA is invalid after the object is removed.
        entry = st.find_entry(sid)
        settings = bpy.context.scene.fdt_rebar_settings
        settings.rebar_active_sid = sid
        assert st.tagged_faces_for_entry(entry)

        # One click in the Basics section replaces Clear -> Delete.
        assert bpy.ops.fdt.rebar_remove() == {"FINISHED"}
        assert st.tagged_faces_for_entry(entry) == []
        assert st.find_entry(sid) is None
        assert bpy.data.objects.get(obj_name) is None
        assert settings.rebar_active_sid == ""
    finally:
        _reset()
        _cleanup(host)


def test_d26_remove_explicit_sid_keeps_newer_set() -> None:
    """An explicit sid from the generator list overrides viewport selection."""
    _reset()
    host = _make_host("RemoveTwoHost")
    try:
        sid1, obj1 = _create(host, selected=(0,))
        name1 = obj1.name
        sid2, _obj2 = _create(host, selected=(1,))
        settings = bpy.context.scene.fdt_rebar_settings
        assert settings.rebar_active_sid == sid2

        # An explicit sid (from the generator list) overrides the selection,
        # which currently points at the newer set after Create.
        assert bpy.ops.fdt.rebar_remove(sid=sid1) == {"FINISHED"}
        assert st.find_entry(sid1) is None
        assert st.find_entry(sid2) is not None
        assert bpy.data.objects.get(name1) is None
    finally:
        _reset()
        _cleanup(host)


def test_d27_remove_follows_viewport_selection() -> None:
    """The big Remove button targets the rebar selected in the viewport."""
    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    host = _make_host("RemoveSelectedHost")
    try:
        sid1, obj1 = _create(host, selected=(0,))
        sid8_1 = st.find_entry(sid1).sid8
        sid2, _obj2 = _create(host, selected=(1,))

        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        obj1.select_set(True)
        bpy.context.view_layer.objects.active = obj1

        settings.ui_show_rebar = True
        layout = _MockLayout()
        draw_rebar(layout, bpy.context)
        assert any("Target:" in text and sid8_1 in text for text in layout.labels)

        assert bpy.ops.fdt.rebar_remove() == {"FINISHED"}
        assert st.find_entry(sid1) is None
        assert st.find_entry(sid2) is not None
    finally:
        settings.ui_show_rebar = False
        _reset()
        _cleanup(host)


def test_d28_basics_has_quick_edit_actions_without_system_section() -> None:
    """Edit/Faces/Clear must be reachable without opening the generator list."""
    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    host = _make_host("QuickEditHost")
    try:
        _create(host, selected=(0,))
        settings.ui_show_rebar = True
        settings.ui_show_basics = True
        settings.ui_show_look = False
        settings.ui_show_advanced = False
        settings.ui_show_system = False

        layout = _MockLayout()
        draw_rebar(layout, bpy.context)
        for op_id in (
            "fdt.rebar_reedit",
            "fdt.rebar_select_faces",
            "fdt.rebar_clear_faces",
        ):
            assert op_id in layout.operators, f"missing quick action {op_id}"
    finally:
        settings.ui_show_rebar = False
        _reset()
        _cleanup(host)


def test_d29_quick_actions_target_the_active_selected_rebar() -> None:
    """Edit/Faces/Clear follow the active rebar even with several selected."""
    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    host = _make_host("QuickTargetHost")
    try:
        sid1, obj1 = _create(host, selected=(0,))
        _sid2, obj2 = _create(host, selected=(1,))
        for obj in (obj1, obj2):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = obj1

        settings.ui_show_rebar = True
        settings.ui_show_basics = True
        settings.ui_show_system = False
        layout = _MockLayout()
        draw_rebar(layout, bpy.context)

        quick = ("fdt.rebar_reedit", "fdt.rebar_select_faces", "fdt.rebar_clear_faces")
        found = {
            op_id: getattr(props, "sid", "")
            for op_id, props in layout.op_instances
            if op_id in quick
        }
        assert set(found) == set(quick)
        assert all(value == sid1 for value in found.values()), found
    finally:
        settings.ui_show_rebar = False
        _reset()
        _cleanup(host)


def test_d24_panel_only_sets_real_operator_properties() -> None:
    """Real operator RNA rejects .enabled; that used to hide all other headers."""
    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    host = _make_host("StrictPanelHost")
    try:
        settings.ui_show_rebar = True
        settings.ui_show_basics = True
        settings.ui_show_look = True
        settings.ui_show_advanced = True
        settings.ui_show_system = True

        # Idle branch (no generator yet).
        draw_rebar(_StrictLayout(), bpy.context)

        # Active branch (Bake / Refresh / Create buttons present).
        _create(host, selected=(0,))
        draw_rebar(_StrictLayout(), bpy.context)
    finally:
        settings.ui_show_rebar = False
        settings.ui_show_basics = True
        settings.ui_show_look = False
        settings.ui_show_advanced = False
        settings.ui_show_system = False
        _reset()
        _cleanup(host)


def test_d21_rebar_icons_are_valid() -> None:
    """Every icon name must exist in this Blender build, not be a guessed string."""
    from FullDestruc_Tool.ui.panel_rebar import REBAR_ICONS

    enum = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items
    valid = {item.identifier for item in enum}
    missing = sorted(name for name in REBAR_ICONS.values() if name not in valid)
    assert not missing, f"unsupported Blender icons: {missing}"


def test_d22_rebar_tooltips_are_plain_language() -> None:
    """Settings tooltips must exist and avoid programmer jargon / variables."""
    from FullDestruc_Tool.props import FDT_RebarSettings

    public_props = (
        "ui_show_rebar", "ui_show_basics", "ui_show_look", "ui_show_advanced",
        "ui_show_system", "target_material", "rebar_label", "density_preset",
        "bar_size", "sink_depth", "min_length", "max_length", "bending_strength",
        "bend_scale_mm",
        "random_seed", "diameter_mm", "align_x", "align_y", "align_z",
        "spacing_x", "spacing_y", "spacing_z", "grid_offset", "angle_threshold",
        "normal_bias", "gravity_bias", "avoidance_strength", "smooth_radius",
        "subdivisions", "rebar_material", "max_bars", "fast_drag",
        "enable_debris", "debris_density", "debris_min_size", "debris_max_size",
        "debris_irregularity", "debris_concentration", "debris_material",
    )
    banned = ("BVH", "spline", "polygon", "ray", "threshold", "capped", "subdivision", "n =", "n=")
    for name in public_props:
        prop = FDT_RebarSettings.bl_rna.properties[name]
        description = (prop.description or "").strip()
        assert description, f"missing tooltip for {name}"
        assert not [word for word in banned if word.lower() in description.lower()], (name, description)


def test_d30_defaults_match_red_panda_metal_rebar() -> None:
    """Fresh buffers must match RP: 400 mm grid, 50 mm sink, 100–500 mm bars, Ø40."""
    _reset()
    params = st.default_params()
    expected = {
        "spacing_x": 0.4,
        "spacing_y": 0.4,
        "spacing_z": 0.4,
        "sink_depth": 0.05,
        "min_length": 0.1,
        "max_length": 0.5,
        "bending_strength": 0.03,
        "bend_scale": 0.4,
        "smooth_radius": 0.3,
        "angle_threshold": 0.5,
        "normal_bias": 0.5,
        "gravity_bias": 0.5,
        "avoidance_strength": 0.5,
        "subdivisions": 6,
        "random_seed": 0,
    }
    for key, value in expected.items():
        assert math.isclose(params[key], value, rel_tol=TOL), (key, params[key], value)
    assert params["bar_size"] == "CUSTOM"
    assert params["diameter_mm"] == 40.0
    # RP stores bevel_depth as a 0.02 m radius; our diameter maps to the same.
    assert math.isclose(rs.diameter_to_radius(params["diameter_mm"]), 0.02, rel_tol=TOL)

    settings = bpy.context.scene.fdt_rebar_settings
    assert settings.density_preset == "SPARSE"
    assert settings.spacing_x == 400.0
    assert settings.spacing_y == 400.0
    assert settings.spacing_z == 400.0
    assert settings.sink_depth == 50.0
    assert settings.min_length == 100.0
    assert settings.max_length == 500.0
    assert settings.bar_size == "CUSTOM"
    assert settings.diameter_mm == 40.0


def test_d31_reroll_seed_changes_only_the_seed() -> None:
    """The dice operator must reassign the seed and touch nothing else."""
    _reset()
    settings = bpy.context.scene.fdt_rebar_settings
    settings.random_seed = 42
    settings.spacing_x = 321.0
    settings.bending_strength = 160.0

    assert bpy.ops.fdt.rebar_reroll_seed() == {"FINISHED"}

    assert settings.random_seed != 42
    assert 0 <= settings.random_seed < 1000000
    assert settings.spacing_x == 321.0, "reroll must not disturb other settings"
    assert settings.bending_strength == 160.0


ALL_TESTS = [
    test_d01_settings_to_params_converts_mm_to_m,
    test_d02_params_to_settings_roundtrip,
    test_d03_resolve_faces_prefers_selected_matching_faces,
    test_d04_resolve_faces_falls_back_to_all_matching_faces,
    test_d05_resolve_faces_rejects_unassigned_material,
    test_d06_estimate_for_counts_grid_rays,
    test_d07_create_builds_tagged_live_generator,
    test_d08_create_rejects_missing_target_material,
    test_d09_rebuild_is_deterministic_and_seed_sensitive,
    test_d10_bake_freezes_params_and_moves_to_collection,
    test_d11_reedit_loads_params_into_buffer,
    test_d12_select_faces_selects_only_tagged_faces,
    test_d13_delete_guard_then_clear_then_delete,
    test_d14_no_auto_bake_on_setting_change,
    test_d15_fast_drag_uses_overlay_then_commits_mesh,
    test_d16_material_appends_from_bundled_blend,
    test_d17_material_does_not_fall_back_to_grey_stub,
    test_d18_overlay_state_is_module_local,
    test_d19_expanded_panel_draws_all_operators,
    test_d20_label_is_cosmetic_identity_is_sid,
    test_d21_rebar_icons_are_valid,
    test_d22_rebar_tooltips_are_plain_language,
    test_d23_density_preset_drives_spacing_and_returns_to_custom,
    test_d24_panel_only_sets_real_operator_properties,
    test_d25_remove_set_clears_tags_and_deletes_in_one_step,
    test_d26_remove_explicit_sid_keeps_newer_set,
    test_d27_remove_follows_viewport_selection,
    test_d28_basics_has_quick_edit_actions_without_system_section,
    test_d29_quick_actions_target_the_active_selected_rebar,
    test_d30_defaults_match_red_panda_metal_rebar,
    test_d31_reroll_seed_changes_only_the_seed,
]
