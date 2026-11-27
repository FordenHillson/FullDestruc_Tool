"""Phase E tests: Red Panda-style concrete debris on generated rebars."""

from __future__ import annotations

import math

from mathutils import Vector

from FullDestruc_Tool.core import rebar_debris as rd
from FullDestruc_Tool.core import rebar_generate as rg
from FullDestruc_Tool.core import rebar_spline as rs
from FullDestruc_Tool.core import rebar_state as st

TOL = 1e-5


def _params(**overrides):
    params = st.default_params()
    params.update(overrides)
    return st.normalize_params(params)


def _straight_bar(length: float = 1.0):
    return [Vector((0.0, 0.0, 0.0)), Vector((length, 0.0, 0.0))]


# ---------------------------------------------------------------------------
# Defaults and settings round-trip
# ---------------------------------------------------------------------------


def test_e01_defaults_match_red_panda_debris() -> None:
    params = st.default_params()
    assert params["enable_debris"] is False
    assert math.isclose(params["debris_density"], 0.5, rel_tol=TOL)
    assert math.isclose(params["debris_min_size"], 0.005, rel_tol=TOL)
    assert math.isclose(params["debris_max_size"], 0.04, rel_tol=TOL)
    assert math.isclose(params["debris_irregularity"], 0.4, rel_tol=TOL)
    assert math.isclose(params["debris_concentration"], 0.7, rel_tol=TOL)
    assert params["debris_material"] == ""


def test_e02_normalize_clamps_debris_and_keeps_empty_material() -> None:
    params = _params(
        debris_density=9.0,
        debris_min_size=0.09,
        debris_max_size=0.01,
        debris_irregularity=5.0,
        debris_concentration=-2.0,
        debris_material="",
    )
    assert math.isclose(params["debris_density"], 1.0, rel_tol=TOL)
    assert math.isclose(params["debris_min_size"], 0.01, rel_tol=TOL)
    assert math.isclose(params["debris_max_size"], 0.09, rel_tol=TOL)
    assert math.isclose(params["debris_irregularity"], 1.0, rel_tol=TOL)
    assert math.isclose(params["debris_concentration"], 0.0, rel_tol=TOL)
    assert params["debris_material"] == ""


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_e03_centers_empty_when_disabled_or_zero_density() -> None:
    splines = [(_straight_bar(), 1)]
    off = st.params_to_generation(_params(enable_debris=False))
    assert rd.debris_centers(splines, off) == []

    zero = st.params_to_generation(_params(enable_debris=True, debris_density=0.0))
    assert rd.debris_centers(splines, zero) == []


def test_e04_centers_are_deterministic_and_surface_biased() -> None:
    params = st.params_to_generation(
        _params(enable_debris=True, debris_density=1.0, sink_depth=0.05)
    )
    splines = [(_straight_bar(1.0), 1)]
    first = rd.debris_centers(splines, params)
    second = rd.debris_centers(splines, params)
    assert first == second, "debris placement must be seed-deterministic"
    assert first, "a 1 m exposed bar at density 1 should produce chunks"

    # Concentration 0.7 biases t toward the embedded fraction (x=0.05).
    near_surface = sum(1 for pos, _size in first if pos.x < 0.5)
    assert near_surface >= len(first) * 0.6, (near_surface, len(first))


def test_e05_seed_changes_placement() -> None:
    base = _params(enable_debris=True, debris_density=1.0)
    a = rd.debris_centers([(_straight_bar(), 1)], st.params_to_generation(base))
    b = rd.debris_centers(
        [(_straight_bar(), 1)], st.params_to_generation(_params(**{**base, "random_seed": 7}))
    )
    assert a != b


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def test_e06_build_debris_chunk_topology_and_uvs() -> None:
    params = st.params_to_generation(_params(enable_debris=True, debris_density=1.0))
    result = rd.build_debris([(_straight_bar(), 1)], params, uv_scale=0.5)
    assert result.chunk_count > 0
    assert len(result.payload.vertices) == result.chunk_count * 12
    assert len(result.payload.faces) == result.chunk_count * 20
    assert len(result.payload.uvs) == result.chunk_count * 20 * 3
    assert len(result.payload.smooth_faces) == result.chunk_count * 20
    assert all(flag for flag in result.payload.smooth_faces)
    assert set(result.payload.face_materials) == {rd.DEBRIS_MATERIAL_INDEX}
    assert not result.payload.is_empty()


def test_e07_merge_payload_offsets_all_indices() -> None:
    bar = rs.build_spline_mesh([(_straight_bar(), 1)], radius=0.02, uv_scale=0.5)
    result = rd.build_debris([(_straight_bar(), 1)], st.params_to_generation(_params(enable_debris=True)), 0.5)
    bar_count = len(bar.vertices)
    bar_faces = len(bar.faces)
    rd.merge_payload(bar, result.payload)
    assert len(bar.vertices) == bar_count + len(result.payload.vertices)
    assert len(bar.faces) == bar_faces + len(result.payload.faces)
    assert all(min(face) >= 0 and max(face) < len(bar.vertices) for face in bar.faces)
    assert len(bar.face_materials) == len(bar.faces)
    assert set(bar.face_materials[:bar_faces]) == {0}, "bar faces must stay on slot 0"
    assert set(bar.face_materials[bar_faces:]) == {rd.DEBRIS_MATERIAL_INDEX}


def test_e08_bar_payload_alone_has_no_debris_faces() -> None:
    bar = rs.build_spline_mesh([(_straight_bar(), 1)], radius=0.02, uv_scale=0.5)
    assert bar.face_materials == []


def test_e09_apply_spline_mesh_writes_debris_material_index() -> None:
    import bpy

    mesh = bpy.data.meshes.new("FDT_Debris_Test_Mesh")
    try:
        mat0 = bpy.data.materials.get("FDT_Debris_Test_A") or bpy.data.materials.new("FDT_Debris_Test_A")
        mat1 = bpy.data.materials.get("FDT_Debris_Test_B") or bpy.data.materials.new("FDT_Debris_Test_B")
        mesh.materials.append(mat0)
        mesh.materials.append(mat1)
        payload = rs.SplineMesh(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            faces=[(0, 1, 2)],
            uvs=[(0, 0), (1, 0), (0, 1)],
            smooth_faces=[True],
            face_materials=[rd.DEBRIS_MATERIAL_INDEX],
        )
        rs.apply_spline_mesh(mesh, payload)
        assert mesh.polygons[0].material_index == rd.DEBRIS_MATERIAL_INDEX
    finally:
        bpy.data.meshes.remove(mesh)


def test_e10_debris_tooltips_are_plain_language() -> None:
    from FullDestruc_Tool.props import FDT_RebarSettings

    names = (
        "enable_debris", "debris_density", "debris_min_size", "debris_max_size",
        "debris_irregularity", "debris_concentration", "debris_material",
    )
    banned = ("BVH", "spline", "polygon", "ray", "threshold", "cap", "vertex")
    for name in names:
        description = (FDT_RebarSettings.bl_rna.properties[name].description or "").strip()
        assert description, name
        assert not [w for w in banned if w.lower() in description.lower()], (name, description)
        assert "เช่น" in description or "คือ" in description, (name, description)


def _make_host():
    import bmesh
    import bpy

    mesh = bpy.data.meshes.new("FDT_Debris_Host_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    mat = bpy.data.materials.get("FDT_Debris_Host_Mat") or bpy.data.materials.new("FDT_Debris_Host_Mat")
    mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.material_index = 0
    host = bpy.data.objects.new("FDT_Debris_Host", mesh)
    bpy.context.scene.collection.objects.link(host)
    bpy.context.view_layer.objects.active = host
    host.select_set(True)
    return host, mat


def test_e11_toggle_changes_generated_mesh_integration() -> None:
    import bpy

    from FullDestruc_Tool.core import rebar_overlay, rebar_state as st
    from FullDestruc_Tool.ops import rebar_ops

    scene = bpy.context.scene
    try:
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Rebar_"):
                bpy.data.objects.remove(obj, do_unlink=True)
        if st.REGISTRY_PROP in scene:
            del scene[st.REGISTRY_PROP]

        host, mat = _make_host()
        settings = scene.fdt_rebar_settings
        rebar_ops.params_to_settings(st.default_params(), settings)
        settings.target_material = mat
        settings.enable_debris = False
        rebar_overlay.clear()

        assert bpy.ops.fdt.rebar_create() == {"FINISHED"}
        obj = bpy.context.view_layer.objects.active
        bare_vertices = len(obj.data.vertices)
        bare_indices = {poly.material_index for poly in obj.data.polygons}
        assert bare_indices == {0}

        settings.enable_debris = True
        settings.debris_density = 1.0
        assert bpy.ops.fdt.rebar_rebuild() == {"FINISHED"}
        obj = bpy.context.view_layer.objects.active
        assert len(obj.data.vertices) > bare_vertices, "debris must add chunks to the set mesh"
        indices = {poly.material_index for poly in obj.data.polygons}
        assert 0 in indices and 1 in indices, "bars use slot 0, debris uses slot 1"
        assert len(obj.data.materials) >= 2

        settings.enable_debris = False
        assert bpy.ops.fdt.rebar_rebuild() == {"FINISHED"}
        obj = bpy.context.view_layer.objects.active
        assert len(obj.data.vertices) == bare_vertices
        assert {poly.material_index for poly in obj.data.polygons} == {0}
    finally:
        if st.REGISTRY_PROP in scene:
            del scene[st.REGISTRY_PROP]
        rebar_overlay.clear()
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Rebar_") or obj.name.startswith("FDT_Debris_Host"):
                bpy.data.objects.remove(obj, do_unlink=True)


def test_e12_panel_exposes_debris_toggle_and_details() -> None:
    import bpy

    from FullDestruc_Tool.ui.panel_rebar import draw as draw_rebar

    class _Layout:
        def __init__(self):
            self.labels = []
            self.props = []
            self.prop_kwargs = {}
            self.op_slots = []
            self.events = []
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

        def label(self, text="", icon="NONE"):
            self.labels.append(text)

        def prop(self, data, prop_name, **kwargs):
            self.props.append(prop_name)
            self.events.append(("prop", prop_name))
            self.prop_kwargs.setdefault(prop_name, kwargs)

        def operator(self, op_id, text="", **kwargs):
            class _Op:
                slot = ""

            instance = _Op()
            self.op_slots.append((op_id, instance))
            self.events.append(("operator", op_id))
            return instance

    settings = bpy.context.scene.fdt_rebar_settings
    previous = (settings.ui_show_rebar, settings.ui_show_basics, settings.ui_show_advanced, settings.enable_debris)
    try:
        settings.ui_show_rebar = True
        settings.ui_show_basics = True
        settings.ui_show_advanced = True
        settings.enable_debris = True
        layout = _Layout()
        draw_rebar(layout, bpy.context)
        assert layout.props[0] == "ui_show_rebar"
        for flag in ("ui_show_basics", "ui_show_look", "ui_show_advanced", "ui_show_system"):
            assert flag in layout.props, flag

        assert "enable_debris" in layout.props
        for name in (
            "debris_density", "debris_min_size", "debris_max_size",
            "debris_irregularity", "debris_concentration", "debris_material",
        ):
            assert name in layout.props, name
        assert any("Concrete Debris" in text for text in layout.labels)
        assert any(getattr(op, "slot", "") == "DEBRIS" for _op, op in layout.op_slots)

        # Exact order inside the action group:
        # Create / Bake → Edit · Faces · Clear → Concrete Debris → Remove Set.
        ordered = [name for _kind, name in layout.events if (
            (_kind == "operator" and name in {
                "fdt.rebar_create", "fdt.rebar_bake", "fdt.rebar_reedit",
                "fdt.rebar_select_faces", "fdt.rebar_clear_faces", "fdt.rebar_remove",
            }) or (_kind == "prop" and name == "enable_debris")
        )]
        assert ordered[0] in {"fdt.rebar_create", "fdt.rebar_bake"}, ordered
        assert ordered[-5:] == [
            "fdt.rebar_reedit",
            "fdt.rebar_select_faces",
            "fdt.rebar_clear_faces",
            "enable_debris",
            "fdt.rebar_remove",
        ], ordered
    finally:
        (
            settings.ui_show_rebar,
            settings.ui_show_basics,
            settings.ui_show_advanced,
            settings.enable_debris,
        ) = previous


def test_e13_main_panel_has_tool_rail_on_the_left() -> None:
    import bpy

    from FullDestruc_Tool.ui.panel_n import FDT_PT_main_panel

    class _Op:
        def __init__(self):
            self.tool = ""
            self.depress = False

    class _Layout:
        def __init__(self):
            self.tool_ops = []
            self.op_ids = []
            self.props = []
            self.popovers = []
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.ui_units_x = 0.0
            self.enabled = True
            self.alert = False

        def box(self):
            return self

        def row(self, align=False):
            return self

        def column(self, align=False):
            return self

        def grid_flow(self, **kwargs):
            return self

        def split(self, factor=0.0, align=False):
            assert factor == 0.16
            return self

        def separator(self, **kwargs):
            return None

        def label(self, text="", **kwargs):
            pass

        def popover(self, panel="", **kwargs):
            self.popovers.append(str(panel))

        def prop(self, data, prop_name, **kwargs):
            assert hasattr(data, prop_name), prop_name
            self.props.append(prop_name)

        def operator(self, op_id, text="", **kwargs):
            self.op_ids.append(op_id)
            op = _Op()
            op.depress = bool(kwargs.get("depress", False))
            if op_id == "fdt.set_active_tool":
                ico = kwargs.get("icon", "") or kwargs.get("icon_value", 0)
                self.tool_ops.append((ico, op))
            return op

    class _DummyPanel:
        _draw_step1_idle = FDT_PT_main_panel._draw_step1_idle
        _draw_step2_and_step3_active = FDT_PT_main_panel._draw_step2_and_step3_active

    settings = bpy.context.scene.fdt_settings
    previous_tool = settings.ui_active_tool
    panel = _DummyPanel()
    try:
        settings.ui_active_tool = "REBAR"
        layout = _Layout()
        panel.layout = layout
        FDT_PT_main_panel.draw(panel, bpy.context)

        assert all(bool(ico) for ico, _op in layout.tool_ops), layout.tool_ops
        tools = [op.tool for _icon, op in layout.tool_ops]
        assert tools == ["APPLE_BITE", "REBAR", "CLEANUP", "BATCH"], tools
        assert [op.depress for _icon, op in layout.tool_ops] == [False, True, False, False]
        assert "fdt.rebar_create" in layout.op_ids
        assert "fdt.applebite_smart_start" not in layout.op_ids
        # Engine belongs to Apple Bite Settings popover only — never floats above Rebar.
        assert "boolean_solver" not in layout.props
        assert "FDT_PT_popover_settings" not in layout.popovers

        settings.ui_active_tool = "APPLE_BITE"
        layout = _Layout()
        panel.layout = layout
        FDT_PT_main_panel.draw(panel, bpy.context)
        assert "fdt.applebite_smart_start" in layout.op_ids
        assert "fdt.rebar_create" not in layout.op_ids
        assert "FDT_PT_popover_settings" in layout.popovers
    finally:
        settings.ui_active_tool = previous_tool


def test_e14_debris_follows_bent_bars_with_twist_tuples() -> None:
    """Generation hands debris (points, subdivisions, twist) triples; the
    chunks must sit on the final (bent) points, identical to the 2-tuple case."""
    points = [Vector((0.0, 0.0, 0.0)), Vector((0.1, 0.02, 0.4)), Vector((1.0, 0.0, 0.0))]
    params = st.params_to_generation(_params(enable_debris=True, debris_density=1.0))
    as_twotuple = rd.debris_centers([(points, 2)], params)
    as_triple = rd.debris_centers([(list(points), 2, 0.9)], params)
    assert as_twotuple, "expected chunks from a 1 m exposed bar"
    assert as_triple == as_twotuple, "twist must not move the debris chunks"


ALL_TESTS = [
    test_e01_defaults_match_red_panda_debris,
    test_e02_normalize_clamps_debris_and_keeps_empty_material,
    test_e03_centers_empty_when_disabled_or_zero_density,
    test_e04_centers_are_deterministic_and_surface_biased,
    test_e05_seed_changes_placement,
    test_e06_build_debris_chunk_topology_and_uvs,
    test_e07_merge_payload_offsets_all_indices,
    test_e08_bar_payload_alone_has_no_debris_faces,
    test_e09_apply_spline_mesh_writes_debris_material_index,
    test_e10_debris_tooltips_are_plain_language,
    test_e11_toggle_changes_generated_mesh_integration,
    test_e12_panel_exposes_debris_toggle_and_details,
    test_e13_main_panel_has_tool_rail_on_the_left,
    test_e14_debris_follows_bent_bars_with_twist_tuples,
]
