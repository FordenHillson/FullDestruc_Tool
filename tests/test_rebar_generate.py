"""Phase B tests: face tagging, selection resolution and grid generation.

Kept in a separate module so the pure-math suite (test_rebar.py) stays free of
scene setup, while ``run_tests.py`` collects both.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector

from FullDestruc_Tool.core import rebar_generate as rg

VEC_TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plane(name: str = "Host", size: float = 2.0, divisions: int = 4) -> bpy.types.Object:
    """A flat grid in XY at z=0 with two materials on separate face groups."""
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=divisions, y_segments=divisions, size=size * 0.5)
    bm.to_mesh(mesh)
    bm.free()

    mat_a = bpy.data.materials.get("FDT_Test_MatA") or bpy.data.materials.new("FDT_Test_MatA")
    mat_b = bpy.data.materials.get("FDT_Test_MatB") or bpy.data.materials.new("FDT_Test_MatB")
    mesh.materials.append(mat_a)
    mesh.materials.append(mat_b)
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = 0 if index % 2 == 0 else 1

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_box(name: str = "HostBox", size: float = 2.0) -> bpy.types.Object:
    """A closed cube, so raycasts along every axis have something to hit."""
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bm.to_mesh(mesh)
    bm.free()

    mat = bpy.data.materials.get("FDT_Test_MatA") or bpy.data.materials.new("FDT_Test_MatA")
    mesh.materials.append(mat)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _cleanup(*objects) -> None:
    for obj in objects:
        if obj is None:
            continue
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def test_b01_group_attribute_is_created_at_no_group() -> None:
    obj = _make_plane("TagHost")
    try:
        attr = rg.ensure_group_attribute(obj.data)
        assert attr.domain == "FACE"
        assert attr.data_type == "INT"
        assert all(item.value == rg.NO_GROUP for item in attr.data)
    finally:
        _cleanup(obj)


def test_b02_tag_and_read_roundtrip() -> None:
    obj = _make_plane("TagRoundtrip")
    try:
        faces = [0, 2, 4]
        assert rg.tag_faces(obj.data, faces, 3) == 3
        assert sorted(rg.tagged_face_indices(obj.data, 3)) == faces
        assert rg.tagged_face_count(obj.data, 3) == 3
        # Other groups stay empty.
        assert rg.tagged_face_count(obj.data, 4) == 0
    finally:
        _cleanup(obj)


def test_b03_tag_two_groups_are_independent() -> None:
    obj = _make_plane("TagGroups")
    try:
        rg.tag_faces(obj.data, [0], 0)
        rg.tag_faces(obj.data, [1, 2], 1)
        assert rg.tagged_face_indices(obj.data, 0) == [0]
        assert sorted(rg.tagged_face_indices(obj.data, 1)) == [1, 2]
        assert rg.count_any_tagged(obj.data) == 3
    finally:
        _cleanup(obj)


def test_b04_untag_faces_clears_only_targets() -> None:
    obj = _make_plane("TagClear")
    try:
        rg.tag_faces(obj.data, [0, 1, 2], 0)
        assert rg.untag_faces(obj.data, [1]) == 1
        assert sorted(rg.tagged_face_indices(obj.data, 0)) == [0, 2]
        assert rg.count_any_tagged(obj.data) == 2
    finally:
        _cleanup(obj)


def test_b05_next_group_index_reuses_lowest_free_slot() -> None:
    obj = _make_plane("TagNext")
    try:
        assert rg.next_group_index(obj.data) == 0
        rg.tag_faces(obj.data, [0], 0)
        assert rg.next_group_index(obj.data) == 1
        rg.tag_faces(obj.data, [1], 2)
        assert rg.next_group_index(obj.data) == 1
    finally:
        _cleanup(obj)


def test_b06_tag_rejects_negative_group() -> None:
    obj = _make_plane("TagNegative")
    try:
        try:
            rg.tag_faces(obj.data, [0], -1)
        except ValueError:
            pass
        else:
            raise AssertionError("negative group index must raise")
    finally:
        _cleanup(obj)


def test_b07_untag_on_untagged_mesh_is_a_noop() -> None:
    obj = _make_plane("TagNoop")
    try:
        assert rg.untag_faces(obj.data, [0, 1]) == 0
        assert rg.tagged_face_indices(obj.data, 0) == []
        assert rg.count_any_tagged(obj.data) == 0
    finally:
        _cleanup(obj)


# ---------------------------------------------------------------------------
# Material + selection resolution
# ---------------------------------------------------------------------------


def test_b08_material_index_for_resolves_slots() -> None:
    obj = _make_plane("MatHost")
    try:
        mat_a = obj.data.materials[0]
        mat_b = obj.data.materials[1]
        assert rg.material_index_for(obj, mat_a) == 0
        assert rg.material_index_for(obj, mat_b) == 1
        assert rg.material_index_for(obj, None) == -1
        other = bpy.data.materials.new("FDT_Unused_Mat")
        try:
            assert rg.material_index_for(obj, other) == -1
        finally:
            bpy.data.materials.remove(other)
    finally:
        _cleanup(obj)


def test_b09_selected_faces_object_mode() -> None:
    obj = _make_plane("SelHost")
    try:
        for polygon in obj.data.polygons:
            polygon.select = False
        obj.data.polygons[1].select = True
        obj.data.polygons[3].select = True
        assert sorted(rg.selected_face_indices(bpy.context, obj)) == [1, 3]
    finally:
        _cleanup(obj)


def test_b10_selected_faces_filtered_by_material() -> None:
    obj = _make_plane("SelFilter")
    try:
        for polygon in obj.data.polygons:
            polygon.select = True
        even = rg.selected_face_indices(bpy.context, obj, material_index=0)
        assert even and all(obj.data.polygons[i].material_index == 0 for i in even)
        odd = rg.selected_face_indices(bpy.context, obj, material_index=1)
        assert odd and all(obj.data.polygons[i].material_index == 1 for i in odd)
        assert len(even) + len(odd) == len(obj.data.polygons)
    finally:
        _cleanup(obj)


def test_b11_selected_faces_edit_mode() -> None:
    obj = _make_plane("SelEdit")
    try:
        bpy.context.view_layer.objects.active = obj
        for polygon in obj.data.polygons:
            polygon.select = False
        obj.data.polygons[2].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            selected = rg.selected_face_indices(bpy.context, obj)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        assert selected == [2], selected
    finally:
        _cleanup(obj)


# ---------------------------------------------------------------------------
# Bounds / BVH
# ---------------------------------------------------------------------------


def test_b12_face_bounds_covers_tagged_faces_only() -> None:
    obj = _make_plane("BoundsHost", size=2.0, divisions=4)
    try:
        full = rg.face_bounds(obj.data, list(range(len(obj.data.polygons))))
        assert full is not None
        min_co, max_co = full
        assert math.isclose(min_co.x, -1.0, abs_tol=VEC_TOL), min_co
        assert math.isclose(max_co.x, 1.0, abs_tol=VEC_TOL), max_co

        single = rg.face_bounds(obj.data, [0])
        assert single is not None
        s_min, s_max = single
        assert s_max.x - s_min.x < (max_co.x - min_co.x)
    finally:
        _cleanup(obj)


def test_b13_face_bounds_empty_set_is_none() -> None:
    obj = _make_plane("BoundsEmpty")
    try:
        assert rg.face_bounds(obj.data, []) is None
        assert rg.face_bounds(obj.data, [9999]) is None
    finally:
        _cleanup(obj)


def test_b14_bvh_builds_from_subset_and_complement() -> None:
    obj = _make_plane("BvhHost")
    try:
        faces = [0, 1, 2]
        bvh = rg.build_bvh(obj.data, faces)
        assert bvh is not None
        complement = rg.build_complement_bvh(obj.data, faces)
        assert complement is not None
        # A BVH over everything must be rejected as a "complement".
        assert rg.build_complement_bvh(obj.data, range(len(obj.data.polygons))) is None
        assert rg.build_bvh(obj.data, []) is None
    finally:
        _cleanup(obj)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _params(**overrides) -> rg.RebarParams:
    base = dict(
        spacing=(0.5, 0.5, 0.5),
        align=(False, False, True),
        offset=(0.0, 0.0, 0.0),
        radius=0.006,
        sink_depth=0.03,
        min_length=0.2,
        max_length=0.2,
        bending_strength=0.16,
        subdivisions=3,
        seed=1,
        angle_threshold=0.5,
        normal_bias=0.5,
        gravity_bias=0.5,
        smooth_radius=0.3,
        avoidance_strength=0.5,
        max_bars=5000,
    )
    base.update(overrides)
    return rg.RebarParams(**base)


def test_b15_generate_plane_produces_bars() -> None:
    obj = _make_plane("GenPlane", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        result = rg.generate(obj.data, faces, _params())
        assert result.bar_count > 0, "no bars generated from a tagged plane"
        assert not result.truncated
        for points, subdivisions, _twist in result.splines:
            # Bent bars may be densified for smoothness (rebar_spline).
            assert len(points) >= subdivisions + 1
    finally:
        _cleanup(obj)


def test_b16_generate_only_uses_tagged_faces() -> None:
    obj = _make_plane("GenSubset", size=2.0, divisions=4)
    try:
        all_faces = list(range(len(obj.data.polygons)))
        full = rg.generate(obj.data, all_faces, _params())
        one_face = rg.generate(obj.data, [0], _params())
        assert full.bar_count > 0
        assert one_face.bar_count < full.bar_count, (one_face.bar_count, full.bar_count)
    finally:
        _cleanup(obj)


def test_b17_generate_respects_max_bars() -> None:
    obj = _make_plane("GenCap", size=2.0, divisions=8)
    try:
        faces = list(range(len(obj.data.polygons)))
        result = rg.generate(obj.data, faces, _params(max_bars=2))
        assert result.truncated, "cap was not reported"
        assert result.bar_count <= 2, result.bar_count
    finally:
        _cleanup(obj)


def test_b18_generate_is_deterministic_for_a_seed() -> None:
    obj = _make_plane("GenSeed", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        first = rg.generate(obj.data, faces, _params(seed=7))
        second = rg.generate(obj.data, faces, _params(seed=7))
        third = rg.generate(obj.data, faces, _params(seed=8))

        assert first.bar_count == second.bar_count
        for (pa, _sa, ta), (pb, _sb, tb) in zip(first.splines, second.splines):
            assert math.isclose(ta, tb, abs_tol=1e-12), "twist must be seed-reproducible"
            for a, b in zip(pa, pb):
                assert (a - b).length < VEC_TOL

        assert first.bar_count == third.bar_count
        differs = any(
            (a - b).length > VEC_TOL
            for (pa, _sa, _ta), (pb, _sb, _tb) in zip(first.splines, third.splines)
            for a, b in zip(pa, pb)
        )
        assert differs, "different seeds produced identical geometry"
    finally:
        _cleanup(obj)


def test_b19_angle_threshold_rejects_grazing_hits() -> None:
    obj = _make_plane("GenGrazing", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        # A flat plane's normals are parallel to the Z rays, so requiring a
        # near-parallel dot lets them through; a threshold above 1 rejects all.
        permissive = rg.generate(obj.data, faces, _params(angle_threshold=0.5))
        impossible = rg.generate(obj.data, faces, _params(angle_threshold=1.01))
        assert permissive.bar_count > 0
        assert impossible.bar_count == 0, impossible.bar_count
        assert impossible.skipped_grazing > 0
    finally:
        _cleanup(obj)


def test_b20_generate_handles_empty_face_set() -> None:
    obj = _make_plane("GenEmpty")
    try:
        result = rg.generate(obj.data, [], _params())
        assert result.is_empty
        assert result.bar_count == 0
    finally:
        _cleanup(obj)


def test_b21_y_axis_bars_are_lifted_by_one_diameter() -> None:
    obj = _make_box("GenLift", size=2.0)
    try:
        faces = list(range(len(obj.data.polygons)))
        # Y-directed bars sample a Z grid shifted up by one diameter (2r) so
        # they rest on top of the X-directed bars.  With spacing 0.5 every
        # start Z must therefore land on 0.012 + k*0.5, never on the raw grid.
        lift = 2.0 * 0.006
        result = rg.generate(obj.data, faces, _params(align=(False, True, False)))
        assert result.bar_count > 0, "no Y bars generated from the box"
        for points, _sub, _twist in result.splines:
            offset = (points[0].z - lift) / 0.5
            assert abs(offset - round(offset)) < VEC_TOL, points[0]
    finally:
        _cleanup(obj)


def test_b22_zero_bend_gives_straight_un_twisted_bars() -> None:
    obj = _make_plane("GenStraight", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        result = rg.generate(obj.data, faces, _params(bending_strength=0.0))
        assert result.bar_count > 0
        for points, _sub, twist in result.splines:
            assert twist == 0.0, "a straight bar must never twist"
            axis = (points[-1] - points[0]).normalized()
            for point in points:
                assert (point - points[0]).cross(axis).length < VEC_TOL
    finally:
        _cleanup(obj)


def test_b23_bend_scale_changes_the_shapes() -> None:
    obj = _make_plane("GenScale", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        tight = rg.generate(obj.data, faces, _params(bend_scale=0.05))
        wide = rg.generate(obj.data, faces, _params(bend_scale=1.0))
        assert tight.bar_count == wide.bar_count
        differs = any(
            (a - b).length > VEC_TOL
            for (pa, _sa, _ta), (pb, _sb, _tb) in zip(tight.splines, wide.splines)
            for a, b in zip(pa, pb)
        )
        assert differs, "bend scale had no effect on the shapes"
    finally:
        _cleanup(obj)


def test_b24_twist_stays_within_the_bend_amount_limit() -> None:
    from FullDestruc_Tool.core import rebar_spline as rs

    obj = _make_plane("GenTwist", size=2.0, divisions=4)
    try:
        faces = list(range(len(obj.data.polygons)))
        result = rg.generate(obj.data, faces, _params(bending_strength=0.16))
        assert result.bar_count > 0
        limit = rs.twist_limit(0.16)
        assert limit > 0.0
        assert all(abs(twist) <= limit + VEC_TOL for _p, _s, twist in result.splines)
        # With 0.16 m of bend the half-turn (pi) cap must not be reached.
        assert all(abs(twist) < math.pi - 0.1 for _p, _s, twist in result.splines)
    finally:
        _cleanup(obj)


ALL_TESTS = [
    test_b01_group_attribute_is_created_at_no_group,
    test_b02_tag_and_read_roundtrip,
    test_b03_tag_two_groups_are_independent,
    test_b04_untag_faces_clears_only_targets,
    test_b05_next_group_index_reuses_lowest_free_slot,
    test_b06_tag_rejects_negative_group,
    test_b07_untag_on_untagged_mesh_is_a_noop,
    test_b08_material_index_for_resolves_slots,
    test_b09_selected_faces_object_mode,
    test_b10_selected_faces_filtered_by_material,
    test_b11_selected_faces_edit_mode,
    test_b12_face_bounds_covers_tagged_faces_only,
    test_b13_face_bounds_empty_set_is_none,
    test_b14_bvh_builds_from_subset_and_complement,
    test_b15_generate_plane_produces_bars,
    test_b16_generate_only_uses_tagged_faces,
    test_b17_generate_respects_max_bars,
    test_b18_generate_is_deterministic_for_a_seed,
    test_b19_angle_threshold_rejects_grazing_hits,
    test_b20_generate_handles_empty_face_set,
    test_b21_y_axis_bars_are_lifted_by_one_diameter,
    test_b22_zero_bend_gives_straight_un_twisted_bars,
    test_b23_bend_scale_changes_the_shapes,
    test_b24_twist_stays_within_the_bend_amount_limit,
]
