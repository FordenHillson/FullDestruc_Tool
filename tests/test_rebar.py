"""Headless Blender 5.2 tests for the Concrete Rebars tool.

Run with:
    blender.exe --factory-startup --background --python tests/run_tests.py
"""

from __future__ import annotations

import math

import bpy
from mathutils import Vector

from FullDestruc_Tool.core import rebar_spline as rs

# mathutils.Vector stores single-precision floats, so anything read back out of a
# Vector has ~1e-7 relative error.  Geometry assertions use this tolerance.
VEC_TOL = 1e-5


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_r01_mm_conversion_roundtrip() -> None:
    for mm in (0.0, 6.0, 12.0, 150.0, 5000.0):
        assert math.isclose(rs.m_to_mm(rs.mm_to_m(mm)), mm, rel_tol=1e-12), mm


def test_r02_diameter_to_radius_is_metres_and_nonzero() -> None:
    assert math.isclose(rs.diameter_to_radius(12.0), 0.006, rel_tol=1e-12)
    assert math.isclose(rs.diameter_to_radius(25.0), 0.0125, rel_tol=1e-12)
    # Degenerate diameters must never collapse the profile.
    assert rs.diameter_to_radius(0.0) > 0.0


def test_r03_texel_scale_matches_fdt_convention() -> None:
    assert math.isclose(rs.texel_scale(512.0, 1024.0), 0.5, rel_tol=1e-12)
    # Guard against zero texture size.
    assert math.isclose(rs.texel_scale(512.0, 0.0), 0.5, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Grid lattice
# ---------------------------------------------------------------------------


def test_r04_grid_coords_basic() -> None:
    coords = rs.get_grid_coords(0.0, 1.0, 0.4)
    assert len(coords) == 3, coords
    for got, want in zip(coords, (0.0, 0.4, 0.8)):
        assert math.isclose(got, want, abs_tol=1e-9), coords


def test_r05_grid_coords_honours_offset() -> None:
    coords = rs.get_grid_coords(0.0, 1.0, 0.4, 0.1)
    assert len(coords) == 3, coords
    for got, want in zip(coords, (0.1, 0.5, 0.9)):
        assert math.isclose(got, want, abs_tol=1e-9), coords


def test_r06_grid_coords_rejects_zero_spacing() -> None:
    assert rs.get_grid_coords(0.0, 1.0, 0.0) == []
    assert rs.get_grid_coords(0.0, 1.0, -1.0) == []


def test_r07_grid_coords_handles_reversed_range() -> None:
    assert rs.get_grid_coords(1.0, 0.0, 0.5) == rs.get_grid_coords(0.0, 1.0, 0.5)


def test_r08_grid_coords_empty_when_valid_lines_absent() -> None:
    # Range sits between two grid lines.
    assert rs.get_grid_coords(0.51, 0.99, 0.5) == []


def test_r09_estimate_bar_count_multiplies_active_axes() -> None:
    count = rs.estimate_bar_count(
        Vector((0.0, 0.0, 0.0)),
        Vector((1.0, 1.0, 1.0)),
        spacings=(0.5, 0.5, 0.5),
        align=(True, True, True),
    )
    assert count == 3 * 3 * 3, count
    one_axis = rs.estimate_bar_count(
        Vector((0.0, 0.0, 0.0)),
        Vector((1.0, 1.0, 1.0)),
        spacings=(0.5, 0.5, 0.5),
        align=(False, False, True),
    )
    assert one_axis == 3, one_axis


# ---------------------------------------------------------------------------
# Bend + sampling
# ---------------------------------------------------------------------------


def test_r10_proportional_subdivision_scales_with_length() -> None:
    assert rs.proportional_subdivision(1.0, 1.0, 6) == 6
    assert rs.proportional_subdivision(0.5, 1.0, 6) == 3
    assert rs.proportional_subdivision(0.0, 1.0, 6) == 1
    # Never drops below one segment.
    assert rs.proportional_subdivision(0.01, 1.0, 3) == 1


def _spec(**overrides) -> rs.BendSpec:
    base = dict(
        direction=Vector((1.0, 0.0, 0.0)),
        lateral=Vector((0.0, 1.0, 0.0)),
        amplitude=0.2,
        wavelength=0.3,
        phase1=0.7,
        phase2=1.9,
        phase3=2.6,
        wave_scale1=0.7,
        wave_scale2=0.4,
    )
    base.update(overrides)
    return rs.BendSpec(**base)


def test_r11_build_rebar_points_count_and_embedded_section() -> None:
    pts = rs.build_rebar_points(
        start=Vector((0.0, 0.0, 0.0)),
        direction=Vector((0.0, 0.0, 1.0)),
        total_len=1.0,
        sink_depth=0.2,
        spec=_spec(),
        subdivisions=10,
    )
    # A bent bar is densified beyond the requested subdivisions (capped).
    n = len(pts) - 1
    assert 11 <= len(pts) <= 41, len(pts)

    embedded_end = round(0.2 * n)
    for i in range(embedded_end + 1):
        assert math.isclose(pts[i].x, 0.0, abs_tol=VEC_TOL), f"bend leaked into embedded section at {i}"
        assert math.isclose(pts[i].z, i / n, abs_tol=VEC_TOL)


def test_r12_build_rebar_points_bends_only_after_embed() -> None:
    pts = rs.build_rebar_points(
        start=Vector((0.0, 0.0, 0.0)),
        direction=Vector((0.0, 0.0, 1.0)),
        total_len=1.0,
        sink_depth=0.2,
        spec=_spec(),
        subdivisions=10,
    )
    # Right at the surface the offset is exactly zero (no jump out of the hole).
    assert math.isclose(pts[2].x, 0.0, abs_tol=VEC_TOL), pts[2].x
    # Further out the wave actually moves the bar sideways.
    assert any(abs(p.x) > 1e-3 for p in pts[3:]), "bend never left the axis"


def test_r13_build_rebar_points_survives_degenerate_input() -> None:
    assert len(rs.build_rebar_points(Vector(), Vector((0, 0, 1)), 0.0, 0.0, _spec(), 3)) == 1
    zero_dir = rs.build_rebar_points(Vector(), Vector((0, 0, 0)), 1.0, 0.0, _spec(), 3)
    assert len(zero_dir) == 1


def test_r14_compose_bend_spec_keeps_amplitude_and_unit_direction() -> None:
    spec = rs.compose_bend_spec(
        normal_perp=Vector((0.0, 0.0, 0.0)),
        gravity_perp=Vector((0.0, 0.0, 0.0)),
        avoidance_perp=Vector((1.0, 0.0, 0.0)),
        random_b1=Vector((1.0, 0.0, 0.0)),
        random_b2=Vector((0.0, 1.0, 0.0)),
        normal_bias=0.5,
        gravity_bias=0.5,
        avoidance_strength=5.0,
        bending_strength=0.16,
        bend_scale=0.2,
        phase1=0.1, phase2=0.2, phase3=0.3,
        wave_scale1=0.7, wave_scale2=0.4,
        twist=0.9,
    )
    assert math.isclose(spec.amplitude, 0.16, rel_tol=VEC_TOL), spec.amplitude
    assert math.isclose(spec.direction.length, 1.0, rel_tol=VEC_TOL), spec.direction
    assert math.isclose(spec.wavelength, 0.2, rel_tol=VEC_TOL)
    assert math.isclose(spec.twist, 0.9, rel_tol=VEC_TOL)
    # The lateral wobble must stay out of the dominant bend plane.
    assert abs(spec.lateral.dot(spec.direction)) < VEC_TOL


def test_r15_compose_bend_spec_applies_shared_bias() -> None:
    # Normal bias dwarfs the random component, so the dominant direction
    # points along +X.
    spec = rs.compose_bend_spec(
        normal_perp=Vector((1.0, 0.0, 0.0)),
        gravity_perp=Vector((0.0, 0.0, 0.0)),
        avoidance_perp=Vector((0.0, 0.0, 0.0)),
        random_b1=Vector((0.0, 1.0, 0.0)),
        random_b2=Vector((0.0, 1.0, 0.0)),
        normal_bias=100.0,
        gravity_bias=0.0,
        avoidance_strength=0.0,
        bending_strength=10.0,
        bend_scale=0.2,
        phase1=0.0, phase2=0.0, phase3=0.0,
        wave_scale1=0.7, wave_scale2=0.4,
        twist=0.0,
    )
    assert math.isclose(spec.amplitude, 10.0, rel_tol=1e-4), spec.amplitude
    assert spec.direction.x > 0.99, spec.direction
    assert spec.direction.y > 0.0, spec.direction


def test_r15c_twist_limit_scales_with_bend_amount() -> None:
    assert rs.twist_limit(0.0) == 0.0
    assert math.isclose(rs.twist_limit(rs.TWIST_REF_BEND), math.pi, rel_tol=VEC_TOL)
    assert math.isclose(rs.twist_limit(rs.TWIST_REF_BEND * 0.5), math.pi / 2, rel_tol=VEC_TOL)
    # Above the reference bend the twist saturates at a half turn.
    assert math.isclose(rs.twist_limit(10.0), math.pi, rel_tol=VEC_TOL)
    assert math.isclose(rs.twist_limit(-3.0), 0.0, rel_tol=VEC_TOL)


def test_r15d_bend_offset_zero_at_surface_and_bounded() -> None:
    spec = _spec()
    assert rs.bend_offset(spec, 0.0, 0.5).length < 1e-12
    assert rs.bend_offset(spec, -1.0, 0.5).length < 1e-12
    assert rs.bend_offset(_spec(amplitude=0.0), 0.25, 0.5).length < 1e-12
    max_off = max(rs.bend_offset(spec, s, 0.5).length for s in (i / 100.0 * 0.5 for i in range(1, 100)))
    assert max_off <= 1.5 * spec.amplitude + VEC_TOL, max_off
    # The same spec always produces the same offset (reproducible shapes).
    a = rs.bend_offset(spec, 0.31, 0.5)
    assert (a - rs.bend_offset(spec, 0.31, 0.5)).length < 1e-12


def test_r15b_perpendicular_component_strips_the_axis() -> None:
    perp = rs.perpendicular_component(Vector((0.0, 0.0, 1.0)), Vector((0.0, 0.0, 1.0)))
    assert perp.length < 1e-9, perp
    sideways = rs.perpendicular_component(Vector((1.0, 0.0, 1.0)), Vector((0.0, 0.0, 1.0)))
    assert math.isclose(sideways.x, 1.0, abs_tol=VEC_TOL), sideways


# ---------------------------------------------------------------------------
# Mesher
# ---------------------------------------------------------------------------


def _straight_bar(subdivisions: int = 4, length: float = 1.0) -> tuple[list[Vector], int]:
    pts = [Vector((0.0, 0.0, length * i / subdivisions)) for i in range(subdivisions + 1)]
    return pts, subdivisions


def test_r16_spline_mesh_vertex_and_face_counts() -> None:
    payload = rs.build_spline_mesh([_straight_bar(4)], radius=0.006, uv_scale=0.5)
    assert len(payload.vertices) == rs.spline_vertex_count(4) == 20
    assert len(payload.faces) == rs.spline_face_count(4) == 17
    assert len(payload.smooth_faces) == len(payload.faces)


def test_r17_spline_mesh_face_indices_in_range() -> None:
    payload = rs.build_spline_mesh([_straight_bar(6)], radius=0.01, uv_scale=0.5)
    vcount = len(payload.vertices)
    for face in payload.faces:
        for index in face:
            assert 0 <= index < vcount, (face, vcount)


def test_r18_spline_mesh_uv_count_matches_loop_count() -> None:
    payload = rs.build_spline_mesh([_straight_bar(5)], radius=0.006, uv_scale=0.5)
    expected_loops = sum(len(face) for face in payload.faces)
    assert len(payload.uvs) == expected_loops, (len(payload.uvs), expected_loops)


def test_r19_spline_mesh_uv_islands_do_not_overlap() -> None:
    bars = [_straight_bar(3), _straight_bar(3)]
    radius = 0.006
    scale = 0.5
    payload = rs.build_spline_mesh(bars, radius=radius, uv_scale=scale)
    side_width = math.sqrt(2.0) * radius * scale

    first = payload.uvs[: rs.spline_face_count(3) * 4]
    second_offset = rs.spline_face_count(3) * 4
    second = payload.uvs[second_offset:]

    assert max(u for u, _v in first) <= min(u for u, _v in second) + VEC_TOL, "UV islands overlap"
    assert min(u for u, _v in second) > max(u for u, _v in first)
    assert math.isclose(min(u for u, _v in second) - max(u for u, _v in first), 0.02, abs_tol=VEC_TOL)


def test_r20_spline_mesh_marks_tip_cap_flat_and_sides_smooth() -> None:
    payload = rs.build_spline_mesh([_straight_bar(3)], radius=0.006, uv_scale=0.5)
    assert payload.smooth_faces[-1] is False, "tip cap must be flat shaded"
    assert all(payload.smooth_faces[:-1]), "side quads must be smooth shaded"


def test_r21_spline_mesh_seam_chain_runs_along_the_bar() -> None:
    subdivisions = 4
    payload = rs.build_spline_mesh([_straight_bar(subdivisions)], radius=0.006, uv_scale=0.5)
    # One seam edge per segment along corner 0.
    for i in range(subdivisions):
        assert tuple(sorted((i * 4, (i + 1) * 4))) in payload.seam_edges
    assert len(payload.seam_edges) >= subdivisions + 3


def test_r22_spline_mesh_skips_degenerate_bars() -> None:
    assert rs.build_spline_mesh([], radius=0.006, uv_scale=0.5).is_empty()
    assert rs.build_spline_mesh([([Vector()], 1)], radius=0.006, uv_scale=0.5).is_empty()
    flat = [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))]
    assert rs.build_spline_mesh([(flat, 1)], radius=0.006, uv_scale=0.5).is_empty()


def test_r23_spline_mesh_radius_controls_profile_size() -> None:
    payload = rs.build_spline_mesh([_straight_bar(2)], radius=0.01, uv_scale=1.0)
    # First ring of the first bar: four corners at distance r from the axis.
    ring = payload.vertices[0:4]
    for x, y, z in ring:
        assert math.isclose(math.hypot(x, y), 0.01, abs_tol=VEC_TOL), (x, y, z)


def test_r25_mesh_twist_rotates_profile_along_the_bar() -> None:
    payload = rs.build_spline_mesh([(_straight_bar(4)[0], 4, math.pi / 2)], radius=0.01, uv_scale=1.0)
    c = math.sqrt(2.0) / 2.0 * 0.01
    # Base ring stays untwisted; the tip ring is rotated +90° around the axis.
    base0 = payload.vertices[0]
    assert math.isclose(base0[0], c, abs_tol=VEC_TOL) and math.isclose(base0[1], c, abs_tol=VEC_TOL)
    tip0 = payload.vertices[16]
    assert math.isclose(tip0[0], -c, abs_tol=VEC_TOL), tip0
    assert math.isclose(tip0[1], c, abs_tol=VEC_TOL), tip0
    assert math.isclose(tip0[2], 1.0, abs_tol=VEC_TOL)

    # Negative twist rotates the other way.
    p_neg = rs.build_spline_mesh([(_straight_bar(4)[0], 4, -math.pi / 2)], radius=0.01, uv_scale=1.0)
    tip_neg = p_neg.vertices[16]
    assert math.isclose(tip_neg[0], c, abs_tol=VEC_TOL) and math.isclose(tip_neg[1], -c, abs_tol=VEC_TOL)


def test_r26_mesh_legacy_twotuple_means_no_twist() -> None:
    explicit = rs.build_spline_mesh([(_straight_bar(4)[0], 4, 0.0)], radius=0.006, uv_scale=0.5)
    legacy = rs.build_spline_mesh([_straight_bar(4)], radius=0.006, uv_scale=0.5)
    assert [Vector(v) for v in explicit.vertices] == [Vector(v) for v in legacy.vertices]


def test_r24_apply_spline_mesh_writes_uvs_and_seams() -> None:
    payload = rs.build_spline_mesh([_straight_bar(4)], radius=0.006, uv_scale=0.5)
    mesh = bpy.data.meshes.new("FDT_Rebar_TestMesh")
    try:
        rs.apply_spline_mesh(mesh, payload)
        assert len(mesh.vertices) == len(payload.vertices)
        assert len(mesh.polygons) == len(payload.faces)
        assert mesh.uv_layers.get("UVMap0") is not None
        assert any(edge.use_seam for edge in mesh.edges), "no seam edges were marked"
        assert mesh.polygons[-1].use_smooth is False
    finally:
        bpy.data.meshes.remove(mesh)


ALL_TESTS = [
    test_r01_mm_conversion_roundtrip,
    test_r02_diameter_to_radius_is_metres_and_nonzero,
    test_r03_texel_scale_matches_fdt_convention,
    test_r04_grid_coords_basic,
    test_r05_grid_coords_honours_offset,
    test_r06_grid_coords_rejects_zero_spacing,
    test_r07_grid_coords_handles_reversed_range,
    test_r08_grid_coords_empty_when_valid_lines_absent,
    test_r09_estimate_bar_count_multiplies_active_axes,
    test_r10_proportional_subdivision_scales_with_length,
    test_r11_build_rebar_points_count_and_embedded_section,
    test_r12_build_rebar_points_bends_only_after_embed,
    test_r13_build_rebar_points_survives_degenerate_input,
    test_r14_compose_bend_spec_keeps_amplitude_and_unit_direction,
    test_r15_compose_bend_spec_applies_shared_bias,
    test_r15c_twist_limit_scales_with_bend_amount,
    test_r15d_bend_offset_zero_at_surface_and_bounded,
    test_r15b_perpendicular_component_strips_the_axis,
    test_r16_spline_mesh_vertex_and_face_counts,
    test_r17_spline_mesh_face_indices_in_range,
    test_r18_spline_mesh_uv_count_matches_loop_count,
    test_r19_spline_mesh_uv_islands_do_not_overlap,
    test_r20_spline_mesh_marks_tip_cap_flat_and_sides_smooth,
    test_r21_spline_mesh_seam_chain_runs_along_the_bar,
    test_r22_spline_mesh_skips_degenerate_bars,
    test_r23_spline_mesh_radius_controls_profile_size,
    test_r25_mesh_twist_rotates_profile_along_the_bar,
    test_r26_mesh_legacy_twotuple_means_no_twist,
    test_r24_apply_spline_mesh_writes_uvs_and_seams,
]
