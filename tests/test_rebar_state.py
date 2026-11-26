"""Phase C tests: registry, params snapshots, collection placement and delete guard."""

from __future__ import annotations

import math

import bmesh
import bpy

from FullDestruc_Tool.core import rebar_generate as rg
from FullDestruc_Tool.core import rebar_state as st
from FullDestruc_Tool.core import rebar_spline as rs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reset_scene() -> None:
    scene = bpy.context.scene
    if st.REGISTRY_PROP in scene:
        del scene[st.REGISTRY_PROP]


def _make_host(name: str = "Host") -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_rebar(sid: str, name: str | None = None) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"RebarMesh_{sid}")
    obj = bpy.data.objects.new(name or st.object_name(sid), mesh)
    bpy.context.scene.collection.objects.link(obj)
    st.mark_object(obj, sid)
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


def _entry(sid: str, host, group: int = 0, **kwargs) -> st.RebarEntry:
    values = {
        "sid": sid,
        "host": getattr(host, "name", host),
        "object": st.object_name(sid),
        "collection": st.collection_name(sid),
        "group": group,
    }
    values.update(kwargs)
    return st.RebarEntry(**values)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def test_c01_default_params_match_red_panda() -> None:
    params = st.default_params()
    assert params["bar_size"] == "CUSTOM"
    assert params["diameter_mm"] == 40.0
    assert math.isclose(params["sink_depth"], 0.05, rel_tol=1e-12)
    assert math.isclose(params["min_length"], 0.1, rel_tol=1e-12)
    assert math.isclose(params["max_length"], 0.5, rel_tol=1e-12)
    assert math.isclose(params["bending_strength"], 0.03, rel_tol=1e-12)
    assert math.isclose(params["bend_scale"], 0.4, rel_tol=1e-12)
    assert params["subdivisions"] == 6
    assert params["random_seed"] == 0
    assert params["max_bars"] == 5000
    assert params["fast_drag"] is True
    assert params["ui_show_rebar"] is False


def test_c02_normalize_params_repairs_bad_values() -> None:
    params = st.normalize_params(
        {
            "spacing_x": 0.0,
            "spacing_y": -3.0,
            "subdivisions": 999,
            "max_bars": -1,
            "diameter_mm": 0.0,
            "min_length": 0.9,
            "max_length": 0.2,
            "angle_threshold": 7.0,
            "align_z": "not-a-bool",
            "bend_scale": 99.0,
            "unknown_key": 123,
        }
    )
    assert params["spacing_x"] > 0.0 and params["spacing_y"] > 0.0
    assert params["subdivisions"] <= 12
    # Bend Scale is a wavelength: clamped into a sane 50–1000 mm band.
    assert 0.05 <= params["bend_scale"] <= 1.0, params["bend_scale"]
    assert params["max_bars"] >= 1
    assert params["diameter_mm"] > 0.0
    assert params["min_length"] <= params["max_length"]
    assert params["angle_threshold"] <= 1.0
    assert "unknown_key" not in params
    # Every key from the defaults must still be present.
    assert set(params) == set(st.default_params())


def test_c03_normalize_params_accepts_empty_input() -> None:
    assert st.normalize_params(None) == st.default_params()
    assert st.normalize_params("garbage") == st.default_params()


def test_c04_params_to_generation_maps_units_and_radius() -> None:
    params = st.default_params()
    gen = st.params_to_generation(params)
    assert math.isclose(gen.radius, rs.diameter_to_radius(40.0), rel_tol=1e-12)
    assert gen.spacing == (0.4, 0.4, 0.4)
    assert math.isclose(gen.sink_depth, 0.05, rel_tol=1e-12)
    assert gen.subdivisions == 6
    assert gen.seed == 0
    assert gen.max_bars == 5000
    assert isinstance(gen, rg.RebarParams)


def test_c05_params_snapshot_roundtrips_on_object() -> None:
    obj = _make_rebar("snap")
    try:
        custom = st.normalize_params({"spacing_x": 0.4, "bar_size": "DB20", "random_seed": 9})
        st.save_params(obj, custom)
        loaded = st.load_params(obj)
        assert loaded["spacing_x"] == 0.4
        assert loaded["bar_size"] == "DB20"
        assert loaded["random_seed"] == 9
        assert set(loaded) == set(st.default_params())
    finally:
        _cleanup(obj)


def test_c06_corrupt_object_snapshot_falls_back_to_defaults() -> None:
    obj = _make_rebar("bad")
    try:
        obj[st.OBJ_PARAMS_PROP] = "{not json"
        assert st.load_params(obj) == st.default_params()
    finally:
        _cleanup(obj)


# ---------------------------------------------------------------------------
# Names and ids
# ---------------------------------------------------------------------------


def test_c07_sid8_and_names_are_bounded() -> None:
    sid = "abcdef1234567890"
    assert st.short_sid(sid) == "abcdef12"
    assert len(st.collection_name(sid)) <= 63
    assert st.collection_name(sid) == "FDT_Rebar_abcdef12"
    assert st.object_name(sid) == "Rebar_abcdef12"
    assert len(st.collection_name("x" * 200)) <= 63


def test_c08_new_sid_avoids_used_values() -> None:
    used = {st.new_sid() for _ in range(32)}
    assert len(used) == 32
    assert st.new_sid(used) not in used


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_c09_registry_roundtrip() -> None:
    _reset_scene()
    try:
        entry = st.RebarEntry(
            sid="00112233",
            label="Wall_A · 00112233",
            host="Wall",
            target_material="Concrete_Ruin",
            rebar_material="ST_Metal_Rebar_Ruin_01",
            collection="FDT_Rebar_00112233",
            object="Rebar_00112233",
            group=2,
            baked=True,
            params=st.normalize_params({"spacing_x": 0.25}),
        )
        assert st.add_entry(entry)
        loaded = st.load_registry()
        assert len(loaded) == 1
        restored = loaded[0]
        assert restored.sid == entry.sid
        assert restored.label == entry.label
        assert restored.group == 2
        assert restored.baked is True
        assert restored.params["spacing_x"] == 0.25
        assert st.find_entry(entry.sid) is not None
        assert st.find_entry("nope") is None
    finally:
        _reset_scene()


def test_c10_add_entry_rejects_duplicate_and_update_works() -> None:
    _reset_scene()
    try:
        entry = _entry("aabbccdd", _make_host("DupHost"))
        assert st.add_entry(entry)
        assert not st.add_entry(entry)
        entry.label = "renamed"
        assert st.update_entry(entry)
        assert st.find_entry(entry.sid).label == "renamed"
        assert st.remove_entry(entry.sid)
        assert st.find_entry(entry.sid) is None
        assert not st.remove_entry(entry.sid)
    finally:
        _reset_scene()
        for obj in list(bpy.data.objects):
            if obj.name.startswith("DupHost"):
                _cleanup(obj)


def test_c11_registry_survives_corrupt_json() -> None:
    _reset_scene()
    try:
        bpy.context.scene[st.REGISTRY_PROP] = "[invalid"
        assert st.load_registry() == []
        bpy.context.scene[st.REGISTRY_PROP] = '{"not": "a list"}'
        assert st.load_registry() == []
    finally:
        _reset_scene()


def test_c12_entry_skips_records_without_sid() -> None:
    _reset_scene()
    try:
        bpy.context.scene[st.REGISTRY_PROP] = '[{"label": "no sid"}, {"sid": "ok123456"}]'
        loaded = st.load_registry()
        assert len(loaded) == 1
        assert loaded[0].sid == "ok123456"
    finally:
        _reset_scene()


# ---------------------------------------------------------------------------
# Object / collection resolution
# ---------------------------------------------------------------------------


def test_c13_object_sid_back_reference() -> None:
    obj = _make_rebar("11223344")
    try:
        assert st.object_sid(obj) == "11223344"
        assert st.load_params(obj) == st.default_params()
        st.mark_object(obj, "99999999")
        assert st.object_sid(obj) == "99999999"
    finally:
        _cleanup(obj)


def test_c14_resolve_object_prefers_sid_over_cosmetic_name() -> None:
    host = _make_host("ResolveHost")
    rebar = _make_rebar("44556677", name="Totally_Renamed")
    entry = _entry("44556677", host, object=rebar.name)
    try:
        assert st.resolve_object(entry) is rebar
        # Cosmetic rename must not break ownership.
        rebar.name = "Another_Random_Name"
        assert st.resolve_object(entry) is rebar
        assert st.resolve_host(entry) is host
    finally:
        _cleanup(host, rebar)


def test_c15_ensure_collection_and_move() -> None:
    host = _make_host("ColHost")
    rebar = _make_rebar("a1b2c3d4")
    entry = _entry("a1b2c3d4", host)
    try:
        col = st.ensure_collection(entry)
        assert col.name == "FDT_Rebar_a1b2c3d4"
        assert col.name in bpy.context.scene.collection.children
        st.move_to_collection(rebar, entry)
        assert [c.name for c in rebar.users_collection] == [col.name]
        # Idempotent: calling again must not duplicate the link.
        st.move_to_collection(rebar, entry)
        assert len(rebar.users_collection) == 1
    finally:
        _cleanup(host, rebar)


def test_c16_next_group_for_host_accounts_for_registry_and_tags() -> None:
    _reset_scene()
    host = _make_host("GroupHost")
    try:
        assert st.next_group_for_host(host) == 0
        entry = _entry("11111111", host, group=0)
        st.add_entry(entry)
        assert st.next_group_for_host(host) == 1
        rg.ensure_group_attribute(host.data)
        rg.tag_faces(host.data, [0], 1)
        assert st.next_group_for_host(host) == 2
    finally:
        _reset_scene()
        _cleanup(host)


# ---------------------------------------------------------------------------
# Delete guard, tags, stale
# ---------------------------------------------------------------------------


def test_c17_delete_guard_blocks_while_faces_are_tagged() -> None:
    _reset_scene()
    host = _make_host("GuardHost")
    entry = _entry("22222222", host, group=3)
    try:
        st.add_entry(entry)
        rg.tag_faces(host.data, [0, 1], 3)
        ok, reason = st.delete_guard(entry.sid)
        assert not ok
        assert "face" in reason
        assert st.clear_tags_for_entry(entry) == 2
        ok, _reason = st.delete_guard(entry.sid)
        assert ok
    finally:
        _reset_scene()
        _cleanup(host)


def test_c18_clear_tags_only_touches_its_own_group() -> None:
    _reset_scene()
    host = _make_host("ClearHost")
    first = _entry("33333333", host, group=0)
    second = _entry("44444444", host, group=1)
    try:
        rg.tag_faces(host.data, [0, 1], 0)
        rg.tag_faces(host.data, [2], 1)
        assert st.clear_tags_for_entry(first) == 2
        assert rg.tagged_face_indices(host.data, 0) == []
        assert rg.tagged_face_indices(host.data, 1) == [2]
    finally:
        _reset_scene()
        _cleanup(host)


def test_c19_mark_stale_for_host_flags_only_that_host() -> None:
    _reset_scene()
    host_a = _make_host("StaleA")
    host_b = _make_host("StaleB")
    try:
        st.add_entry(_entry("55555555", host_a))
        st.add_entry(_entry("66666666", host_b))
        stale = st.mark_stale_for_host(host_a)
        assert stale == ["55555555"]
        assert st.is_stale("55555555")
        assert not st.is_stale("66666666")
        assert st.mark_stale_for_host(host_a) == []
        assert st.set_stale("55555555", False)
        assert not st.is_stale("55555555")
    finally:
        _reset_scene()
        _cleanup(host_a, host_b)


def test_c20_load_all_generators_reports_missing_host() -> None:
    _reset_scene()
    host = _make_host("LiveHost")
    rebar = _make_rebar("77777777")
    try:
        st.add_entry(_entry("77777777", host, baked=True))
        states = st.load_all_generators()
        assert states["77777777"].valid
        assert states["77777777"].obj is rebar
        assert states["77777777"].label == st.object_name("77777777")

        st.mark_object(rebar, "")
        states = st.load_all_generators()
        assert not states["77777777"].valid, "name-only match must not be adopted"

        st.remove_entry("77777777")
        st.add_entry(_entry("88888888", "MissingHost", baked=True))
        assert not st.load_all_generators()["88888888"].valid
    finally:
        _reset_scene()
        _cleanup(host, rebar)


def test_c21_prune_missing_keeps_entries_with_live_tags() -> None:
    _reset_scene()
    host = _make_host("PruneHost")
    try:
        rg.tag_faces(host.data, [0], 5)
        st.add_entry(_entry("99999999", host, group=5))
        st.add_entry(_entry("aaaaaaaa", "GoneHost", group=0, baked=True))
        removed = st.prune_missing()
        assert removed == ["aaaaaaaa"]
        assert st.find_entry("99999999") is not None
    finally:
        _reset_scene()
        _cleanup(host)


ALL_TESTS = [
    test_c01_default_params_match_red_panda,
    test_c02_normalize_params_repairs_bad_values,
    test_c03_normalize_params_accepts_empty_input,
    test_c04_params_to_generation_maps_units_and_radius,
    test_c05_params_snapshot_roundtrips_on_object,
    test_c06_corrupt_object_snapshot_falls_back_to_defaults,
    test_c07_sid8_and_names_are_bounded,
    test_c08_new_sid_avoids_used_values,
    test_c09_registry_roundtrip,
    test_c10_add_entry_rejects_duplicate_and_update_works,
    test_c11_registry_survives_corrupt_json,
    test_c12_entry_skips_records_without_sid,
    test_c13_object_sid_back_reference,
    test_c14_resolve_object_prefers_sid_over_cosmetic_name,
    test_c15_ensure_collection_and_move,
    test_c16_next_group_for_host_accounts_for_registry_and_tags,
    test_c17_delete_guard_blocks_while_faces_are_tagged,
    test_c18_clear_tags_only_touches_its_own_group,
    test_c19_mark_stale_for_host_flags_only_that_host,
    test_c20_load_all_generators_reports_missing_host,
    test_c21_prune_missing_keeps_entries_with_live_tags,
]
