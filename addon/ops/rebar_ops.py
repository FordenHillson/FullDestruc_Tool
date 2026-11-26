"""Operators and live-preview glue for the Concrete Rebars tool.

Design rules enforced here:

* Property callbacks never touch geometry — they only call
  :func:`request_rebuild`, which debounces through ``bpy.app.timers``.
* Bake is the deliberate commit step and the only place that marks an entry
  baked.  No auto-bake timer exists.
* Ownership is always ``sid``-based; the Outliner label is cosmetic.
"""

from __future__ import annotations

import math

import bpy
from bpy.props import EnumProperty, StringProperty
from mathutils import Vector

from ..core import rebar_debris as rd
from ..core import rebar_generate as rg
from ..core import rebar_overlay, rebar_state as st
from ..core import rebar_spline as rs

DEBOUNCE_SEC = 0.05
COMMIT_SEC = 0.25

_REBUILD_HANDLE = None
_COMMIT_HANDLE = None
_LOADING = False


# ---------------------------------------------------------------------------
# Settings <-> params boundary (mm in RNA, metres everywhere downstream)
# ---------------------------------------------------------------------------


def settings_to_params(settings) -> dict:
    """Snapshot the Scene rebar buffer into a metres-based params dict."""
    params = st.default_params()
    params.update(
        {
            "bar_size": str(getattr(settings, "bar_size", "CUSTOM") or "CUSTOM"),
            "diameter_mm": float(getattr(settings, "diameter_mm", 40.0)),
            "align_x": bool(getattr(settings, "align_x", True)),
            "align_y": bool(getattr(settings, "align_y", True)),
            "align_z": bool(getattr(settings, "align_z", True)),
            "spacing_x": rs.mm_to_m(getattr(settings, "spacing_x", 400.0)),
            "spacing_y": rs.mm_to_m(getattr(settings, "spacing_y", 400.0)),
            "spacing_z": rs.mm_to_m(getattr(settings, "spacing_z", 400.0)),
            "offset_x": rs.mm_to_m(getattr(settings, "grid_offset", (0.0, 0.0, 0.0))[0]),
            "offset_y": rs.mm_to_m(getattr(settings, "grid_offset", (0.0, 0.0, 0.0))[1]),
            "offset_z": rs.mm_to_m(getattr(settings, "grid_offset", (0.0, 0.0, 0.0))[2]),
            "sink_depth": rs.mm_to_m(getattr(settings, "sink_depth", 50.0)),
            "min_length": rs.mm_to_m(getattr(settings, "min_length", 100.0)),
            "max_length": rs.mm_to_m(getattr(settings, "max_length", 500.0)),
            "bending_strength": rs.mm_to_m(getattr(settings, "bending_strength", 30.0)),
            "bend_scale": rs.mm_to_m(float(getattr(settings, "bend_scale_mm", 400.0))),
            "subdivisions": int(getattr(settings, "subdivisions", 6)),
            "random_seed": int(getattr(settings, "random_seed", 0)),
            "angle_threshold": float(getattr(settings, "angle_threshold", 0.5)),
            "normal_bias": float(getattr(settings, "normal_bias", 0.5)),
            "gravity_bias": float(getattr(settings, "gravity_bias", 0.5)),
            "smooth_radius": rs.mm_to_m(getattr(settings, "smooth_radius", 300.0)),
            "avoidance_strength": float(getattr(settings, "avoidance_strength", 0.5)),
            "max_bars": int(getattr(settings, "max_bars", 5000)),
            "fast_drag": bool(getattr(settings, "fast_drag", True)),
            "enable_debris": bool(getattr(settings, "enable_debris", False)),
            "debris_density": float(getattr(settings, "debris_density", 0.5)),
            "debris_min_size": rs.mm_to_m(float(getattr(settings, "debris_min_size", 5.0))),
            "debris_max_size": rs.mm_to_m(float(getattr(settings, "debris_max_size", 40.0))),
            "debris_irregularity": float(getattr(settings, "debris_irregularity", 0.4)),
            "debris_concentration": float(getattr(settings, "debris_concentration", 0.7)),
            "debris_material": str(
                getattr(getattr(settings, "debris_material", None), "name", "") or ""
            ),
            "ui_show_rebar": bool(getattr(settings, "ui_show_rebar", False)),
        }
    )
    return st.normalize_params(params)


def params_to_settings(params, settings, *, include_label: bool = False) -> None:
    """Push a metres-based snapshot back into the mm-based RNA buffer."""
    global _LOADING
    params = st.normalize_params(params)
    _LOADING = True
    try:
        size = params["bar_size"]
        valid_sizes = {item[0] for item in _BAR_SIZE_ITEMS}
        settings.bar_size = size if size in valid_sizes else "CUSTOM"
        settings.diameter_mm = params["diameter_mm"]
        settings.align_x = params["align_x"]
        settings.align_y = params["align_y"]
        settings.align_z = params["align_z"]
        settings.spacing_x = rs.m_to_mm(params["spacing_x"])
        settings.spacing_y = rs.m_to_mm(params["spacing_y"])
        settings.spacing_z = rs.m_to_mm(params["spacing_z"])
        # Set the friendly dropdown *after* the detailed fields: editing a
        # spacing field marks it Custom, so restoring a preset must come last.
        settings.density_preset = _density_preset_for(rs.m_to_mm(params["spacing_x"]))
        settings.grid_offset = (
            rs.m_to_mm(params["offset_x"]),
            rs.m_to_mm(params["offset_y"]),
            rs.m_to_mm(params["offset_z"]),
        )
        settings.sink_depth = rs.m_to_mm(params["sink_depth"])
        settings.min_length = rs.m_to_mm(params["min_length"])
        settings.max_length = rs.m_to_mm(params["max_length"])
        settings.bending_strength = rs.m_to_mm(params["bending_strength"])
        settings.bend_scale_mm = rs.m_to_mm(params["bend_scale"])
        settings.subdivisions = params["subdivisions"]
        settings.random_seed = params["random_seed"]
        settings.angle_threshold = params["angle_threshold"]
        settings.normal_bias = params["normal_bias"]
        settings.gravity_bias = params["gravity_bias"]
        settings.smooth_radius = rs.m_to_mm(params["smooth_radius"])
        settings.avoidance_strength = params["avoidance_strength"]
        settings.max_bars = params["max_bars"]
        settings.fast_drag = params["fast_drag"]
        settings.enable_debris = params["enable_debris"]
        settings.debris_density = params["debris_density"]
        settings.debris_min_size = rs.m_to_mm(params["debris_min_size"])
        settings.debris_max_size = rs.m_to_mm(params["debris_max_size"])
        settings.debris_irregularity = params["debris_irregularity"]
        settings.debris_concentration = params["debris_concentration"]
        settings.debris_material = bpy.data.materials.get(params["debris_material"])
        settings.ui_show_rebar = params["ui_show_rebar"]
        if include_label:
            settings.rebar_label = ""
    finally:
        _LOADING = False


_BAR_SIZE_ITEMS = (
    ("RB6", "RB6", ""),
    ("RB9", "RB9", ""),
    ("DB10", "DB10", ""),
    ("DB12", "DB12", ""),
    ("DB16", "DB16", ""),
    ("DB20", "DB20", ""),
    ("DB25", "DB25", ""),
    ("CUSTOM", "Custom", ""),
)


def _density_preset_for(spacing_mm: float) -> str:
    """Pick the friendly dropdown entry that matches a saved spacing value."""
    from ..props import DENSITY_SPACING_MM

    for key, value in DENSITY_SPACING_MM.items():
        if math.isclose(float(spacing_mm), float(value), abs_tol=0.5):
            return key
    return "CUSTOM"


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _rebar_settings(context=None):
    ctx = context or bpy.context
    scene = getattr(ctx, "scene", None)
    return getattr(scene, "fdt_rebar_settings", None) if scene is not None else None


def _apply_label(entry, obj, label: str) -> None:
    """Set the cosmetic Outliner label; identity stays the sid.

    Blender allows `#`, `!`, `-` and `_` in object names, so labels such as
    ``ID-A_VIS-!B#A`` round-trip intact.  Every lookup still goes through
    ``sid`` / ``fdt_rebar_id``, never through ``Object.name``.
    """
    name = str(label or "").strip()[:63] or st.object_name(entry.sid)
    entry.label = name
    if obj is not None:
        obj.name = name


def active_entry(context=None):
    """``(entry, object)`` for the current generator, or ``(None, None)``."""
    ctx = context or bpy.context
    settings = _rebar_settings(ctx)
    sid = str(getattr(settings, "rebar_active_sid", "") or "")
    obj = getattr(ctx, "active_object", None)
    entry = st.find_entry(sid, ctx.scene) if sid else None
    if entry is None and obj is not None:
        entry = st.find_entry_for_object(obj, ctx.scene)
    if entry is None:
        return None, None
    return entry, st.resolve_object(entry)


def resolve_faces(context, obj, settings) -> list[int]:
    """Selected faces ∩ target material, falling back to all matching faces."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return []
    material = getattr(settings, "target_material", None)
    if material is None:
        return []
    mat_index = rg.material_index_for(obj, material)
    if mat_index < 0:
        return []
    selected = rg.selected_face_indices(context, obj, material_index=mat_index)
    if selected:
        return selected
    return [p.index for p in obj.data.polygons if p.material_index == mat_index]


def estimate_for(mesh, faces, params) -> int:
    bounds = rg.face_bounds(mesh, faces)
    if bounds is None:
        return 0
    gen = st.params_to_generation(params)
    min_co, max_co = bounds
    return rs.estimate_bar_count(
        min_co,
        max_co,
        gen.spacing,
        gen.align,
        Vector(gen.offset),
    )


def _uv_scale(context) -> float:
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "fdt_settings", None) if scene is not None else None
    tpm = float(getattr(settings, "texel_per_meter", 512.0) or 512.0)
    size = float(getattr(settings, "texture_size", 1024.0) or 1024.0)
    return rs.texel_scale(tpm, size)


def _assign_material(entry, obj, params=None) -> None:
    if obj is None or obj.data is None:
        return
    mat = bpy.data.materials.get(entry.rebar_material) if entry.rebar_material else None
    if mat is None:
        mat, _source = st.ensure_rebar_material()
    if mat is not None:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        entry.rebar_material = mat.name

    # Slot 1 is reserved for concrete debris. Keeping a stable slot layout
    # means face_materials indices never need rewriting on later rebuilds.
    debris_name = (params or {}).get("debris_material", "") if params else ""
    debris_mat = bpy.data.materials.get(debris_name) if debris_name else None
    if debris_mat is None and params and params.get("enable_debris"):
        host = st.resolve_host(entry)
        target_name = getattr(entry, "target_material", "")
        if host is not None and target_name:
            index = host.data.materials.find(target_name)
            if index >= 0:
                debris_mat = host.data.materials[index]
    if debris_mat is not None:
        while len(obj.data.materials) < 2:
            obj.data.materials.append(debris_mat)
        obj.data.materials[1] = debris_mat


def build_preview(entry, params, context=None, *, overlay_only: bool = False):
    """Generate splines for ``entry`` and either draw lines or rebuild its mesh."""
    ctx = context or bpy.context
    host = st.resolve_host(entry)
    obj = st.resolve_object(entry)
    if host is None or getattr(host, "data", None) is None or obj is None:
        return rg.GenerationResult()

    mesh, evaluated, eval_obj = rg.source_mesh(ctx, host, entry.group)
    try:
        faces = (
            rg.tagged_face_indices(mesh, entry.group)
            if mesh is not None
            else []
        )
        result = rg.generate(
            mesh,
            faces,
            st.params_to_generation(params),
            matrix_world=host.matrix_world,
        )
    finally:
        rg.release_source(ctx, evaluated, eval_obj)

    if overlay_only:
        world = host.matrix_world
        lines = [[world @ point for point in points] for points, _sub, _twist in result.splines]
        rebar_overlay.set_lines(lines)
        return result

    rebar_overlay.clear()
    generation_params = st.params_to_generation(params)
    payload = rs.build_spline_mesh(
        result.splines, radius=generation_params.radius, uv_scale=_uv_scale(ctx)
    )
    if generation_params.debris_enabled and result.splines:
        debris = rd.build_debris(result.splines, generation_params, _uv_scale(ctx))
        rd.merge_payload(payload, debris.payload)
    if payload.is_empty():
        obj.data.clear_geometry()
    else:
        rs.apply_spline_mesh(obj.data, payload)
    obj.matrix_world = host.matrix_world.copy()
    try:
        obj.data.update()
    except Exception:
        pass
    _assign_material(entry, obj, params)
    return result


# ---------------------------------------------------------------------------
# Debounced rebuild (the only geometry work triggered by property changes)
# ---------------------------------------------------------------------------


def _cancel(handle_name: str) -> None:
    global _REBUILD_HANDLE, _COMMIT_HANDLE
    handle = _REBUILD_HANDLE if handle_name == "rebuild" else _COMMIT_HANDLE
    if handle is not None:
        try:
            bpy.app.timers.unregister(handle)
        except Exception:
            pass
    if handle_name == "rebuild":
        _REBUILD_HANDLE = None
    else:
        _COMMIT_HANDLE = None


def request_rebuild() -> None:
    """Schedule a 50 ms debounced preview. Never builds geometry synchronously."""
    global _REBUILD_HANDLE
    if _LOADING:
        return
    settings = _rebar_settings()
    if settings is None or not getattr(settings, "rebar_active_sid", ""):
        return
    entry = st.find_entry(settings.rebar_active_sid)
    if entry is None or entry.baked:
        return
    if _REBUILD_HANDLE is None:
        _REBUILD_HANDLE = bpy.app.timers.register(_rebuild_timer, first_interval=DEBOUNCE_SEC)


def _rebuild_timer():
    global _REBUILD_HANDLE, _COMMIT_HANDLE
    _REBUILD_HANDLE = None
    settings = _rebar_settings()
    if settings is None:
        return None
    entry = st.find_entry(settings.rebar_active_sid)
    if entry is None or entry.baked:
        rebar_overlay.clear()
        return None

    params = settings_to_params(settings)
    st.update_entry(st.replace_params(entry, params))
    try:
        if bool(params["fast_drag"]):
            build_preview(entry, params, overlay_only=True)
            _cancel("commit")
            _COMMIT_HANDLE = bpy.app.timers.register(_commit_timer, first_interval=COMMIT_SEC)
        else:
            _cancel("commit")
            result = build_preview(entry, params, overlay_only=False)
            settings.rebar_estimate = result.bar_count
            settings.rebar_status = f"Bars: {result.bar_count}" + (" (capped)" if result.truncated else "")
    except Exception as exc:
        settings.rebar_status = f"Rebar rebuild error: {exc}"
        print(f"[FullDestruc_Tool] Rebar rebuild error: {exc}")
    return None


def _commit_timer():
    global _COMMIT_HANDLE
    _COMMIT_HANDLE = None
    settings = _rebar_settings()
    if settings is None:
        return None
    entry = st.find_entry(settings.rebar_active_sid)
    if entry is None or entry.baked:
        rebar_overlay.clear()
        return None
    params = settings_to_params(settings)
    try:
        result = build_preview(entry, params, overlay_only=False)
        settings.rebar_estimate = result.bar_count
        settings.rebar_status = f"Bars: {result.bar_count}" + (" (capped)" if result.truncated else "")
    except Exception as exc:
        settings.rebar_status = f"Rebar rebuild error: {exc}"
    return None


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class FDT_OT_rebar_create(bpy.types.Operator):
    bl_idname = "fdt.rebar_create"
    bl_label = "Create Rebar"
    bl_description = "เลือกเฉพาะแผ่นที่ตรงกับวัสดุที่เลือก แล้วทำชุดเหล็กขึ้นมาให้ปรับค่าก่อนตรึง (Bake)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        scene = context.scene
        settings = _rebar_settings(context)
        host = context.active_object
        if settings is None or settings.target_material is None:
            self.report({"WARNING"}, "เลือก Target Material ก่อนสร้าง Rebar")
            return {"CANCELLED"}
        if host.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

        params = settings_to_params(settings)
        faces = resolve_faces(context, host, settings)
        if not faces:
            self.report({"WARNING"}, "ไม่พบ face ที่ material ตรงกัน — เลือก face หรือเปลี่ยน Target Material")
            return {"CANCELLED"}

        estimate = estimate_for(host.data, faces, params)
        settings.rebar_estimate = estimate
        if estimate > params["max_bars"]:
            self.report(
                {"WARNING"},
                f"คาดการณ์ ~{estimate} แท่ง เกิน cap {params['max_bars']} — ตัดที่ cap อัตโนมัติ",
            )

        sid = st.new_sid({e.sid for e in st.load_registry(scene)})
        label = str(settings.rebar_label or "").strip()
        group = st.next_group_for_host(host, scene)
        rebar_mat, source = st.ensure_rebar_material()
        if rebar_mat is not None:
            settings.rebar_material = rebar_mat

        mesh = bpy.data.meshes.new(st.object_name(sid))
        obj = bpy.data.objects.new(st.object_name(sid), mesh)
        scene.collection.objects.link(obj)
        obj.matrix_world = host.matrix_world.copy()
        st.mark_object(obj, sid)
        st.save_params(obj, params)

        entry = st.RebarEntry(
            sid=sid,
            label=label,
            host=host.name,
            target_material=settings.target_material.name,
            rebar_material=rebar_mat.name if rebar_mat is not None else "",
            object=obj.name,
            group=group,
            baked=False,
            stale=False,
            params=params,
        )
        if not st.add_entry(entry, scene):
            bpy.data.objects.remove(obj, do_unlink=True)
            self.report({"ERROR"}, "ไม่สามารถลงทะเบียน generator ใหม่ได้")
            return {"CANCELLED"}

        _apply_label(entry, obj, label)
        rg.tag_faces(host.data, faces, group)
        settings.rebar_active_sid = sid
        settings.rebar_label = ""
        _cancel("rebuild")
        _cancel("commit")
        try:
            result = build_preview(entry, params, context, overlay_only=False)
            settings.rebar_status = (
                f"สร้าง {result.bar_count} แท่ง ({sid[:8]})"
                + (" · วัสดุโหลดจาก blend" if source == "appended" else "")
            )
        except Exception as exc:
            settings.rebar_status = f"Rebar create error: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        try:
            for other in bpy.context.selected_objects:
                other.select_set(False)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        except Exception:
            pass
        self.report({"INFO"}, f"สร้าง Rebar {sid[:8]} แล้ว — ปรับค่าแล้วกด Bake")
        return {"FINISHED"}


class FDT_OT_rebar_reroll_seed(bpy.types.Operator):
    bl_idname = "fdt.rebar_reroll_seed"
    bl_label = "Reroll Seed"
    bl_description = "สุ่ม Seed ใหม่ — ได้ลายเหล็กชุดใหม่โดยค่าอื่นทั้งหมดไม่เปลี่ยน"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = _rebar_settings(context)
        if settings is None:
            self.report({"WARNING"}, "ไม่พบ buffer ของ Rebar")
            return {"CANCELLED"}
        import random as _random

        # Assigning the property fires the usual debounced-preview callback;
        # the dice never rebuilds geometry synchronously.
        settings.random_seed = _random.randrange(0, 1000000)
        return {"FINISHED"}


class FDT_OT_rebar_rebuild(bpy.types.Operator):
    bl_idname = "fdt.rebar_rebuild"
    bl_label = "Re-apply Rebar to Surface"
    bl_description = "ใช้เฉพาะตอนผิวถูกตัดใหม่หรือล้างรอย — ปรับค่าปกติระบบอัปเดตเอง ไม่ต้องกดปุ่มนี้"
    bl_options = {"REGISTER"}

    sid: StringProperty(name="Generator sid", default="")

    def execute(self, context):
        _cancel("rebuild")
        _cancel("commit")
        sid = str(self.sid or "").strip()
        settings = _rebar_settings(context)
        if sid:
            entry = st.find_entry(sid, context.scene)
            if settings is not None:
                settings.rebar_active_sid = sid
        else:
            entry, _obj = active_entry(context)
        if entry is None:
            self.report({"WARNING"}, "ไม่พบ generator ที่กำลังใช้งาน")
            return {"CANCELLED"}
        params = settings_to_params(settings)
        st.update_entry(st.replace_params(entry, params))
        try:
            result = build_preview(entry, params, context, overlay_only=False)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        settings.rebar_estimate = result.bar_count
        settings.rebar_status = f"ปรับเหล็ก {result.bar_count} แท่งให้เข้ากับผิวใหม่แล้ว — กด Bake เพื่อบันทึก"
        return {"FINISHED"}


class FDT_OT_rebar_bake(bpy.types.Operator):
    bl_idname = "fdt.rebar_bake"
    bl_label = "Bake Rebar"
    bl_description = "บันทึกค่าที่ปรับไว้เป็นชุดถาวร และยุบหมวดลุคเหล็ก/ขั้นสูง — ถ้าผิวเปลี่ยนให้กด Re-apply ก่อน"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _cancel("rebuild")
        _cancel("commit")
        rebar_overlay.clear()
        entry, obj = active_entry(context)
        if entry is None or obj is None:
            self.report({"WARNING"}, "ไม่พบ generator ที่กำลังใช้งาน")
            return {"CANCELLED"}
        settings = _rebar_settings(context)
        params = settings_to_params(settings)
        st.save_params(obj, params)
        entry.params = params
        entry.baked = True
        entry.stale = False
        new_label = str(settings.rebar_label or "").strip() if settings is not None else ""
        # A rename done directly in the Outliner is also a cosmetic label edit;
        # only an explicit buffer label takes precedence.
        effective_label = new_label or (obj.name if obj.name != entry.label else entry.label)
        _apply_label(entry, obj, effective_label)
        entry.object = obj.name
        if settings is not None:
            settings.rebar_label = ""
            # Edit opened these two sections; Bake closes them again so the
            # panel returns to its compact default state.
            settings.ui_show_look = False
            settings.ui_show_advanced = False
        st.move_to_collection(obj, entry, context.scene)
        st.update_entry(entry, context.scene)
        if settings is not None:
            settings.rebar_status = f"บันทึกชุด {entry.sid8} แล้ว"
        self.report({"INFO"}, f"Bake rebar {entry.sid8} แล้ว")
        return {"FINISHED"}


class FDT_OT_rebar_reedit(bpy.types.Operator):
    bl_idname = "fdt.rebar_reedit"
    bl_label = "Re-edit Rebar"
    bl_description = "เอาค่าที่ตรึงไว้กลับมาในแผง แล้วเปิดหมวดลุคเหล็กและขั้นสูงให้ปรับต่อทันที"
    bl_options = {"REGISTER"}

    sid: StringProperty(name="Generator sid", default="")

    def execute(self, context):
        sid = str(self.sid or "").strip()
        settings = _rebar_settings(context)
        if not sid and settings is not None:
            sid = str(settings.rebar_active_sid or "")
        entry = st.find_entry(sid, context.scene)
        if entry is None:
            self.report({"WARNING"}, "ไม่พบ generator ที่ต้องการแก้ไข")
            return {"CANCELLED"}
        obj = st.resolve_object(entry)
        params = st.load_params(obj) if obj is not None else entry.params
        params_to_settings(params, settings)
        if settings is not None:
            settings.rebar_label = entry.label if entry.label != st.object_name(entry.sid) else ""
            settings.rebar_active_sid = entry.sid
            # Opening the detail sections is the visible feedback that Edit
            # worked — otherwise the click looks silent to the artist.
            settings.ui_show_rebar = True
            settings.ui_show_basics = True
            settings.ui_show_look = True
            settings.ui_show_advanced = True
            settings.rebar_status = f"กำลังแก้ชุด {entry.sid8} · เปิด ลุคเหล็ก/ขั้นสูงให้แล้ว"
        entry.baked = False
        entry.stale = False
        st.update_entry(entry, context.scene)
        if obj is not None:
            try:
                for other in context.selected_objects:
                    other.select_set(False)
                obj.select_set(True)
                context.view_layer.objects.active = obj
            except Exception:
                pass
        rebar_overlay.clear()
        return {"FINISHED"}


class FDT_OT_rebar_select_faces(bpy.types.Operator):
    bl_idname = "fdt.rebar_select_faces"
    bl_label = "Select Rebar Faces"
    bl_description = "เข้าโหมดแก้ไขแล้วเลือกเฉพาะแผ่นของชุดนี้ · กด Shift เพื่อเพิ่ม, Alt เพื่อลบ"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Selection Mode",
        items=[
            ("REPLACE", "Replace", "Select only this generator's tagged faces"),
            ("ADD", "Add", "Add tagged faces to the current selection (Shift)"),
            ("REMOVE", "Remove", "Remove tagged faces from the current selection (Alt)"),
        ],
        default="REPLACE",
    )
    sid: StringProperty(name="Generator sid", default="")

    def invoke(self, context, event):
        self.mode = "ADD" if event.shift else ("REMOVE" if event.alt else "REPLACE")
        return self.execute(context)

    def execute(self, context):
        sid = str(self.sid or "").strip()
        settings = _rebar_settings(context)
        if not sid and settings is not None:
            sid = str(settings.rebar_active_sid or "")
        entry = st.find_entry(sid, context.scene)
        host = st.resolve_host(entry) if entry is not None else None
        if host is None or getattr(host, "data", None) is None:
            self.report({"WARNING"}, "ไม่พบ host mesh ของ generator นี้")
            return {"CANCELLED"}

        if context.object is not None and context.object.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        for other in context.selected_objects:
            other.select_set(False)
        host.select_set(True)
        context.view_layer.objects.active = host

        tagged = set(rg.tagged_face_indices(host.data, entry.group))
        currently = {p.index for p in host.data.polygons if p.select}
        if self.mode == "ADD":
            wanted = currently | tagged
        elif self.mode == "REMOVE":
            wanted = currently - tagged
        else:
            wanted = set(tagged)
        for polygon in host.data.polygons:
            polygon.select = polygon.index in wanted
        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # Set the selection on the BMesh itself: Blender derives Edit Mode face
        # selection from vertices/edges, so doing this before mode_set gets
        # flushed away.  select_flush keeps all three domains consistent.
        import bmesh

        bm = bmesh.from_edit_mesh(host.data)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        for face in bm.faces:
            face.select_set(face.index in wanted)
        bmesh.update_edit_mesh(host.data, loop_triangles=False, destructive=False)
        self.report({"INFO"}, f"เลือก {len(tagged)} face ของ {entry.sid8} แล้ว")
        return {"FINISHED"}


class FDT_OT_rebar_delete(bpy.types.Operator):
    bl_idname = "fdt.rebar_delete"
    bl_label = "Delete Rebar"
    bl_description = "ลบชุดเหล็ก — ถ้ายังมีรอยบนผิวค้างอยู่ระบบจะไม่ให้ลบ ให้ลบรอยก่อน"
    bl_options = {"REGISTER", "UNDO"}

    sid: StringProperty(name="Generator sid", default="")

    def execute(self, context):
        sid = str(self.sid or "").strip()
        settings = _rebar_settings(context)
        if not sid and settings is not None:
            sid = str(settings.rebar_active_sid or "")
        entry = st.find_entry(sid, context.scene)
        if entry is None:
            self.report({"WARNING"}, "ไม่พบ generator ที่ต้องการลบ")
            return {"CANCELLED"}
        ok, reason = st.delete_guard(entry.sid, context.scene)
        if not ok:
            self.report({"WARNING"}, reason)
            return {"CANCELLED"}
        obj = st.resolve_object(entry)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        st.remove_entry(entry.sid, context.scene)
        if settings is not None and settings.rebar_active_sid == entry.sid:
            settings.rebar_active_sid = ""
            settings.rebar_estimate = 0
            settings.rebar_status = ""
        rebar_overlay.clear()
        self.report({"INFO"}, f"ลบ generator {entry.sid8} แล้ว")
        return {"FINISHED"}


def selected_rebar_entries(context) -> list:
    """Registry entries whose rebar object is selected in the viewport."""
    scene = getattr(context, "scene", None)
    entries = []
    seen = set()
    for obj in getattr(context, "selected_objects", []) or []:
        entry = st.find_entry_for_object(obj, scene)
        if entry is not None and entry.sid not in seen:
            seen.add(entry.sid)
            entries.append(entry)
    return entries


def resolve_remove_target(context, explicit_sid: str = ""):
    """``(entry, hint)`` for the big Remove Set button.

    Priority: explicit sid (generator list) > active rebar object > the only
    selected rebar > the only generator when there is no ambiguity.  Never
    silently pick the newest generator when several exist.
    """
    sid = str(explicit_sid or "").strip()
    if sid:
        entry = st.find_entry(sid, getattr(context, "scene", None))
        if entry is None:
            return None, "ไม่พบชุดเหล็กที่ต้องการลบ"
        return entry, ""

    # The active object wins even when several rebars are selected: it is the
    # one Blender treats as "the selected target".
    view_layer = getattr(context, "view_layer", None)
    active_obj = getattr(view_layer, "objects", None)
    active_obj = getattr(active_obj, "active", None) if active_obj is not None else None
    if active_obj is not None:
        entry = st.find_entry_for_object(
            active_obj, getattr(context, "scene", None)
        )
        if entry is not None:
            return entry, ""

    selected = selected_rebar_entries(context)
    if len(selected) > 1:
        return None, "เลือกเฉพาะเหล็กชุดเดียวที่จะลบ"
    if len(selected) == 1:
        return selected[0], ""

    states = st.load_all_generators(getattr(context, "scene", None))
    if len(states) == 1:
        return next(iter(states.values())).entry, ""
    if len(states) > 1:
        return None, "เลือกเหล็กชุดที่ต้องการลบในวิวพอร์ตก่อน"

    settings = _rebar_settings(context)
    sid = str(getattr(settings, "rebar_active_sid", "") or "") if settings is not None else ""
    entry = st.find_entry(sid, getattr(context, "scene", None)) if sid else None
    if entry is None:
        return None, "ยังไม่มีชุดเหล็กที่จะลบ"
    return entry, ""


class FDT_OT_rebar_remove(bpy.types.Operator):
    bl_idname = "fdt.rebar_remove"
    bl_label = "Remove Rebar Set"
    bl_description = "ลบชุดเหล็กที่เลือกอยู่ในวิวพอร์ตในคลิกเดียว — ลบรอยที่ทิ้งไว้บนผิวให้ด้วย กด Ctrl+Z เพื่อคืน"
    bl_options = {"REGISTER", "UNDO"}

    sid: StringProperty(name="Generator sid", default="")

    def execute(self, context):
        settings = _rebar_settings(context)
        entry, hint = resolve_remove_target(context, self.sid)
        if entry is None:
            self.report({"WARNING"}, hint)
            return {"CANCELLED"}

        # Wipe the face marks first, then reuse the normal delete guard so any
        # other refusal reason is still respected.  This is the one-click
        # equivalent of Clear Faces -> Delete in the generator list.
        cleared = st.clear_tags_for_entry(entry)
        ok, reason = st.delete_guard(entry.sid, context.scene)
        if not ok:
            self.report({"WARNING"}, reason)
            return {"CANCELLED"}

        obj = st.resolve_object(entry)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        st.remove_entry(entry.sid, context.scene)
        if settings is not None and settings.rebar_active_sid == entry.sid:
            settings.rebar_active_sid = ""
            settings.rebar_estimate = 0
            settings.rebar_status = ""
        rebar_overlay.clear()
        self.report({"INFO"}, f"ลบชุด {entry.sid8} แล้ว (ล้างรอยบนผิว {cleared} จุด)")
        return {"FINISHED"}


class FDT_OT_rebar_clear_faces(bpy.types.Operator):
    bl_idname = "fdt.rebar_clear_faces"
    bl_label = "Clear Rebar Faces"
    bl_description = "ลบรอยบนผิวของชุดนี้โดยไม่ลบเหล็ก แล้วกด Re-apply และ Bake เพื่อบันทึกใหม่"
    bl_options = {"REGISTER", "UNDO"}

    sid: StringProperty(name="Generator sid", default="")

    def execute(self, context):
        sid = str(self.sid or "").strip()
        settings = _rebar_settings(context)
        if not sid and settings is not None:
            sid = str(settings.rebar_active_sid or "")
        entry = st.find_entry(sid, context.scene)
        if entry is None:
            self.report({"WARNING"}, "ไม่พบ generator ที่ต้องการล้าง tag")
            return {"CANCELLED"}
        count = st.clear_tags_for_entry(entry)
        st.set_stale(entry.sid, True, context.scene)
        if settings is not None:
            settings.rebar_status = f"ล้างรอย {count} จุด ({entry.sid8}) — Re-apply แล้ว Bake"
        self.report({"INFO"}, f"ล้างรอย {count} จุดแล้ว — กด Re-apply แล้ว Bake เพื่อบันทึกใหม่")
        return {"FINISHED"}


class FDT_OT_rebar_load_material(bpy.types.Operator):
    bl_idname = "fdt.rebar_load_material"
    bl_label = "Load Rebar Material"
    bl_description = "นำผิวเหล็กที่แนบมากับตัวโปรแกรมมาใส่ช่อง Rebar Mat ให้เลย"
    bl_options = {"REGISTER"}

    def execute(self, context):
        mat, source = st.ensure_rebar_material()
        if mat is None:
            self.report({"WARNING"}, "ไม่พบไฟล์ material ที่แนบมากับ addon")
            return {"CANCELLED"}
        settings = _rebar_settings(context)
        if settings is not None:
            settings.rebar_material = mat
            settings.rebar_status = f"Rebar material: {mat.name} ({source})"
        self.report({"INFO"}, f"โหลด {mat.name} ({source}) แล้ว")
        return {"FINISHED"}


class FDT_OT_rebar_pick_material(bpy.types.Operator):
    bl_idname = "fdt.rebar_pick_material"
    bl_label = "Pick Rebar Material from Surface"
    bl_description = "คลิกที่ผิวในหน้าต่างสามมิติ เพื่อดูดวัสดุมาใส่ช่องที่เลือก"
    bl_options = {"REGISTER", "UNDO"}

    slot: EnumProperty(
        name="Slot",
        items=[("TARGET", "Target", ""), ("REBAR", "Rebar", ""), ("DEBRIS", "Debris", "")],
        default="TARGET",
    )

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "กรุณากดหลอดดูดจากหน้าต่าง 3D View")
            return {"CANCELLED"}
        context.window.cursor_modal_set("EYEDROPPER")
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "คลิกซ้ายบนผิวโมเดลเพื่อดูด Material (คลิกขวา / ESC เพื่อยกเลิก)")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            context.window.cursor_modal_restore()
            return {"CANCELLED"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            context.window.cursor_modal_restore()
            from .material_picker import _pick_material_name_from_mouse

            name, _hit = _pick_material_name_from_mouse(context, event)
            if not name:
                self.report({"WARNING"}, "ไม่พบ Material บนพื้นผิวที่คลิก")
                return {"CANCELLED"}
            slot = str(self.slot)
            picked = str(name)

            def _assign():
                mat = bpy.data.materials.get(picked)
                settings = _rebar_settings()
                if mat is not None and settings is not None:
                    if slot == "TARGET":
                        settings.target_material = mat
                    elif slot == "REBAR":
                        settings.rebar_material = mat
                    else:
                        settings.debris_material = mat
                return None

            bpy.app.timers.register(_assign, first_interval=0.05)
            return {"FINISHED"}
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


CLASSES = (
    FDT_OT_rebar_create,
    FDT_OT_rebar_reroll_seed,
    FDT_OT_rebar_rebuild,
    FDT_OT_rebar_bake,
    FDT_OT_rebar_reedit,
    FDT_OT_rebar_select_faces,
    FDT_OT_rebar_delete,
    FDT_OT_rebar_remove,
    FDT_OT_rebar_clear_faces,
    FDT_OT_rebar_load_material,
    FDT_OT_rebar_pick_material,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    rebar_overlay.register()


def unregister():
    _cancel("rebuild")
    _cancel("commit")
    rebar_overlay.unregister()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
