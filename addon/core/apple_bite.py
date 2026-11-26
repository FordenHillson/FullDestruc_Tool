"""Apple Bite v2.1 core engine (guided flow, multi-object, zone colors, mute/delete, re-edit).

Zero UI code here — called by N-Panel, Floating Menu, PySide6 Window, and automated tests.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Sequence

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import live_sync, profiler, rust_bridge, zone_registry
from .geonodes import (
    ORIG_ATTR_NAME,
    ensure_cut_face_uv_group,
    ensure_mark_orig_group,
    ensure_rough_cutter_group,
    set_modifier_socket,
)
from .session import (
    OBJ_SESSION_ID_PROP,
    AppleBiteSession,
    CutterEntry,
    collect_related_session_family,
    find_session_id_for_object,
    load_all_stored_sessions,
    load_session,
    load_stored_session_by_id,
    save_session,
)
from .vault import (
    create_vault_collection,
    get_or_create_collection,
    move_object_to_collection,
    restore_originals_from_vault,
    set_collection_excluded,
    set_collection_hidden,
    store_originals_in_vault,
    temporary_vault_evaluation,
)
from .zone_shield import is_protected_object

BITE_ROOT_PREFIX = "FDT_Bite_"
FINAL_ROOT_PREFIX = "FDT_Final_"
CUTTER_PROP = "fdt_is_cutter"
CUTTER_ZONE_PROP = "fdt_cutter_zone"
CUTTER_ROUGH_PROP = "fdt_cutter_rough"
RESULT_SOURCE_PROP = "fdt_source_name"
PIECE_ZONE_PROP = "fdt_piece_zone"
MAX_BLENDER_NAME = 63

# Custom zone tokens may contain '#', '-' and '_' (e.g. _FDST_ID-ID-A#B).
_FDST_ID_RE = re.compile(r"_FDST_ID-[A-Za-z0-9_#-]+")

# Palette lives in the registry module; keep the old name for existing callers.
ZONE_COLORS = zone_registry.BASE_COLORS


def get_zone_color(zone: str) -> tuple[float, float, float, float]:
    """Return the registered RGBA color for a zone key (or a legacy zone letter)."""
    return zone_registry.zone_color(zone, getattr(bpy.context, "scene", None))


def _resolve_zone_id(scene: bpy.types.Scene | None, zone: str | None) -> str:
    """Map a requested zone/token/legacy letter onto a valid registry key."""
    raw = zone if zone else None
    if raw is None:
        settings = _get_settings(scene)
        raw = settings.active_zone if settings else "A"
    return zone_registry.resolve_zone(raw, scene).key


_ROUGH_WIRE_PREVIEW_UNTIL: dict[str, float] = {}
_ROUGH_WIRE_TIMER_ACTIVE: bool = False


def is_rough_wire_visible(cutter: bpy.types.Object, is_rough: bool) -> bool:
    """Return True if the cutter should render its full WIRE mesh; False hides rough mesh as clean BOUNDS."""
    if not is_rough:
        return True
    if cutter.mode == "EDIT":
        return True
    until = _ROUGH_WIRE_PREVIEW_UNTIL.get(cutter.name, 0.0)
    return until > 0.0 and time.monotonic() < until


def _rough_wire_hide_timer() -> float | None:
    """Debounced timer that hides rough wireframes back to BOUNDS once setting tweaks stop."""
    global _ROUGH_WIRE_TIMER_ACTIVE
    now = time.monotonic()
    next_wait: float | None = None
    for cname in list(_ROUGH_WIRE_PREVIEW_UNTIL.keys()):
        rem = _ROUGH_WIRE_PREVIEW_UNTIL[cname] - now
        if rem > 0.01:
            next_wait = rem if next_wait is None else min(next_wait, rem)
            continue
        _ROUGH_WIRE_PREVIEW_UNTIL.pop(cname, None)
        obj = bpy.data.objects.get(cname)
        if obj is not None and obj.type == "MESH":
            is_rough = bool(obj.get(CUTTER_ROUGH_PROP, False))
            want_disp = "WIRE" if is_rough_wire_visible(obj, is_rough) else "BOUNDS"
            if obj.display_type != want_disp:
                obj.display_type = want_disp
    if next_wait is not None:
        return min(max(next_wait, 0.05), 1.0)
    _ROUGH_WIRE_TIMER_ACTIVE = False
    wm = getattr(bpy.context, "window_manager", None)
    if wm is not None:
        for win in wm.windows:
            if win.screen is None:
                continue
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    return None


def flash_rough_wire_preview(cutter_name: str | None = None, duration: float = 1.5) -> None:
    """Temporarily reveal a rough cutter's wireframe while actively tweaking its Rough settings."""
    global _ROUGH_WIRE_TIMER_ACTIVE
    if cutter_name:
        _ROUGH_WIRE_PREVIEW_UNTIL[cutter_name] = time.monotonic() + float(duration)
        obj = bpy.data.objects.get(cutter_name)
        if obj is not None and obj.type == "MESH" and obj.display_type != "WIRE":
            obj.display_type = "WIRE"
    if not bpy.app.background and not _ROUGH_WIRE_TIMER_ACTIVE and _ROUGH_WIRE_PREVIEW_UNTIL:
        _ROUGH_WIRE_TIMER_ACTIVE = True
        try:
            bpy.app.timers.register(_rough_wire_hide_timer, first_interval=min(float(duration), 0.5))
        except Exception:
            _ROUGH_WIRE_TIMER_ACTIVE = False


def apply_cutter_visuals(
    cutter: bpy.types.Object,
    zone: str,
    muted: bool = False,
    is_rough: bool | None = None,
) -> None:
    """Style a cutter matching its Zone ID; hide noisy black rough wireframe when idle."""
    rough_flag = bool(cutter.get(CUTTER_ROUGH_PROP, False)) if is_rough is None else bool(is_rough)
    want_disp = "WIRE" if is_rough_wire_visible(cutter, rough_flag) else "BOUNDS"
    if cutter.display_type != want_disp:
        cutter.display_type = want_disp
    cutter.show_in_front = cutter.mode == "EDIT"
    cutter.hide_render = True
    r, g, b, _ = get_zone_color(zone)
    if muted:
        cutter.color = (0.45, 0.45, 0.45, 0.45)
    else:
        cutter.color = (r, g, b, 1.0)


def _sanitize_id(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return clean[:24] or "Session"


def _strip_existing_fdst_id(name: str) -> str:
    return _FDST_ID_RE.sub("", name)


def _safe_zone_token(value: str) -> str:
    """Keep '#', '-' and '_' in generated names; replace everything else."""
    return re.sub(r"[^A-Za-z0-9_#-]+", "_", str(value or "A").strip())[:MAX_BLENDER_NAME] or "A"


def _zone_name_token(zone: str, scene: bpy.types.Scene | None = None) -> str:
    """Resolve a stored zone key to its editable token for object naming."""
    return _safe_zone_token(zone_registry.resolve_zone(zone, scene).token)


def _format_zone_name(base_name: str, zone: str) -> str:
    clean = _strip_existing_fdst_id(base_name)
    token = _safe_zone_token(zone)
    prefix = "_FDST_ID-"
    suffix = f"{prefix}{token}"[:MAX_BLENDER_NAME]
    max_base = max(0, MAX_BLENDER_NAME - len(suffix))
    return f"{clean[:max_base]}{suffix}"


def _set_boolean_solver(mod: bpy.types.BooleanModifier, prefer: str = "EXACT") -> None:
    """Assign a valid Boolean solver enum for Blender 5.2+ safely."""
    prop = mod.bl_rna.properties.get("solver")
    if prop is None:
        return
    valid = {item.identifier for item in prop.enum_items}
    candidates = [prefer, "EXACT", "FLOAT", "FAST", "MANIFOLD"]
    for cand in candidates:
        if cand in valid:
            mod.solver = cand
            if cand == "EXACT" and hasattr(mod, "use_hole_tolerant"):
                mod.use_hole_tolerant = True
            return


def safe_select_objects(
    context: bpy.types.Context,
    objects: Iterable[bpy.types.Object],
    active: bpy.types.Object | None = None,
) -> None:
    """Safely select only objects that are visible in the current view layer."""
    vl_objs = context.view_layer.objects
    for obj in list(context.selected_objects):
        try:
            obj.select_set(False)
        except RuntimeError:
            pass

    first_selected = None
    for obj in objects:
        if obj is None or obj.name not in vl_objs:
            continue
        if obj.hide_viewport or obj.hide_get():
            continue
        try:
            obj.select_set(True)
            if first_selected is None:
                first_selected = obj
        except RuntimeError:
            continue

    target_active = active if (active is not None and active.name in vl_objs) else first_selected
    if target_active is not None:
        try:
            vl_objs.active = target_active
        except RuntimeError:
            pass


def is_valid_target_mesh(obj: bpy.types.Object | None) -> bool:
    """Return True if obj can be used as a fresh Apple Bite target mesh."""
    if obj is None or obj.type != "MESH":
        return False
    if is_protected_object(obj):
        return False
    if bool(obj.get(CUTTER_PROP, False)):
        return False
    if RESULT_SOURCE_PROP in obj:
        return False
    return True


def detect_step1_action(context: bpy.types.Context) -> dict:
    """Inspect current selection for Step 1.

    Returns dict with keys:
    - mode: 'BOTH' | 'REEDIT' | 'START' | 'NONE'
    - session_id: str | None (when mode in ('REEDIT', 'BOTH'))
    - root_session_id: str | None (earliest root original session when mode in ('REEDIT', 'BOTH'))
    - targets: list[Object] (when mode in ('START', 'BOTH'))
    - button_label: str
    - reedit_label: str
    - reset_label: str
    - guide_text: str
    """
    selected = list(context.selected_objects)
    valid = [o for o in selected if is_valid_target_mesh(o)]

    found_sid = None
    for obj in selected:
        sid = find_session_id_for_object(obj)
        if sid is not None:
            found_sid = sid
            break

    root_sid = None
    reset_label = ""
    if found_sid is not None:
        root_ids, _family_ids = collect_related_session_family(context.scene, found_sid)
        root_sid = root_ids[0] if root_ids else found_sid
        reset_label = f"Reset to Original ({root_sid})"

    if found_sid is not None and valid:
        return {
            "mode": "BOTH",
            "session_id": found_sid,
            "root_session_id": root_sid,
            "targets": valid,
            "button_label": f"Cut Further ({len(valid)})",
            "reedit_label": f"Re-edit Previous ({found_sid})",
            "reset_label": reset_label,
            "guide_text": f"From '{found_sid}': cut further into new zones, or re-edit old cut",
        }

    if found_sid is not None:
        return {
            "mode": "REEDIT",
            "session_id": found_sid,
            "root_session_id": root_sid,
            "targets": [],
            "button_label": f"Re-edit Previous ({found_sid})",
            "reedit_label": f"Re-edit Previous ({found_sid})",
            "reset_label": reset_label,
            "guide_text": f"Found session '{found_sid}' — click to re-open old cutters",
        }

    if valid:
        preview_names = ", ".join(o.name for o in valid[:3])
        if len(valid) > 3:
            preview_names += f" +{len(valid) - 3}"
        return {
            "mode": "START",
            "session_id": None,
            "root_session_id": None,
            "targets": valid,
            "button_label": f"Start Apple Bite ({len(valid)})",
            "reedit_label": "",
            "reset_label": "",
            "guide_text": f"Ready to cut: {preview_names}",
        }

    return {
        "mode": "NONE",
        "session_id": None,
        "root_session_id": None,
        "targets": [],
        "button_label": "Select Meshes to Start",
        "reedit_label": "",
        "reset_label": "",
        "guide_text": "Step 1: Select 1 or more building meshes in 3D View",
    }


def _compute_true_matrix_world(obj: bpy.types.Object) -> Matrix:
    """Compute world matrix from matrix_basis + parent chain (never stale even inside hidden/excluded Vaults)."""
    mat = obj.matrix_basis.copy()
    cur = obj
    while cur.parent is not None:
        mat = cur.parent.matrix_basis @ cur.matrix_parent_inverse @ mat
        cur = cur.parent
    return mat


def compute_targets_bounds(targets: Iterable[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    """Compute world-space `(min_v, max_v)` bounding box across all target meshes."""
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in targets:
        if obj is None or obj.type != "MESH":
            continue
        mw = _compute_true_matrix_world(obj)
        for corner in obj.bound_box:
            wc = mw @ Vector(corner)
            min_v.x = min(min_v.x, wc.x)
            min_v.y = min(min_v.y, wc.y)
            min_v.z = min(min_v.z, wc.z)
            max_v.x = max(max_v.x, wc.x)
            max_v.y = max(max_v.y, wc.y)
            max_v.z = max(max_v.z, wc.z)
            found = True
    if not found:
        return None
    return min_v, max_v


def compute_targets_center(targets: Iterable[bpy.types.Object]) -> Vector:
    """Compute world-space bounding-box center across all target meshes."""
    bounds = compute_targets_bounds(targets)
    if bounds is None:
        return Vector((0.0, 0.0, 0.0))
    return (bounds[0] + bounds[1]) * 0.5


def get_apply_preview_summary(scene: bpy.types.Scene) -> dict:
    """Return a clean pre-Apply summary of what meshes & zones will be created."""
    session = load_session(scene)
    if session is None:
        return {"main_count": 0, "zone_counts": {}, "active_cutters": 0, "muted_cutters": 0}

    settings = _get_settings(scene)
    keep_res = bool(settings.apply_keep_result) if settings else True
    keep_pcs = bool(settings.apply_keep_pieces) if settings else True

    target_count = len(session.get_target_objects())
    zone_counts: dict[str, int] = {}
    active_c = 0
    muted_c = 0
    for entry in session.cutters:
        if entry.name not in bpy.data.objects:
            continue
        if entry.muted:
            muted_c += 1
        else:
            active_c += 1
            if keep_pcs:
                z = zone_registry.resolve_zone(entry.zone, scene).key
                zone_counts[z] = zone_counts.get(z, 0) + target_count

    return {
        "main_count": target_count if keep_res else 0,
        "zone_counts": zone_counts,
        "active_cutters": active_c,
        "muted_cutters": muted_c,
    }


def _get_settings(scene: bpy.types.Scene):
    return getattr(scene, "fdt_settings", None)


def _ensure_default_cut_material_nodes(mat: bpy.types.Material) -> None:
    """Ensure fallback FDT_CutFace_Mat shows a clear terracotta color in Material Preview & Solid."""
    mat.diffuse_color = (0.72, 0.32, 0.18, 1.0)
    try:
        mat.use_nodes = True
        nt = mat.node_tree
        if nt is not None:
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                if "Base Color" in bsdf.inputs:
                    bsdf.inputs["Base Color"].default_value = (0.72, 0.32, 0.18, 1.0)
                if "Roughness" in bsdf.inputs:
                    bsdf.inputs["Roughness"].default_value = 0.85
    except Exception:
        pass


def _get_cut_material(scene: bpy.types.Scene) -> bpy.types.Material | None:
    settings = _get_settings(scene)
    if settings is not None and settings.cut_material is not None:
        return settings.cut_material
    mat = bpy.data.materials.get("FDT_CutFace_Mat")
    if mat is None:
        mat = bpy.data.materials.new(name="FDT_CutFace_Mat")
    _ensure_default_cut_material_nodes(mat)
    return mat


def assign_cut_material_to_objects(
    context: bpy.types.Context,
    objects: list[bpy.types.Object],
    material: bpy.types.Material | None,
) -> tuple[bool, str, int]:
    """Assign/replace cut-face material on already-applied meshes (only faces with fdt_orig < 0.5)."""
    if material is None:
        return False, "กรุณาเลือก Cut Material ก่อนครับ", 0

    mesh_objs = [o for o in objects if o is not None and o.type == "MESH" and o.data is not None]
    if not mesh_objs:
        return False, "กรุณาเลือกโมเดลที่ต้องการเปลี่ยน Material หน้าตัด", 0

    updated_objs = 0
    total_faces = 0
    fallback_mat = bpy.data.materials.get("FDT_CutFace_Mat")

    for obj in mesh_objs:
        mesh = obj.data
        if len(mesh.polygons) == 0:
            continue

        target_poly_indices: list[int] = []
        orig_attr = mesh.attributes.get(ORIG_ATTR_NAME)
        if orig_attr is not None and orig_attr.domain == "FACE" and len(orig_attr.data) == len(mesh.polygons):
            vals = [1.0] * len(mesh.polygons)
            orig_attr.data.foreach_get("value", vals)
            target_poly_indices = [i for i, v in enumerate(vals) if v < 0.5]
        elif fallback_mat is not None:
            fb_idx = next((i for i, m in enumerate(mesh.materials) if m == fallback_mat), -1)
            if fb_idx >= 0:
                target_poly_indices = [p.index for p in mesh.polygons if p.material_index == fb_idx]

        if not target_poly_indices:
            continue

        # Find or append material slot (replace FDT_CutFace_Mat slot if it exists and is unused elsewhere)
        slot_idx = next((i for i, m in enumerate(mesh.materials) if m == material), -1)
        if slot_idx < 0:
            mesh.materials.append(material)
            slot_idx = len(mesh.materials) - 1

        for pi in target_poly_indices:
            mesh.polygons[pi].material_index = slot_idx

        mesh.update()
        updated_objs += 1
        total_faces += len(target_poly_indices)

    if updated_objs == 0:
        return False, "ไม่พบหน้าตัด (fdt_orig) บนชิ้นที่เลือก", 0

    return True, f"ใส่ Material '{material.name}' ให้หน้าตัด {total_faces} หน้า ({updated_objs} ชิ้น) แล้ว", updated_objs


def init_cutter_rough_settings(
    cutter: bpy.types.Object,
    scene: bpy.types.Scene | None,
    ref_cutter: bpy.types.Object | None = None,
) -> None:
    """Initialize a newly created cutter's per-object fdt_rough from ref_cutter or scene defaults."""
    cr = getattr(cutter, "fdt_rough", None)
    if cr is None:
        return
    from .. import props as _props

    prev = getattr(_props, "_LOADING_CUTTER_ROUGH", False)
    _props._LOADING_CUTTER_ROUGH = True
    try:
        ref_r = getattr(ref_cutter, "fdt_rough", None) if ref_cutter is not None else None
        if ref_r is not None:
            cr.inward_depth = float(ref_r.inward_depth)
            cr.jaggedness = float(ref_r.jaggedness)
            cr.noise_scale = float(ref_r.noise_scale)
            cr.detail = max(1, min(3, int(ref_r.detail)))
            cr.seed = float(ref_r.seed)
            return
        settings = _get_settings(scene) if scene is not None else None
        if settings is not None:
            cr.inward_depth = float(getattr(settings, "rough_inward_ratio", 0.0))
            cr.jaggedness = float(getattr(settings, "rough_jaggedness", 0.45))
            cr.noise_scale = float(getattr(settings, "rough_scale", 2.5))
            cr.detail = max(1, min(3, int(getattr(settings, "rough_subdiv", 2))))
            cr.seed = float(getattr(settings, "rough_seed", 0.0))
    finally:
        _props._LOADING_CUTTER_ROUGH = prev


def get_active_rough_cutter(
    context: bpy.types.Context | None,
    session: AppleBiteSession | None,
) -> tuple[bpy.types.Object | None, bool]:
    """Return (cutter_obj, is_rough) for the active/selected cutter in the session (or first rough cutter)."""
    if session is None or not session.cutters:
        return None, False
    by_name = {c.name: c for c in session.cutters}

    active = getattr(context, "active_object", None) if context is not None else None
    if active is not None and active.name in by_name:
        return active, bool(by_name[active.name].rough)

    selected = list(getattr(context, "selected_objects", []) or []) if context is not None else []
    for obj in selected:
        if obj is not None and obj.name in by_name:
            return obj, bool(by_name[obj.name].rough)

    # Fallback: first rough cutter in the session
    for entry in session.cutters:
        if entry.rough:
            obj = bpy.data.objects.get(entry.name)
            if obj is not None:
                return obj, True
    return None, False


def _ensure_rough_modifier(
    cutter: bpy.types.Object,
    scene: bpy.types.Scene,
    is_rough: bool,
    preview_mode: bool = True,
) -> None:
    settings = _get_settings(scene)
    mod_name = "FDT_RoughCutter"
    mod = cutter.modifiers.get(mod_name)

    if not is_rough:
        if mod is not None:
            cutter.modifiers.remove(mod)
        return

    if mod is None or mod.type != "NODES":
        if mod is not None:
            cutter.modifiers.remove(mod)
        mod = cutter.modifiers.new(name=mod_name, type="NODES")
    mod.node_group = ensure_rough_cutter_group()

    cr = getattr(cutter, "fdt_rough", None)
    if cr is not None:
        inward = float(cr.inward_depth)
        jagged = float(cr.jaggedness)
        scale = float(cr.noise_scale)
        detail = max(1, min(3, int(cr.detail)))
        seed = float(cr.seed)
    elif settings is not None:
        inward = float(getattr(settings, "rough_inward_ratio", 0.0))
        jagged = float(getattr(settings, "rough_jaggedness", 0.45))
        scale = float(getattr(settings, "rough_scale", 1.5))
        detail = max(1, min(3, int(getattr(settings, "rough_subdiv", 2))))
        seed = float(getattr(settings, "rough_seed", 0.0))
    else:
        return

    set_modifier_socket(mod, "Inward Depth", inward)
    set_modifier_socket(mod, "Jaggedness", jagged)
    set_modifier_socket(mod, "Noise Scale", scale)
    set_modifier_socket(mod, "Detail", detail)
    set_modifier_socket(mod, "Seed", seed)


def _ensure_mark_and_uv_stack(
    obj: bpy.types.Object,
    scene: bpy.types.Scene,
    bool_ops: list[tuple[str, bpy.types.Object]],
    solver: str = "EXACT",
) -> None:
    """Rebuild clean non-destructive stack: MarkOrig -> Booleans -> CutFaceUV."""
    obj.modifiers.clear()

    m_orig = obj.modifiers.new(name="FDT_MarkOrig", type="NODES")
    m_orig.node_group = ensure_mark_orig_group()

    for idx, (op_type, cutter_obj) in enumerate(bool_ops):
        m_bool = obj.modifiers.new(name=f"FDT_Bool_{idx:02d}_{op_type}", type="BOOLEAN")
        m_bool.operation = op_type
        m_bool.operand_type = "OBJECT"
        m_bool.object = cutter_obj
        _set_boolean_solver(m_bool, prefer=solver)

    cut_mat = _get_cut_material(scene)
    settings = _get_settings(scene)
    tpm = float(settings.texel_per_meter) if settings else 512.0
    tex_size = float(settings.texture_size) if settings else 1024.0

    m_uv = obj.modifiers.new(name="FDT_CutFaceUV", type="NODES")
    m_uv.node_group = ensure_cut_face_uv_group()
    if cut_mat is not None:
        set_modifier_socket(m_uv, "Cut Material", cut_mat)
    set_modifier_socket(m_uv, "Texel Per Meter", tpm)
    set_modifier_socket(m_uv, "Texture Size", tex_size)


def _create_working_copy(
    source_obj: bpy.types.Object,
    name: str,
    target_col: bpy.types.Collection,
    session_id: str,
) -> bpy.types.Object:
    mesh_copy = source_obj.data.copy()
    mesh_copy.name = f"{name}_Mesh"
    obj = bpy.data.objects.new(name, mesh_copy)
    obj.matrix_world = _compute_true_matrix_world(source_obj)
    obj[RESULT_SOURCE_PROP] = source_obj.name
    obj[OBJ_SESSION_ID_PROP] = session_id
    target_col.objects.link(obj)
    return obj


def enter_session(
    context: bpy.types.Context,
    target_objects: list[bpy.types.Object] | None = None,
    session_id: str | None = None,
) -> tuple[bool, str, AppleBiteSession | None]:
    """Start a new Apple Bite v2.1 session on 1 or multiple mesh objects."""
    scene = context.scene
    existing = load_session(scene)
    if existing is not None:
        return True, f"Resumed active session '{existing.session_id}'", existing

    if target_objects is None:
        candidates = list(context.selected_objects)
        if not candidates and context.active_object is not None:
            candidates = [context.active_object]
    else:
        candidates = list(target_objects)

    valid_targets = [obj for obj in candidates if is_valid_target_mesh(obj)]
    if not valid_targets:
        return False, "กรุณาเลือกโมเดลตึกอย่างน้อย 1 ชิ้นก่อนเริ่มตัด", None

    # Ensure matrix_world is up-to-date before cloning transforms
    context.view_layer.update()

    if session_id is None:
        base = _sanitize_id(valid_targets[0].name)
        session_id = base
        counter = 1
        while f"{BITE_ROOT_PREFIX}{session_id}" in bpy.data.collections:
            counter += 1
            session_id = f"{base}_{counter:02d}"

    root_col = get_or_create_collection(f"{BITE_ROOT_PREFIX}{session_id}", parent=scene.collection, color_tag="COLOR_02")
    vault_col = create_vault_collection(context.view_layer, session_id, parent_col=root_col)
    cutters_col = get_or_create_collection(f"FDT_Cutters_{session_id}", parent=root_col, color_tag="COLOR_03")
    results_col = get_or_create_collection(f"FDT_Results_{session_id}", parent=root_col, color_tag="COLOR_05")
    pieces_col = get_or_create_collection(f"FDT_Pieces_{session_id}", parent=root_col, color_tag="COLOR_06")

    # Store originals safely in the protected vault (hidden)
    store_originals_in_vault(context.view_layer, vault_col, valid_targets)

    # Create working Result objects for each target
    result_objs: list[bpy.types.Object] = []
    for orig in valid_targets:
        res_name = f"FDT_Res_{orig.name}"[:MAX_BLENDER_NAME]
        res_obj = _create_working_copy(orig, res_name, results_col, session_id)
        result_objs.append(res_obj)

    session = AppleBiteSession(
        session_id=session_id,
        root_col_name=root_col.name,
        vault_col_name=vault_col.name,
        cutters_col_name=cutters_col.name,
        results_col_name=results_col.name,
        pieces_col_name=pieces_col.name,
        final_col_name=f"{FINAL_ROOT_PREFIX}{session_id}",
        target_names=[obj.name for obj in valid_targets],
        cutters=[],
    )
    save_session(scene, session)
    profiler.record_marker("ENTER_SESSION", f"{session_id} ({len(valid_targets)} targets)")
    with profiler.action_scope("ENTER_SESSION", session_id):
        rebuild_session(context)
    safe_select_objects(context, result_objs)
    return True, f"เริ่มตัด {len(valid_targets)} ชิ้นแล้ว — กด '+ วางกล่องตัด' เพื่อเริ่มได้เลย", session


def resume_session(
    context: bpy.types.Context,
    session_id: str,
) -> tuple[bool, str, AppleBiteSession | None]:
    """Re-open an exited or already-applied Apple Bite session for further editing.

    Temporarily hides FDT_Final_<id> (if present) and unhides FDT_Bite_<id>,
    while keeping FDT_Vault_<id> strictly hidden & protected.
    """
    scene = context.scene
    sess = load_stored_session_by_id(session_id)
    if sess is None:
        return False, f"ไม่พบข้อมูลงานตัด '{session_id}'", None

    cols = sess.get_collections()
    # Exclude & hide baked Final collection so user ONLY sees live working cuts (zero overlap!)
    if cols["final"] is not None:
        set_collection_excluded(context.view_layer, cols["final"], True)

    # Un-exclude & show working root + cutters/results/pieces, NEVER touch vault (stays hidden!)
    if cols["root"] is not None:
        set_collection_excluded(context.view_layer, cols["root"], False)
    for key in ("cutters", "results", "pieces"):
        if cols[key] is not None:
            set_collection_excluded(context.view_layer, cols[key], False)
    if cols["vault"] is not None:
        set_collection_hidden(context.view_layer, cols["vault"], True)

    save_session(scene, sess)
    rebuild_session(context)

    # Select cutters (or results if no cutters) so user can immediately edit
    cutters = [c for c, _ in sess.get_valid_cutters(include_muted=True)]
    if cutters:
        safe_select_objects(context, [cutters[-1]], active=cutters[-1])
    elif cols["results"] is not None:
        safe_select_objects(context, list(cols["results"].objects))

    return True, f"เปิดงานตัด '{session_id}' กลับมาแก้ไขแล้ว", sess


def _remove_objects_in_collection(col: bpy.types.Collection | None) -> None:
    if col is None:
        return
    for obj in list(col.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)


def _uv_scale_for_scene(scene: bpy.types.Scene) -> float:
    settings = _get_settings(scene)
    if settings is None:
        return 0.5
    texel = float(getattr(settings, "texel_per_meter", 512.0) or 512.0)
    tex_size = max(float(getattr(settings, "texture_size", 1024.0) or 1024.0), 1.0)
    return texel / tex_size


def _clear_fdt_modifiers(obj: bpy.types.Object) -> None:
    for mod in list(obj.modifiers):
        if mod.name.startswith("FDT_"):
            obj.modifiers.remove(mod)


def _sync_rust_session(
    scene: bpy.types.Scene,
    session: AppleBiteSession,
    depsgraph: bpy.types.Depsgraph | None,
    recreate_pieces: bool = False,
    is_live: bool = False,
    force_write: bool = False,
) -> bool:
    """Evaluate all targets & active cutters via the Rust Boolean Engine (O(N) single-pass split)."""
    cols = session.get_collections()
    results_col = cols["results"]
    pieces_col = cols["pieces"]
    if not results_col or not pieces_col:
        return False

    active_cutters = session.get_valid_cutters(include_muted=False)
    cutter_objs = [c_obj for c_obj, _ in active_cutters]
    targets = session.get_target_objects()
    cut_mat = _get_cut_material(scene)
    uv_scale = _uv_scale_for_scene(scene)

    # Map existing Result objects by source name
    res_by_source: dict[str, bpy.types.Object] = {}
    for obj in list(results_col.objects):
        src = obj.get(RESULT_SOURCE_PROP)
        if src:
            res_by_source[str(src)] = obj

    # Map existing Piece objects by (source_name, cutter_name) so live dragging updates in-place
    if recreate_pieces:
        _remove_objects_in_collection(pieces_col)
    piece_map: dict[tuple[str, str], bpy.types.Object] = {}
    for obj in list(pieces_col.objects):
        src = str(obj.get(RESULT_SOURCE_PROP, ""))
        cname = str(obj.get("fdt_piece_cutter", ""))
        if src and cname:
            piece_map[(src, cname)] = obj

    valid_piece_keys: set[tuple[str, str]] = set()

    for orig in targets:
        res_obj = res_by_source.get(orig.name)
        if res_obj is None:
            res_name = f"FDT_Res_{orig.name}"[:MAX_BLENDER_NAME]
            res_obj = _create_working_copy(orig, res_name, results_col, session.session_id)
        _clear_fdt_modifiers(res_obj)
        res_obj.matrix_world = orig.matrix_world.copy()

        target_piece_objs: list[bpy.types.Object] = []
        for idx, (cutter_obj, zone) in enumerate(active_cutters):
            key = (orig.name, cutter_obj.name)
            valid_piece_keys.add(key)
            p_obj = piece_map.get(key)
            if p_obj is None:
                piece_name = f"FDT_Piece_{orig.name}_ID-{_zone_name_token(zone, scene)}_{idx + 1:02d}"[:MAX_BLENDER_NAME]
                p_obj = _create_working_copy(orig, piece_name, pieces_col, session.session_id)
                piece_map[key] = p_obj
            _clear_fdt_modifiers(p_obj)
            p_obj.matrix_world = orig.matrix_world.copy()
            p_obj[PIECE_ZONE_PROP] = zone
            p_obj["fdt_piece_cutter"] = cutter_obj.name
            target_piece_objs.append(p_obj)

        rust_bridge.evaluate_target_with_cutters(
            session_id=session.session_id,
            target_obj=orig,
            cutter_objs=cutter_objs,
            depsgraph=depsgraph,
            cut_material=cut_mat,
            uv_scale=uv_scale,
            res_obj=res_obj,
            piece_objs=target_piece_objs,
            is_live=is_live,
            force_write=force_write,
        )

    # Remove stale piece objects (e.g., muted or deleted cutters)
    for key, obj in list(piece_map.items()):
        if key not in valid_piece_keys:
            _remove_object_and_orphan_mesh(obj)

    # Refresh Zone Grid GPU overlay immediately so cut area always tracks the moving cutter
    try:
        from ..ops import apple_bite_ops

        apple_bite_ops.invalidate_zone_grid_cache(tag_redraw=True)
    except Exception:
        pass

    return True


def sync_rust_session_live(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph | None) -> bool:
    """Fast real-time entry point called by `live_sync.py` when dragging a cutter (classic mode)."""
    session = load_session(scene)
    if session is None:
        return False
    return _sync_rust_session(scene, session, depsgraph, recreate_pieces=False, is_live=True)


def submit_rust_session_async(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph | None) -> dict[str, int]:
    """Non-blocking async job submission for Zero-Switch Textured GPU Mode (< 0.3 ms on main thread)."""
    session = load_session(scene)
    if session is None:
        return {}
    active_cutters = session.get_valid_cutters(include_muted=False)
    cutter_objs = [c_obj for c_obj, _ in active_cutters]
    targets = session.get_target_objects()
    cut_mat = _get_cut_material(scene)
    uv_scale = _uv_scale_for_scene(scene)

    job_map: dict[str, int] = {}
    for orig in targets:
        jid = rust_bridge.submit_target_async(
            session_id=session.session_id,
            target_obj=orig,
            cutter_objs=cutter_objs,
            depsgraph=depsgraph,
            cut_material=cut_mat,
            uv_scale=uv_scale,
        )
        job_map[orig.name] = jid
    return job_map


def commit_rust_gpu_drag(scene: bpy.types.Scene) -> bool:
    """Commit the already-computed Rust async cache ONCE into real Blender meshes (0 ms Rust recompute)."""
    session = load_session(scene)
    if session is None:
        return False
    cols = session.get_collections()
    results_col = cols["results"]
    pieces_col = cols["pieces"]
    if not results_col or not pieces_col:
        return False

    active_cutters = session.get_valid_cutters(include_muted=False)
    targets = session.get_target_objects()
    cut_mat = _get_cut_material(scene)

    res_by_source: dict[str, bpy.types.Object] = {}
    for obj in list(results_col.objects):
        src = obj.get(RESULT_SOURCE_PROP)
        if src:
            res_by_source[str(src)] = obj

    piece_map: dict[tuple[str, str], bpy.types.Object] = {}
    for obj in list(pieces_col.objects):
        src = str(obj.get(RESULT_SOURCE_PROP, ""))
        cname = str(obj.get("fdt_piece_cutter", ""))
        if src and cname:
            piece_map[(src, cname)] = obj

    for orig in targets:
        res_obj = res_by_source.get(orig.name)
        if res_obj is None:
            res_name = f"FDT_Res_{orig.name}"[:MAX_BLENDER_NAME]
            res_obj = _create_working_copy(orig, res_name, results_col, session.session_id)
        _clear_fdt_modifiers(res_obj)
        res_obj.matrix_world = orig.matrix_world.copy()

        target_piece_objs: list[bpy.types.Object] = []
        for idx, (cutter_obj, zone) in enumerate(active_cutters):
            key = (orig.name, cutter_obj.name)
            p_obj = piece_map.get(key)
            if p_obj is None:
                piece_name = f"FDT_Piece_{orig.name}_ID-{_zone_name_token(zone, scene)}_{idx + 1:02d}"[:MAX_BLENDER_NAME]
                p_obj = _create_working_copy(orig, piece_name, pieces_col, session.session_id)
                piece_map[key] = p_obj
            _clear_fdt_modifiers(p_obj)
            p_obj.matrix_world = orig.matrix_world.copy()
            p_obj[PIECE_ZONE_PROP] = zone
            p_obj["fdt_piece_cutter"] = cutter_obj.name
            target_piece_objs.append(p_obj)

        rust_bridge.commit_target_cached_eval(
            session_id=session.session_id,
            target_obj=orig,
            cut_material=cut_mat,
            res_obj=res_obj,
            piece_objs=target_piece_objs,
        )

    try:
        from ..ops import apple_bite_ops

        apple_bite_ops.invalidate_zone_grid_cache(tag_redraw=True)
    except Exception:
        pass
    return True


def rebuild_session(
    context: bpy.types.Context,
    solver_override: str | None = None,
    for_apply: bool = False,
) -> bool:
    """Rebuild all non-destructive Results and Pieces (uses Rust Engine when selected, else Blender modifiers)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False

    cols = session.get_collections()
    vault_col = cols["vault"]
    results_col = cols["results"]
    pieces_col = cols["pieces"]
    if not results_col or not pieces_col:
        return False

    settings = _get_settings(scene)
    solver = solver_override or (settings.boolean_solver if settings else "RUST")

    # Style all cutters (including muted) and update per-cutter rough modifier
    for entry in session.cutters:
        c_obj = bpy.data.objects.get(entry.name)
        if c_obj is None or c_obj.type != "MESH":
            continue
        zone = str(c_obj.get(CUTTER_ZONE_PROP, entry.zone) or entry.zone)
        entry.zone = zone
        is_rough = bool(entry.rough)
        c_obj[CUTTER_ROUGH_PROP] = is_rough
        apply_cutter_visuals(c_obj, zone, muted=entry.muted)
        _ensure_rough_modifier(c_obj, scene, is_rough=is_rough, preview_mode=not for_apply)

    # Fast path: Rust Boolean Engine (O(N) single-pass split + prefix cache)
    if solver == "RUST" and rust_bridge.is_available():
        depsgraph = context.evaluated_depsgraph_get()
        use_gpu_mode = (
            not for_apply
            and settings is not None
            and bool(getattr(settings, "gpu_fast_drag", True))
            and len(session.get_valid_cutters(include_muted=False)) > 0
        )
        if use_gpu_mode:
            # Zero-Switch Textured GPU Mode: stay on GPU seamlessly without flickering!
            if vault_col is not None:
                set_collection_hidden(context.view_layer, vault_col, True)
            save_session(scene, session)
            live_sync.record_current_cutter_states(scene)
            live_sync.start_gpu_async_update(scene, depsgraph)
            return True

        # Direct real-mesh sync (Apply, 0 cutters, or GPU Fast Drag turned OFF)
        live_sync.reset_sync_cache(session.session_id)
        _sync_rust_session(
            scene,
            session,
            depsgraph,
            recreate_pieces=False,
            is_live=not for_apply,
            force_write=for_apply,
        )
        if vault_col is not None:
            set_collection_hidden(context.view_layer, vault_col, True)
        save_session(scene, session)
        live_sync.record_current_cutter_states(scene)
        context.view_layer.update()
        return True

    # Fallback: legacy Blender modifier stack (EXACT / FLOAT)
    if solver == "RUST":
        solver = "EXACT"

    active_cutters = session.get_valid_cutters(include_muted=False)

    # Map existing result objects by source name
    res_by_source: dict[str, bpy.types.Object] = {}
    for obj in list(results_col.objects):
        src = obj.get(RESULT_SOURCE_PROP)
        if src:
            res_by_source[str(src)] = obj

    # Rebuild Results stack using only unmuted cutters
    diff_ops = [("DIFFERENCE", c_obj) for c_obj, _ in active_cutters]
    targets = session.get_target_objects()
    for orig in targets:
        res_obj = res_by_source.get(orig.name)
        if res_obj is None:
            res_name = f"FDT_Res_{orig.name}"[:MAX_BLENDER_NAME]
            res_obj = _create_working_copy(orig, res_name, results_col, session.session_id)
        _ensure_mark_and_uv_stack(res_obj, scene, diff_ops, solver=solver)

    # Rebuild Pieces cleanly (one piece per target x active cutter, non-overlapping)
    _remove_objects_in_collection(pieces_col)
    for orig in targets:
        for idx, (cutter_obj, zone) in enumerate(active_cutters):
            piece_name = f"FDT_Piece_{orig.name}_ID-{_zone_name_token(zone, scene)}_{idx + 1:02d}"[:MAX_BLENDER_NAME]
            piece_obj = _create_working_copy(orig, piece_name, pieces_col, session.session_id)
            piece_obj[PIECE_ZONE_PROP] = zone
            piece_obj["fdt_piece_cutter"] = cutter_obj.name
            bool_ops: list[tuple[str, bpy.types.Object]] = [("INTERSECT", cutter_obj)]
            for prev_cutter, _ in active_cutters[:idx]:
                bool_ops.append(("DIFFERENCE", prev_cutter))
            _ensure_mark_and_uv_stack(piece_obj, scene, bool_ops, solver=solver)

    # Ensure Vault stays hidden and protected
    if vault_col is not None:
        set_collection_hidden(context.view_layer, vault_col, True)

    save_session(scene, session)
    context.view_layer.update()
    return True


def add_box_cutter(
    context: bpy.types.Context,
    location: Vector | tuple[float, float, float] | None = None,
    normal: Vector | None = None,
    size: float = 2.0,
    zone: str | None = None,
) -> tuple[bool, str, bpy.types.Object | None]:
    """Spawn a new colored-wireframe Box Cutter at the target center (or given location)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite", None

    cols = session.get_collections()
    cutters_col = cols["cutters"]
    if cutters_col is None:
        return False, "ไม่พบโฟลเดอร์ตัวตัด", None

    settings = _get_settings(scene)
    zone_id = _resolve_zone_id(scene, zone)
    idx = len(session.cutters) + 1
    zone_in_name = _zone_name_token(zone_id, scene)
    cutter_name = f"Cutter_{session.session_id}_ID-{zone_in_name}_{idx:02d}"[:MAX_BLENDER_NAME]

    mesh = bpy.data.meshes.new(f"{cutter_name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=float(size))
    bm.to_mesh(mesh)
    bm.free()

    cutter = bpy.data.objects.new(cutter_name, mesh)
    cutters_col.objects.link(cutter)

    if location is None:
        bounds = compute_targets_bounds(session.get_target_objects())
        if bounds is not None:
            min_v, max_v = bounds
            cx = (min_v.x + max_v.x) * 0.5
            cy = (min_v.y + max_v.y) * 0.5
            # Place near the top of the building so the box is immediately visible without X-ray
            cz = max_v.z - float(size) * 0.15
            loc = Vector((cx, cy, cz))
        else:
            loc = Vector((0.0, 0.0, 0.0))
    else:
        loc = Vector(location)

    if normal is not None and normal.length > 1e-5:
        rot_quat = normal.normalized().to_track_quat("Z", "Y")
        cutter.matrix_world = Matrix.Translation(loc) @ rot_quat.to_matrix().to_4x4()
    else:
        cutter.location = loc

    default_rough = bool(settings.enable_rough) if settings else False
    cutter[CUTTER_PROP] = True
    cutter[CUTTER_ZONE_PROP] = zone_id
    cutter[CUTTER_ROUGH_PROP] = default_rough
    init_cutter_rough_settings(cutter, scene)
    cutter[OBJ_SESSION_ID_PROP] = session.session_id
    apply_cutter_visuals(cutter, zone_id, muted=False)

    session.cutters.insert(0, CutterEntry(name=cutter.name, zone=zone_id, muted=False, rough=default_rough))
    save_session(scene, session)
    with profiler.action_scope("ADD_BOX", cutter.name):
        rebuild_session(context)
    safe_select_objects(context, [cutter], active=cutter)
    return True, f"วางกล่องตัด '{cutter.name}' (โซน {zone_registry.zone_token(zone_id, scene)}) แล้ว — กด G/R/S ขยับได้เลย", cutter


def add_mesh_data_cutter(
    context: bpy.types.Context,
    vertices,
    faces,
    zone: str | None = None,
    name_prefix: str = "DrawCutter",
) -> tuple[bool, str, bpy.types.Object | None]:
    """Create a Blender cutter object from plain (vertices, faces) produced by the standalone Draw Engine."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite", None

    cols = session.get_collections()
    cutters_col = cols["cutters"]
    if cutters_col is None:
        return False, "ไม่พบโฟลเดอร์ตัวตัด", None

    verts_list = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    faces_list = [tuple(int(i) for i in f) for f in faces]
    if len(verts_list) < 4 or len(faces_list) < 4:
        return False, "รูปทรงที่วาดมีจุดไม่พอสำหรับสร้างตัวตัด", None

    # Center local geometry at bounding-box center so G/R/S and 3D Gear Gizmo sit neatly on the cutter
    xs = [v[0] for v in verts_list]
    ys = [v[1] for v in verts_list]
    zs = [v[2] for v in verts_list]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    local_verts = [(v[0] - cx, v[1] - cy, v[2] - cz) for v in verts_list]

    settings = _get_settings(scene)
    zone_id = _resolve_zone_id(scene, zone)
    idx = len(session.cutters) + 1
    zone_in_name = _zone_name_token(zone_id, scene)
    cutter_name = f"{name_prefix}_{session.session_id}_ID-{zone_in_name}_{idx:02d}"[:MAX_BLENDER_NAME]

    mesh = bpy.data.meshes.new(f"{cutter_name}_Mesh")
    mesh.from_pydata(local_verts, [], faces_list)
    mesh.update(calc_edges=True)

    cutter = bpy.data.objects.new(cutter_name, mesh)
    cutter.location = Vector((cx, cy, cz))
    cutters_col.objects.link(cutter)

    default_rough = bool(settings.enable_rough) if settings else False
    cutter[CUTTER_PROP] = True
    cutter[CUTTER_ZONE_PROP] = zone_id
    cutter[CUTTER_ROUGH_PROP] = default_rough
    init_cutter_rough_settings(cutter, scene)
    cutter[OBJ_SESSION_ID_PROP] = session.session_id
    apply_cutter_visuals(cutter, zone_id, muted=False)

    session.cutters.insert(0, CutterEntry(name=cutter.name, zone=zone_id, muted=False, rough=default_rough))
    save_session(scene, session)
    with profiler.action_scope("ADD_DRAW_CUTTER", cutter.name):
        rebuild_session(context)
    safe_select_objects(context, [cutter], active=cutter)
    return True, f"สร้างตัวตัดจากเส้นวาด '{cutter.name}' (โซน {zone_registry.zone_token(zone_id, scene)}) แล้ว", cutter


_SCULPT_OP_LABELS: dict[str, str] = {
    "DIFFERENCE": "ปาดออก (Carve Out)",
    "INTERSECT": "เก็บเฉพาะข้างใน (Keep Inside)",
    "UNION": "ต่อเติมเพิ่ม (Add Shape)",
}


def _build_face_mask_mesh(
    cutter: bpy.types.Object,
    mask_face_indices: Sequence[int],
    mask_depth: float = 1.5,
    outward_pad: float = 0.25,
) -> bpy.types.Mesh | None:
    """Build a watertight world-space slab volume extruded inward/outward from selected cutter faces."""
    src_mesh = cutter.data
    if not isinstance(src_mesh, bpy.types.Mesh) or len(src_mesh.polygons) == 0:
        return None

    mw = cutter.matrix_world.copy()
    depth_in = max(float(mask_depth), 0.05)
    pad_out = max(float(outward_pad), 0.05)

    bm = bmesh.new()
    built_any = False
    n_polys = len(src_mesh.polygons)

    for raw_idx in mask_face_indices:
        fi = int(raw_idx)
        if fi < 0 or fi >= n_polys:
            continue
        poly = src_mesh.polygons[fi]
        if len(poly.vertices) < 3:
            continue

        world_pts = [mw @ src_mesh.vertices[vi].co for vi in poly.vertices]
        center = sum(world_pts, Vector((0.0, 0.0, 0.0))) / len(world_pts)

        # Compute true world-space face normal via Newell's method (robust under non-uniform scale)
        normal = Vector((0.0, 0.0, 0.0))
        k_len = len(world_pts)
        for i in range(k_len):
            p0 = world_pts[i]
            p1 = world_pts[(i + 1) % k_len]
            normal.x += (p0.y - p1.y) * (p0.z + p1.z)
            normal.y += (p0.z - p1.z) * (p0.x + p1.x)
            normal.z += (p0.x - p1.x) * (p0.y + p1.y)
        if normal.length <= 1e-8:
            continue
        normal.normalize()

        # Slightly expand face perimeter in-plane (2 cm) to avoid coplanar boolean slivers at face edges
        expanded = []
        for p in world_pts:
            radial = p - center
            r_len = radial.length
            if r_len > 1e-6:
                expanded.append(p + (radial / r_len) * 0.02)
            else:
                expanded.append(p.copy())

        out_verts = [bm.verts.new(p + normal * pad_out) for p in expanded]
        in_verts = [bm.verts.new(p - normal * depth_in) for p in expanded]

        try:
            bm.faces.new(out_verts)
            bm.faces.new(list(reversed(in_verts)))
            for i in range(k_len):
                j = (i + 1) % k_len
                bm.faces.new((out_verts[i], in_verts[i], in_verts[j], out_verts[j]))
            built_any = True
        except ValueError:
            continue

    if not built_any or len(bm.faces) < 4:
        bm.free()
        return None

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mask_mesh = bpy.data.meshes.new("_FDT_TempFaceMask_Mesh")
    bm.to_mesh(mask_mesh)
    bm.free()
    mask_mesh.update(calc_edges=True)
    return mask_mesh


def sculpt_existing_cutter_with_mesh_data(
    context: bpy.types.Context,
    cutter_name: str,
    vertices,
    faces,
    operation: str = "DIFFERENCE",
    mask_face_indices: Sequence[int] | None = None,
    mask_depth: float = 1.5,
) -> tuple[bool, str, bpy.types.Object | None]:
    """Boolean-sculpt an existing cutter mesh in place (DIFFERENCE / INTERSECT / UNION).

    Temporarily disables FDT_RoughCutter during the boolean so we always sculpt the clean
    base mesh, preserves cutter transform/origin centering, and aborts safely if the result
    would become empty.

    When mask_face_indices is provided, restricts the sculpt strictly to the slab volume
    extruded from the selected cutter faces so other parts of the cutter stay 100% untouched.
    """
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite", None

    cutter = bpy.data.objects.get(cutter_name)
    if cutter is None or cutter.type != "MESH" or not bool(cutter.get(CUTTER_PROP, False)):
        return False, f"ไม่พบตัวตัด '{cutter_name}'", None

    op_type = (operation or "DIFFERENCE").upper()
    if op_type not in ("DIFFERENCE", "INTERSECT", "UNION"):
        op_type = "DIFFERENCE"

    verts_list = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    faces_list = [tuple(int(i) for i in f) for f in faces]
    if len(verts_list) < 4 or len(faces_list) < 4:
        return False, "รูปทรงที่วาดมีจุดไม่พอสำหรับแต่งตัวตัด", None

    cols = session.get_collections()
    temp_col = cols.get("cutters") or scene.collection

    # Create temporary world-space prism mesh object for exact boolean
    temp_mesh = bpy.data.meshes.new("_FDT_TempSculpt_Mesh")
    temp_mesh.from_pydata(verts_list, [], faces_list)
    temp_mesh.update(calc_edges=True)
    temp_obj = bpy.data.objects.new("_FDT_TempSculpt", temp_mesh)
    temp_col.objects.link(temp_obj)

    mask_obj = None
    mask_mesh = None
    tool_obj = temp_obj
    cutter_bool_op = op_type
    used_mask = False

    # Temporarily disable FDT_RoughCutter so boolean operates on clean base geometry
    rough_mod = cutter.modifiers.get("FDT_RoughCutter")
    prev_rough_show = rough_mod.show_viewport if rough_mod is not None else False
    if rough_mod is not None:
        rough_mod.show_viewport = False

    bool_mod = None
    new_mesh = None
    temp_cleanup_objs: list[bpy.types.Object] = [temp_obj]
    temp_cleanup_meshes: list[bpy.types.Mesh] = [temp_mesh]

    try:
        # Optional Face Mask clamping: protect everything outside the selected cutter face slab(s)
        if mask_face_indices:
            out_pad = max(float(mask_depth), 0.5) if op_type == "UNION" else 0.25
            mask_mesh = _build_face_mask_mesh(
                cutter,
                mask_face_indices=mask_face_indices,
                mask_depth=mask_depth,
                outward_pad=out_pad,
            )
            if mask_mesh is not None:
                mask_obj = bpy.data.objects.new("_FDT_TempFaceMask", mask_mesh)
                temp_col.objects.link(mask_obj)
                temp_cleanup_objs.append(mask_obj)
                temp_cleanup_meshes.append(mask_mesh)
                used_mask = True

                # Masked set algebra:
                # - Carve Out (DIFFERENCE):   tool = P ∩ M,  cutter = cutter - tool
                # - Add Shape (UNION):        tool = P ∩ M,  cutter = cutter ∪ tool
                # - Keep Inside (INTERSECT):  tool = M - P,  cutter = cutter - tool
                if op_type == "INTERSECT":
                    m_mod = mask_obj.modifiers.new(name="_FDT_MaskSub", type="BOOLEAN")
                    m_mod.operation = "DIFFERENCE"
                    m_mod.operand_type = "OBJECT"
                    m_mod.object = temp_obj
                    _set_boolean_solver(m_mod, prefer="EXACT")
                    context.view_layer.update()
                    deps_m = context.evaluated_depsgraph_get()
                    eff_mesh = bpy.data.meshes.new_from_object(mask_obj.evaluated_get(deps_m), depsgraph=deps_m)
                    cutter_bool_op = "DIFFERENCE"
                else:
                    m_mod = temp_obj.modifiers.new(name="_FDT_MaskClamp", type="BOOLEAN")
                    m_mod.operation = "INTERSECT"
                    m_mod.operand_type = "OBJECT"
                    m_mod.object = mask_obj
                    _set_boolean_solver(m_mod, prefer="EXACT")
                    context.view_layer.update()
                    deps_m = context.evaluated_depsgraph_get()
                    eff_mesh = bpy.data.meshes.new_from_object(temp_obj.evaluated_get(deps_m), depsgraph=deps_m)
                    cutter_bool_op = op_type

                if eff_mesh is None or len(eff_mesh.polygons) == 0:
                    if eff_mesh is not None:
                        bpy.data.meshes.remove(eff_mesh)
                    return False, "เส้นที่วาดไม่ซ้อนกับพื้นที่ Face Mask ที่เลือก (ยกเลิกอัตโนมัติ)", None

                eff_obj = bpy.data.objects.new("_FDT_TempMaskedTool", eff_mesh)
                temp_col.objects.link(eff_obj)
                temp_cleanup_objs.append(eff_obj)
                temp_cleanup_meshes.append(eff_mesh)
                tool_obj = eff_obj

        bool_mod = cutter.modifiers.new(name="_FDT_TempSculptBool", type="BOOLEAN")
        bool_mod.operation = cutter_bool_op
        bool_mod.operand_type = "OBJECT"
        bool_mod.object = tool_obj
        _set_boolean_solver(bool_mod, prefer="EXACT")

        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        eval_cutter = cutter.evaluated_get(depsgraph)
        new_mesh = bpy.data.meshes.new_from_object(eval_cutter, depsgraph=depsgraph)
    finally:
        if bool_mod is not None:
            try:
                cutter.modifiers.remove(bool_mod)
            except Exception:
                pass
        if rough_mod is not None:
            try:
                rough_mod.show_viewport = prev_rough_show
            except Exception:
                pass
        for obj in temp_cleanup_objs:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        for m in temp_cleanup_meshes:
            if m is not None and m.users == 0:
                try:
                    bpy.data.meshes.remove(m)
                except Exception:
                    pass

    if new_mesh is None or len(new_mesh.polygons) == 0:
        if new_mesh is not None:
            bpy.data.meshes.remove(new_mesh)
        return False, "รูปที่วาดไม่ซ้อนกับตัวตัดเดิม หรือปาดตัวตัดหายไปทั้งหมด (ยกเลิกอัตโนมัติ)", None

    # Recenter local vertices around bbox center and ensure clean outward normals
    bm = bmesh.new()
    bm.from_mesh(new_mesh)
    bm.verts.ensure_lookup_table()
    if len(bm.verts) < 4 or len(bm.faces) < 4:
        bm.free()
        bpy.data.meshes.remove(new_mesh)
        return False, "ผลลัพธ์หลังตัดแต่งมีเหลี่ยมไม่พอ (ยกเลิกอัตโนมัติ)", None

    min_l = Vector((float("inf"), float("inf"), float("inf")))
    max_l = Vector((float("-inf"), float("-inf"), float("-inf")))
    for v in bm.verts:
        min_l.x = min(min_l.x, v.co.x)
        min_l.y = min(min_l.y, v.co.y)
        min_l.z = min(min_l.z, v.co.z)
        max_l.x = max(max_l.x, v.co.x)
        max_l.y = max(max_l.y, v.co.y)
        max_l.z = max(max_l.z, v.co.z)
    local_center = (min_l + max_l) * 0.5
    for v in bm.verts:
        v.co -= local_center

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(new_mesh)
    bm.free()
    new_mesh.update(calc_edges=True)

    mw = cutter.matrix_world.copy()
    world_center = mw @ local_center

    old_mesh = cutter.data
    base_mesh_name = old_mesh.name if old_mesh is not None else f"{cutter.name}_Mesh"
    cutter.data = new_mesh
    cutter.matrix_world.translation = world_center

    if old_mesh is not None and old_mesh.users == 0 and isinstance(old_mesh, bpy.types.Mesh):
        bpy.data.meshes.remove(old_mesh)
    new_mesh.name = base_mesh_name[:MAX_BLENDER_NAME]

    rebuild_session(context)
    safe_select_objects(context, [cutter], active=cutter)
    op_label = _SCULPT_OP_LABELS.get(op_type, op_type)
    mask_tag = f" [Mask {len(mask_face_indices)} หน้า / ลึก {float(mask_depth):.2f}m]" if (used_mask and mask_face_indices) else ""
    return True, f"แต่งตัวตัด '{cutter.name}' แบบ {op_label}{mask_tag} เรียบร้อย (กด Ctrl+Z ย้อนกลับได้)", cutter


_DIR_VECTORS: dict[str, tuple[int, float]] = {
    "+X": (0, 1.0),
    "-X": (0, -1.0),
    "+Y": (1, 1.0),
    "-Y": (1, -1.0),
    "+Z": (2, 1.0),
    "-Z": (2, -1.0),
}


def _next_zone_letter(zone: str) -> str:
    """Next zone in registry order: A..Z then custom zones, wrapping to A."""
    return zone_registry.next_zone_key(zone, getattr(bpy.context, "scene", None))


def add_adjacent_box_cutter(
    context: bpy.types.Context,
    direction: str = "+X",
    overlap: float = 0.05,
    auto_next_zone: bool = True,
) -> tuple[bool, str, bpy.types.Object | None]:
    """Clone the active/last cutter flush-adjacent along direction with slight overlap to prevent slivers.

    Because Pieces subtract all prior cutters (DIFFERENCE cutter_0..i-1), overlapping the
    previous cutter by a few cm eliminates both gaps and coplanar boolean precision slivers.
    """
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite", None

    cols = session.get_collections()
    cutters_col = cols["cutters"]
    if cutters_col is None:
        return False, "ไม่พบโฟลเดอร์ตัวตัด", None

    # Find reference cutter: active/selected cutter, else last cutter in session
    ref_cutter = None
    active = context.active_object
    if active is not None and bool(active.get(CUTTER_PROP, False)):
        ref_cutter = active
    else:
        for obj in context.selected_objects:
            if bool(obj.get(CUTTER_PROP, False)):
                ref_cutter = obj
                break
    if ref_cutter is None:
        valid = session.get_valid_cutters(include_muted=True)
        if valid:
            ref_cutter = valid[-1][0]
    if ref_cutter is None:
        return False, "กรุณาวางกล่องตัดอันแรกก่อน แล้วค่อยกดต่อกล่องชิดขอบ", None

    context.view_layer.update()
    axis_idx, sign = _DIR_VECTORS.get(direction.upper(), (0, 1.0))

    # Measure reference cutter extent along its local axis
    extent = float(ref_cutter.dimensions[axis_idx])
    if extent <= 1e-4:
        extent = 2.0
    safe_overlap = min(max(float(overlap), 0.005), extent * 0.25)
    step_dist = extent - safe_overlap

    local_axis = Vector((0.0, 0.0, 0.0))
    local_axis[axis_idx] = sign
    rot_mat = ref_cutter.matrix_world.to_3x3().normalized()
    world_offset = rot_mat @ (local_axis * step_dist)

    ref_zone = zone_registry.resolve_zone(ref_cutter.get(CUTTER_ZONE_PROP, "A"), scene).key
    settings = _get_settings(scene)
    if auto_next_zone:
        zone_id = _next_zone_letter(ref_zone)
        if settings is not None:
            settings.active_zone = zone_id
    else:
        zone_id = _resolve_zone_id(scene, settings.active_zone if settings else ref_zone)

    idx = len(session.cutters) + 1
    cutter_name = f"Cutter_{session.session_id}_ID-{_zone_name_token(zone_id, scene)}_{idx:02d}"[:MAX_BLENDER_NAME]

    mesh_copy = ref_cutter.data.copy()
    mesh_copy.name = f"{cutter_name}_Mesh"
    cutter = bpy.data.objects.new(cutter_name, mesh_copy)
    cutters_col.objects.link(cutter)

    cutter.matrix_world = ref_cutter.matrix_world.copy()
    cutter.matrix_world.translation = ref_cutter.matrix_world.translation + world_offset

    default_rough = bool(settings.enable_rough) if settings else False
    cutter[CUTTER_PROP] = True
    cutter[CUTTER_ZONE_PROP] = zone_id
    cutter[CUTTER_ROUGH_PROP] = default_rough
    init_cutter_rough_settings(cutter, scene, ref_cutter=ref_cutter)
    cutter[OBJ_SESSION_ID_PROP] = session.session_id
    apply_cutter_visuals(cutter, zone_id, muted=False)

    session.cutters.insert(0, CutterEntry(name=cutter.name, zone=zone_id, muted=False, rough=default_rough))
    save_session(scene, session)
    rebuild_session(context)
    safe_select_objects(context, [cutter], active=cutter)
    return True, f"ต่อกล่องชิดขอบ ({direction}) โซน {zone_registry.zone_token(zone_id, scene)} เรียบร้อย (เหลื่อมกัน {safe_overlap*100:.1f} ซม. ไม่เหลือเศษ)", cutter


def add_custom_mesh_cutters(
    context: bpy.types.Context,
    mesh_objects: list[bpy.types.Object],
    zone: str | None = None,
) -> tuple[bool, str, list[bpy.types.Object]]:
    """Register existing custom mesh object(s) as Apple Bite cutters with a Zone ID."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ยังไม่ได้เริ่มโหมด Apple Bite", []

    cols = session.get_collections()
    cutters_col = cols["cutters"]
    if cutters_col is None:
        return False, "ไม่พบโฟลเดอร์ตัวตัด", []

    settings = _get_settings(scene)
    zone_id = _resolve_zone_id(scene, zone)
    existing_names = {c.name for c in session.cutters}

    added: list[bpy.types.Object] = []
    for obj in mesh_objects:
        if obj is None or obj.type != "MESH":
            continue
        if is_protected_object(obj) or RESULT_SOURCE_PROP in obj:
            continue
        if obj.name in existing_names:
            continue
        default_rough = bool(settings.enable_rough) if settings else False
        move_object_to_collection(obj, cutters_col)
        obj[CUTTER_PROP] = True
        obj[CUTTER_ZONE_PROP] = zone_id
        obj[CUTTER_ROUGH_PROP] = default_rough
        init_cutter_rough_settings(obj, scene)
        obj[OBJ_SESSION_ID_PROP] = session.session_id
        apply_cutter_visuals(obj, zone_id, muted=False)
        session.cutters.insert(0, CutterEntry(name=obj.name, zone=zone_id, muted=False, rough=default_rough))
        existing_names.add(obj.name)
        added.append(obj)

    if not added:
        return False, "กรุณาเลือกโมเดลที่ปั้นเองก่อนกดเพิ่มเป็นตัวตัด", []

    save_session(scene, session)
    rebuild_session(context)
    safe_select_objects(context, added, active=added[-1])
    return True, f"เพิ่มโมเดลตัดเอง {len(added)} ชิ้น (โซน {zone_registry.zone_token(zone_id, scene)}) แล้ว", added


def set_cutters_zone(
    context: bpy.types.Context,
    cutters: list[bpy.types.Object],
    zone: str,
) -> int:
    """Update the Zone ID and wireframe color on one or more cutters."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return 0

    zone_id = zone_registry.resolve_zone(zone, scene).key
    updated = 0
    cutter_map = {c.name: c for c in session.cutters}
    for obj in cutters:
        if obj is None or not bool(obj.get(CUTTER_PROP, False)):
            continue
        obj[CUTTER_ZONE_PROP] = zone_id
        muted = False
        if obj.name in cutter_map:
            cutter_map[obj.name].zone = zone_id
            muted = cutter_map[obj.name].muted
        apply_cutter_visuals(obj, zone_id, muted=muted)
        updated += 1

    if updated:
        save_session(scene, session)
        rebuild_session(context)
    return updated


def cycle_cutter_zone(
    context: bpy.types.Context, cutter_name: str, zones: str = "ABCDEFGH"
) -> tuple[str, str] | None:
    """Cycle a cutter to the next registry zone and return ``(key, new_object_name)``."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return None
    obj = bpy.data.objects.get(cutter_name)
    if obj is None:
        return None

    cur = zone_registry.resolve_zone(obj.get(CUTTER_ZONE_PROP, "A"), scene).key
    next_z = zone_registry.next_zone_key(cur, scene)
    if not set_cutters_zone(context, [obj], next_z):
        return None

    # Load the session while the object still has its old name; otherwise
    # load_session() would prune the entry before we can rename it.
    old_name = obj.name
    session = load_session(scene)
    if session is None:
        return None
    entry = next((c for c in session.cutters if c.name == old_name), None)
    if entry is None:
        return None
    entry.zone = next_z

    # Keep the readable zone token in the object name while the stable key stays in props.
    old_token = _zone_name_token(cur, scene)
    new_token = _zone_name_token(next_z, scene)
    tag_re = re.compile(r"_ID-" + re.escape(old_token) + r"(?=_)")
    new_name = tag_re.sub(f"_ID-{new_token}", old_name, count=1)[:MAX_BLENDER_NAME]
    renamed = new_name != old_name
    if renamed:
        obj.name = new_name
    entry.name = obj.name
    for piece in bpy.data.objects:
        if str(piece.get("fdt_piece_cutter", "") or "") == old_name:
            piece["fdt_piece_cutter"] = obj.name
    save_session(scene, session)
    if renamed:
        rebuild_session(context)
    return next_z, obj.name


def toggle_cutter_mute(context: bpy.types.Context, cutter_name: str) -> tuple[bool, bool]:
    """Pause or resume a single cutter without deleting it. Returns (ok, new_muted_state)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, False
    for entry in session.cutters:
        if entry.name == cutter_name:
            entry.muted = not entry.muted
            obj = bpy.data.objects.get(cutter_name)
            if obj is not None:
                apply_cutter_visuals(obj, entry.zone, muted=entry.muted)
            save_session(scene, session)
            rebuild_session(context)
            return True, entry.muted
    return False, False


def toggle_cutter_rough(context: bpy.types.Context, cutter_name: str) -> tuple[bool, bool]:
    """Toggle rough cut edge independently on a single cutter. Returns (ok, new_rough_state)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, False
    for entry in session.cutters:
        if entry.name == cutter_name:
            entry.rough = not entry.rough
            if not entry.rough:
                _ROUGH_WIRE_PREVIEW_UNTIL.pop(cutter_name, None)
            obj = bpy.data.objects.get(cutter_name)
            if obj is not None:
                obj[CUTTER_ROUGH_PROP] = entry.rough
            save_session(scene, session)
            with profiler.action_scope("TOGGLE_ROUGH", f"{cutter_name}={entry.rough}"):
                rebuild_session(context)
            return True, entry.rough
    return False, False


def set_all_cutters_rough(context: bpy.types.Context, enabled: bool) -> int:
    """Apply rough cut edge ON/OFF across all cutters in the active session."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return 0
    val = bool(enabled)
    for entry in session.cutters:
        entry.rough = val
        if not val:
            _ROUGH_WIRE_PREVIEW_UNTIL.pop(entry.name, None)
        obj = bpy.data.objects.get(entry.name)
        if obj is not None:
            obj[CUTTER_ROUGH_PROP] = val
    save_session(scene, session)
    rebuild_session(context)
    return len(session.cutters)


def delete_cutter(context: bpy.types.Context, cutter_name: str) -> tuple[bool, str]:
    """Remove a single cutter cleanly from the session and rebuild stacks."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัดที่เปิดอยู่"

    before = len(session.cutters)
    session.cutters = [c for c in session.cutters if c.name != cutter_name]
    obj = bpy.data.objects.get(cutter_name)
    if obj is not None:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)

    if len(session.cutters) == before and obj is None:
        return False, f"ไม่พบตัวตัด '{cutter_name}'"

    save_session(scene, session)
    rebuild_session(context)
    return True, f"ลบตัวตัด '{cutter_name}' แล้ว"


def select_cutter(context: bpy.types.Context, cutter_name: str) -> bool:
    """Select a cutter by name in the 3D Viewport."""
    obj = bpy.data.objects.get(cutter_name)
    if obj is None:
        return False
    safe_select_objects(context, [obj], active=obj)
    return True


def move_cutter_layer(
    context: bpy.types.Context,
    cutter_name: str,
    direction: str = "UP",
) -> tuple[bool, str]:
    """Move a cutter up or down by 1 layer (index 0 = top priority) and rebuild cuts."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัดที่เปิดอยู่"

    idx = next((i for i, c in enumerate(session.cutters) if c.name == cutter_name), -1)
    if idx < 0:
        return False, f"ไม่พบตัวตัด '{cutter_name}'"

    step = -1 if direction.upper() == "UP" else 1
    new_idx = idx + step
    if new_idx < 0 or new_idx >= len(session.cutters):
        return False, "ตัวตัดอยู่ตำแหน่งสุดขอบแล้ว"

    session.cutters[idx], session.cutters[new_idx] = session.cutters[new_idx], session.cutters[idx]
    save_session(scene, session)
    rebuild_session(context)
    return True, f"ย้าย '{cutter_name}' ไป Layer #{new_idx + 1} แล้ว"


def move_cutter_relative_to(
    context: bpy.types.Context,
    source_name: str,
    target_name: str,
    position: str = "ABOVE",
) -> tuple[bool, str]:
    """Move a single cutter directly ABOVE or BELOW another target cutter (index 0 = top priority)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัดที่เปิดอยู่"
    if not source_name or not target_name or source_name == target_name:
        return False, "กรุณาเลือกกล่องตัดเป้าหมายที่ต่างจากกล่องเดิม"

    src_idx = next((i for i, c in enumerate(session.cutters) if c.name == source_name), -1)
    if src_idx < 0:
        return False, f"ไม่พบตัวตัด '{source_name}'"
    if not any(c.name == target_name for c in session.cutters):
        return False, f"ไม่พบตัวตัดเป้าหมาย '{target_name}'"

    src_entry = session.cutters.pop(src_idx)
    tgt_idx = next((i for i, c in enumerate(session.cutters) if c.name == target_name), -1)
    if tgt_idx < 0:
        session.cutters.insert(src_idx, src_entry)
        return False, f"ไม่พบตัวตัดเป้าหมาย '{target_name}'"

    pos = str(position or "ABOVE").upper()
    insert_idx = tgt_idx if pos == "ABOVE" else (tgt_idx + 1)
    session.cutters.insert(insert_idx, src_entry)

    save_session(scene, session)
    rebuild_session(context)
    rel_word = "เหนือ" if pos == "ABOVE" else "ใต้"
    return True, f"ย้าย '{source_name}' ไปอยู่{rel_word} '{target_name}' (Layer #{insert_idx + 1}) แล้ว"


def reorder_cutter_layers(
    context: bpy.types.Context,
    ordered_names: list[str],
) -> tuple[bool, str]:
    """Reorder session.cutters to match ordered_names (from Drag & Drop) and rebuild cuts."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัดที่เปิดอยู่"

    by_name = {c.name: c for c in session.cutters}
    new_order: list[CutterEntry] = []
    seen: set[str] = set()
    for name in ordered_names:
        if name in by_name and name not in seen:
            new_order.append(by_name[name])
            seen.add(name)
    # Preserve any cutter not included in ordered_names
    for entry in session.cutters:
        if entry.name not in seen:
            new_order.append(entry)

    if [c.name for c in new_order] == [c.name for c in session.cutters]:
        return True, "ลำดับ Layer เหมือนเดิม"

    session.cutters = new_order
    save_session(scene, session)
    rebuild_session(context)
    return True, f"อัปเดตลำดับ Layer ({len(new_order)} ชั้น) แล้ว"


def _postprocess_baked_mesh(
    obj: bpy.types.Object,
    cut_material: bpy.types.Material | None,
    cut_planes: Sequence[tuple[Vector, Vector]] | None = None,
    uv_scale: float = 0.5,
    is_cut_main: bool = False,
    line_backfaces: bool = False,
    cutter_bboxes_local: Sequence[tuple[Vector, Vector]] | None = None,
) -> int:
    """Clean degenerate slivers, auto-fill unfilled solid cut-plane cavities, line exposed
    single-sided backfaces, safely dissolve collinear 2-way vertices (preserving UVs & normals),
    mark cut border as `freestyle_edge`, reconstruct `sharp_edge`, and restore exact custom split normals.

    Returns the number of auto-filled/lined triangles added.
    """
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh) or len(mesh.polygons) == 0:
        return 0

    import numpy as np
    from . import cap_fill

    cut_mat_slot = cap_fill._ensure_cut_material_slot(mesh, cut_material) if (cut_planes or line_backfaces) else 0

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    uv_layers = list(bm.loops.layers.uv.values())
    orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
    nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)

    # 1. Remove microscopic zero-area degenerate sliver faces cleanly
    degen_faces = [f for f in bm.faces if f.calc_area() < 1e-8]
    if degen_faces:
        bmesh.ops.delete(bm, geom=degen_faces, context="FACES")

    # 1b. Auto-fill remaining unfilled solid holes strictly on active cut planes (never windows/rooms!)
    auto_filled_tris = 0
    if cut_planes:
        _, auto_filled_tris = cap_fill.fill_unclosed_cut_planes_in_bmesh(
            bm=bm,
            matrix_world=obj.matrix_world,
            cut_mat_slot=cut_mat_slot,
            uv_scale=uv_scale,
            planes=cut_planes,
            is_cut_main=is_cut_main,
        )
        orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
        nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)

    # 1c. Line exposed single-sided backfaces inside cut-open attics/walls (fixes Backface Culling see-through!)
    if line_backfaces:
        bm.normal_update()
        auto_filled_tris += cap_fill.line_exposed_backfaces_in_bmesh(
            bm=bm,
            matrix_world=obj.matrix_world,
            cut_mat_slot=cut_mat_slot,
            uv_scale=uv_scale,
            planes=cut_planes or (),
            is_cut_main=is_cut_main,
            planes_are_cap_outward=not bool(cut_planes),
            cutter_bboxes_local=cutter_bboxes_local,
        )
        orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
        nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)

    # 2. Dissolve straight 2-way vertices ONLY when safe (skip UV seams, sharp normal breaks,
    #    material boundaries, and cut-border vertices so shape & normals stay 100% intact).
    straight_verts = []
    for v in bm.verts:
        if len(v.link_edges) == 2:
            e1, e2 = v.link_edges[0], v.link_edges[1]
            v1 = e1.other_vert(v).co - v.co
            v2 = e2.other_vert(v).co - v.co
            if v1.length > 1e-6 and v2.length > 1e-6:
                if v1.normalized().dot(v2.normalized()) < -0.9999:
                    skip = False
                    if len(v.link_loops) > 1:
                        # Protect UV seams
                        for uv_l in uv_layers:
                            ref_uv = v.link_loops[0][uv_l].uv
                            if any((lp[uv_l].uv - ref_uv).length > 1e-4 for lp in v.link_loops[1:]):
                                skip = True
                                break
                        # Protect sharp normal breaks
                        if not skip and nrm_layer is not None:
                            ref_n = v.link_loops[0][nrm_layer]
                            if ref_n.length > 1e-5:
                                ref_nu = ref_n.normalized()
                                for lp in v.link_loops[1:]:
                                    ln = lp[nrm_layer]
                                    if ln.length > 1e-5 and ref_nu.dot(ln.normalized()) < 0.998:
                                        skip = True
                                        break
                        # Protect material / orig-vs-cut boundaries
                        if not skip and len(v.link_faces) > 1:
                            ref_mat = v.link_faces[0].material_index
                            if any(f.material_index != ref_mat for f in v.link_faces[1:]):
                                skip = True
                            elif orig_layer is not None:
                                ref_o = v.link_faces[0][orig_layer] >= 0.5
                                if any((f[orig_layer] >= 0.5) != ref_o for f in v.link_faces[1:]):
                                    skip = True
                    if not skip:
                        straight_verts.append(v)
    if straight_verts:
        bmesh.ops.dissolve_verts(bm, verts=straight_verts)

    # 3. Identify outer cut border edges (`orig != cut`) AND sharp edges from preserved split normals
    border_edge_indices: set[int] = set()
    sharp_edge_indices: set[int] = set()
    bm.edges.ensure_lookup_table()

    for idx, edge in enumerate(bm.edges):
        if len(edge.link_faces) != 2:
            continue
        f0, f1 = edge.link_faces[0], edge.link_faces[1]
        is_border = False
        if orig_layer is not None:
            a = f0[orig_layer] >= 0.5
            b = f1[orig_layer] >= 0.5
            if a != b:
                is_border = True
                border_edge_indices.add(idx)
                sharp_edge_indices.add(idx)

        if not is_border and nrm_layer is not None:
            # Compare preserved split normals across the edge at both shared vertices
            v_a, v_b = edge.verts[0], edge.verts[1]
            is_sharp = False
            for sv in (v_a, v_b):
                lp0 = next((lp for lp in f0.loops if lp.vert == sv), None)
                lp1 = next((lp for lp in f1.loops if lp.vert == sv), None)
                if lp0 is not None and lp1 is not None:
                    n0 = lp0[nrm_layer]
                    n1 = lp1[nrm_layer]
                    if n0.length > 1e-5 and n1.length > 1e-5:
                        if n0.normalized().dot(n1.normalized()) < 0.995:
                            is_sharp = True
                            break
            if is_sharp:
                sharp_edge_indices.add(idx)
        elif not is_border and nrm_layer is None:
            # Fallback for non-Rust (Blender solver) meshes: sharp if dihedral > 35 deg
            if f0.normal.dot(f1.normal) < 0.819:
                sharp_edge_indices.add(idx)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # 4. Write `freestyle_edge` on cut border edges
    num_edges = len(mesh.edges)
    if border_edge_indices and num_edges > 0:
        fs_attr = mesh.attributes.get("freestyle_edge")
        if fs_attr is None:
            fs_attr = mesh.attributes.new(name="freestyle_edge", type="BOOLEAN", domain="EDGE")
        flags = np.zeros(num_edges, dtype=bool)
        flags[list(border_edge_indices)] = True
        fs_attr.data.foreach_set("value", flags)

    # 5. Reconstruct `sharp_edge` boolean attribute (matches original Enfusion / game mesh export!)
    if sharp_edge_indices and num_edges > 0:
        sh_attr = mesh.attributes.get("sharp_edge")
        if sh_attr is not None and (sh_attr.data_type != "BOOLEAN" or sh_attr.domain != "EDGE"):
            mesh.attributes.remove(sh_attr)
            sh_attr = None
        if sh_attr is None:
            sh_attr = mesh.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
        sflags = np.zeros(num_edges, dtype=bool)
        sflags[list(sharp_edge_indices)] = True
        sh_attr.data.foreach_set("value", sflags)

    # 6. Restore exact preserved custom split normals (`has_custom_normals = True`) and remove temp attr
    nrm_attr = mesh.attributes.get(rust_bridge.PRESERVED_NRM_ATTR)
    if nrm_attr is not None and nrm_attr.domain == "CORNER" and len(mesh.loops) > 0:
        try:
            nl = len(mesh.loops)
            nrm_flat = np.empty(nl * 3, dtype=np.float32)
            nrm_attr.data.foreach_get("vector", nrm_flat)
            nrm_3 = nrm_flat.reshape(-1, 3)
            # Normalize and replace any zero vectors with polygon geometric normal
            lens = np.linalg.norm(nrm_3, axis=1)
            good = lens > 1e-5
            nrm_3[good] /= lens[good, None]
            if not np.all(good):
                for poly in mesh.polygons:
                    pn = poly.normal
                    for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        if not good[li]:
                            nrm_3[li] = (pn.x, pn.y, pn.z)
            mesh.attributes.remove(nrm_attr)
            mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
            mesh.normals_split_custom_set(nrm_3)
        except Exception as exc:
            print(f"[FullDestruc_Tool] Custom split normal restore warning on {obj.name}: {exc}")

    # 7. Ensure cut material is present on mesh if needed
    if cut_material is not None and cut_material.name not in mesh.materials:
        orig_attr = mesh.attributes.get(ORIG_ATTR_NAME)
        if orig_attr is not None and orig_attr.domain == "FACE":
            vals = [0.0] * len(mesh.polygons)
            orig_attr.data.foreach_get("value", vals)
            if any(v < 0.5 for v in vals):
                mesh.materials.append(cut_material)
                slot_idx = len(mesh.materials) - 1
                for poly, val in zip(mesh.polygons, vals):
                    if val < 0.5:
                        poly.material_index = slot_idx
    return auto_filled_tris


def fill_cut_caps(context: bpy.types.Context) -> tuple[bool, str, int]:
    """1-click Manual Fill Cut Cap (Edit Mode: selected open edges; Object Mode: auto cut-plane scan)."""
    from . import cap_fill

    scene = context.scene
    settings = _get_settings(scene)
    cut_mat = _get_cut_material(scene)
    uv_scale = (
        float(settings.texel_per_meter) / max(1.0, float(settings.texture_size))
        if settings is not None
        else 0.5
    )
    return cap_fill.manual_fill_cut_caps(context, cut_mat, uv_scale)


def apply_session(
    context: bpy.types.Context,
    keep_result: bool = True,
    keep_pieces: bool = True,
) -> tuple[bool, str, list[bpy.types.Object]]:
    """Bake Results and/or Pieces into Final collection (cleanly overwriting previous Final on re-edit)."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัด Apple Bite ที่เปิดอยู่", []

    # A re-cut invalidates any rebar generator whose tagged faces live on the
    # affected hosts.  Mark them stale only — the user must press Re-apply
    # and then Bake.
    try:
        from . import rebar_state

        for name in list(session.target_names):
            host = bpy.data.objects.get(name)
            if host is not None:
                rebar_state.mark_stale_for_host(host, scene)
    except Exception:
        pass

    # Flush any active Edit Mode edits (e.g. resized cutter box face) before baking
    if context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    cols = session.get_collections()
    root_col = cols["root"]
    vault_col = cols["vault"]
    cutters_col = cols["cutters"]
    results_col = cols["results"]
    pieces_col = cols["pieces"]

    final_col = get_or_create_collection(
        session.final_col_name or f"{FINAL_ROOT_PREFIX}{session.session_id}",
        parent=scene.collection,
        color_tag="COLOR_04",
    )
    # Cleanly overwrite any old baked objects in FDT_Final_<id> (zero duplicates on re-edit!)
    _remove_objects_in_collection(final_col)
    set_collection_excluded(context.view_layer, final_col, False)

    cut_mat = _get_cut_material(scene)
    baked_objects: list[bpy.types.Object] = []

    settings = _get_settings(scene)
    use_rust = (
        settings is not None
        and getattr(settings, "boolean_solver", "RUST") == "RUST"
        and rust_bridge.is_available()
    )
    auto_fill_caps = bool(getattr(settings, "apply_auto_fill_caps", True)) if settings is not None else True
    line_backfaces = bool(getattr(settings, "apply_line_backfaces", True)) if settings is not None else True
    uv_scale = (
        float(settings.texel_per_meter) / max(1.0, float(settings.texture_size))
        if settings is not None
        else 0.5
    )
    total_auto_filled_tris = 0

    t_apply_start = time.perf_counter()
    profiler.record_marker("APPLY_START", f"{session.session_id} (rust={use_rust})")

    with live_sync.suspend():
        with temporary_vault_evaluation(
            context,
            [root_col, vault_col, cutters_col, results_col, pieces_col],
        ):
            with profiler.action_scope("APPLY_EVAL", session.session_id):
                rebuild_session(context, solver_override="RUST" if use_rust else "EXACT", for_apply=True)
            depsgraph = context.evaluated_depsgraph_get()

            from . import cap_fill

            active_cutter_objs = [c_obj for c_obj, _ in session.get_valid_cutters(include_muted=False)]

            def _bake_one(
                src_obj: bpy.types.Object,
                final_name: str,
                is_cut_main: bool = False,
            ) -> bpy.types.Object | None:
                nonlocal total_auto_filled_tris
                if use_rust and src_obj.type == "MESH" and src_obj.data is not None:
                    # Rust already wrote the final evaluated mesh directly into src_obj.data
                    baked_mesh = src_obj.data.copy()
                    baked_mesh.validate(verbose=False, clean_customdata=False)
                else:
                    eval_obj = src_obj.evaluated_get(depsgraph)
                    baked_mesh = bpy.data.meshes.new_from_object(
                        eval_obj,
                        preserve_all_data_layers=True,
                        depsgraph=depsgraph,
                    )
                if len(baked_mesh.polygons) == 0:
                    bpy.data.meshes.remove(baked_mesh)
                    return None
                baked_mesh.name = f"{final_name}_Mesh"
                new_obj = bpy.data.objects.new(final_name, baked_mesh)
                new_obj.matrix_world = src_obj.matrix_world.copy()
                new_obj[OBJ_SESSION_ID_PROP] = session.session_id
                final_col.objects.link(new_obj)
                planes = (
                    cap_fill.extract_cutter_planes_local(active_cutter_objs, new_obj.matrix_world, depsgraph)
                    if (auto_fill_caps or line_backfaces) and active_cutter_objs
                    else None
                )
                cutter_bboxes = (
                    cap_fill.extract_cutter_bboxes_local(active_cutter_objs, new_obj.matrix_world, depsgraph)
                    if line_backfaces and active_cutter_objs
                    else None
                )
                total_auto_filled_tris += _postprocess_baked_mesh(
                    new_obj,
                    cut_mat,
                    cut_planes=planes if auto_fill_caps else None,
                    uv_scale=uv_scale,
                    is_cut_main=is_cut_main,
                    line_backfaces=line_backfaces,
                    cutter_bboxes_local=cutter_bboxes,
                )
                return new_obj

            if keep_result and results_col is not None:
                for res_obj in list(results_col.objects):
                    orig_name = str(res_obj.get(RESULT_SOURCE_PROP, res_obj.name))
                    final_name = f"{_strip_existing_fdst_id(orig_name)}_Cut"[:MAX_BLENDER_NAME]
                    baked = _bake_one(res_obj, final_name, is_cut_main=True)
                    if baked is not None:
                        baked_objects.append(baked)

            if keep_pieces and pieces_col is not None:
                for piece_obj in list(pieces_col.objects):
                    orig_name = str(piece_obj.get(RESULT_SOURCE_PROP, piece_obj.name))
                    zone = str(piece_obj.get(PIECE_ZONE_PROP, "A") or "A")
                    final_name = _format_zone_name(orig_name, _zone_name_token(zone, scene))
                    baked = _bake_one(piece_obj, final_name, is_cut_main=False)
                    if baked is not None:
                        baked_objects.append(baked)

    # Persist metadata on root_col BEFORE clearing active scene flag so Re-edit works later!
    save_session(scene, session, persist_meta=True)

    # Exclude & hide ALL working collections so ONLY FDT_Final_<id> is visible after Apply!
    for key in ("cutters", "results", "pieces", "vault"):
        if cols[key] is not None:
            set_collection_hidden(context.view_layer, cols[key], True)
    if root_col is not None:
        set_collection_excluded(context.view_layer, root_col, True)

    # Ensure Final collection is visible and ready to focus on
    set_collection_excluded(context.view_layer, final_col, False)
    context.view_layer.update()

    save_session(scene, None)
    _s_reset = _get_settings(scene)
    if _s_reset is not None and getattr(_s_reset, "preview_display_mode", "ALL") != "ALL":
        _s_reset.preview_display_mode = "ALL"
    rust_bridge.destroy_session(session.session_id)
    live_sync.reset_sync_cache(session.session_id)
    safe_select_objects(context, baked_objects)
    apply_ms = (time.perf_counter() - t_apply_start) * 1000.0
    profiler.record_marker("APPLY_DONE", f"{len(baked_objects)} baked pieces", duration_ms=apply_ms)
    return True, f"Apply เสร็จแล้ว: ได้ชิ้นส่วนทั้งหมด {len(baked_objects)} ชิ้น", baked_objects


def exit_session(context: bpy.types.Context) -> tuple[bool, str]:
    """Exit Apple Bite mode cleanly while keeping non-destructive setup intact for later re-edit."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return True, "ไม่มีงานตัดที่เปิดอยู่"

    from . import gpu_preview

    if gpu_preview.is_active():
        with live_sync.suspend():
            commit_rust_gpu_drag(scene)

    cols = session.get_collections()
    if cols["vault"] is not None:
        set_collection_hidden(context.view_layer, cols["vault"], True)

    save_session(scene, session, persist_meta=True)
    save_session(scene, None)
    _s_reset = _get_settings(scene)
    if _s_reset is not None and getattr(_s_reset, "preview_display_mode", "ALL") != "ALL":
        _s_reset.preview_display_mode = "ALL"
    rust_bridge.destroy_session(session.session_id)
    live_sync.reset_sync_cache(session.session_id)
    return True, f"พักงานตัด '{session.session_id}' แล้ว (คลิกชิ้นงานเดิมเพื่อกลับมาทำต่อได้)"


def _remove_object_and_orphan_mesh(obj: bpy.types.Object) -> None:
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)


def _cleanup_session_collections_and_strays(
    session_ids: Iterable[str],
    all_sessions: dict[str, AppleBiteSession],
    keep_objects: set[bpy.types.Object],
) -> None:
    """Remove all working/final collections and any stray objects belonging to session_ids."""
    sid_set = set(session_ids)

    # 1. Remove objects & collections recorded in each session
    for sid in sid_set:
        sess = all_sessions.get(sid)
        if sess is not None:
            cols = sess.get_collections()
            for key in ("results", "pieces", "cutters", "final", "vault", "root"):
                col = cols.get(key)
                if col is None:
                    continue
                for obj in list(col.objects):
                    if obj not in keep_objects:
                        _remove_object_and_orphan_mesh(obj)
                try:
                    bpy.data.collections.remove(col)
                except ReferenceError:
                    pass

        # Fallback: also remove any FDT_Final_<sid> / FDT_Bite_<sid> by exact name
        for col_name in (
            f"{FINAL_ROOT_PREFIX}{sid}",
            f"{BITE_ROOT_PREFIX}{sid}",
            f"FDT_Results_{sid}",
            f"FDT_Pieces_{sid}",
            f"FDT_Cutters_{sid}",
            f"FDT_Vault_{sid}",
        ):
            col = bpy.data.collections.get(col_name)
            if col is None:
                continue
            for obj in list(col.objects):
                if obj not in keep_objects:
                    _remove_object_and_orphan_mesh(obj)
            try:
                bpy.data.collections.remove(col)
            except ReferenceError:
                pass

    # 2. Sweep any stray objects still tagged with fdt_session_id in sid_set
    for obj in list(bpy.data.objects):
        if obj in keep_objects:
            continue
        if str(obj.get(OBJ_SESSION_ID_PROP, "")) in sid_set:
            _remove_object_and_orphan_mesh(obj)


def cancel_session(context: bpy.types.Context) -> tuple[bool, str]:
    """Cancel active Apple Bite session, restore its vault originals, and remove working + final collections."""
    scene = context.scene
    session = load_session(scene)
    if session is None:
        return False, "ไม่มีงานตัดที่เปิดอยู่"

    cols = session.get_collections()
    restored = restore_originals_from_vault(scene, cols["vault"])
    keep_set = set(restored)

    _cleanup_session_collections_and_strays(
        [session.session_id],
        {session.session_id: session},
        keep_set,
    )

    save_session(scene, None)
    _s_reset = _get_settings(scene)
    if _s_reset is not None and getattr(_s_reset, "preview_display_mode", "ALL") != "ALL":
        _s_reset.preview_display_mode = "ALL"
    rust_bridge.destroy_session(session.session_id)
    live_sync.reset_sync_cache(session.session_id)
    context.view_layer.update()
    safe_select_objects(context, restored)
    return True, f"ยกเลิกงานตัดและคืนต้นฉบับ {len(restored)} ชิ้นแล้ว"


def get_reset_preview_info(scene: bpy.types.Scene | None, session_id: str) -> dict:
    """Return preview details for resetting an entire Apple Bite cut chain back to its root original(s)."""
    root_ids, family_ids = collect_related_session_family(scene, session_id)
    all_sessions = load_all_stored_sessions(scene)
    original_names: list[str] = []
    for rsid in root_ids:
        sess = all_sessions.get(rsid)
        if sess is None:
            continue
        vault_col = sess.get_collections().get("vault")
        if vault_col is not None and len(vault_col.objects) > 0:
            for obj in vault_col.objects:
                if obj.name not in original_names:
                    original_names.append(obj.name)
        else:
            for name in sess.target_names:
                if name not in original_names:
                    original_names.append(name)
    return {
        "root_ids": root_ids,
        "family_ids": family_ids,
        "original_names": original_names,
    }


def reset_to_original(
    context: bpy.types.Context,
    session_id: str,
) -> tuple[bool, str, list[bpy.types.Object]]:
    """Delete all related cuts (including Cut Further chains & baked Finals) and restore root original(s)."""
    scene = context.scene
    root_ids, family_ids = collect_related_session_family(scene, session_id)
    if not family_ids:
        return False, f"ไม่พบงานตัดที่เกี่ยวข้องกับ '{session_id}'", []

    all_sessions = load_all_stored_sessions(scene)

    # 1. Restore true root original(s) from root vault(s) back to the main scene collection
    restored: list[bpy.types.Object] = []
    restored_set: set[bpy.types.Object] = set()
    for rsid in root_ids:
        sess = all_sessions.get(rsid)
        if sess is None:
            continue
        cols = sess.get_collections()
        objs = restore_originals_from_vault(scene, cols.get("vault"))
        for obj in objs:
            if OBJ_SESSION_ID_PROP in obj:
                del obj[OBJ_SESSION_ID_PROP]
            if obj not in restored_set:
                restored_set.add(obj)
                restored.append(obj)

    # 2. Delete all cut pieces, cutters, working results, baked finals, intermediate vaults, strays, and collections
    _cleanup_session_collections_and_strays(family_ids, all_sessions, restored_set)

    # 3. Clear active session if it belonged to this family
    for fid in family_ids:
        rust_bridge.destroy_session(fid)
        live_sync.reset_sync_cache(fid)
    active_sess = load_session(scene)
    if active_sess is not None and active_sess.session_id in family_ids:
        save_session(scene, None)

    context.view_layer.update()
    safe_select_objects(context, restored)
    return (
        True,
        f"ลบงานตัดทั้งหมด ({len(family_ids)} รอบ) และคืนต้นฉบับ {len(restored)} ชิ้นแล้ว",
        restored,
    )
