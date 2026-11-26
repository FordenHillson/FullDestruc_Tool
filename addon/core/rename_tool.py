"""User-driven object renaming for FullDestruc_Tool (independent of Zone Registry).

Zero UI code here — called by operators, N-Panel preview, and automated tests.
Supports:
- ``FULL`` mode: rename selected object(s) directly to user-entered text (with
  optional ``_01, _02`` numbering when multiple objects are selected).
- ``TAG`` mode: replace only the embedded ``_ID-<old>`` or ``_FDST_ID-<old>``
  segment with any user-entered text while preserving prefixes, indices, and
  ``_LOD*`` suffixes (not tied to the Scene zone registry).

When renamed objects belong to an active or stored Apple Bite session (cutters
or targets), all internal session references and working-piece links stay in
sync without modifying zone assignments.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Sequence

import bpy

from . import apple_bite, live_sync
from .session import (
    SESSION_COL_META_PROP,
    SESSION_SCENE_PROP,
    AppleBiteSession,
    load_session,
)

MAX_BLENDER_NAME = 63

# Cutter / working-piece tag: ..._ID-<old>_18 (ignores _FDST_ID-)
_CUTTER_ID_IDX_RE = re.compile(r"(?<!_FDST)_ID-(.+?)(_\d+(?:\.\d+)?)$")
# Generic non-FDST _ID- tag with optional _LOD suffix: ..._ID-<old>(_LOD0)?
_GENERIC_ID_RE = re.compile(r"(?<!_FDST)_ID-(.+?)(_LOD\d+(?:_[A-Za-z0-9]+)?)?(?:\.\d+)?$")
# Baked Enfusion FDST tag with optional _LOD / _Cut suffix: ..._FDST_ID-<old>(_LOD0|_Cut)?
_FDST_ID_TAG_RE = re.compile(r"_FDST_ID-(.+?)(_LOD\d+(?:_[A-Za-z0-9]+)?|_Cut)?(?:\.\d+)?$")
# Trailing LOD suffix on plain names without an ID tag
_TRAILING_LOD_RE = re.compile(r"(_LOD\d+(?:_[A-Za-z0-9]+)?)(?:\.\d+)?$")


def sanitize_id_text(value: str) -> str:
    """Clean user-entered ID text while allowing Enfusion characters (#, -, _, !, .)."""
    cleaned = re.sub(r"\s+", "_", str(value or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9_#!.-]+", "_", cleaned)
    return cleaned.strip("_")[:MAX_BLENDER_NAME]


def extract_id_from_name(name: str) -> str:
    """Extract the editable ID segment from an object name (empty string if none)."""
    text = str(name or "").strip()
    if not text:
        return ""
    m = _CUTTER_ID_IDX_RE.search(text)
    if m is not None:
        return m.group(1)
    m = _GENERIC_ID_RE.search(text)
    if m is not None:
        return m.group(1)
    m = _FDST_ID_TAG_RE.search(text)
    if m is not None:
        return m.group(1)
    return ""


def _assemble_with_limit(prefix: str, tag: str, suffix: str) -> str:
    max_prefix = max(0, MAX_BLENDER_NAME - len(tag) - len(suffix))
    if len(prefix) > max_prefix:
        prefix = prefix[:max_prefix]
    return f"{prefix}{tag}{suffix}"[:MAX_BLENDER_NAME]


def replace_id_in_name(old_name: str, new_id: str) -> str:
    """Replace the ID tag inside ``old_name`` with free-form ``new_id`` text.

    Examples:
    - ``Cutter_PubVillage_E_2L01_V2_dst_ID-B_18`` + ``A``
      -> ``Cutter_PubVillage_E_2L01_V2_dst_ID-A_18``
    - ``PubVillage_E_2L01_V2_dst_FDST_ID-C_LOD0`` + ``A``
      -> ``PubVillage_E_2L01_V2_dst_FDST_ID-A_LOD0``
    - ``Wall_LOD0`` + ``A`` -> ``Wall_ID-A_LOD0``
    - ``Wall`` + ``A`` -> ``Wall_ID-A``
    """
    clean_id = sanitize_id_text(new_id)
    if not clean_id:
        return old_name

    # 1. Cutter / working-piece pattern: ..._ID-<old>_18
    m = _CUTTER_ID_IDX_RE.search(old_name)
    if m is not None:
        if m.group(1) == clean_id:
            return old_name
        prefix = old_name[: m.start()]
        suffix = re.sub(r"\.\d+$", "", m.group(2))
        return _assemble_with_limit(prefix, f"_ID-{clean_id}", suffix)

    # 2. Generic non-FDST _ID- pattern: ..._ID-<old>(_LOD0)?
    m = _GENERIC_ID_RE.search(old_name)
    if m is not None:
        if m.group(1) == clean_id:
            return old_name
        prefix = old_name[: m.start()]
        suffix = m.group(2) or ""
        return _assemble_with_limit(prefix, f"_ID-{clean_id}", suffix)

    # 3. Baked / Enfusion pattern: ..._FDST_ID-<old>(_LOD0|_Cut)?
    m = _FDST_ID_TAG_RE.search(old_name)
    if m is not None:
        if m.group(1) == clean_id:
            return old_name
        prefix = old_name[: m.start()]
        suffix = m.group(2) or ""
        return _assemble_with_limit(prefix, f"_FDST_ID-{clean_id}", suffix)

    # 4. Plain name without any ID tag: insert before trailing _LOD* or append
    base = re.sub(r"\.\d+$", "", old_name)
    m_lod = _TRAILING_LOD_RE.search(base)
    if m_lod is not None:
        prefix = base[: m_lod.start()]
        suffix = m_lod.group(1)
        return _assemble_with_limit(prefix, f"_ID-{clean_id}", suffix)

    return _assemble_with_limit(base, f"_ID-{clean_id}", "")


def get_target_objects(context: bpy.types.Context | None) -> list[bpy.types.Object]:
    """Return currently selected objects (falling back to active_object if none selected)."""
    if context is None:
        return []
    seen: set[bpy.types.Object] = set()
    result: list[bpy.types.Object] = []
    for obj in getattr(context, "selected_objects", ()) or ():
        if obj is not None and obj not in seen:
            seen.add(obj)
            result.append(obj)
    if not result:
        active = getattr(context, "active_object", None)
        if active is not None:
            result.append(active)
    return result


def build_target_names(
    objects: Sequence[bpy.types.Object],
    mode: str = "FULL",
    text: str = "",
    auto_number: bool = True,
    start_index: int = 1,
) -> list[tuple[bpy.types.Object, str]]:
    """Compute ``[(obj, desired_name), ...]`` without mutating Blender data."""
    objs = [o for o in objects if o is not None]
    if not objs:
        return []

    mode_key = str(mode or "FULL").upper()
    if mode_key == "TAG":
        clean_id = sanitize_id_text(text)
        if not clean_id:
            return []
        return [(obj, replace_id_in_name(obj.name, clean_id)) for obj in objs]

    # FULL mode: direct user-typed rename (auto-numbered when 2+ objects are selected)
    base = str(text or "").strip()
    if not base:
        return []

    ordered = sorted(objs, key=lambda o: o.name) if len(objs) > 1 else objs
    use_num = bool(auto_number) and len(ordered) > 1
    start = max(0, int(start_index))
    pad = max(2, len(str(start + len(ordered) - 1)))

    plan: list[tuple[bpy.types.Object, str]] = []
    for idx, obj in enumerate(ordered):
        if use_num:
            suffix = f"_{start + idx:0{pad}d}"
            max_base = max(0, MAX_BLENDER_NAME - len(suffix))
            desired = f"{base[:max_base]}{suffix}"[:MAX_BLENDER_NAME]
        else:
            desired = base[:MAX_BLENDER_NAME]
        plan.append((obj, desired))
    return plan


def _sync_apple_bite_references(
    scene: bpy.types.Scene | None,
    renamed_map: dict[str, str],
) -> None:
    """Keep active and stored Apple Bite sessions synced after manual object renames.

    Never alters zone IDs or colors — only updates object name references so
    cutters, targets, and working pieces stay attached.
    """
    if not renamed_map:
        return

    # 1. Update working piece / result custom properties referencing old names
    for obj in bpy.data.objects:
        old_cutter = str(obj.get("fdt_piece_cutter", "") or "")
        if old_cutter in renamed_map:
            obj["fdt_piece_cutter"] = renamed_map[old_cutter]
        old_src = str(obj.get(apple_bite.RESULT_SOURCE_PROP, "") or "")
        if old_src in renamed_map:
            obj[apple_bite.RESULT_SOURCE_PROP] = renamed_map[old_src]

    # 2. Migrate rough wireframe preview timer keys
    for old_name, new_name in renamed_map.items():
        if old_name in apple_bite._ROUGH_WIRE_PREVIEW_UNTIL:
            apple_bite._ROUGH_WIRE_PREVIEW_UNTIL[new_name] = apple_bite._ROUGH_WIRE_PREVIEW_UNTIL.pop(old_name)

    if scene is None:
        return

    # 3. Update active session JSON (load raw JSON BEFORE prune_missing could drop renamed names)
    def _patch_session_json(raw_json: str | None) -> str | None:
        if not raw_json:
            return None
        try:
            data = json.loads(raw_json)
            sess = AppleBiteSession.from_dict(data)
        except Exception:
            return None
        changed = False
        for i, tname in enumerate(sess.target_names):
            if tname in renamed_map:
                sess.target_names[i] = renamed_map[tname]
                changed = True
        for entry in sess.cutters:
            if entry.name in renamed_map:
                entry.name = renamed_map[entry.name]
                changed = True
        return json.dumps(sess.to_dict()) if changed else None

    patched_scene = _patch_session_json(scene.get(SESSION_SCENE_PROP))
    if patched_scene is not None:
        scene[SESSION_SCENE_PROP] = patched_scene

    # 4. Update stored session metadata on every FDT_Bite_* root collection
    for col in bpy.data.collections:
        if not col.name.startswith(apple_bite.BITE_ROOT_PREFIX):
            continue
        patched_col = _patch_session_json(col.get(SESSION_COL_META_PROP))
        if patched_col is not None:
            col[SESSION_COL_META_PROP] = patched_col

    # 5. Refresh live_sync cutter state map and N-Panel UIList so dragging & display stay synced
    active_sess = load_session(scene)
    if active_sess is not None:
        try:
            from ..props import sync_cutter_layer_items

            sync_cutter_layer_items(scene, session=active_sess)
        except Exception:
            pass
        try:
            live_sync.record_current_cutter_states(scene)
        except Exception:
            pass


def rename_objects(
    context: bpy.types.Context | None,
    objects: Sequence[bpy.types.Object],
    mode: str = "FULL",
    text: str = "",
    auto_number: bool = True,
    start_index: int = 1,
    sync_mesh_data: bool = True,
) -> tuple[bool, str, list[tuple[str, str]]]:
    """Rename ``objects`` according to ``mode`` and ``text``, keeping Apple Bite sessions intact.

    Returns ``(ok, message, [(old_name, new_name), ...])``.
    """
    objs = [o for o in objects if o is not None]
    if not objs:
        return False, "กรุณาเลือก Object ที่ต้องการเปลี่ยนชื่อก่อน", []

    plan = build_target_names(
        objs,
        mode=mode,
        text=text,
        auto_number=auto_number,
        start_index=start_index,
    )
    if not plan:
        return False, "กรุณากรอกชื่อใหม่ก่อนกด Rename", []

    to_rename = [(obj, obj.name, desired) for obj, desired in plan if desired and desired != obj.name]
    if not to_rename:
        return False, "ชื่อ Object เหมือนเดิมอยู่แล้ว", []

    # Two-phase rename prevents spurious .001 collisions when target names overlap within the batch.
    batch_old_names = {old for _, old, _ in to_rename}
    batch_desired_names = {desired for _, _, desired in to_rename}
    overlap = batch_old_names & batch_desired_names
    if overlap:
        token = uuid.uuid4().hex[:6]
        for idx, (obj, old, _desired) in enumerate(to_rename):
            if old in overlap:
                obj.name = f"__fdt_tmp_{token}_{idx}__"

    renamed_pairs: list[tuple[str, str]] = []
    renamed_map: dict[str, str] = {}
    for obj, old_name, desired in to_rename:
        obj.name = desired
        actual_name = obj.name
        if actual_name != old_name:
            renamed_pairs.append((old_name, actual_name))
            renamed_map[old_name] = actual_name
            if sync_mesh_data:
                data = getattr(obj, "data", None)
                if data is not None and getattr(data, "users", 0) == 1:
                    suffix = "_Mesh" if getattr(obj, "type", "") == "MESH" else ""
                    new_data_name = f"{actual_name}{suffix}"[:MAX_BLENDER_NAME]
                    if data.name != new_data_name:
                        data.name = new_data_name

    if not renamed_pairs:
        return False, "ไม่มี Object ที่ถูกเปลี่ยนชื่อ", []

    scene = getattr(context, "scene", None) if context is not None else getattr(bpy.context, "scene", None)
    _sync_apple_bite_references(scene, renamed_map)

    if len(renamed_pairs) == 1:
        old_n, new_n = renamed_pairs[0]
        msg = f"เปลี่ยนชื่อ '{old_n}' -> '{new_n}' แล้ว"
    else:
        first_old, first_new = renamed_pairs[0]
        msg = f"เปลี่ยนชื่อ {len(renamed_pairs)} ชิ้นแล้ว (เช่น {first_old} -> {first_new})"
    return True, msg, renamed_pairs
