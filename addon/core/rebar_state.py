"""Scene/object state for Concrete Rebars generators.

Three layers of ownership (invariant: the cosmetic label never identifies a
generator):

1. ``Scene.fdt_rebar_registry_json`` — the authoritative list of generators.
2. ``Object.fdt_rebar_id`` + ``Object.fdt_rebar_params_json`` — per-object
   back-reference and a params snapshot that survives registry drift.
3. ``FDT_Rebar_<sid8>`` collection — where a *baked* generator lives.

Faces on the host mesh carry ``fdt_rebar_group`` (an INT index, not the sid), so
this module owns the index -> sid mapping in each registry entry.

Nothing here registers operators or touches the UI.  In particular, there is
no ``WindowManager`` singleton and no ``load_post`` handler: stale state is
resolved explicitly by :func:`load_all_generators` / :func:`is_stale`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import bpy

from . import rebar_generate as rg
from . import rebar_spline as rs

REGISTRY_PROP = "fdt_rebar_registry_json"
OBJ_ID_PROP = "fdt_rebar_id"
OBJ_PARAMS_PROP = "fdt_rebar_params_json"

COLLECTION_PREFIX = "FDT_Rebar_"
OBJECT_PREFIX = "Rebar_"
MAX_NAME_LEN = 63  # Blender's ID name limit is 63 bytes; keep ASCII so len == bytes.

DEFAULT_LABEL = ""

# All lengths are metres internally; the UI converts to/from mm.
BAR_SIZES = {
    "RB6": 6.0,
    "RB9": 9.0,
    "DB10": 10.0,
    "DB12": 12.0,
    "DB16": 16.0,
    "DB20": 20.0,
    "DB25": 25.0,
}
DEFAULT_BAR_SIZE = "CUSTOM"  # Red Panda bevel_depth 0.02 m == Ø40 mm
DEFAULT_DIAMETER_MM = 40.0


def default_params() -> dict:
    """Fresh parameter snapshot matched to Red Panda's metal rebar defaults."""
    return {
        "bar_size": DEFAULT_BAR_SIZE,
        "diameter_mm": DEFAULT_DIAMETER_MM,
        "align_x": True,
        "align_y": True,
        "align_z": True,
        "spacing_x": 0.4,
        "spacing_y": 0.4,
        "spacing_z": 0.4,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_z": 0.0,
        "sink_depth": 0.05,
        "min_length": 0.1,
        "max_length": 0.5,
        "bending_strength": 0.03,
        "bend_scale": 0.4,
        "subdivisions": 6,
        "random_seed": 0,
        "angle_threshold": 0.5,
        "normal_bias": 0.5,
        "gravity_bias": 0.5,
        "smooth_radius": 0.3,
        "avoidance_strength": 0.5,
        "max_bars": 5000,
        "fast_drag": True,
        "enable_debris": False,
        "debris_density": 0.5,
        "debris_min_size": 0.005,
        "debris_max_size": 0.04,
        "debris_irregularity": 0.4,
        "debris_concentration": 0.7,
        "debris_material": "",
        "ui_show_rebar": False,
    }


# Type/coercion table used when reading old or hand-edited JSON.
_BOOL_KEYS = {"align_x", "align_y", "align_z", "fast_drag", "enable_debris", "ui_show_rebar"}
_INT_KEYS = {"subdivisions", "random_seed", "max_bars"}
_FLOAT_KEYS = {
    "diameter_mm",
    "spacing_x", "spacing_y", "spacing_z",
    "offset_x", "offset_y", "offset_z",
    "sink_depth", "min_length", "max_length", "bending_strength", "bend_scale",
    "angle_threshold", "normal_bias", "gravity_bias", "smooth_radius",
    "avoidance_strength",
    "debris_density", "debris_min_size", "debris_max_size",
    "debris_irregularity", "debris_concentration",
}
_STR_KEYS = {"bar_size", "debris_material"}


def normalize_params(data) -> dict:
    """Coerce a params mapping into a complete, well-typed snapshot."""
    result = default_params()
    if not isinstance(data, dict):
        return result
    for key, value in data.items():
        if key in _BOOL_KEYS:
            result[key] = bool(value)
        elif key in _INT_KEYS:
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                pass
        elif key in _FLOAT_KEYS:
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                pass
        elif key == "debris_material":
            result[key] = str(value or "")[:64]
        elif key in _STR_KEYS:
            result[key] = str(value or DEFAULT_BAR_SIZE)[:32]
    # Guard against pathological values that would hang a raycast or collapse
    # the square profile.
    for key in ("spacing_x", "spacing_y", "spacing_z", "smooth_radius"):
        result[key] = max(float(result[key]), 1e-4)
    # Bend Scale is a wavelength: below 50 mm the bar aliases its own waves,
    # above 1000 mm the effect is invisible anyway.
    result["bend_scale"] = min(max(float(result["bend_scale"]), 0.05), 1.0)
    result["subdivisions"] = max(1, min(int(result["subdivisions"]), 12))
    result["max_bars"] = max(1, min(int(result["max_bars"]), 200000))
    result["diameter_mm"] = max(float(result["diameter_mm"]), 0.1)
    result["angle_threshold"] = max(-1.0, min(float(result["angle_threshold"]), 1.0))
    if result["max_length"] < result["min_length"]:
        result["min_length"], result["max_length"] = result["max_length"], result["min_length"]
    result["debris_density"] = min(max(float(result["debris_density"]), 0.0), 1.0)
    if result["debris_max_size"] < result["debris_min_size"]:
        result["debris_min_size"], result["debris_max_size"] = (
            result["debris_max_size"], result["debris_min_size"]
        )
    for key in ("debris_irregularity", "debris_concentration"):
        result[key] = min(max(float(result[key]), 0.0), 1.0)
    return result


def params_to_generation(params: dict) -> rg.RebarParams:
    """Project a full params snapshot onto the generation engine's data class."""
    params = normalize_params(params)
    return rg.RebarParams(
        spacing=(params["spacing_x"], params["spacing_y"], params["spacing_z"]),
        align=(params["align_x"], params["align_y"], params["align_z"]),
        offset=(params["offset_x"], params["offset_y"], params["offset_z"]),
        radius=rs.diameter_to_radius(params["diameter_mm"]),
        sink_depth=max(float(params["sink_depth"]), 0.0),
        min_length=max(float(params["min_length"]), 0.0),
        max_length=max(float(params["max_length"]), 0.0),
        bending_strength=max(float(params["bending_strength"]), 0.0),
        bend_scale=float(params["bend_scale"]),
        subdivisions=int(params["subdivisions"]),
        seed=int(params["random_seed"]),
        angle_threshold=float(params["angle_threshold"]),
        normal_bias=float(params["normal_bias"]),
        gravity_bias=float(params["gravity_bias"]),
        smooth_radius=float(params["smooth_radius"]),
        avoidance_strength=float(params["avoidance_strength"]),
        max_bars=int(params["max_bars"]),
        debris_enabled=bool(params["enable_debris"]),
        debris_density=float(params["debris_density"]),
        debris_min_size=float(params["debris_min_size"]),
        debris_max_size=float(params["debris_max_size"]),
        debris_irregularity=float(params["debris_irregularity"]),
        debris_concentration=float(params["debris_concentration"]),
    )


# ---------------------------------------------------------------------------
# Names + ids
# ---------------------------------------------------------------------------


def short_sid(sid) -> str:
    """Eight-character display token derived from a stable sid."""
    return str(sid or "").replace("-", "")[:8] or "00000000"


def bounded_name(value: str) -> str:
    """Trim to Blender's 63-character ID name limit."""
    return str(value or "")[:MAX_NAME_LEN]


def collection_name(sid) -> str:
    return bounded_name(f"{COLLECTION_PREFIX}{short_sid(sid)}")


def object_name(sid) -> str:
    return bounded_name(f"{OBJECT_PREFIX}{short_sid(sid)}")


def new_sid(used: set[str] | None = None) -> str:
    """Generate a collision-free 8-hex-character sid."""
    used = set(used or ())
    for _ in range(64):
        sid = uuid.uuid4().hex[:8]
        if sid not in used:
            return sid
    # Deterministic fallback if the RNG somehow keeps colliding.
    index = 0
    while f"{index:08x}" in used:
        index += 1
    return f"{index:08x}"


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


@dataclass
class RebarEntry:
    sid: str
    label: str = DEFAULT_LABEL
    host: str = ""
    target_material: str = ""
    rebar_material: str = ""
    collection: str = ""
    object: str = ""
    group: int = 0
    baked: bool = False
    stale: bool = False
    params: dict = field(default_factory=default_params)

    def to_dict(self) -> dict:
        return {
            "sid": str(self.sid),
            "label": str(self.label or ""),
            "host": str(self.host or ""),
            "target_material": str(self.target_material or ""),
            "rebar_material": str(self.rebar_material or ""),
            "collection": str(self.collection or ""),
            "object": str(self.object or ""),
            "group": int(self.group),
            "baked": bool(self.baked),
            "stale": bool(self.stale),
            "params": normalize_params(self.params),
        }

    @classmethod
    def from_dict(cls, data) -> "RebarEntry | None":
        if not isinstance(data, dict):
            return None
        sid = str(data.get("sid") or "").strip()
        if not sid:
            return None
        try:
            group = int(data.get("group", 0))
        except (TypeError, ValueError):
            group = 0
        return cls(
            sid=sid,
            label=str(data.get("label") or ""),
            host=str(data.get("host") or ""),
            target_material=str(data.get("target_material") or ""),
            rebar_material=str(data.get("rebar_material") or ""),
            collection=str(data.get("collection") or collection_name(sid)),
            object=str(data.get("object") or object_name(sid)),
            group=max(group, 0),
            baked=bool(data.get("baked", False)),
            stale=bool(data.get("stale", False)),
            params=normalize_params(data.get("params")),
        )

    @property
    def sid8(self) -> str:
        return short_sid(self.sid)

    @property
    def display_label(self) -> str:
        return self.label.strip() or object_name(self.sid)


def _scene_of(scene=None):
    return scene if scene is not None else getattr(bpy.context, "scene", None)


def load_registry(scene=None) -> list[RebarEntry]:
    sc = _scene_of(scene)
    if sc is None:
        return []
    raw = sc.get(REGISTRY_PROP)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    result: list[RebarEntry] = []
    seen: set[str] = set()
    for item in data:
        entry = RebarEntry.from_dict(item)
        if entry is None or entry.sid in seen:
            continue
        seen.add(entry.sid)
        result.append(entry)
    return result


def save_registry(entries, scene=None) -> None:
    sc = _scene_of(scene)
    if sc is None:
        return
    sc[REGISTRY_PROP] = json.dumps([e.to_dict() for e in entries], ensure_ascii=False)


def find_entry(sid, scene=None) -> RebarEntry | None:
    key = str(sid or "")
    return next((e for e in load_registry(scene) if e.sid == key), None)


def find_entry_for_object(obj, scene=None) -> RebarEntry | None:
    sid = object_sid(obj)
    if sid is None:
        return None
    return find_entry(sid, scene)


def add_entry(entry: RebarEntry, scene=None) -> bool:
    entries = load_registry(scene)
    if any(e.sid == entry.sid for e in entries):
        return False
    entries.append(entry)
    save_registry(entries, scene)
    return True


def replace_params(entry: RebarEntry, params) -> RebarEntry:
    """Return ``entry`` with a normalised params snapshot (live-preview helper)."""
    entry.params = normalize_params(params)
    return entry


def update_entry(entry: RebarEntry, scene=None) -> bool:
    entries = load_registry(scene)
    for index, existing in enumerate(entries):
        if existing.sid == entry.sid:
            entries[index] = entry
            save_registry(entries, scene)
            return True
    return False


def remove_entry(sid, scene=None) -> bool:
    entries = load_registry(scene)
    kept = [e for e in entries if e.sid != str(sid)]
    if len(kept) == len(entries):
        return False
    save_registry(kept, scene)
    return True


def next_group_for_host(host_obj, scene=None) -> int:
    """Lowest free rebar tag index on ``host_obj`` across all registry entries."""
    used = {
        e.group
        for e in load_registry(scene)
        if e.host == getattr(host_obj, "name", "") and e.group >= 0
    }
    if host_obj is not None and getattr(host_obj, "data", None) is not None:
        attr = host_obj.data.attributes.get(rs.GROUP_ATTR_NAME)
        if attr is not None and attr.domain == "FACE" and attr.data_type == "INT":
            used.update(item.value for item in attr.data if item.value >= 0)
    index = 0
    while index in used:
        index += 1
    return index


# ---------------------------------------------------------------------------
# Object side of the state
# ---------------------------------------------------------------------------


def object_sid(obj) -> str | None:
    if obj is None:
        return None
    value = obj.get(OBJ_ID_PROP)
    return str(value) if value else None


def mark_object(obj, sid) -> None:
    if obj is not None:
        obj[OBJ_ID_PROP] = str(sid)


def save_params(obj, params) -> None:
    if obj is not None:
        obj[OBJ_PARAMS_PROP] = json.dumps(normalize_params(params), ensure_ascii=False)


def load_params(obj) -> dict:
    if obj is None:
        return default_params()
    raw = obj.get(OBJ_PARAMS_PROP)
    if not raw:
        return default_params()
    try:
        return normalize_params(json.loads(raw))
    except Exception:
        return default_params()


def resolve_object(entry: RebarEntry):
    """Find the generated object by sid first, then by stored name (migration only)."""
    for obj in bpy.data.objects:
        if object_sid(obj) == entry.sid:
            return obj
    return bpy.data.objects.get(entry.object)


def resolve_host(entry: RebarEntry):
    return bpy.data.objects.get(entry.host)


def resolve_material(name: str):
    return bpy.data.materials.get(name) if name else None


# ---------------------------------------------------------------------------
# Rebar material asset (offline-first, EBT sync is strictly optional)
# ---------------------------------------------------------------------------

REBAR_MATERIAL_NAME = "ST_Metal_Rebar_Ruin_01"
REBAR_RESOURCE_NAME = "{1ED917F76283BA69}Assets/_SharedMaterials/Metal/ST_Metal_Rebar_Ruin_01.emat"
REBAR_ASSET_PATH = Path(__file__).parents[1] / "resources" / "rebar_material.blend"


def _ebt_resource_name(mat) -> str:
    """Read the GUID from either the RNA property or the raw ``ebt`` ID group."""
    direct = mat.get("ebt_resource_name") or getattr(mat, "ebt_resource_name", "")
    if direct:
        return str(direct)
    group = mat.get("ebt")
    if group is not None:
        try:
            return str(group.get("resource_name", ""))
        except Exception:
            return ""
    return ""


def resolve_rebar_material() -> tuple[object | None, str]:
    """Resolve the real EBT rebar material, returning ``(material, source)``.

    Order: existing name -> matching EBT resource id -> append the bundled
    ``rebar_material.blend``.  There is deliberately **no grey fallback**: an
    unavailable asset must be visible to the user as an unset material.
    """
    mat = bpy.data.materials.get(REBAR_MATERIAL_NAME)
    if mat is not None:
        return mat, "existing"

    for candidate in bpy.data.materials:
        if _ebt_resource_name(candidate) == REBAR_RESOURCE_NAME:
            return candidate, "resource"

    path = Path(REBAR_ASSET_PATH)
    if path.is_file():
        try:
            with bpy.data.libraries.load(str(path), link=False) as (src, dst):
                wanted = [n for n in src.materials if n == REBAR_MATERIAL_NAME]
                dst.materials = wanted or (list(src.materials)[:1] if src.materials else [])
            if dst.materials and dst.materials[0] is not None:
                mat = dst.materials[0]
                mat.use_fake_user = True
                return mat, "appended"
        except Exception as exc:
            print(f"[FullDestruc_Tool] Rebar material append failed: {exc}")
    return None, ""


def sync_rebar_material(mat) -> bool:
    """Optionally refresh an appended material from a live Enfusion Workbench.

    The import is lazy on purpose: EnfusionBlenderTools may not be installed at
    all, and ``update_blender_material`` requires a running Workbench NetAPI
    connection.  Failure is non-fatal — the appended material stays usable.
    """
    if mat is None:
        return False
    try:
        from EnfusionBlenderTools.core.materials import material_io
    except Exception:
        return False
    try:
        material_io.update_blender_material(mat)
        return True
    except Exception:
        return False


def ensure_rebar_material() -> tuple[object | None, str]:
    """Resolve, then try the optional Workbench sync (never blocks generation)."""
    mat, source = resolve_rebar_material()
    if mat is not None and source == "appended":
        sync_rebar_material(mat)
    return mat, source


def ensure_collection(entry: RebarEntry, scene=None):
    """Get or create the entry's collection, linked directly under the Scene."""
    name = bounded_name(entry.collection or collection_name(entry.sid))
    entry.collection = name
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        sc = _scene_of(scene)
        if sc is not None and col.name not in sc.collection.children:
            try:
                sc.collection.children.link(col)
            except Exception:
                pass
    return col


def move_to_collection(obj, entry: RebarEntry, scene=None):
    """Put ``obj`` in the generator collection and unlink it from the others."""
    col = ensure_collection(entry, scene)
    if col is None or obj is None:
        return col
    if col.name not in {c.name for c in obj.users_collection}:
        col.objects.link(obj)
    for other in list(obj.users_collection):
        if other.name != col.name:
            other.objects.unlink(obj)
    return col


# ---------------------------------------------------------------------------
# Delete guard + tag maintenance
# ---------------------------------------------------------------------------


def tagged_faces_for_entry(entry: RebarEntry) -> list[int]:
    host = resolve_host(entry)
    if host is None or getattr(host, "data", None) is None:
        return []
    return rg.tagged_face_indices(host.data, entry.group)


def delete_guard(sid, scene=None) -> tuple[bool, str]:
    """``(can_delete, reason)`` — refusing while tagged faces still remain."""
    entry = find_entry(sid, scene)
    if entry is None:
        return True, ""
    remaining = tagged_faces_for_entry(entry)
    if remaining:
        return (
            False,
            f"ลบไม่ได้ — ยังมี {len(remaining)} face ที่ถูก tag อยู่ ให้กด Clear Faces ก่อน",
        )
    return True, ""


def clear_tags_for_entry(entry: RebarEntry) -> int:
    host = resolve_host(entry)
    if host is None or getattr(host, "data", None) is None:
        return 0
    faces = rg.tagged_face_indices(host.data, entry.group)
    return rg.untag_faces(host.data, faces)


def mark_stale_for_host(host_obj, scene=None) -> list[str]:
    """Flag every generator on ``host_obj`` after a re-cut.  No automatic rebuild."""
    if host_obj is None:
        return []
    stale: list[str] = []
    host_name = getattr(host_obj, "name", "")
    entries = load_registry(scene)
    changed = False
    for entry in entries:
        if entry.host == host_name and not entry.stale:
            entry.stale = True
            stale.append(entry.sid)
            changed = True
    if changed:
        save_registry(entries, scene)
    return stale


def set_stale(sid, stale: bool, scene=None) -> bool:
    entry = find_entry(sid, scene)
    if entry is None:
        return False
    entry.stale = bool(stale)
    return update_entry(entry, scene)


def is_stale(sid, scene=None) -> bool:
    entry = find_entry(sid, scene)
    return bool(entry and entry.stale)


# ---------------------------------------------------------------------------
# Load-time aggregation (explicit, never a handler)
# ---------------------------------------------------------------------------


@dataclass
class GeneratorState:
    entry: RebarEntry
    obj: object | None = None
    host: object | None = None
    valid: bool = True

    @property
    def sid(self) -> str:
        return self.entry.sid

    @property
    def label(self) -> str:
        return self.entry.display_label


def load_all_generators(scene=None) -> dict[str, GeneratorState]:
    """Resolve every registry entry against the current blend file.

    Missing object or host does not discard the entry: it is surfaced as
    ``valid=False`` so the UI can offer a Re-apply / Bake / delete repair path.
    """
    states: dict[str, GeneratorState] = {}
    for entry in load_registry(scene):
        obj = resolve_object(entry)
        host = resolve_host(entry)
        valid = True
        if host is None or getattr(host, "data", None) is None:
            valid = False
        if obj is None and entry.baked:
            valid = False
        if obj is not None and object_sid(obj) != entry.sid and entry.baked:
            # Name matched but the sid did not: treat as unowned, never adopt.
            valid = False
        states[entry.sid] = GeneratorState(entry=entry, obj=obj, host=host, valid=valid)
    return states


def prune_missing(scene=None) -> list[str]:
    """Drop only entries whose generated object is gone and whose tags are gone."""
    removed: list[str] = []
    entries = load_registry(scene)
    kept: list[RebarEntry] = []
    for entry in entries:
        if not entry.baked or resolve_object(entry) is not None or tagged_faces_for_entry(entry):
            kept.append(entry)
        else:
            removed.append(entry.sid)
    if removed:
        save_registry(kept, scene)
    return removed
