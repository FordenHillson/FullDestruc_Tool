"""Clean session state machine for Apple Bite v2.1 (multi-object, re-editable, muteable cutters)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import bpy

SESSION_SCENE_PROP = "fdt_applebite_session_json"
SESSION_COL_META_PROP = "fdt_session_meta_json"
OBJ_SESSION_ID_PROP = "fdt_session_id"


@dataclass
class CutterEntry:
    name: str
    zone: str = "A"
    muted: bool = False
    rough: bool = False


@dataclass
class AppleBiteSession:
    session_id: str
    root_col_name: str
    vault_col_name: str
    cutters_col_name: str
    results_col_name: str
    pieces_col_name: str
    final_col_name: str = ""
    target_names: list[str] = field(default_factory=list)
    cutters: list[CutterEntry] = field(default_factory=list)
    reedit_hidden_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "root_col_name": self.root_col_name,
            "vault_col_name": self.vault_col_name,
            "cutters_col_name": self.cutters_col_name,
            "results_col_name": self.results_col_name,
            "pieces_col_name": self.pieces_col_name,
            "final_col_name": self.final_col_name,
            "target_names": list(self.target_names),
            "cutters": [
                {"name": c.name, "zone": c.zone, "muted": bool(c.muted), "rough": bool(c.rough)}
                for c in self.cutters
            ],
            "reedit_hidden_names": list(self.reedit_hidden_names),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppleBiteSession":
        cutters = [
            CutterEntry(
                name=item["name"],
                zone=item.get("zone", "A"),
                muted=bool(item.get("muted", False)),
                rough=bool(item.get("rough", False)),
            )
            for item in data.get("cutters", [])
            if isinstance(item, dict) and "name" in item
        ]
        sid = data["session_id"]
        return cls(
            session_id=sid,
            root_col_name=data["root_col_name"],
            vault_col_name=data["vault_col_name"],
            cutters_col_name=data["cutters_col_name"],
            results_col_name=data["results_col_name"],
            pieces_col_name=data["pieces_col_name"],
            final_col_name=data.get("final_col_name") or f"FDT_Final_{sid}",
            target_names=list(data.get("target_names", [])),
            cutters=cutters,
            reedit_hidden_names=list(data.get("reedit_hidden_names", [])),
        )

    def get_collections(self) -> dict[str, bpy.types.Collection | None]:
        cols = bpy.data.collections
        return {
            "root": cols.get(self.root_col_name),
            "vault": cols.get(self.vault_col_name),
            "cutters": cols.get(self.cutters_col_name),
            "results": cols.get(self.results_col_name),
            "pieces": cols.get(self.pieces_col_name),
            "final": cols.get(self.final_col_name) if self.final_col_name else None,
        }

    def get_target_objects(self) -> list[bpy.types.Object]:
        return [
            bpy.data.objects[name]
            for name in self.target_names
            if name in bpy.data.objects
        ]

    def get_valid_cutters(self, include_muted: bool = False) -> list[tuple[bpy.types.Object, str]]:
        valid: list[tuple[bpy.types.Object, str]] = []
        for entry in self.cutters:
            obj = bpy.data.objects.get(entry.name)
            if obj is not None and obj.type == "MESH":
                zone = str(obj.get("fdt_cutter_zone", entry.zone) or entry.zone)
                entry.zone = zone
                if include_muted or not entry.muted:
                    valid.append((obj, zone))
        return valid

    def prune_missing(self) -> None:
        self.target_names = [n for n in self.target_names if n in bpy.data.objects]
        self.cutters = [c for c in self.cutters if c.name in bpy.data.objects]
        self.reedit_hidden_names = [n for n in self.reedit_hidden_names if n in bpy.data.objects]

    def is_valid(self) -> bool:
        self.prune_missing()
        return bool(self.target_names) and (self.vault_col_name in bpy.data.collections)


# In-memory cache: (scene_key, raw_json_str) -> parsed dict (immutable blueprint)
# Eliminates 6-10x redundant json.loads() per viewport frame / depsgraph tick.
_SESSION_JSON_CACHE: dict[int, tuple[str, dict]] = {}
_READONLY_SESSION_CACHE: dict[int, tuple[str, int, AppleBiteSession]] = {}


def _scene_cache_key(scene: bpy.types.Scene) -> int:
    try:
        return int(scene.as_pointer())
    except Exception:
        return id(scene)


def invalidate_session_cache(scene: bpy.types.Scene | None = None) -> None:
    if scene is None:
        _SESSION_JSON_CACHE.clear()
        _READONLY_SESSION_CACHE.clear()
    else:
        key = _scene_cache_key(scene)
        _SESSION_JSON_CACHE.pop(key, None)
        _READONLY_SESSION_CACHE.pop(key, None)


def save_session(scene: bpy.types.Scene, session: AppleBiteSession | None, persist_meta: bool = True) -> None:
    """Persist active session JSON on the Scene and optionally on its root collection."""
    invalidate_session_cache(scene)
    if session is None:
        if SESSION_SCENE_PROP in scene:
            del scene[SESSION_SCENE_PROP]
    else:
        data_dict = session.to_dict()
        payload = json.dumps(data_dict)
        scene[SESSION_SCENE_PROP] = payload
        _SESSION_JSON_CACHE[_scene_cache_key(scene)] = (payload, data_dict)
        if persist_meta:
            root_col = bpy.data.collections.get(session.root_col_name)
            if root_col is not None:
                root_col[SESSION_COL_META_PROP] = payload
    try:
        from ..props import sync_cutter_layer_items

        sync_cutter_layer_items(scene, session=session)
    except Exception:
        pass


def _parse_session_json_cached(scene: bpy.types.Scene, raw: str) -> dict | None:
    skey = _scene_cache_key(scene)
    cached = _SESSION_JSON_CACHE.get(skey)
    if cached is not None and cached[0] == raw:
        return cached[1]
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        _SESSION_JSON_CACHE[skey] = (raw, data)
        return data
    except Exception:
        _SESSION_JSON_CACHE.pop(skey, None)
        return None


def load_session(scene: bpy.types.Scene | None) -> AppleBiteSession | None:
    """Load active session from Scene, auto-clearing if stale/corrupted."""
    if scene is None:
        return None
    raw = scene.get(SESSION_SCENE_PROP)
    if not raw:
        return None
    data = _parse_session_json_cached(scene, raw)
    if data is None:
        save_session(scene, None)
        return None
    try:
        session = AppleBiteSession.from_dict(data)
    except Exception:
        save_session(scene, None)
        return None
    if not session.is_valid():
        save_session(scene, None)
        return None
    return session


def load_session_readonly(scene: bpy.types.Scene | None) -> AppleBiteSession | None:
    """Ultra-fast read-only session accessor for hot paths (viewport draw / depsgraph).

    Reuses the constructed AppleBiteSession when the raw JSON and Blender object count
    have not changed, avoiding both json.loads() and O(N) CutterEntry list rebuilds.
    Callers MUST NOT mutate the returned object.
    """
    if scene is None:
        return None
    raw = scene.get(SESSION_SCENE_PROP)
    if not raw:
        return None
    skey = _scene_cache_key(scene)
    obj_count = len(bpy.data.objects)
    cached = _READONLY_SESSION_CACHE.get(skey)
    if cached is not None and cached[0] == raw and cached[1] == obj_count:
        sess = cached[2]
        if sess.vault_col_name in bpy.data.collections:
            return sess
    sess = load_session(scene)
    if sess is None:
        _READONLY_SESSION_CACHE.pop(skey, None)
        return None
    _READONLY_SESSION_CACHE[skey] = (raw, obj_count, sess)
    return sess


def load_stored_session_by_id(session_id: str) -> AppleBiteSession | None:
    """Load a previously saved session from FDT_Bite_<id>, auto-recovering from collections if metadata drifted."""
    root_col = bpy.data.collections.get(f"FDT_Bite_{session_id}")
    if root_col is None:
        return None

    vault_name = f"FDT_Vault_{session_id}"
    cutters_name = f"FDT_Cutters_{session_id}"
    results_name = f"FDT_Results_{session_id}"
    pieces_name = f"FDT_Pieces_{session_id}"
    final_name = f"FDT_Final_{session_id}"

    sess: AppleBiteSession | None = None
    raw = root_col.get(SESSION_COL_META_PROP)
    if raw:
        try:
            sess = AppleBiteSession.from_dict(json.loads(raw))
        except Exception:
            sess = None

    if sess is None:
        sess = AppleBiteSession(
            session_id=session_id,
            root_col_name=root_col.name,
            vault_col_name=vault_name,
            cutters_col_name=cutters_name,
            results_col_name=results_name,
            pieces_col_name=pieces_name,
            final_col_name=final_name,
            target_names=[],
            cutters=[],
        )

    # Self-heal target_names from FDT_Vault_<id> if missing/renamed
    sess.prune_missing()
    vault_col = bpy.data.collections.get(sess.vault_col_name)
    if vault_col is None:
        return None
    if not sess.target_names:
        sess.target_names = [o.name for o in vault_col.objects if o.type == "MESH"]

    # Self-heal cutters from FDT_Cutters_<id> if missing
    cutters_col = bpy.data.collections.get(sess.cutters_col_name)
    if cutters_col is not None and not sess.cutters:
        for obj in cutters_col.objects:
            if obj.type == "MESH":
                sess.cutters.append(
                    CutterEntry(
                        name=obj.name,
                        zone=str(obj.get("fdt_cutter_zone", "A") or "A"),
                        muted=False,
                        rough=bool(obj.get("fdt_cutter_rough", False)),
                    )
                )

    return sess if sess.is_valid() else None


def find_session_id_for_object(obj: bpy.types.Object | None) -> str | None:
    """Return session_id if obj belongs to any existing/baked Apple Bite session."""
    if obj is None:
        return None
    sid = obj.get(OBJ_SESSION_ID_PROP)
    if sid and load_stored_session_by_id(str(sid)) is not None:
        return str(sid)
    for col in obj.users_collection:
        for prefix in ("FDT_Final_", "FDT_Bite_", "FDT_Results_", "FDT_Pieces_", "FDT_Cutters_", "FDT_Vault_"):
            if col.name.startswith(prefix):
                cand = col.name[len(prefix):]
                if load_stored_session_by_id(cand) is not None:
                    return cand
    return None


def has_active_session(scene: bpy.types.Scene | None) -> bool:
    return load_session(scene) is not None


def load_all_stored_sessions(scene: bpy.types.Scene | None = None) -> dict[str, AppleBiteSession]:
    """Return a mapping of session_id -> AppleBiteSession for all valid sessions in the blend file."""
    sessions: dict[str, AppleBiteSession] = {}
    prefix = "FDT_Bite_"
    for col in list(bpy.data.collections):
        if not col.name.startswith(prefix):
            continue
        sid = col.name[len(prefix):]
        if not sid:
            continue
        sess = load_stored_session_by_id(sid)
        if sess is not None:
            sessions[sess.session_id] = sess

    active = load_session(scene)
    if active is not None and active.session_id not in sessions:
        sessions[active.session_id] = active
    return sessions


def collect_related_session_family(
    scene: bpy.types.Scene | None,
    start_session_id: str,
) -> tuple[list[str], list[str]]:
    """Trace all ancestor (root) and descendant (Cut Further) sessions related to start_session_id.

    Returns (root_ids, all_family_ids) in deterministic order.
    """
    all_sessions = load_all_stored_sessions(scene)
    if start_session_id not in all_sessions:
        return [], []

    parents: dict[str, set[str]] = {sid: set() for sid in all_sessions}
    children: dict[str, set[str]] = {sid: set() for sid in all_sessions}

    for sid, sess in all_sessions.items():
        candidate_objs = list(sess.get_target_objects())
        vault_col = sess.get_collections().get("vault")
        if vault_col is not None:
            for obj in vault_col.objects:
                if obj not in candidate_objs:
                    candidate_objs.append(obj)

        for obj in candidate_objs:
            parent_sid = obj.get(OBJ_SESSION_ID_PROP)
            if parent_sid:
                psid = str(parent_sid)
                if psid in all_sessions and psid != sid:
                    parents[sid].add(psid)
                    children[psid].add(sid)

    # Walk UP from start_session_id to find all root sessions (sessions with no parent session)
    visited_up: set[str] = {start_session_id}
    queue: list[str] = [start_session_id]
    root_set: set[str] = set()
    while queue:
        cur = queue.pop(0)
        if not parents[cur]:
            root_set.add(cur)
        else:
            for p in parents[cur]:
                if p not in visited_up:
                    visited_up.add(p)
                    queue.append(p)

    # Fallback in case of an unexpected cycle
    if not root_set:
        root_set.add(start_session_id)

    # Walk DOWN from all roots to collect every descendant & sibling Cut Further branch
    family_set: set[str] = set(root_set)
    down_queue: list[str] = sorted(root_set)
    while down_queue:
        cur = down_queue.pop(0)
        for ch in sorted(children[cur]):
            if ch not in family_set:
                family_set.add(ch)
                down_queue.append(ch)

    return sorted(root_set), sorted(family_set)
