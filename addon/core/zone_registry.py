"""Scene-level zone registry for FullDestruc_Tool.

A zone has a stable ``key`` (what cutters/pieces/session store) and a mutable
``token`` (what the user sees and types, e.g. ``ID-A#B``).  A–Z are built-in
zones: they can be renamed but never deleted, so old files always have a
fallback.  Custom zones get a generated ``z_NNNN`` key.

This module deliberately has no UI dependencies.  It is imported by the
core engine, operators and all zone pickers so there is one source of truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

import bpy

REGISTRY_PROP = "fdt_zone_registry_json"
BUILTIN_ZONES: tuple[str, ...] = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))
MAX_TOKEN_LEN = 40
CUSTOM_KEY_PREFIX = "z_"

# Palette shared with apple_bite.ZONE_COLORS (kept here so this module can
# resolve colors without importing the engine).
BASE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "A": (1.00, 0.50, 0.12, 1.0),  # Orange
    "B": (0.20, 0.65, 1.00, 1.0),  # Sky Blue
    "C": (0.25, 0.88, 0.42, 1.0),  # Green
    "D": (0.76, 0.36, 1.00, 1.0),  # Purple
    "E": (1.00, 0.86, 0.20, 1.0),  # Yellow
    "F": (0.18, 0.90, 0.88, 1.0),  # Cyan
    "G": (1.00, 0.36, 0.68, 1.0),  # Pink
    "H": (1.00, 0.26, 0.26, 1.0),  # Red
}

_KIND_PREFIX_RE = re.compile(r"\b(ID|VIS)-([A-Za-z])", re.IGNORECASE)
_BASE_HASH_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])#")
_FIRST_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class ZoneDef:
    key: str
    token: str
    kind: str
    base: str
    color: tuple[float, float, float, float]
    builtin: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "token": self.token,
            "kind": self.kind,
            "base": self.base,
            "color": list(self.color),
            "builtin": bool(self.builtin),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneDef | None":
        try:
            key = str(data["key"]).strip()
            token = str(data["token"]).strip()
            if not key or not token:
                return None
            base = str(data.get("base", "A")).upper()[:1]
            if base not in BUILTIN_ZONES:
                base = "A"
            kind = str(data.get("kind", "ID")).upper()
            if kind not in {"ID", "VIS"}:
                kind = "ID"
            raw_color = data.get("color") or list(BASE_COLORS[base])
            color = tuple(float(raw_color[i]) for i in range(4))
            return cls(key=key, token=token[:MAX_TOKEN_LEN], kind=kind, base=base, color=color, builtin=bool(data.get("builtin", False)))
        except Exception:
            return None


def base_color(letter: str) -> tuple[float, float, float, float]:
    """Return the palette color for one base zone letter, wrapping I-Z onto A-H."""
    z = (str(letter or "A").upper()[:1] or "A")
    if z in BASE_COLORS:
        return BASE_COLORS[z]
    idx = (ord(z) - ord("A")) % len(BASE_COLORS)
    return list(BASE_COLORS.values())[idx]


def default_zones() -> list[ZoneDef]:
    return [
        ZoneDef(key=letter, token=letter, kind="ID", base=letter, color=base_color(letter), builtin=True)
        for letter in BUILTIN_ZONES
    ]


def normalize_key(value) -> str:
    """Return a usable registry key; unknown/custom values pass through unchanged."""
    key = str(value or "").strip()
    return key or "A"


def normalize_token(value) -> str:
    return str(value or "").strip()[:MAX_TOKEN_LEN]


def _structured_base(value) -> str | None:
    """Extract a base letter only from a structured token, never from arbitrary text."""
    token = str(value or "")
    match = _KIND_PREFIX_RE.search(token)
    if match:
        return match.group(2).upper()
    match = _BASE_HASH_RE.search(token)
    return match.group(1).upper() if match else None


def parse_token_hint(value) -> tuple[str, str]:
    """Auto-fill ``(kind, base)`` from a typed token; user selection stays authoritative."""
    token = str(value or "")
    match = _KIND_PREFIX_RE.search(token)
    kind = match.group(1).upper() if match else "ID"
    base = _structured_base(token)
    if base is None:
        first = _FIRST_LETTER_RE.search(token)
        base = first.group(0).upper() if first else "A"
    return kind, base


def _scene_of(scene=None) -> bpy.types.Scene | None:
    return scene if scene is not None else getattr(bpy.context, "scene", None)


def load_zones(scene=None) -> list[ZoneDef]:
    """Load the Scene registry, restoring missing built-ins and applying renames."""
    sc = _scene_of(scene)
    defaults = {z.key: z for z in default_zones()}
    saved: dict[str, ZoneDef] = {}
    custom: list[ZoneDef] = []
    raw = sc.get(REGISTRY_PROP) if sc is not None else None
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    zdef = ZoneDef.from_dict(item)
                    if zdef is None:
                        continue
                    if zdef.builtin:
                        if zdef.key in defaults:
                            saved[zdef.key] = zdef
                    else:
                        custom.append(zdef)
        except Exception:
            pass

    # Built-ins always come first; saved entries may rename their token/color.
    zones: list[ZoneDef] = []
    seen_tokens: set[str] = set()
    for letter in BUILTIN_ZONES:
        zdef = saved.get(letter, defaults[letter])
        token = zdef.token if zdef.token.casefold() not in seen_tokens else defaults[letter].token
        if token.casefold() in seen_tokens:
            token = letter
        if token != zdef.token:
            zdef = replace(zdef, token=token)
        seen_tokens.add(token.casefold())
        zones.append(zdef)

    seen_keys = {z.key for z in zones}
    for zdef in custom:
        if zdef.key in seen_keys or zdef.token.casefold() in seen_tokens:
            continue
        seen_keys.add(zdef.key)
        seen_tokens.add(zdef.token.casefold())
        zones.append(zdef)
    return zones


def save_zones(zones: list[ZoneDef], scene=None) -> None:
    sc = _scene_of(scene)
    if sc is None:
        return
    sc[REGISTRY_PROP] = json.dumps([z.to_dict() for z in zones], ensure_ascii=False)
    tag_redraw()


def tag_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        screen = getattr(win, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()


def find_zone(key, scene=None) -> ZoneDef | None:
    key = normalize_key(key)
    for zdef in load_zones(scene):
        if zdef.key == key:
            return zdef
    if len(key) == 1 and key in BUILTIN_ZONES:
        return next((z for z in default_zones() if z.key == key), None)
    return None


def find_zone_by_token(token, scene=None) -> ZoneDef | None:
    value = normalize_token(token).casefold()
    if not value:
        return None
    return next((z for z in load_zones(scene) if z.token.casefold() == value), None)


def resolve_zone(key, scene=None) -> ZoneDef:
    """Resolve a stored key; legacy tokens fall back to their base, then to A."""
    found = find_zone(key, scene)
    if found is not None:
        return found
    zones = load_zones(scene)
    legacy = find_zone_by_token(key, scene)
    if legacy is not None:
        return legacy
    # Only structured tokens carry a reliable base letter; arbitrary text falls back to A.
    base = _structured_base(key)
    if base is not None:
        for zdef in zones:
            if zdef.key == base:
                return zdef
    return zones[0]


def zone_token(key, scene=None) -> str:
    return resolve_zone(key, scene).token


def zone_color(key, scene=None) -> tuple[float, float, float, float]:
    return resolve_zone(key, scene).color


def zone_base(key, scene=None) -> str:
    return resolve_zone(key, scene).base


def zone_keys(scene=None) -> list[str]:
    return [z.key for z in load_zones(scene)]


def next_zone_key(current, scene=None) -> str:
    """Cycle to the next zone in registry order (A-Z then custom zones)."""
    keys = zone_keys(scene)
    cur = normalize_key(current)
    idx = keys.index(cur) if cur in keys else -1
    return keys[(idx + 1) % len(keys)]


def iter_zone_items(scene=None) -> list[tuple[str, str, str]]:
    """EnumProperty items: identifier = stable key, label = editable token."""
    return [
        (z.key, z.token, f"Zone {z.token} (base {z.base}, {z.kind})")
        for z in load_zones(scene)
    ]


def validate_token(token, scene=None, exclude_key: str | None = None) -> str | None:
    """Return an error message, or None when the token may be used."""
    value = normalize_token(token)
    if not value:
        return "Zone token ยังว่างอยู่"
    if len(str(token or "").strip()) > MAX_TOKEN_LEN:
        return f"Zone token ยาวเกิน {MAX_TOKEN_LEN} ตัวอักษร"
    for zdef in load_zones(scene):
        if exclude_key is not None and zdef.key == exclude_key:
            continue
        if zdef.token.casefold() == value.casefold():
            return f"Zone token '{value}' ถูกใช้โดย {zdef.key} แล้ว"
    return None


def next_custom_key(zones: list[ZoneDef] | None = None) -> str:
    used = {z.key for z in (zones or load_zones())}
    num = 1
    while f"{CUSTOM_KEY_PREFIX}{num:04d}" in used:
        num += 1
    return f"{CUSTOM_KEY_PREFIX}{num:04d}"


def create_zone(token, base: str = "A", kind: str = "ID", color=None, scene=None) -> tuple[bool, str, str]:
    error = validate_token(token, scene)
    if error:
        return False, error, ""
    value = normalize_token(token)
    base = str(base or "A").upper()[:1]
    if base not in BUILTIN_ZONES:
        base = "A"
    kind = str(kind or "ID").upper()
    if kind not in {"ID", "VIS"}:
        kind = "ID"
    zones = load_zones(scene)
    key = next_custom_key(zones)
    rgba = tuple(float(color[i]) for i in range(4)) if color is not None else base_color(base)
    zones.append(ZoneDef(key=key, token=value, kind=kind, base=base, color=rgba, builtin=False))
    save_zones(zones, scene)
    return True, f"สร้าง Zone '{value}' ({key}) แล้ว", key


def update_zone(key, token=None, base=None, kind=None, color=None, scene=None) -> tuple[bool, str]:
    zones = load_zones(scene)
    key = normalize_key(key)
    target = next((z for z in zones if z.key == key), None)
    if target is None:
        return False, f"ไม่พบ Zone key '{key}'"
    value = normalize_token(token) if token is not None else target.token
    error = validate_token(token if token is not None else value, scene, exclude_key=key)
    if error:
        return False, error
    new_base = str(base or target.base).upper()[:1] if base is not None else target.base
    if new_base not in BUILTIN_ZONES:
        new_base = target.base
    new_kind = str(kind or target.kind).upper() if kind is not None else target.kind
    if new_kind not in {"ID", "VIS"}:
        new_kind = target.kind
    new_color = tuple(float(color[i]) for i in range(4)) if color is not None else target.color
    zones[zones.index(target)] = ZoneDef(
        key=target.key, token=value, kind=new_kind, base=new_base, color=new_color, builtin=target.builtin
    )
    save_zones(zones, scene)
    return True, f"อัปเดต Zone '{value}' แล้ว"


def zone_usage(key, scene=None) -> tuple[int, int]:
    """Return ``(cutter_count, piece_count)`` currently referencing a zone key."""
    key = normalize_key(key)
    cutters = 0
    pieces = 0
    for obj in bpy.data.objects:
        raw_cutter = obj.get("fdt_cutter_zone")
        if raw_cutter is not None and resolve_zone(raw_cutter, scene).key == key:
            cutters += 1
        raw_piece = obj.get("fdt_piece_zone")
        if raw_piece is not None and resolve_zone(raw_piece, scene).key == key:
            pieces += 1
    from .session import load_session

    session = load_session(_scene_of(scene))
    if session is not None:
        named = {obj.name for obj in bpy.data.objects}
        cutters += sum(
            1 for c in session.cutters if c.name not in named and resolve_zone(c.zone, scene).key == key
        )
    return cutters, pieces


def delete_zone(key, scene=None) -> tuple[bool, str]:
    zones = load_zones(scene)
    key = normalize_key(key)
    target = next((z for z in zones if z.key == key), None)
    if target is None:
        return False, f"ไม่พบ Zone key '{key}'"
    if target.builtin:
        return False, "Zone A-Z เป็น fallback ของไฟล์เก่า ลบไม่ได้"
    nc, np = zone_usage(key, scene)
    if nc or np:
        return False, f"ลบไม่ได้ — มี cutter {nc} อัน และ piece {np} ชิ้นใช้ Zone '{target.token}' อยู่"
    zones.remove(target)
    save_zones(zones, scene)
    return True, f"ลบ Zone '{target.token}' แล้ว"
