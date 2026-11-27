"""Custom colored flat icon registry for FullDestruc_Tool (N-Panel + PySide6)."""

from __future__ import annotations

import os
from pathlib import Path

from ..core import zone_registry

try:
    import bpy.utils.previews as _bpy_previews
except Exception:
    _bpy_previews = None


_ICONS_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"
_PREVIEW_COLLECTIONS: dict[str, object] = {}

# Must match ZONE_DOT_COLORS in scripts/generate_icons.py
_ZONE_SWATCH_RGB: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("zone_c0", (245 / 255.0, 130 / 255.0, 32 / 255.0)),
    ("zone_c1", (56 / 255.0, 176 / 255.0, 255 / 255.0)),
    ("zone_c2", (46 / 255.0, 212 / 255.0, 122 / 255.0)),
    ("zone_c3", (244 / 255.0, 72 / 255.0, 92 / 255.0)),
    ("zone_c4", (176 / 255.0, 98 / 255.0, 250 / 255.0)),
    ("zone_c5", (245 / 255.0, 208 / 255.0, 48 / 255.0)),
    ("zone_c6", (34 / 255.0, 211 / 255.0, 208 / 255.0)),
    ("zone_c7", (244 / 255.0, 114 / 255.0, 182 / 255.0)),
)


def icons_dir() -> Path:
    """Return absolute path to ``addon/resources/icons``."""
    return _ICONS_DIR


def get_icon_path(name: str) -> str:
    """Return absolute PNG file path for ``name`` (without .png), or ``''`` if missing."""
    path = _ICONS_DIR / f"{name}.png"
    return str(path) if path.is_file() else ""


def available_icon_names() -> tuple[str, ...]:
    """Return sorted icon names discovered on disk."""
    if not _ICONS_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in _ICONS_DIR.glob("*.png")))


def get_icon_id(name: str) -> int:
    """Return Blender preview ``icon_id`` for ``name``, or ``0`` if unavailable."""
    pcoll = _PREVIEW_COLLECTIONS.get("main")
    if pcoll is None:
        return 0
    item = pcoll.get(name) if hasattr(pcoll, "get") else None
    if item is None:
        return 0
    return int(getattr(item, "icon_id", 0) or 0)


def icon_kw(name: str, fallback: str = "NONE") -> dict[str, object]:
    """Return ``{'icon_value': id}`` when loaded, else ``{'icon': fallback}``.

    Safe to unpack into any ``UILayout.operator(...)``, ``label(...)``,
    ``prop(...)``, or ``popover(...)`` call across GUI and headless Blender.
    """
    iid = get_icon_id(name)
    if iid > 0:
        return {"icon_value": iid}
    return {"icon": fallback}


def closest_zone_dot_name(rgb: tuple[float, float, float]) -> str:
    """Return the zone-dot icon name closest in RGB space to ``rgb``."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    best_name = _ZONE_SWATCH_RGB[0][0]
    best_dist = float("inf")
    for name, (sr, sg, sb) in _ZONE_SWATCH_RGB:
        dist = (r - sr) ** 2 + (g - sg) ** 2 + (b - sb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def get_zone_dot_icon_kw(zone_key: str, scene=None, fallback: str = "COLOR") -> dict[str, object]:
    """Return ``icon_kw`` for the colored dot matching ``zone_key``'s registered color."""
    try:
        zdef = zone_registry.resolve_zone(zone_key, scene)
        col = zdef.color
        dot_name = closest_zone_dot_name((col[0], col[1], col[2]))
    except Exception:
        dot_name = "zone_c0"
    return icon_kw(dot_name, fallback=fallback)


def register() -> None:
    unregister()
    if _bpy_previews is None or not _ICONS_DIR.is_dir():
        return
    try:
        pcoll = _bpy_previews.new()
    except Exception:
        return
    for png in sorted(_ICONS_DIR.glob("*.png")):
        try:
            pcoll.load(png.stem, os.fspath(png), "IMAGE")
        except Exception:
            pass
    _PREVIEW_COLLECTIONS["main"] = pcoll


def unregister() -> None:
    if _bpy_previews is not None:
        for pcoll in list(_PREVIEW_COLLECTIONS.values()):
            try:
                _bpy_previews.remove(pcoll)
            except Exception:
                pass
    _PREVIEW_COLLECTIONS.clear()
