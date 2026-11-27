"""FullDestruc_Tool — Fast boolean cutting & building zone manager for Blender 5.2+."""

from __future__ import annotations

bl_info = {
    "name": "FullDestruc Tool",
    "author": "FullDestruc Team",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N-Panel) > FullDestruc | Alt+Shift+Q | PySide6 Window",
    "description": "Fast multi-object Apple Bite boolean cutting, protected original vault, and building zone workflow",
    "category": "Object",
}

import importlib
import sys

from . import ops, preferences, props, ui


_RELOAD_ORDER = (
    "core.zone_shield",
    "core.zone_registry",
    "core.vault",
    "core.session",
    "core.geonodes",
    "core.perf_score",
    "core.profiler",
    "core.rust_bridge",
    "core.gpu_imposter",
    "core.gpu_preview",
    "core.live_sync",
    "core.draw_engine",
    "core.cap_fill",
    "core.apple_bite",
    "core.rebar_spline",
    "core.rebar_generate",
    "core.rebar_debris",
    "core.rebar_state",
    "core.rebar_overlay",
    "core",
    "preferences",
    "props",
    "ops.material_picker",
    "ops.apple_bite_ops",
    "ops.zone_ops",
    "ops.rebar_ops",
    "ops",
    "ui.cutter_gizmo",
    "ui.wheel_menu",
    "ui.floating_menu",
    "ui.panel_rebar",
    "ui.panel_n",
    "ui.qt_window",
    "ui.qt_bridge",
    "ui.debug_window",
    "ui.draw_overlay",
    "ui",
)


def _reload_submodules() -> None:
    prefix = f"{__name__}."
    order_map = {f"{prefix}{rel}": idx for idx, rel in enumerate(_RELOAD_ORDER)}
    to_reload = sorted(
        (
            (order_map.get(name, 999), name, mod)
            for name, mod in sys.modules.items()
            if name.startswith(prefix) and mod is not None
        ),
        key=lambda item: (item[0], item[1]),
    )
    for _rank, name, mod in to_reload:
        try:
            importlib.reload(mod)
        except Exception as exc:
            print(f"[FullDestruc_Tool] Reload failed for {name}: {exc}")


def register():
    _reload_submodules()
    from .core import live_sync

    preferences.register()
    props.register()
    ops.register()
    ui.register()
    live_sync.register()


def unregister():
    from .core import live_sync

    live_sync.unregister()
    ui.unregister()
    ops.unregister()
    props.unregister()
    preferences.unregister()


if __name__ == "__main__":
    register()
