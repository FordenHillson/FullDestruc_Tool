"""UI registration (N-Panel + Floating Menu + Custom Radial Wheel + Cutter Gear Gizmo + PySide6 Window)."""

from . import cutter_gizmo, debug_window, floating_menu, panel_n, qt_bridge, wheel_menu


def register():
    floating_menu.register()
    wheel_menu.register()
    cutter_gizmo.register()
    panel_n.register()
    qt_bridge.register()
    debug_window.register()


def unregister():
    debug_window.unregister()
    qt_bridge.unregister()
    panel_n.unregister()
    cutter_gizmo.unregister()
    wheel_menu.unregister()
    floating_menu.unregister()
