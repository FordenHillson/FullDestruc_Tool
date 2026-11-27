"""UI registration (Custom Colored Icons + Viewport Dock + N-Panel + Radial Wheel + Gizmo + Qt)."""

from . import cutter_gizmo, debug_window, floating_menu, icons, panel_n, qt_bridge, viewport_hud, wheel_menu


def register():
    icons.register()
    floating_menu.register()
    wheel_menu.register()
    cutter_gizmo.register()
    viewport_hud.register()
    panel_n.register()
    qt_bridge.register()
    debug_window.register()


def unregister():
    debug_window.unregister()
    qt_bridge.unregister()
    panel_n.unregister()
    viewport_hud.unregister()
    cutter_gizmo.unregister()
    wheel_menu.unregister()
    floating_menu.unregister()
    icons.unregister()
