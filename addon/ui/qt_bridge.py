"""Non-blocking PySide6 <-> Blender 5.2 bridge via bpy.app.timers (no modal lock)."""

from __future__ import annotations

import bpy

PYSIDE_AVAILABLE = False
try:
    from PySide6 import QtWidgets  # type: ignore

    PYSIDE_AVAILABLE = True
except Exception:
    QtWidgets = None  # type: ignore

_qt_window = None
_timer_running = False


def _ensure_qapp():
    if not PYSIDE_AVAILABLE:
        return None
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _qt_event_pump():
    global _timer_running, _qt_window
    if not PYSIDE_AVAILABLE or _qt_window is None or not _qt_window.isVisible():
        _timer_running = False
        return None
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()
    try:
        _qt_window.refresh_from_blender()
    except Exception:
        pass
    return 0.03


def open_detached_window() -> tuple[bool, str]:
    """Show or focus the PySide6 floating tool window safely."""
    global _qt_window, _timer_running
    if not PYSIDE_AVAILABLE:
        return False, "PySide6 is not available in this Blender Python; use N-Panel or Floating Menu."

    _ensure_qapp()
    from .qt_window import FullDestrucWindow

    if _qt_window is None:
        _qt_window = FullDestrucWindow()

    _qt_window.refresh_from_blender()
    _qt_window.show()
    _qt_window.raise_()
    _qt_window.activateWindow()

    if not bpy.app.timers.is_registered(_qt_event_pump):
        _timer_running = True
        bpy.app.timers.register(_qt_event_pump, first_interval=0.03, persistent=False)

    return True, "Opened FullDestruc PySide6 Window."


def close_detached_window() -> None:
    global _qt_window, _timer_running
    _timer_running = False
    if _qt_window is not None:
        try:
            _qt_window.close()
        except Exception:
            pass
        _qt_window = None


class FDT_OT_open_qt_window(bpy.types.Operator):
    bl_idname = "fdt.open_qt_window"
    bl_label = "Open Python Window"
    bl_description = "Open the detached PySide6 floating window (can be moved to a 2nd monitor)"

    def execute(self, context):
        ok, msg = open_detached_window()
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


CLASSES = (FDT_OT_open_qt_window,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    close_detached_window()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
