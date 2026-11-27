"""Minimal dark-themed PySide6 companion window (Apple Bite + Rebar, Sticky Emerald Footer)."""

from __future__ import annotations

import json
from contextlib import contextmanager

import bpy
from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from ..core import apple_bite, rebar_state, zone_registry
from ..core.session import SESSION_SCENE_PROP, load_session, load_session_readonly
from ..ops import rebar_ops
from . import icons


def _qicon(name: str) -> QtGui.QIcon:
    path = icons.get_icon_path(name)
    return QtGui.QIcon(path) if path else QtGui.QIcon()


@contextmanager
def blender_view3d_override():
    """Provide a valid VIEW_3D context override when called from Qt / bpy.app.timers."""
    wm = bpy.context.window_manager
    target_win = None
    target_area = None
    target_region = None
    if wm is not None:
        for win in wm.windows:
            screen = win.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    target_win = win
                    target_area = area
                    for reg in area.regions:
                        if reg.type == "WINDOW":
                            target_region = reg
                            break
                    break
            if target_area is not None:
                break

    if target_win is not None and target_area is not None and target_region is not None:
        with bpy.context.temp_override(
            window=target_win,
            screen=target_win.screen,
            area=target_area,
            region=target_region,
        ):
            yield target_area
    else:
        yield None


def _tag_all_view3d_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        if win.screen is None:
            continue
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


DARK_STYLESHEET = """
QWidget {
    background-color: #16181e;
    color: #e6eaf2;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QFrame#Card {
    background-color: #1e212a;
    border: 1px solid #2b303d;
    border-radius: 8px;
}
QFrame#StickyFooter {
    background-color: #1a1d25;
    border-top: 1px solid #2d3342;
}
QPushButton {
    background-color: #272c38;
    border: 1px solid #353c4c;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6eaf2;
}
QPushButton:hover {
    background-color: #323949;
    border-color: #4a546b;
}
QPushButton:disabled {
    background-color: #1e2129;
    border-color: #282c36;
    color: #636b7a;
}
QPushButton:checked, QPushButton[active="true"] {
    background-color: #2c3a4f;
    border-color: #38bdf8;
    color: #f8fafc;
    font-weight: bold;
}
QPushButton#TabBtn {
    background-color: #1e212a;
    border: 1px solid #2b303d;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    color: #9aa4b6;
}
QPushButton#TabBtn:checked {
    background-color: #2b3140;
    border-color: #525f7a;
    color: #ffffff;
}
QPushButton#ApplyHeroBtn {
    background-color: #1da65c;
    border: 1px solid #30d07a;
    border-radius: 8px;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    padding: 10px 14px;
}
QPushButton#ApplyHeroBtn:hover {
    background-color: #24bf6b;
    border-color: #5ce89c;
}
QPushButton#ApplyHeroBtn:disabled {
    background-color: #22352c;
    border-color: #2c463a;
    color: #708c7d;
}
QPushButton#AmberHeroBtn {
    background-color: #c97a12;
    border: 1px solid #f59e0b;
    border-radius: 7px;
    color: #ffffff;
    font-weight: bold;
    padding: 8px 12px;
}
QPushButton#AmberHeroBtn:hover {
    background-color: #df8a1a;
}
QPushButton#CyanHeroBtn {
    background-color: #1b6d96;
    border: 1px solid #38bdf8;
    border-radius: 7px;
    color: #ffffff;
    font-weight: bold;
    padding: 8px 12px;
}
QPushButton#CyanHeroBtn:hover {
    background-color: #2282b3;
}
QPushButton#DangerBtn {
    background-color: #5a2226;
    border: 1px solid #a0383e;
    color: #fca5a5;
}
QPushButton#DangerBtn:hover {
    background-color: #782b30;
    border-color: #ef4444;
    color: #ffffff;
}
QPushButton#DrawerToggle {
    background-color: transparent;
    border: none;
    text-align: left;
    color: #aeb8ca;
    font-weight: bold;
    padding: 4px 2px;
}
QPushButton#DrawerToggle:hover {
    color: #ffffff;
}
QPushButton#ZoneBtn[active="true"] {
    background-color: #c97a12;
    border-color: #f59e0b;
    color: #ffffff;
    font-weight: bold;
}
QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #212530;
    border: 1px solid #343b4a;
    border-radius: 5px;
    padding: 4px 8px;
    color: #eef2f8;
}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
    border-color: #4b556b;
}
QLabel#StatusPill {
    color: #cbd5e1;
    font-size: 11px;
    padding: 4px 8px;
    background-color: #1c2029;
    border: 1px solid #2c3240;
    border-radius: 6px;
}
QLabel#SummaryLabel {
    color: #a6b2c6;
    font-size: 11px;
}
QScrollBar:vertical {
    background: #16181e;
    width: 7px;
    margin: 2px 0;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #3a4254;
    min-height: 24px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #535f78;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class LayerRowWidget(QtWidgets.QFrame):
    """Single draggable cutter layer card with grip, zone color, move arrows, and quick toggles."""

    def __init__(self, cutter_name: str, parent=None):
        super().__init__(parent)
        self.cutter_name = cutter_name
        self._press_pos_local: QtCore.QPointF | None = None
        self._press_pos_parent: QtCore.QPointF | None = None
        self._is_dragging = False

        self.setObjectName("LayerRow")
        self.setStyleSheet(
            """
            QFrame#LayerRow {
                background-color: #232732;
                border: 1px solid #323846;
                border-radius: 5px;
            }
            QFrame#LayerRow:hover {
                background-color: #2a2f3d;
                border: 1px solid #4d566b;
            }
            QFrame#LayerRow[selected="true"] {
                background-color: #303747;
                border: 1px solid #f59e0b;
            }
            """
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._press_pos_local = event.position()
            self._press_pos_parent = self.mapToParent(event.position().toPoint())
            self._is_dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._press_pos_local is not None and (event.buttons() & QtCore.Qt.LeftButton):
            cur_parent = self.mapToParent(event.position().toPoint())
            if not self._is_dragging:
                delta = cur_parent - self._press_pos_parent
                if abs(delta.y()) >= 4 or abs(delta.x()) >= 4:
                    self._is_dragging = True
                    parent = self.parent()
                    if isinstance(parent, CutterLayerListWidget):
                        parent.begin_row_drag(self, self._press_pos_local.y(), cur_parent.y())
            if self._is_dragging:
                parent = self.parent()
                if isinstance(parent, CutterLayerListWidget):
                    parent.update_row_drag(cur_parent.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._press_pos_local is not None:
            was_drag = self._is_dragging
            self._press_pos_local = None
            self._press_pos_parent = None
            self._is_dragging = False
            parent = self.parent()
            if isinstance(parent, CutterLayerListWidget):
                if was_drag:
                    parent.end_row_drag()
                else:
                    parent.on_row_clicked(self.cutter_name)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CutterLayerListWidget(QtWidgets.QFrame):
    """Animated Photoshop-style Cutter Layer list with translucent ghost drag & smooth row sliding."""

    orderChanged = QtCore.Signal(list)
    rowClicked = QtCore.Signal(str)

    ROW_H = 34
    ROW_GAP = 4
    PAD = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[LayerRowWidget] = []
        self._anims: dict[LayerRowWidget, QtCore.QPropertyAnimation] = {}
        self._dragging_row: LayerRowWidget | None = None
        self._drag_grab_y: float = 0.0
        self._ghost_label: QtWidgets.QLabel | None = None
        self._placeholder: QtWidgets.QFrame | None = None
        self._target_index: int = -1

        self.setObjectName("LayerListBox")
        self.setStyleSheet(
            """
            QFrame#LayerListBox {
                background-color: #181b22;
                border: 1px solid #2c313f;
                border-radius: 6px;
            }
            """
        )
        self.setMinimumHeight(44)

    @property
    def is_dragging(self) -> bool:
        return self._dragging_row is not None

    def _slot_y(self, idx: int) -> int:
        return self.PAD + idx * (self.ROW_H + self.ROW_GAP)

    def clear_rows(self) -> None:
        self._cleanup_drag_visuals()
        for anim in list(self._anims.values()):
            anim.stop()
        self._anims.clear()
        for row in self._rows:
            row.hide()
            row.deleteLater()
        self._rows.clear()
        self.setFixedHeight(44)

    def set_rows(self, rows: list[LayerRowWidget]) -> None:
        self.clear_rows()
        self._rows = list(rows)
        total_h = max(44, self.PAD * 2 + len(self._rows) * self.ROW_H + max(0, len(self._rows) - 1) * self.ROW_GAP)
        self.setFixedHeight(total_h)
        w = max(self.width() - self.PAD * 2, 240)
        for idx, row in enumerate(self._rows):
            row.setParent(self)
            row.setGeometry(self.PAD, self._slot_y(idx), w, self.ROW_H)
            row.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = max(self.width() - self.PAD * 2, 120)
        for idx, row in enumerate(self._rows):
            if row is self._dragging_row:
                continue
            row.setFixedWidth(w)
            row.move(self.PAD, self._slot_y(idx))
        if self._placeholder is not None and self._target_index >= 0:
            self._placeholder.setGeometry(self.PAD, self._slot_y(self._target_index), w, self.ROW_H)

    def _animate_row_to(self, row: LayerRowWidget, target_y: int) -> None:
        old = self._anims.get(row)
        if old is not None:
            old.stop()
        anim = QtCore.QPropertyAnimation(row, b"pos", self)
        anim.setDuration(140)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setStartValue(row.pos())
        anim.setEndValue(QtCore.QPoint(self.PAD, target_y))
        self._anims[row] = anim
        anim.start()

    def begin_row_drag(self, row: LayerRowWidget, grab_y: float, cursor_y: float) -> None:
        if row not in self._rows:
            return
        self._dragging_row = row
        self._drag_grab_y = grab_y
        self._target_index = self._rows.index(row)
        w = max(self.width() - self.PAD * 2, 120)

        self._placeholder = QtWidgets.QFrame(self)
        self._placeholder.setStyleSheet(
            "QFrame { background-color: rgba(245, 158, 11, 28); border: 1.5px dashed #f59e0b; border-radius: 5px; }"
        )
        self._placeholder.setGeometry(self.PAD, self._slot_y(self._target_index), w, self.ROW_H)
        self._placeholder.show()
        self._placeholder.lower()

        pixmap = row.grab()
        self._ghost_label = QtWidgets.QLabel(self)
        self._ghost_label.setPixmap(pixmap)
        self._ghost_label.setStyleSheet(
            "border: 1.5px solid #f59e0b; border-radius: 5px; background-color: rgba(40, 46, 58, 225);"
        )
        effect = QtWidgets.QGraphicsOpacityEffect(self._ghost_label)
        effect.setOpacity(0.82)
        self._ghost_label.setGraphicsEffect(effect)
        self._ghost_label.setFixedSize(w, self.ROW_H)
        ghost_y = max(0, min(self.height() - self.ROW_H, int(round(cursor_y - self._drag_grab_y))))
        self._ghost_label.move(self.PAD, ghost_y)
        self._ghost_label.show()
        self._ghost_label.raise_()

        row.setVisible(False)
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def update_row_drag(self, cursor_y: float) -> None:
        if self._dragging_row is None or self._ghost_label is None:
            return
        ghost_y = max(0, min(self.height() - self.ROW_H, int(round(cursor_y - self._drag_grab_y))))
        self._ghost_label.move(self.PAD, ghost_y)

        center_y = ghost_y + self.ROW_H * 0.5
        step = self.ROW_H + self.ROW_GAP
        new_idx = max(0, min(len(self._rows) - 1, int((center_y - self.PAD) // step)))

        if new_idx != self._target_index:
            old_idx = self._rows.index(self._dragging_row)
            self._rows.pop(old_idx)
            self._rows.insert(new_idx, self._dragging_row)
            self._target_index = new_idx

            w = max(self.width() - self.PAD * 2, 120)
            if self._placeholder is not None:
                self._placeholder.setGeometry(self.PAD, self._slot_y(new_idx), w, self.ROW_H)
            for idx, r in enumerate(self._rows):
                if r is not self._dragging_row:
                    self._animate_row_to(r, self._slot_y(idx))

    def end_row_drag(self) -> None:
        if self._dragging_row is None:
            return
        row = self._dragging_row
        final_idx = self._rows.index(row)
        new_order = [r.cutter_name for r in self._rows]

        self._cleanup_drag_visuals()
        self._dragging_row = None
        self._target_index = -1
        self.setCursor(QtCore.Qt.ArrowCursor)

        row.move(self.PAD, self._slot_y(final_idx))
        row.setVisible(True)
        self.orderChanged.emit(new_order)

    def _cleanup_drag_visuals(self) -> None:
        if self._ghost_label is not None:
            self._ghost_label.hide()
            self._ghost_label.deleteLater()
            self._ghost_label = None
        if self._placeholder is not None:
            self._placeholder.hide()
            self._placeholder.deleteLater()
            self._placeholder = None

    def on_row_clicked(self, cutter_name: str) -> None:
        self.rowClicked.emit(cutter_name)


class FullDestrucWindow(QtWidgets.QWidget):
    """Minimal detached companion window synchronized live with Blender 5.2."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FullDestruc Tool")
        self.setMinimumWidth(380)
        self.resize(400, 560)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_STYLESHEET)

        self._zone_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._last_cutter_sig: tuple = ()
        self._last_mat_sig: tuple = ()
        self._last_rebar_mat_sig: tuple = ()
        self._active_tab: str = "APPLE_BITE"
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- SLIM TOP BAR (Tool Switcher + Collapsible Engine Settings) ---
        top_wrap = QtWidgets.QWidget()
        top_l = QtWidgets.QHBoxLayout(top_wrap)
        top_l.setContentsMargins(10, 8, 10, 6)
        top_l.setSpacing(6)

        self.btn_tab_ab = QtWidgets.QPushButton("Apple Bite")
        self.btn_tab_ab.setObjectName("TabBtn")
        self.btn_tab_ab.setIcon(_qicon("tool_apple_bite"))
        self.btn_tab_ab.setCheckable(True)
        self.btn_tab_ab.setChecked(True)
        self.btn_tab_ab.clicked.connect(lambda: self._switch_tab("APPLE_BITE"))

        self.btn_tab_rb = QtWidgets.QPushButton("Rebar")
        self.btn_tab_rb.setObjectName("TabBtn")
        self.btn_tab_rb.setIcon(_qicon("tool_rebar"))
        self.btn_tab_rb.setCheckable(True)
        self.btn_tab_rb.clicked.connect(lambda: self._switch_tab("REBAR"))

        self.btn_settings = QtWidgets.QPushButton("")
        self.btn_settings.setIcon(_qicon("settings_gear"))
        self.btn_settings.setFixedSize(32, 30)
        self.btn_settings.setCheckable(True)
        self.btn_settings.setToolTip("Engine, GPU Fast Drag & Viewport Dock Settings")
        self.btn_settings.toggled.connect(lambda checked: self.settings_drawer.setVisible(checked))

        top_l.addWidget(self.btn_tab_ab, 1)
        top_l.addWidget(self.btn_tab_rb, 1)
        top_l.addWidget(self.btn_settings)
        outer.addWidget(top_wrap)

        # Collapsible Engine/Settings Drawer (hidden by default!)
        self.settings_drawer = QtWidgets.QFrame()
        self.settings_drawer.setObjectName("Card")
        sd_l = QtWidgets.QVBoxLayout(self.settings_drawer)
        sd_l.setContentsMargins(10, 8, 10, 8)
        sd_l.setSpacing(6)

        eng_row = QtWidgets.QHBoxLayout()
        eng_row.addWidget(QtWidgets.QLabel("Engine:"))
        self.cmb_engine = QtWidgets.QComboBox()
        self.cmb_engine.addItem("Rust Engine (Fast & UV Safe)", "RUST")
        self.cmb_engine.addItem("Blender Exact", "EXACT")
        self.cmb_engine.addItem("Blender Float (Fast)", "FLOAT")
        self.cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        eng_row.addWidget(self.cmb_engine, 1)
        sd_l.addLayout(eng_row)

        self.chk_gpu_fast_drag = QtWidgets.QCheckBox("GPU Fast Drag (60+ FPS)")
        self.chk_gpu_fast_drag.setChecked(True)
        self.chk_gpu_fast_drag.toggled.connect(self._on_gpu_fast_drag_toggled)
        sd_l.addWidget(self.chk_gpu_fast_drag)

        self.chk_viewport_dock = QtWidgets.QCheckBox("Show 3D Viewport Action Dock")
        self.chk_viewport_dock.setChecked(True)
        self.chk_viewport_dock.toggled.connect(self._on_viewport_dock_toggled)
        sd_l.addWidget(self.chk_viewport_dock)

        self.settings_drawer.setVisible(False)
        sd_wrap = QtWidgets.QWidget()
        sd_wl = QtWidgets.QVBoxLayout(sd_wrap)
        sd_wl.setContentsMargins(10, 0, 10, 4)
        sd_wl.addWidget(self.settings_drawer)
        outer.addWidget(sd_wrap)

        # --- SCROLLABLE BODY (Apple Bite / Rebar pages) ---
        self.body_scroll = QtWidgets.QScrollArea()
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_apple_bite_page())
        self.stack.addWidget(self._build_rebar_page())
        self.body_scroll.setWidget(self.stack)
        outer.addWidget(self.body_scroll, 1)

        # --- STICKY BOTTOM ACTION FOOTER (ALWAYS VISIBLE!) ---
        self.sticky_footer = QtWidgets.QFrame()
        self.sticky_footer.setObjectName("StickyFooter")
        sf_l = QtWidgets.QVBoxLayout(self.sticky_footer)
        sf_l.setContentsMargins(10, 8, 10, 10)
        sf_l.setSpacing(6)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setObjectName("SummaryLabel")
        sf_l.addWidget(self.summary_label)

        # Giant Vivid Emerald APPLY / BAKE button
        self.btn_hero_commit = QtWidgets.QPushButton("APPLY")
        self.btn_hero_commit.setObjectName("ApplyHeroBtn")
        self.btn_hero_commit.setIcon(_qicon("apply_green"))
        self.btn_hero_commit.setIconSize(QtCore.QSize(18, 18))
        self.btn_hero_commit.setMinimumHeight(42)
        self.btn_hero_commit.clicked.connect(self._on_hero_commit_clicked)
        sf_l.addWidget(self.btn_hero_commit)

        self.footer_sec_row = QtWidgets.QWidget()
        fs_l = QtWidgets.QHBoxLayout(self.footer_sec_row)
        fs_l.setContentsMargins(0, 0, 0, 0)
        fs_l.setSpacing(6)
        self.btn_exit = QtWidgets.QPushButton("Pause")
        self.btn_exit.setIcon(_qicon("pause_slate"))
        self.btn_exit.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_exit))
        self.btn_cancel = QtWidgets.QPushButton("Cancel & Restore")
        self.btn_cancel.setObjectName("DangerBtn")
        self.btn_cancel.setIcon(_qicon("cancel_red"))
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        fs_l.addWidget(self.btn_exit)
        fs_l.addWidget(self.btn_cancel)
        sf_l.addWidget(self.footer_sec_row)

        outer.addWidget(self.sticky_footer)

    def _build_apple_bite_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(10, 4, 10, 10)
        root.setSpacing(8)

        # Compact status pill
        self.guide_label = QtWidgets.QLabel("Select a mesh in 3D View to start")
        self.guide_label.setObjectName("StatusPill")
        self.guide_label.setWordWrap(True)
        root.addWidget(self.guide_label)

        # --- IDLE CARD ---
        self.step1_widget = QtWidgets.QFrame()
        self.step1_widget.setObjectName("Card")
        s1_l = QtWidgets.QVBoxLayout(self.step1_widget)
        s1_l.setContentsMargins(10, 10, 10, 10)
        s1_l.setSpacing(6)

        pair_row = QtWidgets.QHBoxLayout()
        pair_row.setSpacing(6)
        self.btn_reedit_old = QtWidgets.QPushButton("Re-edit Cut")
        self.btn_reedit_old.setObjectName("CyanHeroBtn")
        self.btn_reedit_old.setIcon(_qicon("reedit_cyan"))
        self.btn_reedit_old.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_reedit))
        self.btn_reedit_old.setVisible(False)

        self.btn_smart_start = QtWidgets.QPushButton("Start Cut")
        self.btn_smart_start.setObjectName("AmberHeroBtn")
        self.btn_smart_start.setIcon(_qicon("start_orange"))
        self.btn_smart_start.setMinimumHeight(38)
        self.btn_smart_start.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_smart_start))

        pair_row.addWidget(self.btn_reedit_old, 1)
        pair_row.addWidget(self.btn_smart_start, 1)
        s1_l.addLayout(pair_row)

        self.btn_reset_original = QtWidgets.QPushButton("Reset to Original")
        self.btn_reset_original.setObjectName("DangerBtn")
        self.btn_reset_original.setIcon(_qicon("trash_red"))
        self.btn_reset_original.clicked.connect(self._on_reset_original_clicked)
        self.btn_reset_original.setVisible(False)
        s1_l.addWidget(self.btn_reset_original)
        root.addWidget(self.step1_widget)

        # --- ACTIVE SESSION CARD ---
        self.step2_widget = QtWidgets.QFrame()
        self.step2_widget.setObjectName("Card")
        s2_l = QtWidgets.QVBoxLayout(self.step2_widget)
        s2_l.setContentsMargins(10, 10, 10, 10)
        s2_l.setSpacing(7)

        # 1-Row Zone Strip + collapsible Quick Zones drawer
        z_row = QtWidgets.QHBoxLayout()
        z_row.setSpacing(5)
        self.zone_swatch = QtWidgets.QLabel()
        self.zone_swatch.setFixedSize(14, 14)
        self.zone_swatch.setStyleSheet("border-radius: 7px; background-color: #f59e0b;")

        self.cmb_zone = QtWidgets.QComboBox()
        self.cmb_zone.setEditable(True)
        self.cmb_zone.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        completer = self.cmb_zone.completer()
        if completer is not None:
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.cmb_zone.currentIndexChanged.connect(self._on_zone_combo_changed)

        self.btn_zone_drawer = QtWidgets.QPushButton("")
        self.btn_zone_drawer.setIcon(_qicon("zones_grid"))
        self.btn_zone_drawer.setFixedSize(30, 28)
        self.btn_zone_drawer.setCheckable(True)
        self.btn_zone_drawer.setToolTip("Quick Zones (A-H) & Manage Zone Registry")
        self.btn_zone_drawer.toggled.connect(lambda c: self.zone_drawer.setVisible(c))

        z_row.addWidget(self.zone_swatch)
        z_row.addWidget(self.cmb_zone, 1)
        z_row.addWidget(self.btn_zone_drawer)
        s2_l.addLayout(z_row)

        self.zone_drawer = QtWidgets.QWidget()
        zd_l = QtWidgets.QVBoxLayout(self.zone_drawer)
        zd_l.setContentsMargins(0, 2, 0, 2)
        zd_l.setSpacing(5)
        zone_grid = QtWidgets.QGridLayout()
        zone_grid.setSpacing(4)
        initial_zones = zone_registry.load_zones(getattr(bpy.context, "scene", None))[:8]
        for idx, zdef in enumerate(initial_zones):
            btn = QtWidgets.QPushButton(zdef.token)
            btn.setObjectName("ZoneBtn")
            btn.clicked.connect(lambda _chk=False, z=zdef.key: self._set_zone(z))
            self._zone_buttons[zdef.key] = btn
            zone_grid.addWidget(btn, idx // 4, idx % 4)
        zd_l.addLayout(zone_grid)

        zm_row = QtWidgets.QHBoxLayout()
        zm_row.setSpacing(4)
        btn_z_new = QtWidgets.QPushButton("New Zone")
        btn_z_new.setIcon(_qicon("plus_add"))
        btn_z_new.clicked.connect(
            lambda _c=False: self._run_invoke_op(bpy.ops.fdt.zone_manage, action="NEW", zone_key=self._active_zone_key())
        )
        btn_z_edit = QtWidgets.QPushButton("Edit")
        btn_z_edit.setIcon(_qicon("rename_pen"))
        btn_z_edit.clicked.connect(
            lambda _c=False: self._run_invoke_op(bpy.ops.fdt.zone_manage, action="RENAME", zone_key=self._active_zone_key())
        )
        btn_z_del = QtWidgets.QPushButton("Delete")
        btn_z_del.setIcon(_qicon("trash_red"))
        btn_z_del.clicked.connect(
            lambda _c=False: self._run_invoke_op(bpy.ops.fdt.zone_manage, action="DELETE", zone_key=self._active_zone_key())
        )
        zm_row.addWidget(btn_z_new)
        zm_row.addWidget(btn_z_edit)
        zm_row.addWidget(btn_z_del)
        zd_l.addLayout(zm_row)
        self.zone_drawer.setVisible(False)
        s2_l.addWidget(self.zone_drawer)

        # 2x2 Add Cutters Grid
        add_grid = QtWidgets.QGridLayout()
        add_grid.setSpacing(5)
        btn_box = QtWidgets.QPushButton("+ Box")
        btn_box.setIcon(_qicon("add_box"))
        btn_box.setMinimumHeight(32)
        btn_box.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_add_box))

        btn_draw = QtWidgets.QPushButton("Draw")
        btn_draw.setIcon(_qicon("draw_pen"))
        btn_draw.setMinimumHeight(32)
        btn_draw.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_draw_cutter))

        btn_click = QtWidgets.QPushButton("Click Place")
        btn_click.setIcon(_qicon("click_place"))
        btn_click.clicked.connect(lambda: self._run_invoke_op(bpy.ops.fdt.applebite_interactive))

        btn_mesh = QtWidgets.QPushButton("+ Mesh")
        btn_mesh.setIcon(_qicon("use_mesh"))
        btn_mesh.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_add_mesh))

        add_grid.addWidget(btn_box, 0, 0)
        add_grid.addWidget(btn_draw, 0, 1)
        add_grid.addWidget(btn_click, 1, 0)
        add_grid.addWidget(btn_mesh, 1, 1)
        s2_l.addLayout(add_grid)

        # Phase B controls (hidden when 0 valid cutters)
        self.phase_b_widget = QtWidgets.QWidget()
        pb_l = QtWidgets.QVBoxLayout(self.phase_b_widget)
        pb_l.setContentsMargins(0, 2, 0, 0)
        pb_l.setSpacing(6)

        row_view = QtWidgets.QHBoxLayout()
        row_view.setSpacing(4)
        self._view_mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        for mode_id, lbl in (("ALL", "All"), ("PIECES", "Piece"), ("MAIN", "Hole"), ("MAIN_WIRE", "Wire")):
            b = QtWidgets.QPushButton(lbl)
            b.setCheckable(True)
            b.setProperty("active", mode_id == "ALL")
            b.clicked.connect(lambda _=False, m=mode_id: self._on_preview_view_clicked(m))
            row_view.addWidget(b)
            self._view_mode_buttons[mode_id] = b
        pb_l.addLayout(row_view)

        quick_row = QtWidgets.QHBoxLayout()
        quick_row.setSpacing(6)
        self.chk_zone_grid = QtWidgets.QCheckBox("Zone Grid")
        self.chk_zone_grid.setChecked(True)
        self.chk_zone_grid.toggled.connect(self._on_zone_grid_toggled)
        self.chk_rough = QtWidgets.QCheckBox("Rough Edge")
        self.chk_rough.toggled.connect(self._on_rough_toggled)
        quick_row.addWidget(self.chk_zone_grid)
        quick_row.addWidget(self.chk_rough)
        quick_row.addStretch(1)
        pb_l.addLayout(quick_row)

        # Collapsible Rough Params
        self.rough_params_box = QtWidgets.QFrame()
        self.rough_params_box.setStyleSheet(
            "QFrame { background-color: #181b22; border: 1px solid #2e3444; border-radius: 6px; }"
            "QLabel { border: none; background: transparent; color: #cfd5df; font-size: 11px; }"
        )
        rp_layout = QtWidgets.QFormLayout(self.rough_params_box)
        rp_layout.setContentsMargins(8, 6, 8, 6)
        rp_layout.setSpacing(5)
        self.lbl_rough_target = QtWidgets.QLabel("Cutter: -")
        self.lbl_rough_target.setStyleSheet("color: #c4b5fd; font-weight: bold; font-size: 11px;")
        rp_layout.addRow(self.lbl_rough_target)

        inward_row = QtWidgets.QHBoxLayout()
        self.sld_rough_inward = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_rough_inward.setRange(0, 100)
        self.spn_rough_inward = QtWidgets.QDoubleSpinBox()
        self.spn_rough_inward.setRange(0.0, 100.0)
        self.spn_rough_inward.setSingleStep(5.0)
        self.spn_rough_inward.setDecimals(0)
        self.spn_rough_inward.setSuffix("%")
        self.spn_rough_inward.setFixedWidth(60)
        self.sld_rough_inward.valueChanged.connect(self._on_rough_inward_slider_changed)
        self.spn_rough_inward.valueChanged.connect(self._on_rough_inward_spin_changed)
        inward_row.addWidget(self.sld_rough_inward, 1)
        inward_row.addWidget(self.spn_rough_inward)
        rp_layout.addRow("Inward:", inward_row)

        jagged_row = QtWidgets.QHBoxLayout()
        self.sld_rough_jagged = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_rough_jagged.setRange(0, 100)
        self.sld_rough_jagged.setValue(45)
        self.spn_rough_jagged = QtWidgets.QDoubleSpinBox()
        self.spn_rough_jagged.setRange(0.0, 100.0)
        self.spn_rough_jagged.setSingleStep(5.0)
        self.spn_rough_jagged.setDecimals(0)
        self.spn_rough_jagged.setSuffix("%")
        self.spn_rough_jagged.setValue(45.0)
        self.spn_rough_jagged.setFixedWidth(60)
        self.sld_rough_jagged.valueChanged.connect(self._on_rough_jagged_slider_changed)
        self.spn_rough_jagged.valueChanged.connect(self._on_rough_jagged_spin_changed)
        jagged_row.addWidget(self.sld_rough_jagged, 1)
        jagged_row.addWidget(self.spn_rough_jagged)
        rp_layout.addRow("Jagged:", jagged_row)

        self.spn_rough_scale = QtWidgets.QDoubleSpinBox()
        self.spn_rough_scale.setRange(0.5, 10.0)
        self.spn_rough_scale.setSingleStep(0.25)
        self.spn_rough_scale.setValue(2.5)
        self.spn_rough_scale.valueChanged.connect(lambda v: self._on_rough_param_changed("noise_scale", float(v)))
        rp_layout.addRow("Scale:", self.spn_rough_scale)

        self.spn_rough_subdiv = QtWidgets.QSpinBox()
        self.spn_rough_subdiv.setRange(1, 3)
        self.spn_rough_subdiv.setValue(2)
        self.spn_rough_subdiv.valueChanged.connect(lambda v: self._on_rough_param_changed("detail", int(v)))
        rp_layout.addRow("Detail:", self.spn_rough_subdiv)

        self.spn_rough_seed = QtWidgets.QDoubleSpinBox()
        self.spn_rough_seed.setRange(-10000.0, 10000.0)
        self.spn_rough_seed.setSingleStep(0.5)
        self.spn_rough_seed.valueChanged.connect(lambda v: self._on_rough_param_changed("seed", float(v)))
        rp_layout.addRow("Seed:", self.spn_rough_seed)
        self.rough_params_box.setVisible(False)
        pb_l.addWidget(self.rough_params_box)

        # Collapsible Cutter Layers Drawer (collapsed by default!)
        self.btn_layers_toggle = QtWidgets.QPushButton("▸ Layers (0)")
        self.btn_layers_toggle.setObjectName("DrawerToggle")
        self.btn_layers_toggle.setCheckable(True)
        self.btn_layers_toggle.toggled.connect(self._on_layers_drawer_toggled)
        pb_l.addWidget(self.btn_layers_toggle)

        self.cutter_layer_list = CutterLayerListWidget()
        self.cutter_layer_list.orderChanged.connect(self._on_layers_reordered)
        self.cutter_layer_list.rowClicked.connect(self._on_layer_row_clicked)
        self.layer_scroll = QtWidgets.QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.layer_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.layer_scroll.setMinimumHeight(46)
        self.layer_scroll.setMaximumHeight(210)
        self.layer_scroll.setWidget(self.cutter_layer_list)
        self.layer_scroll.setVisible(False)
        pb_l.addWidget(self.layer_scroll)

        # Collapsible Output Options Drawer
        self.btn_output_toggle = QtWidgets.QPushButton("▸ Output Options")
        self.btn_output_toggle.setObjectName("DrawerToggle")
        self.btn_output_toggle.setCheckable(True)
        self.btn_output_toggle.toggled.connect(self._on_output_drawer_toggled)
        pb_l.addWidget(self.btn_output_toggle)

        self.output_drawer = QtWidgets.QWidget()
        od_l = QtWidgets.QVBoxLayout(self.output_drawer)
        od_l.setContentsMargins(6, 2, 0, 2)
        od_l.setSpacing(4)
        keep_row = QtWidgets.QHBoxLayout()
        self.chk_keep_main = QtWidgets.QCheckBox("Main Mesh")
        self.chk_keep_main.setChecked(True)
        self.chk_keep_main.toggled.connect(self._on_keep_main_toggled)
        self.chk_keep_pieces = QtWidgets.QCheckBox("Zone Pieces")
        self.chk_keep_pieces.setChecked(True)
        self.chk_keep_pieces.toggled.connect(self._on_keep_pieces_toggled)
        keep_row.addWidget(self.chk_keep_main)
        keep_row.addWidget(self.chk_keep_pieces)
        od_l.addLayout(keep_row)

        self.chk_auto_fill = QtWidgets.QCheckBox("Auto-Fill Cut Gaps")
        self.chk_auto_fill.setChecked(True)
        self.chk_auto_fill.toggled.connect(self._on_auto_fill_toggled)
        od_l.addWidget(self.chk_auto_fill)

        self.chk_line_backfaces = QtWidgets.QCheckBox("Close See-Through Backfaces")
        self.chk_line_backfaces.setChecked(True)
        self.chk_line_backfaces.toggled.connect(self._on_line_backfaces_toggled)
        od_l.addWidget(self.chk_line_backfaces)

        btn_ren_obj = QtWidgets.QPushButton("Rename Selected Objects")
        btn_ren_obj.setIcon(_qicon("rename_pen"))
        btn_ren_obj.clicked.connect(lambda: self._run_invoke_op(bpy.ops.fdt.rename_popup))
        od_l.addWidget(btn_ren_obj)
        self.output_drawer.setVisible(False)
        pb_l.addWidget(self.output_drawer)

        s2_l.addWidget(self.phase_b_widget)
        root.addWidget(self.step2_widget)

        # --- CUT FACE MATERIAL CARD (Compact 1-row + optional post-cut tools) ---
        self.cut_mat_box = QtWidgets.QFrame()
        self.cut_mat_box.setObjectName("Card")
        cm_l = QtWidgets.QVBoxLayout(self.cut_mat_box)
        cm_l.setContentsMargins(10, 8, 10, 8)
        cm_l.setSpacing(6)

        cm_row = QtWidgets.QHBoxLayout()
        cm_row.setSpacing(5)
        self.cmb_cut_mat = QtWidgets.QComboBox()
        self.cmb_cut_mat.currentIndexChanged.connect(self._on_cut_mat_changed)
        self.btn_pick_mat = QtWidgets.QPushButton("Pick")
        self.btn_pick_mat.setIcon(_qicon("eyedropper"))
        self.btn_pick_mat.setFixedWidth(72)
        self.btn_pick_mat.clicked.connect(self._on_pick_surface_material)
        cm_row.addWidget(self.cmb_cut_mat, 1)
        cm_row.addWidget(self.btn_pick_mat)
        cm_l.addLayout(cm_row)

        post_row = QtWidgets.QHBoxLayout()
        post_row.setSpacing(5)
        self.btn_assign_cut_mat = QtWidgets.QPushButton("Apply Mat")
        self.btn_assign_cut_mat.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_assign_cut_material))
        self.btn_assign_cut_mat.setVisible(False)
        self.btn_fill_cut_caps = QtWidgets.QPushButton("Fill Cap")
        self.btn_fill_cut_caps.setIcon(_qicon("fill_cap"))
        self.btn_fill_cut_caps.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_fill_cut_caps))
        self.btn_fill_cut_caps.setVisible(False)
        post_row.addWidget(self.btn_assign_cut_mat)
        post_row.addWidget(self.btn_fill_cut_caps)
        cm_l.addLayout(post_row)
        self.cut_mat_box.setVisible(False)
        root.addWidget(self.cut_mat_box)

        root.addStretch(1)
        return page

    def _build_rebar_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(10, 4, 10, 10)
        root.setSpacing(8)

        self.lbl_rebar_stale = QtWidgets.QLabel("")
        self.lbl_rebar_stale.setStyleSheet(
            "background-color: #422b0c; border: 1px solid #f59e0b; border-radius: 6px; "
            "color: #fde68a; padding: 6px 8px; font-weight: bold;"
        )
        self.lbl_rebar_stale.setVisible(False)
        root.addWidget(self.lbl_rebar_stale)

        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        c_l = QtWidgets.QVBoxLayout(card)
        c_l.setContentsMargins(10, 10, 10, 10)
        c_l.setSpacing(7)

        # Target Mat + Eyedropper
        m_row = QtWidgets.QHBoxLayout()
        m_row.setSpacing(5)
        self.cmb_rebar_mat = QtWidgets.QComboBox()
        self.cmb_rebar_mat.currentIndexChanged.connect(self._on_rebar_mat_changed)
        btn_rb_pick = QtWidgets.QPushButton("Pick")
        btn_rb_pick.setIcon(_qicon("eyedropper"))
        btn_rb_pick.setFixedWidth(72)
        btn_rb_pick.clicked.connect(lambda: self._run_invoke_op(bpy.ops.fdt.rebar_pick_material, slot="TARGET"))
        m_row.addWidget(self.cmb_rebar_mat, 1)
        m_row.addWidget(btn_rb_pick)
        c_l.addLayout(m_row)

        # Density + Bar Size
        p_row = QtWidgets.QHBoxLayout()
        p_row.setSpacing(5)
        self.cmb_rebar_density = QtWidgets.QComboBox()
        for key, lbl in (
            ("FINE", "Fine · 75mm"),
            ("STANDARD", "Std · 150mm"),
            ("COARSE", "Coarse · 250mm"),
            ("SPARSE", "Sparse · 400mm"),
            ("CUSTOM", "Custom"),
        ):
            self.cmb_rebar_density.addItem(lbl, key)
        self.cmb_rebar_density.currentIndexChanged.connect(self._on_rebar_density_changed)

        self.cmb_rebar_bar = QtWidgets.QComboBox()
        for key in ("RB6", "RB9", "DB10", "DB12", "DB16", "DB20", "DB25", "CUSTOM"):
            self.cmb_rebar_bar.addItem(key, key)
        self.cmb_rebar_bar.currentIndexChanged.connect(self._on_rebar_bar_changed)
        p_row.addWidget(self.cmb_rebar_density, 1)
        p_row.addWidget(self.cmb_rebar_bar, 1)
        c_l.addLayout(p_row)

        meta_row = QtWidgets.QHBoxLayout()
        self.lbl_rebar_est = QtWidgets.QLabel("~0 bars")
        self.lbl_rebar_est.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.chk_rebar_debris = QtWidgets.QCheckBox("Concrete Debris")
        self.chk_rebar_debris.toggled.connect(self._on_rebar_debris_toggled)
        meta_row.addWidget(self.lbl_rebar_est, 1)
        meta_row.addWidget(self.chk_rebar_debris)
        c_l.addLayout(meta_row)

        self.btn_rebar_create_extra = QtWidgets.QPushButton("Create Another Rebar")
        self.btn_rebar_create_extra.setIcon(_qicon("plus_add"))
        self.btn_rebar_create_extra.clicked.connect(lambda: self._run_op(bpy.ops.fdt.rebar_create))
        self.btn_rebar_create_extra.setVisible(False)
        c_l.addWidget(self.btn_rebar_create_extra)

        self.btn_rebar_rebuild = QtWidgets.QPushButton("Re-apply Stale Rebar")
        self.btn_rebar_rebuild.setObjectName("AmberHeroBtn")
        self.btn_rebar_rebuild.setIcon(_qicon("rebuild_amber"))
        self.btn_rebar_rebuild.clicked.connect(lambda: self._run_op(bpy.ops.fdt.rebar_rebuild))
        self.btn_rebar_rebuild.setVisible(False)
        c_l.addWidget(self.btn_rebar_rebuild)

        self.rebar_target_wrap = QtWidgets.QWidget()
        rt_l = QtWidgets.QVBoxLayout(self.rebar_target_wrap)
        rt_l.setContentsMargins(0, 4, 0, 0)
        rt_l.setSpacing(5)
        self.lbl_rebar_target = QtWidgets.QLabel("")
        self.lbl_rebar_target.setStyleSheet("color: #94a3b8; font-size: 11px;")
        rt_l.addWidget(self.lbl_rebar_target)
        q_row = QtWidgets.QHBoxLayout()
        q_row.setSpacing(4)
        self.btn_rb_edit = QtWidgets.QPushButton("Edit")
        self.btn_rb_edit.setIcon(_qicon("rename_pen"))
        self.btn_rb_edit.clicked.connect(self._on_rebar_edit_clicked)
        self.btn_rb_faces = QtWidgets.QPushButton("Faces")
        self.btn_rb_faces.clicked.connect(self._on_rebar_faces_clicked)
        self.btn_rb_clear = QtWidgets.QPushButton("Clear")
        self.btn_rb_clear.clicked.connect(self._on_rebar_clear_clicked)
        self.btn_rb_remove = QtWidgets.QPushButton("Remove")
        self.btn_rb_remove.setObjectName("DangerBtn")
        self.btn_rb_remove.setIcon(_qicon("trash_red"))
        self.btn_rb_remove.clicked.connect(lambda: self._run_op(bpy.ops.fdt.rebar_remove))
        q_row.addWidget(self.btn_rb_edit)
        q_row.addWidget(self.btn_rb_faces)
        q_row.addWidget(self.btn_rb_clear)
        q_row.addWidget(self.btn_rb_remove)
        rt_l.addLayout(q_row)
        self.rebar_target_wrap.setVisible(False)
        c_l.addWidget(self.rebar_target_wrap)

        root.addWidget(card)
        root.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Tab & Operator Helpers
    # ------------------------------------------------------------------

    def _switch_tab(self, tab: str) -> None:
        self._active_tab = tab
        self.btn_tab_ab.setChecked(tab == "APPLE_BITE")
        self.btn_tab_rb.setChecked(tab == "REBAR")
        self.stack.setCurrentIndex(0 if tab == "APPLE_BITE" else 1)
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and settings.ui_active_tool != tab:
                settings.ui_active_tool = tab
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _run_op(self, op_func, **kwargs) -> None:
        try:
            with blender_view3d_override():
                op_func("EXEC_DEFAULT", **kwargs)
        except Exception as exc:
            self.guide_label.setText(f"Notice: {exc}")
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _run_invoke_op(self, op_func, **kwargs) -> None:
        try:
            with blender_view3d_override():
                op_func("INVOKE_DEFAULT", **kwargs)
        except Exception as exc:
            self.guide_label.setText(f"Notice: {exc}")
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _on_hero_commit_clicked(self) -> None:
        if self._active_tab == "REBAR":
            with blender_view3d_override():
                rset = getattr(bpy.context.scene, "fdt_rebar_settings", None)
                active = rebar_state.find_entry(rset.rebar_active_sid) if (rset and rset.rebar_active_sid) else None
            if active is not None:
                self._run_op(bpy.ops.fdt.rebar_bake)
            else:
                self._run_op(bpy.ops.fdt.rebar_create)
        else:
            self._run_op(bpy.ops.fdt.applebite_apply)

    def _on_cancel_clicked(self) -> None:
        n_cutters = 0
        has_final = False
        with blender_view3d_override():
            session = load_session(bpy.context.scene)
            if session is not None:
                n_cutters = len(session.get_valid_cutters(include_muted=True))
                fcol = session.get_collections().get("final")
                has_final = fcol is not None and len(fcol.objects) > 0
        if n_cutters > 0:
            if has_final:
                msg = (
                    f"WARNING: You are re-editing an applied cut.\n\n"
                    f"Canceling will delete ALL {n_cutters} cutter(s) AND the baked Final cut, "
                    f"restoring the uncut original mesh.\n\n"
                    f"Tip: Click 'Pause' instead if you want to keep your cutters."
                )
            else:
                msg = f"Discard {n_cutters} cutter(s) and restore the original mesh?"
            ans = QtWidgets.QMessageBox.question(
                self,
                "Cancel & Restore",
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        self._run_op(bpy.ops.fdt.applebite_cancel)

    def _on_layers_drawer_toggled(self, checked: bool) -> None:
        self.layer_scroll.setVisible(checked)
        txt = self.btn_layers_toggle.text()
        prefix = "▾" if checked else "▸"
        self.btn_layers_toggle.setText(prefix + txt[1:] if txt else f"{prefix} Layers")

    def _on_output_drawer_toggled(self, checked: bool) -> None:
        self.output_drawer.setVisible(checked)
        self.btn_output_toggle.setText("▾ Output Options" if checked else "▸ Output Options")

    def _active_zone_key(self) -> str:
        idx = self.cmb_zone.currentIndex()
        if idx >= 0:
            data = self.cmb_zone.itemData(idx)
            if data:
                return str(data)
        settings = getattr(bpy.context.scene, "fdt_settings", None)
        return str(getattr(settings, "active_zone", "A") or "A")

    def _set_zone(self, zone: str) -> None:
        self._run_op(bpy.ops.fdt.applebite_set_zone, zone=zone)

    def _on_zone_combo_changed(self, _index: int) -> None:
        key = self._active_zone_key()
        settings = getattr(bpy.context.scene, "fdt_settings", None)
        if key and (settings is None or str(getattr(settings, "active_zone", "A")) != key):
            self._set_zone(key)

    def _on_engine_changed(self, index: int) -> None:
        val = self.cmb_engine.itemData(index)
        if not val:
            return
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and settings.boolean_solver != val:
                settings.boolean_solver = val
        _tag_all_view3d_redraw()

    def _on_gpu_fast_drag_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(settings.gpu_fast_drag) != bool(checked):
                settings.gpu_fast_drag = bool(checked)
        _tag_all_view3d_redraw()

    def _on_viewport_dock_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(getattr(settings, "show_viewport_dock", True)) != bool(checked):
                settings.show_viewport_dock = bool(checked)
        _tag_all_view3d_redraw()

    def _on_rough_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(settings.enable_rough) != bool(checked):
                settings.enable_rough = bool(checked)
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    _ROUGH_SCENE_FALLBACK = {
        "inward_depth": "rough_inward_ratio",
        "jaggedness": "rough_jaggedness",
        "noise_scale": "rough_scale",
        "detail": "rough_subdiv",
        "seed": "rough_seed",
    }

    def _on_rough_param_changed(self, prop_name: str, value) -> None:
        with blender_view3d_override():
            ctx = bpy.context
            scene = ctx.scene
            session = load_session(scene)
            target_cutter, _is_rough = apple_bite.get_active_rough_cutter(ctx, session)
            cr = getattr(target_cutter, "fdt_rough", None) if target_cutter is not None else None
            if cr is not None and hasattr(cr, prop_name):
                cur = getattr(cr, prop_name)
                if isinstance(value, int) and not isinstance(value, bool):
                    if int(cur) != value:
                        setattr(cr, prop_name, value)
                elif abs(float(cur) - float(value)) > 1e-6:
                    setattr(cr, prop_name, float(value))
            else:
                settings = getattr(scene, "fdt_settings", None)
                scene_prop = self._ROUGH_SCENE_FALLBACK.get(prop_name, prop_name)
                if settings is not None and hasattr(settings, scene_prop):
                    cur = getattr(settings, scene_prop)
                    if isinstance(value, int) and not isinstance(value, bool):
                        if int(cur) != value:
                            setattr(settings, scene_prop, value)
                    elif abs(float(cur) - float(value)) > 1e-6:
                        setattr(settings, scene_prop, float(value))
        _tag_all_view3d_redraw()

    def _on_rough_inward_slider_changed(self, pct: int) -> None:
        if abs(self.spn_rough_inward.value() - float(pct)) > 1e-6:
            self.spn_rough_inward.blockSignals(True)
            self.spn_rough_inward.setValue(float(pct))
            self.spn_rough_inward.blockSignals(False)
        self._on_rough_param_changed("inward_depth", float(pct) / 100.0)

    def _on_rough_inward_spin_changed(self, pct: float) -> None:
        ipct = int(round(pct))
        if self.sld_rough_inward.value() != ipct:
            self.sld_rough_inward.blockSignals(True)
            self.sld_rough_inward.setValue(ipct)
            self.sld_rough_inward.blockSignals(False)
        self._on_rough_param_changed("inward_depth", float(pct) / 100.0)

    def _on_rough_jagged_slider_changed(self, pct: int) -> None:
        if abs(self.spn_rough_jagged.value() - float(pct)) > 1e-6:
            self.spn_rough_jagged.blockSignals(True)
            self.spn_rough_jagged.setValue(float(pct))
            self.spn_rough_jagged.blockSignals(False)
        self._on_rough_param_changed("jaggedness", float(pct) / 100.0)

    def _on_rough_jagged_spin_changed(self, pct: float) -> None:
        ipct = int(round(pct))
        if self.sld_rough_jagged.value() != ipct:
            self.sld_rough_jagged.blockSignals(True)
            self.sld_rough_jagged.setValue(ipct)
            self.sld_rough_jagged.blockSignals(False)
        self._on_rough_param_changed("jaggedness", float(pct) / 100.0)

    def _on_keep_main_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(settings.apply_keep_result) != bool(checked):
                settings.apply_keep_result = bool(checked)
        self.refresh_from_blender()

    def _on_keep_pieces_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(settings.apply_keep_pieces) != bool(checked):
                settings.apply_keep_pieces = bool(checked)
        self.refresh_from_blender()

    def _on_auto_fill_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(getattr(settings, "apply_auto_fill_caps", True)) != bool(checked):
                settings.apply_auto_fill_caps = bool(checked)
        _tag_all_view3d_redraw()

    def _on_line_backfaces_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(getattr(settings, "apply_line_backfaces", True)) != bool(checked):
                settings.apply_line_backfaces = bool(checked)
        _tag_all_view3d_redraw()

    def _on_zone_grid_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and bool(settings.show_zone_grid) != bool(checked):
                settings.show_zone_grid = bool(checked)
        _tag_all_view3d_redraw()

    def _on_preview_view_clicked(self, mode_id: str) -> None:
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None and getattr(settings, "preview_display_mode", "ALL") != mode_id:
                settings.preview_display_mode = mode_id
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _on_cut_mat_changed(self, index: int) -> None:
        if index < 0:
            return
        mat_name = self.cmb_cut_mat.itemData(index)
        with blender_view3d_override():
            settings = getattr(bpy.context.scene, "fdt_settings", None)
            if settings is not None:
                new_mat = bpy.data.materials.get(mat_name) if mat_name else None
                if settings.cut_material != new_mat:
                    settings.cut_material = new_mat
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _on_pick_surface_material(self) -> None:
        try:
            with blender_view3d_override():
                bpy.ops.fdt.pick_surface_material("INVOKE_DEFAULT")
            self.guide_label.setText("Eyedropper active: Click any surface in 3D View (ESC / RMB cancel)")
        except Exception as exc:
            self.guide_label.setText(f"Notice: {exc}")

    # --- Rebar callbacks ---

    def _on_rebar_mat_changed(self, index: int) -> None:
        if index < 0:
            return
        mat_name = self.cmb_rebar_mat.itemData(index)
        with blender_view3d_override():
            rset = getattr(bpy.context.scene, "fdt_rebar_settings", None)
            if rset is not None:
                new_mat = bpy.data.materials.get(mat_name) if mat_name else None
                if rset.target_material != new_mat:
                    rset.target_material = new_mat
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _on_rebar_density_changed(self, index: int) -> None:
        val = self.cmb_rebar_density.itemData(index)
        if not val:
            return
        with blender_view3d_override():
            rset = getattr(bpy.context.scene, "fdt_rebar_settings", None)
            if rset is not None and rset.density_preset != val:
                rset.density_preset = val
        _tag_all_view3d_redraw()

    def _on_rebar_bar_changed(self, index: int) -> None:
        val = self.cmb_rebar_bar.itemData(index)
        if not val:
            return
        with blender_view3d_override():
            rset = getattr(bpy.context.scene, "fdt_rebar_settings", None)
            if rset is not None and rset.bar_size != val:
                rset.bar_size = val
        _tag_all_view3d_redraw()

    def _on_rebar_debris_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            rset = getattr(bpy.context.scene, "fdt_rebar_settings", None)
            if rset is not None and bool(rset.enable_debris) != bool(checked):
                rset.enable_debris = bool(checked)
        _tag_all_view3d_redraw()

    def _rebar_target_sid(self) -> str:
        with blender_view3d_override():
            target, _hint = rebar_ops.resolve_remove_target(bpy.context)
            return target.sid if target is not None else ""

    def _on_rebar_edit_clicked(self) -> None:
        sid = self._rebar_target_sid()
        if sid:
            self._run_op(bpy.ops.fdt.rebar_reedit, sid=sid)

    def _on_rebar_faces_clicked(self) -> None:
        sid = self._rebar_target_sid()
        if sid:
            self._run_op(bpy.ops.fdt.rebar_select_faces, sid=sid)

    def _on_rebar_clear_clicked(self) -> None:
        sid = self._rebar_target_sid()
        if sid:
            self._run_op(bpy.ops.fdt.rebar_clear_faces, sid=sid)

    # --- Sync Helpers ---

    def _sync_cut_material_combo(self, settings) -> None:
        if settings is None:
            return
        cur_mat = settings.cut_material
        cur_name = cur_mat.name if cur_mat is not None else ""
        mat_names = sorted(m.name for m in bpy.data.materials if not m.name.startswith("Dots Stroke"))
        sig = (cur_name, tuple(mat_names))
        if sig == self._last_mat_sig:
            return
        self._last_mat_sig = sig

        self.cmb_cut_mat.blockSignals(True)
        self.cmb_cut_mat.clear()
        self.cmb_cut_mat.addItem("-- Cut Material (Unset) --", "")
        sel_idx = 0
        for idx, name in enumerate(mat_names, start=1):
            self.cmb_cut_mat.addItem(name, name)
            if name == cur_name:
                sel_idx = idx
        self.cmb_cut_mat.setCurrentIndex(sel_idx)
        self.cmb_cut_mat.blockSignals(False)

        if cur_mat is None:
            self.cmb_cut_mat.setStyleSheet("QComboBox { border: 1px solid #ef4444; }")
        else:
            self.cmb_cut_mat.setStyleSheet("")
        self.btn_assign_cut_mat.setEnabled(cur_mat is not None)

    def _sync_rebar_mat_combo(self, rset) -> None:
        if rset is None:
            return
        cur_mat = rset.target_material
        cur_name = cur_mat.name if cur_mat is not None else ""
        mat_names = sorted(m.name for m in bpy.data.materials if not m.name.startswith("Dots Stroke"))
        sig = (cur_name, tuple(mat_names))
        if sig == self._last_rebar_mat_sig:
            return
        self._last_rebar_mat_sig = sig

        self.cmb_rebar_mat.blockSignals(True)
        self.cmb_rebar_mat.clear()
        self.cmb_rebar_mat.addItem("-- Target Material --", "")
        sel_idx = 0
        for idx, name in enumerate(mat_names, start=1):
            self.cmb_rebar_mat.addItem(name, name)
            if name == cur_name:
                sel_idx = idx
        self.cmb_rebar_mat.setCurrentIndex(sel_idx)
        self.cmb_rebar_mat.blockSignals(False)

        if cur_mat is None:
            self.cmb_rebar_mat.setStyleSheet("QComboBox { border: 1px solid #ef4444; }")
        else:
            self.cmb_rebar_mat.setStyleSheet("")

    def _on_reset_original_clicked(self) -> None:
        sid = ""
        display_names = ""
        fam_count = 0
        with blender_view3d_override():
            context = bpy.context
            info = apple_bite.detect_step1_action(context)
            sid = info.get("session_id") or ""
            if sid:
                preview = apple_bite.get_reset_preview_info(context.scene, sid)
                orig_names = preview["original_names"]
                fam_count = len(preview["family_ids"])
                if orig_names:
                    display_names = ", ".join(orig_names[:3])
                    if len(orig_names) > 3:
                        display_names += f" +{len(orig_names) - 3}"
                else:
                    display_names = sid

        if not sid:
            self.guide_label.setText("Notice: No cut session found for selected object.")
            return

        extra = "\n(Includes all Cut Further & Final pieces)" if fam_count > 1 else ""
        msg = f"Delete all cuts and restore original?\n\nOriginal: {display_names}\nSessions: {fam_count}{extra}"
        ans = QtWidgets.QMessageBox.question(
            self,
            "Reset to Original",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if ans == QtWidgets.QMessageBox.Yes:
            self._run_op(bpy.ops.fdt.applebite_reset_original, session_id=sid)

    def _on_layers_reordered(self, ordered_names: list[str]) -> None:
        self._last_cutter_sig = ()
        self._run_op(bpy.ops.fdt.applebite_reorder_cutters, order_json=json.dumps(ordered_names))

    def _on_layer_row_clicked(self, cutter_name: str) -> None:
        if cutter_name:
            self._run_op(bpy.ops.fdt.applebite_select_cutter, cutter_name=cutter_name)

    def _rebuild_cutter_rows(self, session) -> None:
        if self.cutter_layer_list.is_dragging:
            return
        active_obj = bpy.context.active_object
        active_name = active_obj.name if active_obj is not None else ""
        sig = (
            active_name,
            tuple(
                (c.name, c.zone, zone_registry.zone_token(c.zone, bpy.context.scene), c.muted, c.rough)
                for c in session.cutters
            ),
        )
        if sig == self._last_cutter_sig:
            return
        self._last_cutter_sig = sig

        total = len(session.cutters)
        prefix = "▾" if self.btn_layers_toggle.isChecked() else "▸"
        self.btn_layers_toggle.setText(f"{prefix} Layers ({total})")
        if total == 0:
            self.cutter_layer_list.clear_rows()
            return

        rows: list[LayerRowWidget] = []
        for idx, entry in enumerate(session.cutters):
            row_w = LayerRowWidget(entry.name)
            row_w.setProperty("selected", entry.name == active_name)
            row_l = QtWidgets.QHBoxLayout(row_w)
            row_l.setContentsMargins(6, 2, 6, 2)
            row_l.setSpacing(4)

            grip = QtWidgets.QLabel(f"☰ #{idx + 1}")
            grip.setFixedWidth(34)
            grip.setStyleSheet("color: #8f9bb0; font-weight: bold; font-size: 11px; background: transparent;")

            r, g, b, _ = apple_bite.get_zone_color(entry.zone)
            rgb_hex = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

            btn_z = QtWidgets.QPushButton(zone_registry.zone_token(entry.zone, bpy.context.scene)[:10])
            btn_z.setFixedWidth(26)
            btn_z.setStyleSheet(f"background-color: {rgb_hex}; color: #111; font-weight: bold; padding: 2px;")
            btn_z.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_cycle_zone, cutter_name=n))

            lbl_name = QtWidgets.QLabel(entry.name)
            lbl_name.setStyleSheet("color: #e6eaf2; padding-left: 2px; background: transparent;")

            btn_up = QtWidgets.QPushButton("▲")
            btn_up.setFixedWidth(22)
            btn_up.setEnabled(idx > 0)
            btn_up.setStyleSheet("padding: 2px;")
            btn_up.clicked.connect(
                lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_move_cutter, cutter_name=n, direction="UP")
            )

            btn_dn = QtWidgets.QPushButton("▼")
            btn_dn.setFixedWidth(22)
            btn_dn.setEnabled(idx < total - 1)
            btn_dn.setStyleSheet("padding: 2px;")
            btn_dn.clicked.connect(
                lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_move_cutter, cutter_name=n, direction="DOWN")
            )

            btn_rough = QtWidgets.QPushButton("R" if entry.rough else "F")
            btn_rough.setFixedWidth(24)
            btn_rough.setStyleSheet(
                "background-color: #9333ea; color: #fff; font-weight: bold; padding: 2px;"
                if entry.rough
                else "padding: 2px;"
            )
            btn_rough.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_toggle_rough, cutter_name=n))

            btn_mute = QtWidgets.QPushButton("Off" if entry.muted else "On")
            btn_mute.setFixedWidth(30)
            btn_mute.setStyleSheet("padding: 2px;")
            btn_mute.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_toggle_mute, cutter_name=n))

            btn_del = QtWidgets.QPushButton("×")
            btn_del.setObjectName("DangerBtn")
            btn_del.setFixedWidth(22)
            btn_del.setStyleSheet("padding: 2px;")
            btn_del.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_delete_cutter, cutter_name=n))

            row_l.addWidget(grip)
            row_l.addWidget(btn_z)
            row_l.addWidget(lbl_name, 1)
            row_l.addWidget(btn_up)
            row_l.addWidget(btn_dn)
            row_l.addWidget(btn_rough)
            row_l.addWidget(btn_mute)
            row_l.addWidget(btn_del)
            rows.append(row_w)

        self.cutter_layer_list.set_rows(rows)
        scroll_h = min(self.cutter_layer_list.height() + 2, 210)
        self.layer_scroll.setFixedHeight(max(46, scroll_h))

    def _refresh_rebar_page(self, context, scene) -> None:
        rset = getattr(scene, "fdt_rebar_settings", None)
        if rset is None:
            self.sticky_footer.setVisible(False)
            return
        self._sync_rebar_mat_combo(rset)

        d_idx = self.cmb_rebar_density.findData(str(rset.density_preset))
        if d_idx >= 0 and self.cmb_rebar_density.currentIndex() != d_idx:
            self.cmb_rebar_density.blockSignals(True)
            self.cmb_rebar_density.setCurrentIndex(d_idx)
            self.cmb_rebar_density.blockSignals(False)

        b_idx = self.cmb_rebar_bar.findData(str(rset.bar_size))
        if b_idx >= 0 and self.cmb_rebar_bar.currentIndex() != b_idx:
            self.cmb_rebar_bar.blockSignals(True)
            self.cmb_rebar_bar.setCurrentIndex(b_idx)
            self.cmb_rebar_bar.blockSignals(False)

        if self.chk_rebar_debris.isChecked() != bool(rset.enable_debris):
            self.chk_rebar_debris.blockSignals(True)
            self.chk_rebar_debris.setChecked(bool(rset.enable_debris))
            self.chk_rebar_debris.blockSignals(False)

        states = rebar_state.load_all_generators(scene)
        stale_count = sum(1 for s in states.values() if s.entry.stale)
        self.lbl_rebar_stale.setVisible(stale_count > 0)
        if stale_count > 0:
            self.lbl_rebar_stale.setText(f"⚠ {stale_count} rebar set(s) stale — press Re-apply then Bake")

        active = rebar_state.find_entry(rset.rebar_active_sid, scene) if rset.rebar_active_sid else None
        obj = context.active_object
        has_mesh = bool(obj is not None and obj.type == "MESH")

        if active is not None:
            est = int(rset.rebar_estimate)
        elif has_mesh and rset.target_material is not None:
            faces = rebar_ops.resolve_faces(context, obj, rset)
            est = rebar_ops.estimate_for(obj.data, faces, rebar_ops.settings_to_params(rset))
        else:
            est = 0
        self.lbl_rebar_est.setText(
            "Pick Target Mat first" if (rset.target_material is None and active is None) else f"~{est:,} bars"
        )

        self.btn_rebar_create_extra.setVisible(active is not None)
        self.btn_rebar_create_extra.setEnabled(has_mesh)
        self.btn_rebar_rebuild.setVisible(active is not None and bool(active.stale))

        target, _hint = rebar_ops.resolve_remove_target(context)
        self.rebar_target_wrap.setVisible(target is not None)
        if target is not None:
            self.lbl_rebar_target.setText(f"Target: {target.display_label} · {target.sid8}")

        # Sticky footer in Rebar mode
        self.sticky_footer.setVisible(True)
        self.footer_sec_row.setVisible(False)
        if active is not None:
            self.summary_label.setText(
                "Host changed · Re-apply then Bake"
                if active.stale
                else ("Baked · click Edit to adjust" if active.baked else "Live preview · press Bake to commit")
            )
            self.btn_hero_commit.setText("BAKE REBAR")
            self.btn_hero_commit.setIcon(_qicon("apply_green"))
            self.btn_hero_commit.setEnabled(has_mesh)
        else:
            self.summary_label.setText(f"Estimated: ~{est:,} bars")
            self.btn_hero_commit.setText("CREATE REBAR")
            self.btn_hero_commit.setIcon(_qicon("start_orange"))
            self.btn_hero_commit.setEnabled(has_mesh and rset.target_material is not None)

    def refresh_from_blender(self) -> None:
        with blender_view3d_override():
            context = bpy.context
            scene = context.scene
            if scene is None:
                return
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None:
                cur_eng = getattr(settings, "boolean_solver", "RUST")
                eng_idx = self.cmb_engine.findData(cur_eng)
                if eng_idx >= 0 and self.cmb_engine.currentIndex() != eng_idx:
                    self.cmb_engine.blockSignals(True)
                    self.cmb_engine.setCurrentIndex(eng_idx)
                    self.cmb_engine.blockSignals(False)
                gpu_fd = bool(getattr(settings, "gpu_fast_drag", True))
                if self.chk_gpu_fast_drag.isChecked() != gpu_fd:
                    self.chk_gpu_fast_drag.blockSignals(True)
                    self.chk_gpu_fast_drag.setChecked(gpu_fd)
                    self.chk_gpu_fast_drag.blockSignals(False)
                vdock = bool(getattr(settings, "show_viewport_dock", True))
                if self.chk_viewport_dock.isChecked() != vdock:
                    self.chk_viewport_dock.blockSignals(True)
                    self.chk_viewport_dock.setChecked(vdock)
                    self.chk_viewport_dock.blockSignals(False)

            if self._active_tab == "REBAR":
                self._last_ab_refresh_sig = None
                self._refresh_rebar_page(context, scene)
                return

            session = load_session_readonly(scene)
            active_obj = getattr(context, "active_object", None)
            active_name = active_obj.name if active_obj is not None else ""
            raw_sess = scene.get(SESSION_SCENE_PROP, "") if session is not None else ""
            raw_zones = scene.get(zone_registry.REGISTRY_PROP, "")
            cut_mat = getattr(settings, "cut_material", None) if settings is not None else None
            cut_mat_name = cut_mat.name if cut_mat is not None else ""
            ab_sig = (
                raw_sess,
                raw_zones,
                len(bpy.data.objects),
                len(bpy.data.materials),
                active_name,
                len(getattr(context, "selected_objects", ())),
                getattr(context, "mode", ""),
                cut_mat_name,
                str(getattr(settings, "active_zone", "A")) if settings else "A",
                bool(getattr(settings, "enable_rough", False)) if settings else False,
                bool(getattr(settings, "apply_keep_result", True)) if settings else True,
                bool(getattr(settings, "apply_keep_pieces", True)) if settings else True,
                str(getattr(settings, "preview_display_mode", "ALL")) if settings else "ALL",
            )
            if getattr(self, "_last_ab_refresh_sig", None) == ab_sig:
                return
            self._last_ab_refresh_sig = ab_sig

            if session is None:
                self.step1_widget.setVisible(True)
                self.step2_widget.setVisible(False)
                self.sticky_footer.setVisible(False)
                info = apple_bite.detect_step1_action(context)
                mode = info["mode"]
                self.guide_label.setText(info["guide_text"])

                show_reedit = mode == "BOTH" and bool(info["session_id"])
                self.btn_reedit_old.setVisible(show_reedit)
                if show_reedit:
                    self.btn_reedit_old.setText("Re-edit Cut")
                    self.btn_smart_start.setText("Cut Further")
                else:
                    self.btn_smart_start.setText(
                        "Resume Cut" if mode == "REEDIT" else ("Start Cut" if mode == "START" else "Select Mesh")
                    )
                self.btn_smart_start.setEnabled(mode != "NONE")

                show_reset = mode in ("BOTH", "REEDIT") and bool(info["session_id"])
                self.btn_reset_original.setVisible(show_reset)
                if show_reset:
                    self.btn_reset_original.setText("Reset to Original")

                has_mesh = (active_obj is not None and active_obj.type == "MESH") or any(
                    o.type == "MESH" for o in context.selected_objects
                )
                show_cut_tools = show_reset or has_mesh
                self.cut_mat_box.setVisible(show_cut_tools)
                self.btn_assign_cut_mat.setVisible(show_reset)
                self.btn_fill_cut_caps.setVisible(has_mesh)
                if has_mesh:
                    self.btn_fill_cut_caps.setText("Fill Selected" if context.mode == "EDIT_MESH" else "Fill Cut Cap")
                if show_cut_tools:
                    self._sync_cut_material_combo(settings)
                self._last_cutter_sig = ()
                return

            # Active Apple Bite Session (single-pass count of valid existing cutters!)
            objs = bpy.data.objects
            total_c = 0
            active_c = 0
            for entry in session.cutters:
                c_obj = objs.get(entry.name)
                if c_obj is not None and c_obj.type == "MESH":
                    total_c += 1
                    if not entry.muted:
                        active_c += 1
            has_cutters = total_c > 0

            self.step1_widget.setVisible(False)
            self.step2_widget.setVisible(True)
            self.phase_b_widget.setVisible(has_cutters)
            self.cut_mat_box.setVisible(has_cutters)
            self.btn_assign_cut_mat.setVisible(False)
            self.btn_fill_cut_caps.setVisible(False)
            self.sticky_footer.setVisible(True)
            self.footer_sec_row.setVisible(True)

            if not has_cutters:
                self.guide_label.setText(f"{session.session_id} · Place a box or draw a cut in 3D View")
                self.summary_label.setText("Waiting for first cutter…")
                self.btn_hero_commit.setVisible(False)
            else:
                self.guide_label.setText(f"{session.session_id} · {total_c} cutter(s) active")
                self._sync_cut_material_combo(settings)
                self.btn_hero_commit.setVisible(True)
                has_output = bool(
                    settings is None or settings.apply_keep_result or settings.apply_keep_pieces
                )
                can_apply = has_output and active_c > 0
                self.btn_hero_commit.setEnabled(can_apply)
                self.btn_hero_commit.setIcon(_qicon("apply_green"))
                cut_word = "Cut" if active_c == 1 else "Cuts"
                self.btn_hero_commit.setText(f"APPLY · {active_c} {cut_word}" if active_c > 0 else "APPLY")

                if not has_output:
                    self.summary_label.setText("⚠ Output: None (enable Main Mesh or Zone Pieces)")
                    self.summary_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                elif active_c == 0:
                    self.summary_label.setText("All cutters muted — unmute to apply")
                    self.summary_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
                else:
                    summary = apple_bite.get_apply_preview_summary(scene)
                    zone_str = ", ".join(
                        f"{zone_registry.zone_token(z, scene)}({n})"
                        for z, n in sorted(summary["zone_counts"].items())
                    ) or "None"
                    self.summary_label.setText(f"Output: Main {summary['main_count']} · {zone_str}")
                    self.summary_label.setStyleSheet("color: #a6b2c6; font-size: 11px;")

            self._rebuild_cutter_rows(session)

            if settings is not None:
                active_z = str(settings.active_zone or "A")
                zdef_cur = zone_registry.resolve_zone(active_z, scene)
                zr, zg, zb = zdef_cur.color[0], zdef_cur.color[1], zdef_cur.color[2]
                self.zone_swatch.setStyleSheet(
                    f"border-radius: 7px; background-color: rgb({int(zr * 255)}, {int(zg * 255)}, {int(zb * 255)});"
                )

                zdefs = zone_registry.load_zones(scene)
                zone_items = [(z.key, z.token, z.key) for z in zdefs]
                sig = tuple(zone_items)
                if getattr(self, "_zone_items_sig", None) != sig:
                    self._zone_items_sig = sig
                    self.cmb_zone.blockSignals(True)
                    self.cmb_zone.clear()
                    for key, label, _desc in zone_items:
                        self.cmb_zone.addItem(label, key)
                    self.cmb_zone.blockSignals(False)
                active_idx = next((i for i, (key, _l, _d) in enumerate(zone_items) if key == active_z), -1)
                if active_idx >= 0 and self.cmb_zone.currentIndex() != active_idx:
                    self.cmb_zone.blockSignals(True)
                    self.cmb_zone.setCurrentIndex(active_idx)
                    self.cmb_zone.blockSignals(False)
                for btn, zdef in zip(self._zone_buttons.values(), zdefs[: len(self._zone_buttons)]):
                    if btn.text() != zdef.token:
                        btn.setText(zdef.token)
                    is_act = zdef.key == active_z
                    if btn.property("active") != is_act:
                        btn.setProperty("active", is_act)
                        btn.style().unpolish(btn)
                        btn.style().polish(btn)

                if self.chk_rough.isChecked() != bool(settings.enable_rough):
                    self.chk_rough.blockSignals(True)
                    self.chk_rough.setChecked(bool(settings.enable_rough))
                    self.chk_rough.blockSignals(False)

                any_rough = bool(settings.enable_rough) or any(c.rough for c in session.cutters)
                if self.rough_params_box.isVisible() != any_rough:
                    self.rough_params_box.setVisible(any_rough)
                if any_rough:
                    target_cutter, target_is_rough = apple_bite.get_active_rough_cutter(bpy.context, session)
                    cr = getattr(target_cutter, "fdt_rough", None) if target_cutter is not None else None
                    if cr is not None:
                        status = "" if target_is_rough else " (OFF)"
                        lbl_txt = f"Cutter: {target_cutter.name}{status}"
                        inward_val = float(cr.inward_depth)
                        jagged_val = float(cr.jaggedness)
                        r_scl = float(cr.noise_scale)
                        r_sub = max(1, min(3, int(cr.detail)))
                        r_seed = float(cr.seed)
                    else:
                        lbl_txt = "Default Rough (New Cutters)"
                        inward_val = float(getattr(settings, "rough_inward_ratio", 0.0))
                        jagged_val = float(getattr(settings, "rough_jaggedness", 0.45))
                        r_scl = float(getattr(settings, "rough_scale", 2.5))
                        r_sub = max(1, min(3, int(getattr(settings, "rough_subdiv", 2))))
                        r_seed = float(getattr(settings, "rough_seed", 0.0))

                    if self.lbl_rough_target.text() != lbl_txt:
                        self.lbl_rough_target.setText(lbl_txt)

                    inward_pct = inward_val * 100.0
                    if not self.sld_rough_inward.isSliderDown() and self.sld_rough_inward.value() != int(round(inward_pct)):
                        self.sld_rough_inward.blockSignals(True)
                        self.sld_rough_inward.setValue(int(round(inward_pct)))
                        self.sld_rough_inward.blockSignals(False)
                    if not self.spn_rough_inward.hasFocus() and abs(self.spn_rough_inward.value() - inward_pct) > 0.5:
                        self.spn_rough_inward.blockSignals(True)
                        self.spn_rough_inward.setValue(inward_pct)
                        self.spn_rough_inward.blockSignals(False)

                    jagged_pct = jagged_val * 100.0
                    if not self.sld_rough_jagged.isSliderDown() and self.sld_rough_jagged.value() != int(round(jagged_pct)):
                        self.sld_rough_jagged.blockSignals(True)
                        self.sld_rough_jagged.setValue(int(round(jagged_pct)))
                        self.sld_rough_jagged.blockSignals(False)
                    if not self.spn_rough_jagged.hasFocus() and abs(self.spn_rough_jagged.value() - jagged_pct) > 0.5:
                        self.spn_rough_jagged.blockSignals(True)
                        self.spn_rough_jagged.setValue(jagged_pct)
                        self.spn_rough_jagged.blockSignals(False)

                    if not self.spn_rough_scale.hasFocus() and abs(self.spn_rough_scale.value() - r_scl) > 1e-4:
                        self.spn_rough_scale.blockSignals(True)
                        self.spn_rough_scale.setValue(r_scl)
                        self.spn_rough_scale.blockSignals(False)
                    if not self.spn_rough_subdiv.hasFocus() and self.spn_rough_subdiv.value() != r_sub:
                        self.spn_rough_subdiv.blockSignals(True)
                        self.spn_rough_subdiv.setValue(r_sub)
                        self.spn_rough_subdiv.blockSignals(False)
                    if not self.spn_rough_seed.hasFocus() and abs(self.spn_rough_seed.value() - r_seed) > 1e-4:
                        self.spn_rough_seed.blockSignals(True)
                        self.spn_rough_seed.setValue(r_seed)
                        self.spn_rough_seed.blockSignals(False)

                for chk, val in (
                    (self.chk_keep_main, bool(getattr(settings, "apply_keep_result", True))),
                    (self.chk_keep_pieces, bool(getattr(settings, "apply_keep_pieces", True))),
                    (self.chk_auto_fill, bool(getattr(settings, "apply_auto_fill_caps", True))),
                    (self.chk_line_backfaces, bool(getattr(settings, "apply_line_backfaces", True))),
                    (self.chk_zone_grid, bool(getattr(settings, "show_zone_grid", True))),
                ):
                    if chk.isChecked() != val:
                        chk.blockSignals(True)
                        chk.setChecked(val)
                        chk.blockSignals(False)

                cur_vm = str(getattr(settings, "preview_display_mode", "ALL") or "ALL")
                for mid, btn in self._view_mode_buttons.items():
                    is_m = mid == cur_vm
                    if btn.isChecked() != is_m:
                        btn.blockSignals(True)
                        btn.setChecked(is_m)
                        btn.blockSignals(False)
                    if btn.property("active") != is_m:
                        btn.setProperty("active", is_m)
                        btn.style().unpolish(btn)
                        btn.style().polish(btn)
