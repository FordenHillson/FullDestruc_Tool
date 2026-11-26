"""Detached dark-themed PySide6 window (English UI, Step-by-Step Guided UX & Live Cutter List)."""

from __future__ import annotations

import json
from contextlib import contextmanager

import bpy
from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from ..core import apple_bite, zone_registry
from ..core.session import load_session


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
    background-color: #23252a;
    color: #e6e8ec;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #3c4049;
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    font-weight: bold;
    color: #f3c669;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #343842;
    border: 1px solid #494f5c;
    border-radius: 5px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #424857;
}
QPushButton:disabled {
    background-color: #2b2e35;
    color: #707785;
}
QPushButton#PrimaryBtn {
    background-color: #2c6e49;
    border-color: #3b9160;
    font-weight: bold;
    padding: 8px 12px;
}
QPushButton#PrimaryBtn:hover {
    background-color: #358458;
}
QPushButton#DangerBtn {
    background-color: #7c2d2d;
    border-color: #a33d3d;
}
QPushButton#ZoneBtn[active="true"] {
    background-color: #c97a1d;
    border-color: #f3a847;
    font-weight: bold;
}
QLabel#GuideBanner {
    color: #ffd97d;
    padding: 8px;
    background-color: #1b1d21;
    border: 1px solid #3c4049;
    border-radius: 5px;
}
QScrollBar:vertical {
    background: #1b1d22;
    width: 8px;
    margin: 2px 0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #4b5263;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #677087;
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
                background-color: #292d36;
                border: 1px solid #3a3f4b;
                border-radius: 5px;
            }
            QFrame#LayerRow:hover {
                background-color: #303542;
                border: 1px solid #586175;
            }
            QFrame#LayerRow[selected="true"] {
                background-color: #363d4d;
                border: 1px solid #f3a847;
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

    ROW_H = 36
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
                background-color: #1b1d22;
                border: 1px solid #3c4049;
                border-radius: 6px;
            }
            """
        )
        self.setMinimumHeight(46)

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
        self.setFixedHeight(46)

    def set_rows(self, rows: list[LayerRowWidget]) -> None:
        self.clear_rows()
        self._rows = list(rows)
        total_h = max(46, self.PAD * 2 + len(self._rows) * self.ROW_H + max(0, len(self._rows) - 1) * self.ROW_GAP)
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

        # Dashed glowing drop placeholder at target slot
        self._placeholder = QtWidgets.QFrame(self)
        self._placeholder.setStyleSheet(
            """
            QFrame {
                background-color: rgba(243, 168, 71, 28);
                border: 1.5px dashed #f3a847;
                border-radius: 5px;
            }
            """
        )
        self._placeholder.setGeometry(self.PAD, self._slot_y(self._target_index), w, self.ROW_H)
        self._placeholder.show()
        self._placeholder.lower()

        # Translucent floating ghost snapshot of the dragged row
        pixmap = row.grab()
        self._ghost_label = QtWidgets.QLabel(self)
        self._ghost_label.setPixmap(pixmap)
        self._ghost_label.setStyleSheet(
            "border: 1.5px solid #f3a847; border-radius: 5px; background-color: rgba(48, 54, 68, 220);"
        )
        effect = QtWidgets.QGraphicsOpacityEffect(self._ghost_label)
        effect.setOpacity(0.80)
        self._ghost_label.setGraphicsEffect(effect)
        self._ghost_label.setFixedSize(w, self.ROW_H)
        ghost_y = int(round(cursor_y - self._drag_grab_y))
        ghost_y = max(0, min(self.height() - self.ROW_H, ghost_y))
        self._ghost_label.move(self.PAD, ghost_y)
        self._ghost_label.show()
        self._ghost_label.raise_()

        # Hide original widget while ghost is floating
        row.setVisible(False)
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def update_row_drag(self, cursor_y: float) -> None:
        if self._dragging_row is None or self._ghost_label is None:
            return
        ghost_y = int(round(cursor_y - self._drag_grab_y))
        ghost_y = max(0, min(self.height() - self.ROW_H, ghost_y))
        self._ghost_label.move(self.PAD, ghost_y)

        # Determine which slot the ghost center is over
        center_y = ghost_y + self.ROW_H * 0.5
        step = self.ROW_H + self.ROW_GAP
        new_idx = int((center_y - self.PAD) // step)
        new_idx = max(0, min(len(self._rows) - 1, new_idx))

        if new_idx != self._target_index:
            old_idx = self._rows.index(self._dragging_row)
            self._rows.pop(old_idx)
            self._rows.insert(new_idx, self._dragging_row)
            self._target_index = new_idx

            w = max(self.width() - self.PAD * 2, 120)
            if self._placeholder is not None:
                self._placeholder.setGeometry(self.PAD, self._slot_y(new_idx), w, self.ROW_H)

            # Smoothly slide all other rows to their new slots!
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FullDestruc Tool — Apple Bite")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_STYLESHEET)

        self._zone_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._last_cutter_sig: tuple = ()
        self._last_mat_sig: tuple = ()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Always-visible Guide Banner
        self.guide_label = QtWidgets.QLabel("Step 1: Select building meshes in 3D View")
        self.guide_label.setObjectName("GuideBanner")
        self.guide_label.setWordWrap(True)
        root.addWidget(self.guide_label)

        # Boolean Engine selector (always visible)
        eng_row = QtWidgets.QHBoxLayout()
        eng_lbl = QtWidgets.QLabel("Boolean Engine:")
        eng_lbl.setFixedWidth(105)
        self.cmb_engine = QtWidgets.QComboBox()
        self.cmb_engine.addItem("Rust Engine (Fast & UV Safe)", "RUST")
        self.cmb_engine.addItem("Blender Exact", "EXACT")
        self.cmb_engine.addItem("Blender Float (Fast)", "FLOAT")
        self.cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        eng_row.addWidget(eng_lbl)
        eng_row.addWidget(self.cmb_engine, 1)
        root.addLayout(eng_row)

        self.chk_gpu_fast_drag = QtWidgets.QCheckBox("GPU Fast Drag (60+ FPS Async Preview)")
        self.chk_gpu_fast_drag.setChecked(True)
        self.chk_gpu_fast_drag.toggled.connect(self._on_gpu_fast_drag_toggled)
        root.addWidget(self.chk_gpu_fast_drag)

        # --- STEP 1 WIDGET (Idle) ---
        self.step1_widget = QtWidgets.QGroupBox("Step 1: Start or Re-edit Cut")
        s1_layout = QtWidgets.QVBoxLayout(self.step1_widget)
        self.btn_smart_start = QtWidgets.QPushButton("Select Meshes to Start")
        self.btn_smart_start.setObjectName("PrimaryBtn")
        self.btn_smart_start.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_smart_start))
        s1_layout.addWidget(self.btn_smart_start)

        self.btn_reedit_old = QtWidgets.QPushButton("Re-edit Previous Cut")
        self.btn_reedit_old.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_reedit))
        self.btn_reedit_old.setVisible(False)
        s1_layout.addWidget(self.btn_reedit_old)

        self.btn_reset_original = QtWidgets.QPushButton("Reset to Original")
        self.btn_reset_original.setObjectName("DangerBtn")
        self.btn_reset_original.clicked.connect(self._on_reset_original_clicked)
        self.btn_reset_original.setVisible(False)
        s1_layout.addWidget(self.btn_reset_original)
        root.addWidget(self.step1_widget)

        # --- STEP 2 WIDGET (Active Session: Cutters & Zones) ---
        self.step2_widget = QtWidgets.QGroupBox("Step 2: Place & Manage Cutters")
        s2_layout = QtWidgets.QVBoxLayout(self.step2_widget)

        zone_grid = QtWidgets.QGridLayout()
        initial_zones = zone_registry.load_zones(getattr(bpy.context, "scene", None))[:8]
        for idx, zdef in enumerate(initial_zones):
            btn = QtWidgets.QPushButton(zdef.token)
            btn.setObjectName("ZoneBtn")
            btn.clicked.connect(lambda _chk=False, z=zdef.key: self._set_zone(z))
            self._zone_buttons[zdef.key] = btn
            zone_grid.addWidget(btn, idx // 4, idx % 4)
        s2_layout.addLayout(zone_grid)

        self.cmb_zone = QtWidgets.QComboBox()
        self.cmb_zone.setEditable(True)
        self.cmb_zone.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        completer = self.cmb_zone.completer()
        if completer is not None:
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.cmb_zone.currentIndexChanged.connect(self._on_zone_combo_changed)
        s2_layout.addWidget(self.cmb_zone)

        zone_manage_row = QtWidgets.QHBoxLayout()
        btn_z_new = QtWidgets.QPushButton("+ New Zone")
        btn_z_new.clicked.connect(
            lambda _checked=False: self._run_op(
                bpy.ops.fdt.zone_manage, action="NEW", zone_key=self._active_zone_key()
            )
        )
        btn_obj_ren = QtWidgets.QPushButton("Rename")
        btn_obj_ren.clicked.connect(
            lambda _checked=False: self._run_invoke_op(bpy.ops.fdt.rename_popup)
        )
        btn_z_del = QtWidgets.QPushButton("Delete")
        btn_z_del.clicked.connect(
            lambda _checked=False: self._run_op(
                bpy.ops.fdt.zone_manage, action="DELETE", zone_key=self._active_zone_key()
            )
        )
        zone_manage_row.addWidget(btn_z_new)
        zone_manage_row.addWidget(btn_obj_ren)
        zone_manage_row.addWidget(btn_z_del)
        s2_layout.addLayout(zone_manage_row)

        row_cut = QtWidgets.QHBoxLayout()
        btn_box = QtWidgets.QPushButton("+ Add Box")
        btn_box.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_add_box))
        btn_mesh = QtWidgets.QPushButton("+ Use Selected Mesh")
        btn_mesh.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_add_mesh))
        row_cut.addWidget(btn_box)
        row_cut.addWidget(btn_mesh)
        s2_layout.addLayout(row_cut)

        btn_draw = QtWidgets.QPushButton("Draw Cutter (2D -> Extrude)")
        btn_draw.setObjectName("PrimaryBtn")
        btn_draw.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_draw_cutter))
        s2_layout.addWidget(btn_draw)

        self.layer_header = QtWidgets.QLabel("Cutter Layers (Drag ☰ to reorder — Top = Priority):")
        self.layer_header.setStyleSheet("color: #c8cdd8; font-size: 11px; margin-top: 4px;")
        s2_layout.addWidget(self.layer_header)

        row_view = QtWidgets.QHBoxLayout()
        row_view.setSpacing(4)
        self._view_mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        for mode_id, lbl in (
            ("ALL", "All"),
            ("PIECES", "Piece Only"),
            ("MAIN", "Hole Only"),
            ("MAIN_WIRE", "Hole + Wire"),
        ):
            b = QtWidgets.QPushButton(lbl)
            b.setCheckable(True)
            b.setProperty("active", mode_id == "ALL")
            b.clicked.connect(lambda _=False, m=mode_id: self._on_preview_view_clicked(m))
            row_view.addWidget(b)
            self._view_mode_buttons[mode_id] = b
        s2_layout.addLayout(row_view)

        row_grid = QtWidgets.QHBoxLayout()
        self.chk_zone_grid = QtWidgets.QCheckBox("Zone Grid Overlay")
        self.chk_zone_grid.setChecked(True)
        self.chk_zone_grid.toggled.connect(self._on_zone_grid_toggled)
        row_grid.addWidget(self.chk_zone_grid)
        s2_layout.addLayout(row_grid)

        self.cutter_layer_list = CutterLayerListWidget()
        self.cutter_layer_list.orderChanged.connect(self._on_layers_reordered)
        self.cutter_layer_list.rowClicked.connect(self._on_layer_row_clicked)

        self.layer_scroll = QtWidgets.QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.layer_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.layer_scroll.setMinimumHeight(48)
        self.layer_scroll.setMaximumHeight(240)
        self.layer_scroll.setWidget(self.cutter_layer_list)
        s2_layout.addWidget(self.layer_scroll)
        root.addWidget(self.step2_widget)

        # --- CUT FACE MATERIAL WIDGET (Live in Step 3 & Post-Apply in Step 1) ---
        self.cut_mat_box = QtWidgets.QGroupBox("Cut Face Material")
        cm_layout = QtWidgets.QVBoxLayout(self.cut_mat_box)
        cm_row = QtWidgets.QHBoxLayout()
        self.cmb_cut_mat = QtWidgets.QComboBox()
        self.cmb_cut_mat.setEditable(False)
        self.cmb_cut_mat.currentIndexChanged.connect(self._on_cut_mat_changed)
        self.btn_pick_mat = QtWidgets.QPushButton("Pick 3D")
        self.btn_pick_mat.setFixedWidth(74)
        self.btn_pick_mat.setToolTip("Click a surface in 3D View to pick its material (Eyedropper)")
        self.btn_pick_mat.clicked.connect(self._on_pick_surface_material)
        cm_row.addWidget(self.cmb_cut_mat, 1)
        cm_row.addWidget(self.btn_pick_mat)
        cm_layout.addLayout(cm_row)

        self.btn_assign_cut_mat = QtWidgets.QPushButton("Apply Cut Material to Selected")
        self.btn_assign_cut_mat.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_assign_cut_material))
        self.btn_assign_cut_mat.setVisible(False)
        cm_layout.addWidget(self.btn_assign_cut_mat)

        self.btn_fill_cut_caps = QtWidgets.QPushButton("Fill Cut Cap (Auto)")
        self.btn_fill_cut_caps.setToolTip(
            "Object Mode: Auto-fill unclosed holes on cut planes.\n"
            "Edit Mode: Fill selected open edges with Cut Material + Box UV."
        )
        self.btn_fill_cut_caps.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_fill_cut_caps))
        self.btn_fill_cut_caps.setVisible(False)
        cm_layout.addWidget(self.btn_fill_cut_caps)
        self.cut_mat_box.setVisible(False)
        root.addWidget(self.cut_mat_box)

        # --- STEP 3 WIDGET (Tune & Apply) ---
        self.step3_widget = QtWidgets.QGroupBox("Step 3: Tune & Apply")
        s3_layout = QtWidgets.QVBoxLayout(self.step3_widget)

        self.chk_rough = QtWidgets.QCheckBox("Rough Cut Edge")
        self.chk_rough.toggled.connect(self._on_rough_toggled)
        s3_layout.addWidget(self.chk_rough)

        self.rough_params_box = QtWidgets.QFrame()
        self.rough_params_box.setStyleSheet(
            "QFrame { background-color: #20232a; border: 1px solid #353944; border-radius: 5px; }"
            "QLabel { border: none; background: transparent; color: #cfd5df; font-size: 11px; }"
        )
        rp_layout = QtWidgets.QFormLayout(self.rough_params_box)
        rp_layout.setContentsMargins(8, 6, 8, 6)
        rp_layout.setSpacing(5)

        self.lbl_rough_target = QtWidgets.QLabel("Cutter: -")
        self.lbl_rough_target.setStyleSheet("color: #c4b5fd; font-weight: bold; font-size: 11px;")
        rp_layout.addRow(self.lbl_rough_target)

        # Inward Depth (0-100%, uniform scale inward preserving jagged pattern)
        inward_row = QtWidgets.QHBoxLayout()
        inward_row.setContentsMargins(0, 0, 0, 0)
        inward_row.setSpacing(6)
        self.sld_rough_inward = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_rough_inward.setRange(0, 100)
        self.sld_rough_inward.setValue(0)
        self.spn_rough_inward = QtWidgets.QDoubleSpinBox()
        self.spn_rough_inward.setRange(0.0, 100.0)
        self.spn_rough_inward.setSingleStep(5.0)
        self.spn_rough_inward.setDecimals(0)
        self.spn_rough_inward.setSuffix("%")
        self.spn_rough_inward.setValue(0.0)
        self.spn_rough_inward.setFixedWidth(64)
        self.sld_rough_inward.valueChanged.connect(self._on_rough_inward_slider_changed)
        self.spn_rough_inward.valueChanged.connect(self._on_rough_inward_spin_changed)
        inward_row.addWidget(self.sld_rough_inward, 1)
        inward_row.addWidget(self.spn_rough_inward)
        rp_layout.addRow("Inward Depth:", inward_row)

        # Jaggedness (0-100%, fracture depth around edges)
        jagged_row = QtWidgets.QHBoxLayout()
        jagged_row.setContentsMargins(0, 0, 0, 0)
        jagged_row.setSpacing(6)
        self.sld_rough_jagged = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_rough_jagged.setRange(0, 100)
        self.sld_rough_jagged.setValue(45)
        self.spn_rough_jagged = QtWidgets.QDoubleSpinBox()
        self.spn_rough_jagged.setRange(0.0, 100.0)
        self.spn_rough_jagged.setSingleStep(5.0)
        self.spn_rough_jagged.setDecimals(0)
        self.spn_rough_jagged.setSuffix("%")
        self.spn_rough_jagged.setValue(45.0)
        self.spn_rough_jagged.setFixedWidth(64)
        self.sld_rough_jagged.valueChanged.connect(self._on_rough_jagged_slider_changed)
        self.spn_rough_jagged.valueChanged.connect(self._on_rough_jagged_spin_changed)
        jagged_row.addWidget(self.sld_rough_jagged, 1)
        jagged_row.addWidget(self.spn_rough_jagged)
        rp_layout.addRow("Jaggedness:", jagged_row)

        # Noise Scale (chunks across cutter)
        self.spn_rough_scale = QtWidgets.QDoubleSpinBox()
        self.spn_rough_scale.setRange(0.5, 10.0)
        self.spn_rough_scale.setSingleStep(0.25)
        self.spn_rough_scale.setDecimals(2)
        self.spn_rough_scale.setValue(2.5)
        self.spn_rough_scale.valueChanged.connect(
            lambda v: self._on_rough_param_changed("noise_scale", float(v))
        )
        rp_layout.addRow("Noise Scale:", self.spn_rough_scale)

        # Low-Poly Detail (1-3)
        self.spn_rough_subdiv = QtWidgets.QSpinBox()
        self.spn_rough_subdiv.setRange(1, 3)
        self.spn_rough_subdiv.setValue(2)
        self.spn_rough_subdiv.valueChanged.connect(
            lambda v: self._on_rough_param_changed("detail", int(v))
        )
        rp_layout.addRow("Detail (1-3):", self.spn_rough_subdiv)

        # Seed
        self.spn_rough_seed = QtWidgets.QDoubleSpinBox()
        self.spn_rough_seed.setRange(-10000.0, 10000.0)
        self.spn_rough_seed.setSingleStep(0.5)
        self.spn_rough_seed.setDecimals(2)
        self.spn_rough_seed.setValue(0.0)
        self.spn_rough_seed.valueChanged.connect(
            lambda v: self._on_rough_param_changed("seed", float(v))
        )
        rp_layout.addRow("Seed:", self.spn_rough_seed)

        self.rough_params_box.setVisible(False)
        s3_layout.addWidget(self.rough_params_box)

        self.chk_auto_fill = QtWidgets.QCheckBox("Auto-Fill Cut Gaps")
        self.chk_auto_fill.setChecked(True)
        self.chk_auto_fill.setToolTip("Automatically fill unclosed solid holes strictly on cut planes on Apply")
        self.chk_auto_fill.toggled.connect(self._on_auto_fill_toggled)
        s3_layout.addWidget(self.chk_auto_fill)

        self.chk_line_backfaces = QtWidgets.QCheckBox("Close See-Through Backfaces")
        self.chk_line_backfaces.setChecked(True)
        self.chk_line_backfaces.setToolTip(
            "Automatically add inner liner faces behind single-sided attics/walls exposed by the cut"
        )
        self.chk_line_backfaces.toggled.connect(self._on_line_backfaces_toggled)
        s3_layout.addWidget(self.chk_line_backfaces)

        self.summary_label = QtWidgets.QLabel("Output: -")
        s3_layout.addWidget(self.summary_label)

        btn_apply = QtWidgets.QPushButton("APPLY")
        btn_apply.setObjectName("PrimaryBtn")
        btn_apply.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_apply))
        s3_layout.addWidget(btn_apply)
        root.addWidget(self.step3_widget)

        # Footer
        self.footer_widget = QtWidgets.QWidget()
        f_layout = QtWidgets.QHBoxLayout(self.footer_widget)
        f_layout.setContentsMargins(0, 0, 0, 0)
        btn_exit = QtWidgets.QPushButton("Pause / Exit")
        btn_exit.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_exit))
        btn_cancel = QtWidgets.QPushButton("Cancel & Restore")
        btn_cancel.setObjectName("DangerBtn")
        btn_cancel.clicked.connect(lambda: self._run_op(bpy.ops.fdt.applebite_cancel))
        f_layout.addWidget(btn_exit)
        f_layout.addWidget(btn_cancel)
        root.addWidget(self.footer_widget)

        root.addStretch(1)

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
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and settings.boolean_solver != val:
                settings.boolean_solver = val
        _tag_all_view3d_redraw()

    def _on_gpu_fast_drag_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and bool(settings.gpu_fast_drag) != bool(checked):
                settings.gpu_fast_drag = bool(checked)
        _tag_all_view3d_redraw()

    def _on_rough_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
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

    def _on_auto_fill_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and bool(getattr(settings, "apply_auto_fill_caps", True)) != bool(checked):
                settings.apply_auto_fill_caps = bool(checked)
        _tag_all_view3d_redraw()

    def _on_line_backfaces_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and bool(getattr(settings, "apply_line_backfaces", True)) != bool(checked):
                settings.apply_line_backfaces = bool(checked)
        _tag_all_view3d_redraw()

    def _on_zone_grid_toggled(self, checked: bool) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and bool(settings.show_zone_grid) != bool(checked):
                settings.show_zone_grid = bool(checked)
        _tag_all_view3d_redraw()

    def _on_preview_view_clicked(self, mode_id: str) -> None:
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
            if settings is not None and getattr(settings, "preview_display_mode", "ALL") != mode_id:
                settings.preview_display_mode = mode_id
        _tag_all_view3d_redraw()
        self.refresh_from_blender()

    def _on_cut_mat_changed(self, index: int) -> None:
        if index < 0:
            return
        mat_name = self.cmb_cut_mat.itemData(index)
        with blender_view3d_override():
            scene = bpy.context.scene
            settings = getattr(scene, "fdt_settings", None)
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
            self.guide_label.setText("Eyedropper active: Click any surface in 3D View (ESC / RMB to cancel)")
        except Exception as exc:
            self.guide_label.setText(f"Notice: {exc}")

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
        self.cmb_cut_mat.addItem("-- Select Cut Material (Default) --", "")
        sel_idx = 0
        for idx, name in enumerate(mat_names, start=1):
            self.cmb_cut_mat.addItem(name, name)
            if name == cur_name:
                sel_idx = idx
        self.cmb_cut_mat.setCurrentIndex(sel_idx)
        self.cmb_cut_mat.blockSignals(False)

        if cur_mat is None:
            self.cmb_cut_mat.setStyleSheet("QComboBox { border: 1px solid #d9534f; }")
        else:
            self.cmb_cut_mat.setStyleSheet("")
        self.btn_assign_cut_mat.setEnabled(cur_mat is not None)

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
        msg = (
            f"Delete all cuts and restore original?\n\n"
            f"Original: {display_names}\n"
            f"Cut sessions to delete: {fam_count}{extra}"
        )
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
        self.layer_header.setVisible(total > 0)
        self.layer_scroll.setVisible(total > 0)
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
            grip.setFixedWidth(36)
            grip.setStyleSheet("color: #9aa3b5; font-weight: bold; font-size: 11px; background: transparent;")
            grip.setToolTip("Drag up/down to change cut priority")

            r, g, b, _ = apple_bite.get_zone_color(entry.zone)
            rgb_hex = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            btn_z = QtWidgets.QPushButton(zone_registry.zone_token(entry.zone, bpy.context.scene)[:12])
            btn_z.setFixedWidth(26)
            btn_z.setStyleSheet(f"background-color: {rgb_hex}; color: #111; font-weight: bold; padding: 3px;")
            btn_z.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_cycle_zone, cutter_name=n))

            lbl_name = QtWidgets.QLabel(entry.name)
            lbl_name.setStyleSheet("color: #e6e8ec; padding-left: 2px; background: transparent;")
            lbl_name.setToolTip("Click to select in 3D View, or drag up/down to reorder layer")

            btn_up = QtWidgets.QPushButton("▲")
            btn_up.setFixedWidth(24)
            btn_up.setEnabled(idx > 0)
            btn_up.setStyleSheet("padding: 2px;")
            btn_up.clicked.connect(
                lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_move_cutter, cutter_name=n, direction="UP")
            )

            btn_dn = QtWidgets.QPushButton("▼")
            btn_dn.setFixedWidth(24)
            btn_dn.setEnabled(idx < total - 1)
            btn_dn.setStyleSheet("padding: 2px;")
            btn_dn.clicked.connect(
                lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_move_cutter, cutter_name=n, direction="DOWN")
            )

            btn_rough = QtWidgets.QPushButton("R" if entry.rough else "F")
            btn_rough.setFixedWidth(26)
            btn_rough.setToolTip("Toggle Rough / Flat Cut Edge")
            if entry.rough:
                btn_rough.setStyleSheet("background-color: #7c3aed; color: #fff; font-weight: bold; padding: 3px;")
            else:
                btn_rough.setStyleSheet("padding: 3px;")
            btn_rough.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_toggle_rough, cutter_name=n))

            btn_mute = QtWidgets.QPushButton("Off" if entry.muted else "On")
            btn_mute.setFixedWidth(32)
            btn_mute.setToolTip("Mute / Unmute Cutter")
            btn_mute.setStyleSheet("padding: 3px;")
            btn_mute.clicked.connect(lambda _=False, n=entry.name: self._run_op(bpy.ops.fdt.applebite_toggle_mute, cutter_name=n))

            btn_del = QtWidgets.QPushButton("X")
            btn_del.setObjectName("DangerBtn")
            btn_del.setFixedWidth(24)
            btn_del.setStyleSheet("padding: 3px;")
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
        scroll_h = min(self.cutter_layer_list.height() + 2, 240)
        self.layer_scroll.setFixedHeight(max(48, scroll_h))

    def refresh_from_blender(self) -> None:
        with blender_view3d_override():
            context = bpy.context
            scene = context.scene
            if scene is None:
                return
            settings = getattr(scene, "fdt_settings", None)
            session = load_session(scene)

            if session is None:
                self.step1_widget.setVisible(True)
                self.step2_widget.setVisible(False)
                self.step3_widget.setVisible(False)
                self.footer_widget.setVisible(False)
                info = apple_bite.detect_step1_action(context)
                self.guide_label.setText(info["guide_text"])
                self.btn_smart_start.setText(info["button_label"])
                self.btn_smart_start.setEnabled(info["mode"] != "NONE")
                show_reedit = info["mode"] == "BOTH" and bool(info["session_id"])
                self.btn_reedit_old.setVisible(show_reedit)
                if show_reedit:
                    self.btn_reedit_old.setText(info["reedit_label"])
                show_reset = info["mode"] in ("BOTH", "REEDIT") and bool(info["session_id"])
                self.btn_reset_original.setVisible(show_reset)
                if show_reset:
                    self.btn_reset_original.setText(info["reset_label"])
                active_obj = context.active_object
                has_mesh = (active_obj is not None and active_obj.type == "MESH") or any(
                    o.type == "MESH" for o in context.selected_objects
                )
                show_cut_tools = show_reset or has_mesh
                self.cut_mat_box.setVisible(show_cut_tools)
                self.btn_assign_cut_mat.setVisible(show_reset)
                self.btn_fill_cut_caps.setVisible(has_mesh)
                if has_mesh:
                    self.btn_fill_cut_caps.setText(
                        "Fill Selected Cut Cap"
                        if context.mode == "EDIT_MESH"
                        else "Fill Cut Cap (Auto)"
                    )
                if show_cut_tools:
                    self._sync_cut_material_combo(settings)
                self._last_cutter_sig = ()
                return

            self.step1_widget.setVisible(False)
            self.step2_widget.setVisible(True)
            self.footer_widget.setVisible(True)

            has_cutters = len(session.cutters) > 0
            self.step3_widget.setVisible(has_cutters)
            self.cut_mat_box.setVisible(has_cutters)
            self.btn_assign_cut_mat.setVisible(False)
            self.btn_fill_cut_caps.setVisible(False)
            if has_cutters:
                self._sync_cut_material_combo(settings)

            if not has_cutters:
                self.guide_label.setText(
                    f"Step 2 ({session.session_id}): Click '+ Add Box' to spawn at center, then move with G / R / S"
                )
            else:
                self.guide_label.setText(
                    f"Step 2 ({session.session_id}): Adjust boxes, use Gear wheel in 3D View, or click APPLY"
                )

            self._rebuild_cutter_rows(session)

            if has_cutters:
                summary = apple_bite.get_apply_preview_summary(scene)
                zone_str = ", ".join(
                    f"{zone_registry.zone_token(z, scene)}({n})"
                    for z, n in sorted(summary["zone_counts"].items())
                ) or "None"
                self.summary_label.setText(f"Output: Main {summary['main_count']} | Zones: {zone_str}")

            if settings is not None:
                active_z = str(settings.active_zone or "A")
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
                active_idx = next((i for i, (key, _label, _d) in enumerate(zone_items) if key == active_z), -1)
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
                    inward_ipct = int(round(inward_pct))
                    if not self.sld_rough_inward.isSliderDown() and self.sld_rough_inward.value() != inward_ipct:
                        self.sld_rough_inward.blockSignals(True)
                        self.sld_rough_inward.setValue(inward_ipct)
                        self.sld_rough_inward.blockSignals(False)
                    if not self.spn_rough_inward.hasFocus() and abs(self.spn_rough_inward.value() - inward_pct) > 0.5:
                        self.spn_rough_inward.blockSignals(True)
                        self.spn_rough_inward.setValue(inward_pct)
                        self.spn_rough_inward.blockSignals(False)

                    jagged_pct = jagged_val * 100.0
                    jagged_ipct = int(round(jagged_pct))
                    if not self.sld_rough_jagged.isSliderDown() and self.sld_rough_jagged.value() != jagged_ipct:
                        self.sld_rough_jagged.blockSignals(True)
                        self.sld_rough_jagged.setValue(jagged_ipct)
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
                auto_fill = bool(getattr(settings, "apply_auto_fill_caps", True))
                if self.chk_auto_fill.isChecked() != auto_fill:
                    self.chk_auto_fill.blockSignals(True)
                    self.chk_auto_fill.setChecked(auto_fill)
                    self.chk_auto_fill.blockSignals(False)
                line_bf = bool(getattr(settings, "apply_line_backfaces", True))
                if self.chk_line_backfaces.isChecked() != line_bf:
                    self.chk_line_backfaces.blockSignals(True)
                    self.chk_line_backfaces.setChecked(line_bf)
                    self.chk_line_backfaces.blockSignals(False)
                if self.chk_zone_grid.isChecked() != bool(settings.show_zone_grid):
                    self.chk_zone_grid.blockSignals(True)
                    self.chk_zone_grid.setChecked(bool(settings.show_zone_grid))
                    self.chk_zone_grid.blockSignals(False)
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
