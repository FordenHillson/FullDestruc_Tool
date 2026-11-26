"""Standalone PySide6 Timeline Performance & Debug Monitor Window for FullDestruc_Tool.

Features:
- Live system stats (CPU%, RAM MB, cores) + Engine status
- Real-time stacked-bar Timeline Graph (Upload / Extract / Rust / Writeback ms)
  with 60 FPS (16.6ms) & 30 FPS (33.3ms) reference lines
- Auto-scrolling Timeline Event Table (Time | Action | Total | Rust | Write | RAM | Polys)
  with spike highlighting
- Crash-safe auto-logging to `FullDestruc_Tool/logs/timeline_latest.jsonl`
- One-click Copy Timeline Summary for AI + Save Full Timeline (.json)
"""

from __future__ import annotations

import os
import subprocess

import bpy

from ..core import profiler, rust_bridge

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    HAS_PYSIDE6 = True
except Exception:
    HAS_PYSIDE6 = False


_DEBUG_STYLESHEET = """
QWidget {
    background-color: #16181d;
    color: #e6e8ec;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2c313c;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: #9bb8ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #282d38;
    border: 1px solid #3c4354;
    border-radius: 5px;
    padding: 6px 10px;
    color: #f0f2f5;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #363d4d;
    border-color: #5b8cff;
}
QPushButton#CopyBtn {
    background-color: #2563eb;
    border-color: #3b82f6;
}
QPushButton#CopyBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#RecOn {
    background-color: #b91c1c;
    border-color: #ef4444;
    color: #ffffff;
}
QPushButton#RecOff {
    background-color: #334155;
    border-color: #475569;
    color: #cbd5e1;
}
QTableWidget {
    background-color: #101216;
    alternate-background-color: #151820;
    border: 1px solid #2b303b;
    border-radius: 4px;
    gridline-color: #232833;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 11px;
    color: #e2e8f0;
}
QHeaderView::section {
    background-color: #1e222b;
    color: #93c5fd;
    padding: 4px;
    border: 1px solid #2b303b;
    font-weight: bold;
}
QPlainTextEdit {
    background-color: #101216;
    border: 1px solid #2b303b;
    border-radius: 4px;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 11px;
    color: #cbd5e1;
}
"""

_DEBUG_WINDOW = None


if HAS_PYSIDE6:

    class TimelineGraphWidget(QtWidgets.QWidget):
        """Lightweight QPainter stacked-bar chart for live frame timings."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(130)
            self._events: list[profiler.TimelineEvent] = []

        def set_events(self, events: list[profiler.TimelineEvent]) -> None:
            self._events = events[-80:]
            self.update()

        def paintEvent(self, _event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

            rect = self.rect()
            w = rect.width()
            h = rect.height()
            pad_l, pad_r, pad_t, pad_b = 44, 8, 14, 18
            plot_w = max(w - pad_l - pad_r, 10)
            plot_h = max(h - pad_t - pad_b, 10)

            # Background
            painter.fillRect(rect, QtGui.QColor("#0f1116"))
            painter.setPen(QtGui.QColor("#262b36"))
            painter.drawRect(pad_l, pad_t, plot_w, plot_h)

            # Dynamic Y scale (at least 40ms so 16.6ms & 33.3ms lines are clearly visible)
            max_ms = 40.0
            for ev in self._events:
                if ev.total_ms > max_ms:
                    max_ms = ev.total_ms * 1.15

            def y_for_ms(ms: float) -> int:
                frac = min(max(ms / max_ms, 0.0), 1.0)
                return int(pad_t + plot_h - frac * plot_h)

            # Reference lines: 16.6ms (60 FPS) & 33.3ms (30 FPS)
            font = QtGui.QFont("Consolas", 8)
            painter.setFont(font)
            for ref_ms, label, col in (
                (16.6, "60fps", "#22c55e"),
                (33.3, "30fps", "#eab308"),
            ):
                if ref_ms <= max_ms:
                    ry = y_for_ms(ref_ms)
                    pen = QtGui.QPen(QtGui.QColor(col))
                    pen.setStyle(QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(pad_l, ry, pad_l + plot_w, ry)
                    painter.setPen(QtGui.QColor(col))
                    painter.drawText(2, ry + 3, label)

            # Top scale label
            painter.setPen(QtGui.QColor("#94a3b8"))
            painter.drawText(2, pad_t + 9, f"{max_ms:.0f}ms")

            # Stacked bars
            n = len(self._events)
            if n > 0:
                slot_w = max(plot_w / 80.0, 2.0)
                bar_w = max(int(slot_w) - 1, 2)
                start_x = pad_l + plot_w - int(n * slot_w)

                c_up = QtGui.QColor("#8b5cf6")
                c_ex = QtGui.QColor("#f59e0b")
                c_rust = QtGui.QColor("#10b981")
                c_wr = QtGui.QColor("#ef4444")
                c_mk = QtGui.QColor("#38bdf8")

                for idx, ev in enumerate(self._events):
                    bx = int(start_x + idx * slot_w)
                    if ev.kind == "MARKER":
                        if ev.total_ms > 0.0:
                            y0 = y_for_ms(0.0)
                            y1 = y_for_ms(ev.total_ms)
                            painter.fillRect(bx, y1, bar_w, max(y0 - y1, 2), c_mk)
                        else:
                            painter.setPen(c_mk)
                            painter.drawLine(bx, pad_t, bx, pad_t + plot_h)
                        continue

                    acc = 0.0
                    for seg_ms, col in (
                        (ev.upload_ms, c_up),
                        (ev.extract_ms, c_ex),
                        (ev.rust_ms, c_rust),
                        (ev.writeback_ms, c_wr),
                    ):
                        if seg_ms <= 0.0:
                            continue
                        y_bot = y_for_ms(acc)
                        acc += seg_ms
                        y_top = y_for_ms(acc)
                        bh = max(y_bot - y_top, 1)
                        painter.fillRect(bx, y_top, bar_w, bh, col)

            # Legend along bottom
            legend_y = h - 4
            lx = pad_l + 4
            for text, col in (
                ("■ Rust", "#10b981"),
                ("■ Writeback", "#ef4444"),
                ("■ Extract", "#f59e0b"),
                ("■ Upload", "#8b5cf6"),
                ("■ Marker/Apply", "#38bdf8"),
            ):
                painter.setPen(QtGui.QColor(col))
                painter.drawText(lx, legend_y, text)
                lx += 78

            painter.end()

    class FDTDebugWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("FullDestruc Tool — Timeline Performance & Debug Monitor")
            self.setMinimumSize(640, 680)
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
            self.setStyleSheet(_DEBUG_STYLESHEET)

            self._last_seq_rendered: int = -1
            self._build_ui()

            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.refresh_stats)
            self._timer.start(250)
            self.refresh_stats()

        def _build_ui(self) -> None:
            root = QtWidgets.QVBoxLayout(self)
            root.setSpacing(6)

            # 0. Recording Control Bar
            ctrl_row = QtWidgets.QHBoxLayout()
            self.btn_rec = QtWidgets.QPushButton("● REC (Auto-Log ON)")
            self.btn_rec.setObjectName("RecOn")
            self.btn_rec.clicked.connect(self._on_toggle_rec)

            btn_new = QtWidgets.QPushButton("New Timeline")
            btn_new.clicked.connect(self._on_clear)

            btn_open_dir = QtWidgets.QPushButton("Open Logs Folder")
            btn_open_dir.clicked.connect(self._on_open_logs_dir)

            ctrl_row.addWidget(self.btn_rec)
            ctrl_row.addWidget(btn_new)
            ctrl_row.addWidget(btn_open_dir)
            ctrl_row.addStretch(1)
            root.addLayout(ctrl_row)

            # 1. System & Engine Info
            sys_box = QtWidgets.QGroupBox("System & Engine Status")
            sys_layout = QtWidgets.QGridLayout(sys_box)
            self.lbl_engine = QtWidgets.QLabel("Engine: -")
            self.lbl_cpu = QtWidgets.QLabel("Blender CPU: -")
            self.lbl_ram = QtWidgets.QLabel("Blender RAM: -")
            self.lbl_total = QtWidgets.QLabel("Last Eval: Waiting...")
            self.lbl_total.setStyleSheet("font-size: 13px; font-weight: bold; color: #34d399;")
            sys_layout.addWidget(self.lbl_engine, 0, 0, 1, 2)
            sys_layout.addWidget(self.lbl_cpu, 1, 0)
            sys_layout.addWidget(self.lbl_ram, 1, 1)
            sys_layout.addWidget(self.lbl_total, 2, 0, 1, 2)
            root.addWidget(sys_box)

            # 2. Live Timeline Stacked Bar Graph
            graph_box = QtWidgets.QGroupBox("Live Timeline Graph (Last 80 Events)")
            g_layout = QtWidgets.QVBoxLayout(graph_box)
            g_layout.setContentsMargins(6, 6, 6, 6)
            self.graph = TimelineGraphWidget()
            g_layout.addWidget(self.graph)
            root.addWidget(graph_box)

            # 3. Timeline Event Table
            tbl_box = QtWidgets.QGroupBox("Timeline Log (Time | Action | Total | Rust | Write | RAM | Polys)")
            t_layout = QtWidgets.QVBoxLayout(tbl_box)
            self.table = QtWidgets.QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels(
                ["Time", "Action", "Total ms", "Rust ms", "Write ms", "RAM MB", "Polys / Note"]
            )
            self.table.setAlternatingRowColors(True)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.verticalHeader().setVisible(False)
            self.table.verticalHeader().setDefaultSectionSize(20)
            hdr = self.table.horizontalHeader()
            hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
            t_layout.addWidget(self.table)
            root.addWidget(tbl_box, 2)

            # 4. Compact AI Summary Preview
            sum_box = QtWidgets.QGroupBox("AI Timeline Summary (Ready to Paste)")
            s_layout = QtWidgets.QVBoxLayout(sum_box)
            self.txt_report = QtWidgets.QPlainTextEdit()
            self.txt_report.setReadOnly(True)
            self.txt_report.setMaximumHeight(130)
            s_layout.addWidget(self.txt_report)
            root.addWidget(sum_box, 1)

            # 5. Export Buttons
            btn_row = QtWidgets.QHBoxLayout()
            self.btn_copy = QtWidgets.QPushButton("Copy Timeline Summary (AI)")
            self.btn_copy.setObjectName("CopyBtn")
            self.btn_copy.clicked.connect(self._on_copy_report)

            self.btn_save = QtWidgets.QPushButton("Save Full Timeline (.json)")
            self.btn_save.clicked.connect(self._on_save_report)

            btn_row.addWidget(self.btn_copy)
            btn_row.addWidget(self.btn_save)
            root.addLayout(btn_row)

            self.lbl_status = QtWidgets.QLabel(
                f"Auto-saving crash-safe log to: {profiler.get_latest_log_path()}"
            )
            self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
            root.addWidget(self.lbl_status)

        def _sync_rec_button(self) -> None:
            rec = profiler.is_recording()
            self.btn_rec.setText("● REC (Auto-Log ON)" if rec else "❚❚ PAUSED")
            self.btn_rec.setObjectName("RecOn" if rec else "RecOff")
            self.btn_rec.style().unpolish(self.btn_rec)
            self.btn_rec.style().polish(self.btn_rec)

        def _on_toggle_rec(self) -> None:
            profiler.set_recording(not profiler.is_recording())
            self._sync_rec_button()

        def _on_open_logs_dir(self) -> None:
            logs_dir = profiler.get_logs_dir()
            try:
                if os.name == "nt":
                    os.startfile(logs_dir)
                else:
                    subprocess.Popen(["xdg-open", logs_dir])
            except Exception as exc:
                self.lbl_status.setText(f"Could not open folder: {exc}")

        def _append_table_row(self, ev: profiler.TimelineEvent) -> None:
            row = self.table.rowCount()
            self.table.insertRow(row)

            if ev.kind == "EVAL":
                info = f"{ev.target_polys:,} -> {ev.out_polys:,} ({ev.num_cutters}c)"
                if ev.note:
                    info = f"{info} [{ev.note}]"
            else:
                info = ev.note or "MARKER"

            vals = [
                f"+{ev.elapsed_sec:6.2f}s",
                ev.action,
                f"{ev.total_ms:.2f}" if ev.total_ms > 0 else "-",
                f"{ev.rust_ms:.2f}" if ev.kind == "EVAL" else "-",
                f"{ev.writeback_ms:.2f}" if ev.kind == "EVAL" else "-",
                f"{ev.ram_mb:.0f}",
                info,
            ]

            bg = None
            if ev.kind == "MARKER":
                bg = QtGui.QColor("#172554")
            elif ev.total_ms >= 50.0:
                bg = QtGui.QColor("#450a0a")
            elif ev.total_ms >= 33.0:
                bg = QtGui.QColor("#422006")

            for col, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                if bg is not None:
                    item.setBackground(bg)
                if col in (2, 3, 4, 5):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(row, col, item)

        def refresh_stats(self) -> None:
            try:
                sys_s = profiler.get_system_stats()
                scene = getattr(bpy.context, "scene", None)
                settings = getattr(scene, "fdt_settings", None) if scene else None
                solver = getattr(settings, "boolean_solver", "?") if settings else "?"
                dll_ok = rust_bridge.is_available()

                self.lbl_engine.setText(
                    f"Engine: {solver}  |  Rust DLL: {'OK' if dll_ok else 'MISSING'}  |  Cores: {sys_s['cpu_cores']}"
                )
                self.lbl_cpu.setText(f"Blender CPU: {sys_s['process_cpu_pct']:.1f}%")
                self.lbl_ram.setText(
                    f"Blender RAM: {sys_s['process_ram_mb']:.0f} MB (Peak {sys_s['process_peak_ram_mb']:.0f} MB)"
                )

                timeline = profiler.get_timeline()
                lat = profiler.get_latest()
                if lat is not None:
                    fps = 1000.0 / max(lat.total_ms, 0.1)
                    color = "#34d399" if lat.total_ms < 20.0 else ("#fbbf24" if lat.total_ms < 50.0 else "#f87171")
                    self.lbl_total.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")
                    self.lbl_total.setText(
                        f"Last Eval: {lat.total_ms:.2f} ms (~{fps:.0f} FPS)  |  "
                        f"Rust: {lat.rust_ms:.2f} ms  |  Write: {lat.writeback_ms:.2f} ms  |  Events: {len(timeline)}"
                    )
                else:
                    self.lbl_total.setText(f"Events: {len(timeline)} — Waiting for cut...")

                latest_seq = timeline[-1].seq if timeline else 0
                if latest_seq != self._last_seq_rendered:
                    if latest_seq < self._last_seq_rendered or self._last_seq_rendered < 0:
                        self.table.setRowCount(0)
                        for ev in timeline:
                            self._append_table_row(ev)
                    else:
                        for ev in timeline:
                            if ev.seq > self._last_seq_rendered:
                                self._append_table_row(ev)
                    self._last_seq_rendered = latest_seq
                    self.table.scrollToBottom()
                    self.graph.set_events(timeline)
                    self.txt_report.setPlainText(profiler.format_timeline_summary_for_ai())
            except Exception as exc:
                self.lbl_status.setText(f"Refresh error: {exc}")

        def _on_copy_report(self) -> None:
            text = profiler.format_timeline_summary_for_ai()
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            self.lbl_status.setText("Copied Timeline Summary! กด Ctrl+V วางส่งให้ AI ได้เลยครับ")

        def _on_save_report(self) -> None:
            try:
                path = profiler.save_report_file()
                self.lbl_status.setText(f"Saved full timeline: {path}")
            except Exception as exc:
                self.lbl_status.setText(f"Save failed: {exc}")

        def _on_clear(self) -> None:
            profiler.start_new_session_log()
            self._last_seq_rendered = -1
            self.table.setRowCount(0)
            self.graph.set_events([])
            self.refresh_stats()
            self.lbl_status.setText("Started a fresh Timeline log.")


def open_debug_window() -> tuple[bool, str]:
    global _DEBUG_WINDOW
    if not HAS_PYSIDE6:
        return False, "PySide6 is not available in this Blender Python."
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    if _DEBUG_WINDOW is None:
        _DEBUG_WINDOW = FDTDebugWindow()
    _DEBUG_WINDOW.show()
    _DEBUG_WINDOW.raise_()
    _DEBUG_WINDOW.activateWindow()
    return True, "Opened Timeline Debug Monitor"


def close_debug_window() -> None:
    global _DEBUG_WINDOW
    if _DEBUG_WINDOW is not None:
        try:
            _DEBUG_WINDOW.close()
        except Exception:
            pass
        _DEBUG_WINDOW = None


class FDT_OT_open_debug_window(bpy.types.Operator):
    bl_idname = "fdt.open_debug_window"
    bl_label = "Debug Monitor"
    bl_description = "Open standalone Timeline Performance & Debug Monitor window"

    def execute(self, context):
        ok, msg = open_debug_window()
        if not ok:
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        return {"FINISHED"}


CLASSES = (FDT_OT_open_debug_window,)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    close_debug_window()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
