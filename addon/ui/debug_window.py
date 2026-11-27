"""Standalone PySide6 Timeline Performance & Debug Monitor Window for FullDestruc_Tool.

Features:
- Uncapped Benchmark Performance Score (10 -> 1,000,000+ pts) with 4 sub-scores,
  Smoothness Grade (S-F), and Automatic Bottleneck Diagnostics
- 2-Tab layout:
  1. Live Monitor: Live Benchmark Banner, System/Engine stats, stacked-bar Timeline Graph
     (with 60 FPS / 30 FPS lines), Timeline Event Table, AI Summary & Export
  2. Score & Compare: Detailed 4-axis breakdown, Bottleneck Diagnostics, and A/B JSON
     Comparison (Baseline .json vs Current Live Session OR File A vs File B)
- Crash-safe auto-logging to `FullDestruc_Tool/logs/timeline_latest.jsonl`
"""

from __future__ import annotations

import os
import subprocess

import bpy

from ..core import perf_score, profiler, rust_bridge

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
QTabWidget::pane {
    border: 1px solid #2c313c;
    border-radius: 6px;
    background-color: #16181d;
    top: -1px;
}
QTabBar::tab {
    background-color: #1e222b;
    color: #94a3b8;
    border: 1px solid #2c313c;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 14px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #2563eb;
    color: #ffffff;
    border-color: #3b82f6;
}
QTabBar::tab:hover:!selected {
    background-color: #282d38;
    color: #e2e8f0;
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
QFrame#ScoreBanner {
    background-color: #111520;
    border: 1px solid #2d3a54;
    border-radius: 8px;
}
QFrame#SubCard {
    background-color: #11141c;
    border: 1px solid #262d3d;
    border-radius: 6px;
}
QFrame#DeltaBanner {
    background-color: #111520;
    border: 1px solid #2d3a54;
    border-radius: 6px;
}
QComboBox {
    background-color: #101216;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e2e8f0;
}
QComboBox QAbstractItemView {
    background-color: #151820;
    color: #e2e8f0;
    selection-background-color: #2563eb;
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
QListWidget {
    background-color: #101216;
    border: 1px solid #2b303b;
    border-radius: 4px;
    font-size: 11px;
    color: #e2e8f0;
}
"""

_GRADE_COLORS = {
    "S": ("#064e3b", "#34d399"),
    "A": ("#14532d", "#4ade80"),
    "B": ("#1e3a8a", "#60a5fa"),
    "C": ("#713f12", "#facc15"),
    "D": ("#7c2d12", "#fb923c"),
    "F": ("#7f1d1d", "#f87171"),
    "-": ("#1e293b", "#94a3b8"),
}

_ALERT_COLORS = {
    "OK": "#34d399",
    "INFO": "#93c5fd",
    "WARN": "#fbbf24",
    "CRITICAL": "#f87171",
}

_LIVE_SENTINEL = "__LIVE_SESSION__"

_DEBUG_WINDOW = None


def _fmt_cmp_val(val: float | None, unit: str) -> str:
    if val is None:
        return "-"
    if unit == "pts":
        return f"{int(round(val)):,}"
    if unit == "MB":
        return f"{val:,.0f} MB"
    if unit == "Hz":
        return f"{val:.2f} Hz"
    return f"{val:,.2f} {unit}"


if HAS_PYSIDE6:

    class TimelineGraphWidget(QtWidgets.QWidget):
        """Lightweight QPainter stacked-bar chart for live frame timings."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(125)
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
            self.setMinimumSize(700, 760)
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
            self.setStyleSheet(_DEBUG_STYLESHEET)

            self._last_seq_rendered: int = -1
            self._loaded_reports: dict[str, dict] = {}
            self._last_comparison: perf_score.ComparisonReport | None = None
            self._build_ui()
            self._refresh_saved_file_combos(select_latest_baseline=True)

            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.refresh_stats)
            self._timer.start(250)
            self.refresh_stats()

        def _build_ui(self) -> None:
            root = QtWidgets.QVBoxLayout(self)
            root.setSpacing(6)

            # 0. Recording Control Bar (always visible)
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

            # Main 2-Tab Widget
            self.tabs = QtWidgets.QTabWidget()
            self.tab_live = QtWidgets.QWidget()
            self.tab_compare = QtWidgets.QWidget()
            self.tabs.addTab(self.tab_live, "Live Monitor")
            self.tabs.addTab(self.tab_compare, "Score & Compare")
            root.addWidget(self.tabs, 1)

            self._build_live_tab()
            self._build_compare_tab()

            self.lbl_status = QtWidgets.QLabel(
                f"Auto-saving crash-safe log to: {profiler.get_latest_log_path()}"
            )
            self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
            root.addWidget(self.lbl_status)

        def _build_live_tab(self) -> None:
            lay = QtWidgets.QVBoxLayout(self.tab_live)
            lay.setSpacing(6)

            # 0. Live Benchmark Score Banner
            banner = QtWidgets.QFrame()
            banner.setObjectName("ScoreBanner")
            b_lay = QtWidgets.QVBoxLayout(banner)
            b_lay.setContentsMargins(10, 8, 10, 8)
            b_lay.setSpacing(5)

            top_row = QtWidgets.QHBoxLayout()
            self.lbl_live_score = QtWidgets.QLabel("0 pts")
            self.lbl_live_score.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")

            self.lbl_live_grade = QtWidgets.QLabel("Grade -")
            self.lbl_live_grade.setStyleSheet(
                "background-color: #1e293b; color: #94a3b8; border-radius: 4px; "
                "padding: 3px 8px; font-weight: 800; font-size: 12px;"
            )

            self.lbl_live_tier = QtWidgets.QLabel("Workload: Waiting...")
            self.lbl_live_tier.setStyleSheet("color: #cbd5e1; font-weight: 600;")

            top_row.addWidget(self.lbl_live_score)
            top_row.addSpacing(8)
            top_row.addWidget(self.lbl_live_grade)
            top_row.addSpacing(8)
            top_row.addWidget(self.lbl_live_tier)
            top_row.addStretch(1)
            b_lay.addLayout(top_row)

            sub_row = QtWidgets.QHBoxLayout()
            self.lbl_live_sub = QtWidgets.QLabel("VP: -  |  Rust: -  |  Apply: Pending  |  Stab: -")
            self.lbl_live_sub.setStyleSheet(
                "font-family: 'Consolas', monospace; font-size: 11px; color: #93c5fd;"
            )
            self.lbl_live_alert = QtWidgets.QLabel("[INFO] Waiting for Cut Events")
            self.lbl_live_alert.setStyleSheet("font-size: 11px; font-weight: 600; color: #93c5fd;")
            sub_row.addWidget(self.lbl_live_sub)
            sub_row.addStretch(1)
            sub_row.addWidget(self.lbl_live_alert)
            b_lay.addLayout(sub_row)
            lay.addWidget(banner)

            # 1. System & Engine Info
            sys_box = QtWidgets.QGroupBox("System & Engine Status")
            sys_layout = QtWidgets.QGridLayout(sys_box)
            self.lbl_engine = QtWidgets.QLabel("Engine: -")
            self.lbl_cpu = QtWidgets.QLabel("Blender CPU: -")
            self.lbl_ram = QtWidgets.QLabel("Blender RAM: -")
            self.lbl_total = QtWidgets.QLabel("Last Eval: Waiting...")
            self.lbl_total.setStyleSheet("font-size: 12px; font-weight: bold; color: #34d399;")
            sys_layout.addWidget(self.lbl_engine, 0, 0, 1, 2)
            sys_layout.addWidget(self.lbl_cpu, 1, 0)
            sys_layout.addWidget(self.lbl_ram, 1, 1)
            sys_layout.addWidget(self.lbl_total, 2, 0, 1, 2)
            lay.addWidget(sys_box)

            # 2. Live Timeline Stacked Bar Graph
            graph_box = QtWidgets.QGroupBox("Live Timeline Graph (Last 80 Events)")
            g_layout = QtWidgets.QVBoxLayout(graph_box)
            g_layout.setContentsMargins(6, 6, 6, 6)
            self.graph = TimelineGraphWidget()
            g_layout.addWidget(self.graph)
            lay.addWidget(graph_box)

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
            for col_idx in range(6):
                hdr.setSectionResizeMode(col_idx, QtWidgets.QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
            t_layout.addWidget(self.table)
            lay.addWidget(tbl_box, 2)

            # 4. Compact AI Summary Preview
            sum_box = QtWidgets.QGroupBox("AI Timeline & Score Summary (Ready to Paste)")
            s_layout = QtWidgets.QVBoxLayout(sum_box)
            self.txt_report = QtWidgets.QPlainTextEdit()
            self.txt_report.setReadOnly(True)
            self.txt_report.setMaximumHeight(125)
            s_layout.addWidget(self.txt_report)
            lay.addWidget(sum_box, 1)

            # 5. Export Buttons
            btn_row = QtWidgets.QHBoxLayout()
            self.btn_copy = QtWidgets.QPushButton("Copy Timeline + Score (AI)")
            self.btn_copy.setObjectName("CopyBtn")
            self.btn_copy.clicked.connect(self._on_copy_report)

            self.btn_save = QtWidgets.QPushButton("Save Full Timeline (.json)")
            self.btn_save.clicked.connect(self._on_save_report)

            btn_row.addWidget(self.btn_copy)
            btn_row.addWidget(self.btn_save)
            lay.addLayout(btn_row)

        def _make_sub_card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel]:
            card = QtWidgets.QFrame()
            card.setObjectName("SubCard")
            c_lay = QtWidgets.QVBoxLayout(card)
            c_lay.setContentsMargins(8, 6, 8, 6)
            c_lay.setSpacing(2)
            lbl_t = QtWidgets.QLabel(title)
            lbl_t.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
            lbl_v = QtWidgets.QLabel("-")
            lbl_v.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 800;")
            lbl_d = QtWidgets.QLabel("-")
            lbl_d.setStyleSheet("color: #93c5fd; font-family: 'Consolas', monospace; font-size: 10px;")
            c_lay.addWidget(lbl_t)
            c_lay.addWidget(lbl_v)
            c_lay.addWidget(lbl_d)
            return card, lbl_v, lbl_d

        def _build_compare_tab(self) -> None:
            lay = QtWidgets.QVBoxLayout(self.tab_compare)
            lay.setSpacing(6)

            # 1. Detailed Target (B) Benchmark Breakdown
            bd_box = QtWidgets.QGroupBox("Benchmark Breakdown (Target B)")
            bd_lay = QtWidgets.QVBoxLayout(bd_box)
            bd_lay.setSpacing(6)

            hdr_row = QtWidgets.QHBoxLayout()
            self.lbl_cmp_total = QtWidgets.QLabel("0 pts")
            self.lbl_cmp_total.setStyleSheet("font-size: 24px; font-weight: 800; color: #38bdf8;")
            self.lbl_cmp_grade = QtWidgets.QLabel("Grade -")
            self.lbl_cmp_grade.setStyleSheet(
                "background-color: #1e293b; color: #94a3b8; border-radius: 4px; "
                "padding: 4px 10px; font-weight: 800;"
            )
            self.lbl_cmp_meta = QtWidgets.QLabel("Workload: -")
            self.lbl_cmp_meta.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            hdr_row.addWidget(self.lbl_cmp_total)
            hdr_row.addSpacing(8)
            hdr_row.addWidget(self.lbl_cmp_grade)
            hdr_row.addSpacing(8)
            hdr_row.addWidget(self.lbl_cmp_meta)
            hdr_row.addStretch(1)
            bd_lay.addLayout(hdr_row)

            grid = QtWidgets.QGridLayout()
            grid.setSpacing(6)
            c_vp, self.lbl_vp_pts, self.lbl_vp_det = self._make_sub_card("Viewport / Writeback")
            c_ru, self.lbl_ru_pts, self.lbl_ru_det = self._make_sub_card("Rust Engine & Cadence")
            c_ap, self.lbl_ap_pts, self.lbl_ap_det = self._make_sub_card("Apply Throughput")
            c_st, self.lbl_st_pts, self.lbl_st_det = self._make_sub_card("Stability & Memory")
            grid.addWidget(c_vp, 0, 0)
            grid.addWidget(c_ru, 0, 1)
            grid.addWidget(c_ap, 1, 0)
            grid.addWidget(c_st, 1, 1)
            bd_lay.addLayout(grid)

            # Bottleneck Diagnostics List
            self.lst_alerts = QtWidgets.QListWidget()
            self.lst_alerts.setMaximumHeight(92)
            bd_lay.addWidget(self.lst_alerts)
            lay.addWidget(bd_box)

            # 2. A/B Comparison Controls + Table
            cmp_box = QtWidgets.QGroupBox("A/B Performance Comparison (Baseline A vs Target B)")
            cmp_lay = QtWidgets.QVBoxLayout(cmp_box)
            cmp_lay.setSpacing(6)

            sel_grid = QtWidgets.QGridLayout()
            sel_grid.addWidget(QtWidgets.QLabel("Baseline (A):"), 0, 0)
            self.cmb_base = QtWidgets.QComboBox()
            self.cmb_base.currentIndexChanged.connect(self._recompute_comparison)
            btn_browse_a = QtWidgets.QPushButton("Browse...")
            btn_browse_a.clicked.connect(self._on_browse_baseline)
            btn_refresh_files = QtWidgets.QPushButton("↻")
            btn_refresh_files.setToolTip("Refresh saved timeline files from logs/")
            btn_refresh_files.setFixedWidth(32)
            btn_refresh_files.clicked.connect(lambda: self._refresh_saved_file_combos(select_latest_baseline=False))
            sel_grid.addWidget(self.cmb_base, 0, 1)
            sel_grid.addWidget(btn_browse_a, 0, 2)
            sel_grid.addWidget(btn_refresh_files, 0, 3)

            sel_grid.addWidget(QtWidgets.QLabel("Target (B):"), 1, 0)
            self.cmb_target = QtWidgets.QComboBox()
            self.cmb_target.currentIndexChanged.connect(self._recompute_comparison)
            btn_browse_b = QtWidgets.QPushButton("Browse...")
            btn_browse_b.clicked.connect(self._on_browse_target)
            sel_grid.addWidget(self.cmb_target, 1, 1)
            sel_grid.addWidget(btn_browse_b, 1, 2, 1, 2)
            cmp_lay.addLayout(sel_grid)

            # Delta Headline Banner
            delta_frame = QtWidgets.QFrame()
            delta_frame.setObjectName("DeltaBanner")
            d_lay = QtWidgets.QHBoxLayout(delta_frame)
            d_lay.setContentsMargins(10, 6, 10, 6)
            self.lbl_delta = QtWidgets.QLabel("Select a Baseline (.json) to compare before/after performance.")
            self.lbl_delta.setStyleSheet("font-size: 13px; font-weight: 700; color: #94a3b8;")
            d_lay.addWidget(self.lbl_delta)
            d_lay.addStretch(1)
            cmp_lay.addWidget(delta_frame)

            # Side-by-side comparison table
            self.tbl_cmp = QtWidgets.QTableWidget(0, 4)
            self.tbl_cmp.setHorizontalHeaderLabels(["Metric", "Baseline (A)", "Target (B)", "Change"])
            self.tbl_cmp.setAlternatingRowColors(True)
            self.tbl_cmp.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.tbl_cmp.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.tbl_cmp.verticalHeader().setVisible(False)
            self.tbl_cmp.verticalHeader().setDefaultSectionSize(21)
            chdr = self.tbl_cmp.horizontalHeader()
            chdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            chdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            chdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            chdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            cmp_lay.addWidget(self.tbl_cmp, 1)

            cmp_btns = QtWidgets.QHBoxLayout()
            self.btn_copy_cmp = QtWidgets.QPushButton("Copy Comparison Summary (AI)")
            self.btn_copy_cmp.setObjectName("CopyBtn")
            self.btn_copy_cmp.clicked.connect(self._on_copy_comparison)
            cmp_btns.addWidget(self.btn_copy_cmp)
            cmp_lay.addLayout(cmp_btns)

            lay.addWidget(cmp_box, 1)

        def _set_grade_badge(self, lbl: QtWidgets.QLabel, grade: str) -> None:
            bg, fg = _GRADE_COLORS.get(grade, _GRADE_COLORS["-"])
            lbl.setText(f"Grade {grade}")
            lbl.setStyleSheet(
                f"background-color: {bg}; color: {fg}; border-radius: 4px; "
                f"padding: 3px 8px; font-weight: 800; font-size: 12px;"
            )

        def _update_live_banner(self, score: perf_score.PerformanceScoreReport) -> None:
            self.lbl_live_score.setText(f"{score.total_score:,} pts")
            self._set_grade_badge(self.lbl_live_grade, score.smoothness_grade)
            if score.target_polys > 0:
                self.lbl_live_tier.setText(
                    f"{score.workload_tier} • {score.target_polys:,} polys • {score.num_cutters}c  |  "
                    f"Preview {score.effective_preview_hz:.1f} Hz"
                )
            else:
                self.lbl_live_tier.setText("Workload: Waiting for cut...")

            ap_str = f"{score.apply_score:,}" if score.apply_score is not None else "Pending"
            self.lbl_live_sub.setText(
                f"VP: {score.viewport_score:,}  |  Rust: {score.rust_score:,}  |  "
                f"Apply: {ap_str}  |  Stab: {score.stability_score:,}"
            )
            if score.alerts:
                top_al = score.alerts[0]
                col = _ALERT_COLORS.get(top_al.severity, "#cbd5e1")
                self.lbl_live_alert.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {col};")
                self.lbl_live_alert.setText(f"[{top_al.severity}] {top_al.title}")

        def _update_breakdown_cards(self, score: perf_score.PerformanceScoreReport) -> None:
            self.lbl_cmp_total.setText(f"{score.total_score:,} pts")
            self._set_grade_badge(self.lbl_cmp_grade, score.smoothness_grade)
            if score.target_polys > 0:
                self.lbl_cmp_meta.setText(
                    f"{score.workload_tier} • {score.target_polys:,} polys • {score.num_cutters}c "
                    f"(W={score.workload_units:,.1f}) • {score.steady_eval_count} live frames"
                )
            else:
                self.lbl_cmp_meta.setText("No cut evaluations recorded yet.")

            self.lbl_vp_pts.setText(f"{score.viewport_score:,} pts")
            self.lbl_vp_det.setText(
                f"P50 {score.wb_p50_ms:.1f}ms | P95 {score.wb_p95_ms:.1f}ms (~{score.main_fps_p50:.0f} FPS)"
                if score.steady_eval_count > 0
                else "Waiting for live drag..."
            )

            self.lbl_ru_pts.setText(f"{score.rust_score:,} pts")
            self.lbl_ru_det.setText(
                f"P50 {score.rust_p50_ms:.1f}ms | {score.effective_preview_hz:.1f} Hz "
                f"(cad {score.cadence_efficiency * 100:.0f}%)"
                if score.steady_eval_count > 0
                else "Waiting for live drag..."
            )

            if score.apply_score is not None and score.apply_done_ms is not None:
                self.lbl_ap_pts.setText(f"{score.apply_score:,} pts")
                post_s = (
                    f" | Post {score.apply_postprocess_ms / 1000.0:.2f}s"
                    if score.apply_postprocess_ms is not None
                    else ""
                )
                self.lbl_ap_det.setText(f"Total {score.apply_done_ms / 1000.0:.2f}s{post_s}")
            else:
                self.lbl_ap_pts.setText("Pending")
                self.lbl_ap_det.setText("Press Apply to measure bake throughput")

            self.lbl_st_pts.setText(f"{score.stability_score:,} pts")
            self.lbl_st_det.setText(
                f"Peak RAM {score.peak_ram_mb:,.0f} MB | Spikes {score.spike_count}/{score.steady_eval_count}"
            )

            self.lst_alerts.clear()
            for al in score.alerts:
                item = QtWidgets.QListWidgetItem(f"[{al.severity}] {al.title} — {al.detail}")
                item.setForeground(QtGui.QColor(_ALERT_COLORS.get(al.severity, "#e2e8f0")))
                self.lst_alerts.addItem(item)

        def _refresh_saved_file_combos(self, select_latest_baseline: bool = False) -> None:
            files = perf_score.list_saved_timeline_files(profiler.get_logs_dir())
            prev_base = self.cmb_base.currentData()
            prev_target = self.cmb_target.currentData()

            self.cmb_base.blockSignals(True)
            self.cmb_target.blockSignals(True)
            try:
                self.cmb_base.clear()
                self.cmb_base.addItem("(None — Select Baseline .json)", "")
                for p in files:
                    self.cmb_base.addItem(os.path.basename(p), p)
                # Preserve externally browsed files
                for p in list(self._loaded_reports.keys()):
                    if p not in files and os.path.isfile(p):
                        self.cmb_base.addItem(os.path.basename(p), p)

                self.cmb_target.clear()
                self.cmb_target.addItem("● Current Live Session", _LIVE_SENTINEL)
                for p in files:
                    self.cmb_target.addItem(os.path.basename(p), p)
                for p in list(self._loaded_reports.keys()):
                    if p not in files and os.path.isfile(p):
                        self.cmb_target.addItem(os.path.basename(p), p)

                # Restore selections
                if prev_base:
                    idx = self.cmb_base.findData(prev_base)
                    if idx >= 0:
                        self.cmb_base.setCurrentIndex(idx)
                elif select_latest_baseline and files:
                    # Pick newest saved .json (preferring timeline_*.json over timeline_latest.jsonl)
                    json_files = [p for p in files if p.lower().endswith(".json")]
                    pick = json_files[0] if json_files else files[0]
                    idx = self.cmb_base.findData(pick)
                    if idx >= 0:
                        self.cmb_base.setCurrentIndex(idx)

                if prev_target:
                    idx = self.cmb_target.findData(prev_target)
                    if idx >= 0:
                        self.cmb_target.setCurrentIndex(idx)
            finally:
                self.cmb_base.blockSignals(False)
                self.cmb_target.blockSignals(False)

            self._recompute_comparison()

        def _load_report_cached(self, path: str) -> dict | None:
            if not path:
                return None
            if path == _LIVE_SENTINEL:
                return profiler.build_report_dict(include_score=False)
            if path in self._loaded_reports:
                return self._loaded_reports[path]
            try:
                rep = perf_score.load_report_file(path)
                self._loaded_reports[path] = rep
                return rep
            except Exception as exc:
                self.lbl_status.setText(f"Failed to load {os.path.basename(path)}: {exc}")
                return None

        def _add_and_select_file(self, combo: QtWidgets.QComboBox, path: str) -> None:
            if not path:
                return
            # Force reload from disk on explicit browse
            self._loaded_reports.pop(path, None)
            if self._load_report_cached(path) is None:
                return
            idx = combo.findData(path)
            if idx < 0:
                combo.addItem(os.path.basename(path), path)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)

        def _on_browse_baseline(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Baseline Timeline (.json / .jsonl)",
                profiler.get_logs_dir(),
                "Timeline Files (*.json *.jsonl);;All Files (*)",
            )
            if path:
                self._add_and_select_file(self.cmb_base, path)

        def _on_browse_target(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Target Timeline (.json / .jsonl)",
                profiler.get_logs_dir(),
                "Timeline Files (*.json *.jsonl);;All Files (*)",
            )
            if path:
                self._add_and_select_file(self.cmb_target, path)

        def _recompute_comparison(self) -> None:
            target_key = self.cmb_target.currentData() or _LIVE_SENTINEL
            base_key = self.cmb_base.currentData() or ""

            target_rep = self._load_report_cached(target_key)
            if target_rep is None:
                return
            target_score = (
                profiler.get_current_score()
                if target_key == _LIVE_SENTINEL
                else perf_score.compute_score_from_report(target_rep)
            )
            self._update_breakdown_cards(target_score)

            if not base_key:
                self._last_comparison = None
                self.lbl_delta.setStyleSheet("font-size: 12px; font-weight: 600; color: #94a3b8;")
                self.lbl_delta.setText("Select a Baseline (.json) above to compare before/after performance.")
                self.tbl_cmp.setRowCount(0)
                return

            base_rep = self._load_report_cached(base_key)
            if base_rep is None:
                return

            base_label = os.path.basename(base_key)
            target_label = "Live Session" if target_key == _LIVE_SENTINEL else os.path.basename(target_key)
            cmp = perf_score.compare_reports(base_rep, target_rep, base_label, target_label)
            self._last_comparison = cmp

            pct_str = f"{cmp.pct_total_score:+.1f}%" if cmp.pct_total_score is not None else "N/A"
            if cmp.delta_total_score > 0:
                d_col = "#34d399"
            elif cmp.delta_total_score < 0:
                d_col = "#f87171"
            else:
                d_col = "#93c5fd"
            self.lbl_delta.setStyleSheet(f"font-size: 13px; font-weight: 800; color: {d_col};")
            self.lbl_delta.setText(
                f"Total Score: {cmp.baseline.total_score:,} → {cmp.current.total_score:,} pts  "
                f"({cmp.delta_total_score:+,} pts / {pct_str})"
            )

            self.tbl_cmp.setRowCount(0)
            for m in cmp.metrics:
                if m.before is None and m.after is None:
                    continue
                r = self.tbl_cmp.rowCount()
                self.tbl_cmp.insertRow(r)
                b_txt = _fmt_cmp_val(m.before, m.unit)
                a_txt = _fmt_cmp_val(m.after, m.unit)
                if m.pct_change is not None:
                    arrow = "▲" if m.improved is True else ("▼" if m.improved is False else "•")
                    c_txt = f"{arrow} {m.pct_change:+.1f}%"
                else:
                    c_txt = "-"

                vals = [m.label, b_txt, a_txt, c_txt]
                for col, text in enumerate(vals):
                    item = QtWidgets.QTableWidgetItem(text)
                    if col in (1, 2, 3):
                        item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    if col == 3 and m.improved is not None:
                        item.setForeground(QtGui.QColor("#34d399" if m.improved else "#f87171"))
                    if r == 0:
                        f = item.font()
                        f.setBold(True)
                        item.setFont(f)
                    self.tbl_cmp.setItem(r, col, item)

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
                    self.lbl_total.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color};")
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

                    live_score = profiler.get_current_score()
                    self._update_live_banner(live_score)
                    if (self.cmb_target.currentData() or _LIVE_SENTINEL) == _LIVE_SENTINEL:
                        self._recompute_comparison()
            except Exception as exc:
                self.lbl_status.setText(f"Refresh error: {exc}")

        def _on_copy_report(self) -> None:
            text = profiler.format_timeline_summary_for_ai()
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            self.lbl_status.setText("Copied Timeline + Benchmark Score! กด Ctrl+V วางส่งให้ AI ได้เลยครับ")

        def _on_copy_comparison(self) -> None:
            if self._last_comparison is None:
                text = perf_score.format_score_summary_text(profiler.get_current_score())
            else:
                text = perf_score.format_comparison_text(self._last_comparison)
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            self.lbl_status.setText("Copied A/B Benchmark Comparison! กด Ctrl+V วางได้เลยครับ")

        def _on_save_report(self) -> None:
            try:
                path = profiler.save_report_file()
                self._refresh_saved_file_combos(select_latest_baseline=False)
                self.lbl_status.setText(f"Saved full timeline + score: {path}")
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
