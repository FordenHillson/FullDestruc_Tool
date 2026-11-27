"""Timeline performance profiler and crash-safe JSONL logger for FullDestruc_Tool.

Tracks per-stage boolean timing (Target Upload, Cutter Extract, Rust Compute,
Blender Writeback), action tags (ENTER_SESSION, ADD_BOX, LIVE_DRAG, APPLY, etc.),
and Blender process CPU/RAM usage using built-in Windows ctypes APIs.

Every recorded event is immediately appended & flushed to:
  FullDestruc_Tool/logs/timeline_latest.jsonl
so diagnostics survive even if Blender crashes or hangs.
"""

from __future__ import annotations

import collections
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import platform
import time

import bpy

MAX_TIMELINE_EVENTS = 400
SPIKE_THRESHOLD_MS = 50.0


@dataclass
class EvalSample:
    timestamp: float
    session_id: str
    target_name: str
    solver: str
    target_verts: int
    target_polys: int
    num_cutters: int
    cutter_total_polys: int
    out_rem_verts: int
    out_rem_polys: int
    out_pieces_verts: int
    out_pieces_polys: int
    target_uploaded: bool
    upload_ms: float
    extract_ms: float
    rust_ms: float
    writeback_ms: float
    total_ms: float


@dataclass
class TimelineEvent:
    seq: int
    elapsed_sec: float
    clock_str: str
    kind: str  # "EVAL" or "MARKER"
    action: str
    note: str
    solver: str
    target_name: str
    target_polys: int
    num_cutters: int
    out_polys: int
    upload_ms: float
    extract_ms: float
    rust_ms: float
    writeback_ms: float
    total_ms: float
    ram_mb: float
    cpu_pct: float
    is_spike: bool


_RECORDING: bool = True
_TIMELINE_START: float = time.perf_counter()
_SEQ: int = 0
_HISTORY: collections.deque[EvalSample] = collections.deque(maxlen=MAX_TIMELINE_EVENTS)
_TIMELINE: collections.deque[TimelineEvent] = collections.deque(maxlen=MAX_TIMELINE_EVENTS)
_ACTION_STACK: list[tuple[str, str]] = []
_LOG_INITIALIZED: bool = False

# ---------------------------------------------------------------------------
# Windows Process CPU & Memory Stats via ctypes (no psutil needed)
# ---------------------------------------------------------------------------

_LAST_PROC_CPU_TIME: float | None = None
_LAST_WALL_TIME: float | None = None
_LAST_CPU_PCT: float = 0.0


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_uint32),
        ("dwHighDateTime", ctypes.c_uint32),
    ]


def _filetime_to_seconds(ft: _FILETIME) -> float:
    ticks = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    return ticks * 1e-7


import atexit

_LAST_STATS_TIME: float = 0.0
_CACHED_SYS_STATS: dict | None = None
_STATS_MIN_INTERVAL_SEC: float = 0.25


def get_system_stats(force: bool = False) -> dict:
    """Return current Blender process RAM (MB), CPU (%), logical cores, and Blender version."""
    global _LAST_PROC_CPU_TIME, _LAST_WALL_TIME, _LAST_CPU_PCT, _LAST_STATS_TIME, _CACHED_SYS_STATS
    now = time.perf_counter()
    if not force and _CACHED_SYS_STATS is not None and (now - _LAST_STATS_TIME) < _STATS_MIN_INTERVAL_SEC:
        return _CACHED_SYS_STATS

    ram_mb = 0.0
    peak_ram_mb = 0.0
    cpu_pct = _LAST_CPU_PCT
    cores = os.cpu_count() or 1

    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            k32.GetProcessTimes.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            k32.GetProcessTimes.restype = ctypes.c_int

            hproc = k32.GetCurrentProcess()

            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(hproc, ctypes.byref(pmc), pmc.cb):
                ram_mb = pmc.WorkingSetSize / (1024.0 * 1024.0)
                peak_ram_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0)

            ft_create = _FILETIME()
            ft_exit = _FILETIME()
            ft_kernel = _FILETIME()
            ft_user = _FILETIME()
            if k32.GetProcessTimes(
                hproc,
                ctypes.byref(ft_create),
                ctypes.byref(ft_exit),
                ctypes.byref(ft_kernel),
                ctypes.byref(ft_user),
            ):
                proc_t = _filetime_to_seconds(ft_kernel) + _filetime_to_seconds(ft_user)
                wall_t = now
                if _LAST_PROC_CPU_TIME is not None and _LAST_WALL_TIME is not None:
                    dt_wall = wall_t - _LAST_WALL_TIME
                    if dt_wall >= 0.10:
                        dt_proc = max(0.0, proc_t - _LAST_PROC_CPU_TIME)
                        cpu_pct = min(100.0, (dt_proc / (dt_wall * cores)) * 100.0)
                        _LAST_CPU_PCT = cpu_pct
                        _LAST_PROC_CPU_TIME = proc_t
                        _LAST_WALL_TIME = wall_t
                else:
                    _LAST_PROC_CPU_TIME = proc_t
                    _LAST_WALL_TIME = wall_t
        except Exception:
            pass

    stats = {
        "os": f"{platform.system()} {platform.release()}",
        "blender_version": bpy.app.version_string,
        "cpu_cores": cores,
        "process_cpu_pct": round(cpu_pct, 1),
        "process_ram_mb": round(ram_mb, 1),
        "process_peak_ram_mb": round(peak_ram_mb, 1),
    }
    _CACHED_SYS_STATS = stats
    _LAST_STATS_TIME = now
    return stats


# ---------------------------------------------------------------------------
# Buffered Crash-Safe JSONL Auto-Logging (Zero Main-Thread Stalls)
# ---------------------------------------------------------------------------

_PENDING_JSONL: list[str] = []
_LAST_FLUSH_TIME: float = 0.0
_FLUSH_INTERVAL_SEC: float = 1.0
_FLUSH_BATCH_SIZE: int = 64


def get_logs_dir() -> str:
    addon_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(addon_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_latest_log_path() -> str:
    return os.path.join(get_logs_dir(), "timeline_latest.jsonl")


def flush_log() -> None:
    global _LOG_INITIALIZED, _LAST_FLUSH_TIME
    if not _PENDING_JSONL:
        return
    lines = list(_PENDING_JSONL)
    _PENDING_JSONL.clear()
    _LAST_FLUSH_TIME = time.perf_counter()
    try:
        path = get_latest_log_path()
        mode = "a" if _LOG_INITIALIZED else "w"
        _LOG_INITIALIZED = True
        with open(path, mode, encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
    except Exception:
        pass


atexit.register(flush_log)


def _append_jsonl(event: TimelineEvent, immediate: bool = False) -> None:
    try:
        _PENDING_JSONL.append(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        now = time.perf_counter()
        if (
            immediate
            or len(_PENDING_JSONL) >= _FLUSH_BATCH_SIZE
            or (now - _LAST_FLUSH_TIME) >= _FLUSH_INTERVAL_SEC
        ):
            flush_log()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recording Controls & Action Tagging
# ---------------------------------------------------------------------------


def is_recording() -> bool:
    return _RECORDING


def set_recording(enabled: bool) -> None:
    global _RECORDING
    _RECORDING = bool(enabled)


def start_new_session_log() -> None:
    """Clear in-memory timeline and start a fresh timeline_latest.jsonl file."""
    global _TIMELINE_START, _SEQ, _LOG_INITIALIZED
    _PENDING_JSONL.clear()
    _HISTORY.clear()
    _TIMELINE.clear()
    _TIMELINE_START = time.perf_counter()
    _SEQ = 0
    _LOG_INITIALIZED = False


@contextmanager
def action_scope(action: str, note: str = ""):
    """Tag all Rust evaluations inside this block with `action`."""
    _ACTION_STACK.append((action, note))
    try:
        yield
    finally:
        if _ACTION_STACK:
            _ACTION_STACK.pop()


def _current_action() -> tuple[str, str]:
    if _ACTION_STACK:
        return _ACTION_STACK[-1]
    return ("EVAL", "")


def record_marker(action: str, note: str = "", duration_ms: float = 0.0) -> None:
    """Record a standalone lifecycle/UI marker on the timeline (e.g. ENTER_SESSION, APPLY_DONE)."""
    global _SEQ
    if not _RECORDING:
        return
    now_perf = time.perf_counter()
    sys_s = get_system_stats(force=True)
    _SEQ += 1
    ev = TimelineEvent(
        seq=_SEQ,
        elapsed_sec=round(now_perf - _TIMELINE_START, 3),
        clock_str=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        kind="MARKER",
        action=action,
        note=note,
        solver=_get_active_solver(),
        target_name="",
        target_polys=0,
        num_cutters=0,
        out_polys=0,
        upload_ms=0.0,
        extract_ms=0.0,
        rust_ms=0.0,
        writeback_ms=0.0,
        total_ms=round(float(duration_ms), 2),
        ram_mb=float(sys_s["process_ram_mb"]),
        cpu_pct=float(sys_s["process_cpu_pct"]),
        is_spike=duration_ms >= SPIKE_THRESHOLD_MS,
    )
    _TIMELINE.append(ev)
    _append_jsonl(ev, immediate=True)


def record_eval(sample: EvalSample) -> None:
    """Record a Rust/Boolean evaluation sample and append it to the Timeline."""
    global _SEQ
    if not _RECORDING:
        return
    _HISTORY.append(sample)
    action, note = _current_action()
    sys_s = get_system_stats()
    _SEQ += 1
    ev = TimelineEvent(
        seq=_SEQ,
        elapsed_sec=round(sample.timestamp - _TIMELINE_START, 3),
        clock_str=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        kind="EVAL",
        action=action,
        note=note,
        solver=sample.solver,
        target_name=sample.target_name,
        target_polys=sample.target_polys,
        num_cutters=sample.num_cutters,
        out_polys=sample.out_rem_polys + sample.out_pieces_polys,
        upload_ms=round(sample.upload_ms, 2),
        extract_ms=round(sample.extract_ms, 2),
        rust_ms=round(sample.rust_ms, 2),
        writeback_ms=round(sample.writeback_ms, 2),
        total_ms=round(sample.total_ms, 2),
        ram_mb=float(sys_s["process_ram_mb"]),
        cpu_pct=float(sys_s["process_cpu_pct"]),
        is_spike=sample.total_ms >= SPIKE_THRESHOLD_MS,
    )
    _TIMELINE.append(ev)
    _append_jsonl(ev)


def clear_history() -> None:
    start_new_session_log()


def get_history() -> list[EvalSample]:
    return list(_HISTORY)


def get_timeline() -> list[TimelineEvent]:
    return list(_TIMELINE)


def get_latest() -> EvalSample | None:
    return _HISTORY[-1] if _HISTORY else None


def _get_active_solver() -> str:
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "fdt_settings", None) if scene else None
    return str(getattr(settings, "boolean_solver", "UNKNOWN")) if settings else "UNKNOWN"


# ---------------------------------------------------------------------------
# Reports for AI & JSON Export
# ---------------------------------------------------------------------------


def build_report_dict() -> dict:
    from . import gpu_preview, rust_bridge

    sys_stats = get_system_stats()
    history = get_history()
    timeline = get_timeline()
    latest = history[-1] if history else None

    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "fdt_settings", None) if scene else None
    active_solver = getattr(settings, "boolean_solver", "UNKNOWN") if settings else "UNKNOWN"
    zone_grid = bool(getattr(settings, "show_zone_grid", False)) if settings else False

    # Group evals by action
    by_action: dict[str, list[TimelineEvent]] = {}
    for ev in timeline:
        by_action.setdefault(ev.action, []).append(ev)

    action_stats = {}
    for act, evs in by_action.items():
        eval_evs = [e for e in evs if e.kind == "EVAL" or e.total_ms > 0.0]
        if eval_evs:
            action_stats[act] = {
                "count": len(evs),
                "avg_total_ms": round(sum(e.total_ms for e in eval_evs) / len(eval_evs), 2),
                "max_total_ms": round(max(e.total_ms for e in eval_evs), 2),
                "avg_rust_ms": round(sum(e.rust_ms for e in eval_evs) / len(eval_evs), 2),
                "avg_writeback_ms": round(sum(e.writeback_ms for e in eval_evs) / len(eval_evs), 2),
            }
        else:
            action_stats[act] = {"count": len(evs)}

    top_spikes = sorted(
        [e for e in timeline if e.total_ms > 0.0],
        key=lambda e: e.total_ms,
        reverse=True,
    )[:5]

    return {
        "system": sys_stats,
        "engine": {
            "selected_solver": active_solver,
            "rust_dll_available": rust_bridge.is_available(),
            "rust_dll_error": rust_bridge.get_last_error(),
            "zone_grid_enabled": zone_grid,
            "auto_log_path": get_latest_log_path(),
            "gpu_materials": gpu_preview.get_material_debug_info(),
        },
        "action_summary": action_stats,
        "top_5_slowest": [asdict(e) for e in top_spikes],
        "latest": asdict(latest) if latest else None,
        "timeline": [asdict(e) for e in timeline],
    }


def format_timeline_summary_for_ai() -> str:
    """Create a compact Timeline summary grouped by Action + Top 5 slowest spikes."""
    rep = build_report_dict()
    sys_s = rep["system"]
    eng = rep["engine"]
    timeline = get_timeline()

    lines = [
        "=== FullDestruc_Tool Timeline Debug Summary ===",
        f"System: {sys_s['os']} | Blender {sys_s['blender_version']} | CPU Cores: {sys_s['cpu_cores']}",
        f"Process: CPU {sys_s['process_cpu_pct']}% | RAM {sys_s['process_ram_mb']} MB (Peak {sys_s['process_peak_ram_mb']} MB)",
        f"Engine: {eng['selected_solver']} | Rust DLL OK: {eng['rust_dll_available']} | ZoneGrid: {eng['zone_grid_enabled']}",
        f"Total Events: {len(timeline)} | Auto-Log: {eng['auto_log_path']}",
    ]
    if eng.get("gpu_materials"):
        lines.append("--- GPU Materials ---")
        for gm in eng["gpu_materials"][:8]:
            lines.append(f"  {gm}")

    if timeline:
        ram_min = min(e.ram_mb for e in timeline if e.ram_mb > 0) if any(e.ram_mb > 0 for e in timeline) else 0.0
        ram_max = max(e.ram_mb for e in timeline)
        lines.append(f"RAM Range: {ram_min:.0f} MB -> {ram_max:.0f} MB")

    lines.append("--- By Action ---")
    for act, st in rep["action_summary"].items():
        if "avg_total_ms" in st:
            lines.append(
                f"  {act} ({st['count']}x): Avg={st['avg_total_ms']}ms (Max={st['max_total_ms']}ms) | "
                f"Rust={st['avg_rust_ms']}ms | Write={st['avg_writeback_ms']}ms"
            )
        else:
            lines.append(f"  {act} ({st['count']}x)")

    spikes = rep["top_5_slowest"]
    if spikes:
        lines.append("--- Top 5 Slowest Events ---")
        for sp in spikes:
            lines.append(
                f"  [{sp['clock_str']}] {sp['action']}: Total={sp['total_ms']}ms "
                f"(Up={sp['upload_ms']} Ex={sp['extract_ms']} Rust={sp['rust_ms']} Wr={sp['writeback_ms']}) "
                f"| Polys={sp['target_polys']}->{sp['out_polys']} | Cutters={sp['num_cutters']} | RAM={sp['ram_mb']:.0f}MB"
            )

    if timeline:
        lines.append("--- Last 8 Timeline Events ---")
        for ev in timeline[-8:]:
            note_part = f" ({ev.note})" if ev.note else ""
            if ev.kind == "EVAL":
                lines.append(
                    f"  +{ev.elapsed_sec:6.2f}s | {ev.action:14s} | {ev.total_ms:6.2f}ms "
                    f"(Rust {ev.rust_ms:.2f} / Wr {ev.writeback_ms:.2f}) | RAM {ev.ram_mb:.0f}MB{note_part}"
                )
            else:
                dur = f" {ev.total_ms:.1f}ms" if ev.total_ms > 0 else ""
                lines.append(
                    f"  +{ev.elapsed_sec:6.2f}s | {ev.action:14s} | MARKER{dur} | RAM {ev.ram_mb:.0f}MB{note_part}"
                )

    return "\n".join(lines)


def format_report_text() -> str:
    return format_timeline_summary_for_ai()


def save_report_file() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(get_logs_dir(), f"timeline_{ts}.json")
    rep = build_report_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    return out_path
