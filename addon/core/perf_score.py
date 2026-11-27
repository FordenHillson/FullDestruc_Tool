"""Uncapped Benchmark Performance Score & A/B Comparison Engine for FullDestruc_Tool.

Pure Python (zero `bpy` or `PySide6` dependency) so it runs identically inside
Blender's live Debug Monitor, exported JSON/JSONL timeline reports, and automated
test suites.

Scoring Model (Cinebench / 3DMark style — uncapped, scales from ~10 to 1,000,000+ pts):
- Workload Weight `W = (polys / 1000.0) * (1.0 + 0.20 * sqrt(cutters))`
- 4 Uncapped Sub-Scores:
  1. `viewport_score`  : Main-thread / GPU writeback responsiveness
  2. `rust_score`      : Rust boolean compute throughput x live streaming cadence
  3. `apply_score`     : Full mesh bake + post-process throughput (None if Apply not run yet)
  4. `stability_score` : Frame pacing (P50/P95), cadence health, spike resistance & RAM efficiency
- Smoothness Grade (`S`/`A`/`B`/`C`/`D`/`F`) & Automatic Bottleneck Diagnostics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import statistics
from typing import Any


SCORE_ENGINE_VERSION = "1.0"
ACTIVE_DRAG_MAX_GAP_MS = 2500.0
SPIKE_MS = 50.0


@dataclass
class BottleneckAlert:
    code: str
    severity: str  # "OK", "INFO", "WARN", "CRITICAL"
    title: str
    detail: str


@dataclass
class PerformanceScoreReport:
    version: str = SCORE_ENGINE_VERSION
    total_score: int = 0
    viewport_score: int = 0
    rust_score: int = 0
    apply_score: int | None = None
    stability_score: int = 0

    workload_units: float = 0.0
    workload_tier: str = "None"
    target_polys: int = 0
    num_cutters: int = 0

    smoothness_grade: str = "-"
    total_events: int = 0
    steady_eval_count: int = 0

    wb_p50_ms: float = 0.0
    wb_p95_ms: float = 0.0
    wb_avg_ms: float = 0.0
    main_fps_p50: float = 0.0

    rust_p50_ms: float = 0.0
    rust_p95_ms: float = 0.0
    rust_avg_ms: float = 0.0

    active_dt_p50_ms: float = 0.0
    effective_preview_hz: float = 0.0
    cadence_efficiency: float = 1.0

    enter_ms: float | None = None
    apply_eval_ms: float | None = None
    apply_done_ms: float | None = None
    apply_postprocess_ms: float | None = None

    peak_ram_mb: float = 0.0
    spike_count: int = 0
    spike_ratio: float = 0.0

    solver: str = "UNKNOWN"
    rust_dll_available: bool = True
    alerts: list[BottleneckAlert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "total_score": self.total_score,
            "viewport_score": self.viewport_score,
            "rust_score": self.rust_score,
            "apply_score": self.apply_score,
            "stability_score": self.stability_score,
            "workload_units": round(self.workload_units, 2),
            "workload_tier": self.workload_tier,
            "target_polys": self.target_polys,
            "num_cutters": self.num_cutters,
            "smoothness_grade": self.smoothness_grade,
            "total_events": self.total_events,
            "steady_eval_count": self.steady_eval_count,
            "wb_p50_ms": round(self.wb_p50_ms, 2),
            "wb_p95_ms": round(self.wb_p95_ms, 2),
            "wb_avg_ms": round(self.wb_avg_ms, 2),
            "main_fps_p50": round(self.main_fps_p50, 1),
            "rust_p50_ms": round(self.rust_p50_ms, 2),
            "rust_p95_ms": round(self.rust_p95_ms, 2),
            "rust_avg_ms": round(self.rust_avg_ms, 2),
            "active_dt_p50_ms": round(self.active_dt_p50_ms, 1),
            "effective_preview_hz": round(self.effective_preview_hz, 2),
            "cadence_efficiency": round(self.cadence_efficiency, 3),
            "enter_ms": round(self.enter_ms, 2) if self.enter_ms is not None else None,
            "apply_eval_ms": round(self.apply_eval_ms, 2) if self.apply_eval_ms is not None else None,
            "apply_done_ms": round(self.apply_done_ms, 2) if self.apply_done_ms is not None else None,
            "apply_postprocess_ms": (
                round(self.apply_postprocess_ms, 2) if self.apply_postprocess_ms is not None else None
            ),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "spike_count": self.spike_count,
            "spike_ratio": round(self.spike_ratio, 3),
            "solver": self.solver,
            "rust_dll_available": self.rust_dll_available,
            "alerts": [asdict(a) for a in self.alerts],
        }


@dataclass
class ComparisonMetric:
    key: str
    label: str
    before: float | None
    after: float | None
    unit: str
    higher_is_better: bool
    delta: float | None = None
    pct_change: float | None = None
    improved: bool | None = None


@dataclass
class ComparisonReport:
    baseline_label: str
    current_label: str
    baseline: PerformanceScoreReport
    current: PerformanceScoreReport
    delta_total_score: int
    pct_total_score: float | None
    metrics: list[ComparisonMetric] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "current_label": self.current_label,
            "delta_total_score": self.delta_total_score,
            "pct_total_score": round(self.pct_total_score, 2) if self.pct_total_score is not None else None,
            "baseline": self.baseline.to_dict(),
            "current": self.current.to_dict(),
            "metrics": [asdict(m) for m in self.metrics],
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return float(s[idx])


def _workload_tier(polys: int) -> str:
    if polys <= 0:
        return "None"
    if polys < 1_000:
        return "Tiny"
    if polys < 50_000:
        return "Low"
    if polys < 500_000:
        return "Mid"
    if polys < 1_500_000:
        return "High"
    return "Ultra"


def _compute_smoothness_grade(
    steady_count: int,
    wb_p50_ms: float,
    wb_p95_ms: float,
    cadence_eff: float,
) -> str:
    if steady_count <= 0:
        return "-"
    if cadence_eff < 0.20:
        return "F"
    if wb_p50_ms <= 16.6 and wb_p95_ms <= 25.0 and cadence_eff >= 0.85:
        return "S"
    if wb_p50_ms <= 20.0 and cadence_eff >= 0.75:
        return "A"
    if wb_p50_ms <= 33.3 and cadence_eff >= 0.65:
        return "B"
    if wb_p50_ms <= 50.0 and cadence_eff >= 0.40:
        return "C"
    if wb_p50_ms <= 100.0:
        return "D"
    return "F"


def _get_field(ev: Any, name: str, default: Any = 0) -> Any:
    if isinstance(ev, dict):
        return ev.get(name, default)
    return getattr(ev, name, default)


def _detect_bottlenecks(
    *,
    steady_count: int,
    wb_p50_ms: float,
    wb_p95_ms: float,
    rust_p50_ms: float,
    active_dt_p50_ms: float,
    cadence_eff: float,
    enter_ms: float | None,
    apply_done_ms: float | None,
    apply_post_ms: float | None,
    peak_ram_mb: float,
    expected_ram_mb: float,
    solver: str,
    rust_dll_available: bool,
) -> list[BottleneckAlert]:
    alerts: list[BottleneckAlert] = []

    if not rust_dll_available or (solver not in ("RUST", "UNKNOWN", "") and "RUST" not in solver.upper()):
        alerts.append(
            BottleneckAlert(
                code="FALLBACK_SOLVER",
                severity="CRITICAL",
                title="Rust Engine Inactive",
                detail=f"Active solver={solver}, rust_dll_available={rust_dll_available} — falling back to slower solver.",
            )
        )

    if steady_count >= 2 and active_dt_p50_ms > 350.0 and cadence_eff < 0.35:
        alerts.append(
            BottleneckAlert(
                code="POLL_STARVATION",
                severity="CRITICAL",
                title="Live Drag Poll Starvation",
                detail=(
                    f"Preview updates only every {active_dt_p50_ms:.0f} ms "
                    f"({1000.0 / max(active_dt_p50_ms, 1.0):.2f} Hz) despite Rust P50={rust_p50_ms:.1f} ms "
                    f"(cadence={cadence_eff * 100:.0f}%)."
                ),
            )
        )

    if apply_done_ms is not None and apply_post_ms is not None:
        if apply_post_ms > 2000.0 and (apply_done_ms <= 0.0 or apply_post_ms >= 0.45 * apply_done_ms):
            alerts.append(
                BottleneckAlert(
                    code="SLOW_APPLY_POSTPROCESS",
                    severity="CRITICAL" if apply_post_ms > 5000.0 else "WARN",
                    title="Slow Apply Post-Process",
                    detail=(
                        f"Post-eval Apply took {apply_post_ms / 1000.0:.2f}s out of "
                        f"{apply_done_ms / 1000.0:.2f}s total (check custom normals / mesh copy)."
                    ),
                )
            )
        elif apply_done_ms > 8000.0:
            alerts.append(
                BottleneckAlert(
                    code="SLOW_APPLY_TOTAL",
                    severity="WARN",
                    title="Heavy Apply Duration",
                    detail=f"Total Apply took {apply_done_ms / 1000.0:.2f}s.",
                )
            )

    if enter_ms is not None and enter_ms > 100.0:
        alerts.append(
            BottleneckAlert(
                code="SLOW_ENTER_SESSION",
                severity="CRITICAL" if enter_ms > 1000.0 else "WARN",
                title="Blocking Cut Mode Startup",
                detail=f"ENTER_SESSION blocked Main Thread for {enter_ms:.0f} ms.",
            )
        )

    if steady_count > 0:
        if wb_p50_ms > 33.3:
            alerts.append(
                BottleneckAlert(
                    code="HIGH_WRITEBACK_STALL",
                    severity="CRITICAL",
                    title="Main-Thread Writeback Stall",
                    detail=f"Steady writeback P50={wb_p50_ms:.1f} ms (P95={wb_p95_ms:.1f} ms) — below 30 FPS budget.",
                )
            )
        elif wb_p95_ms > 33.3:
            alerts.append(
                BottleneckAlert(
                    code="HIGH_WRITEBACK_STALL",
                    severity="WARN",
                    title="Writeback P95 Spikes",
                    detail=f"Steady writeback P50={wb_p50_ms:.1f} ms, P95={wb_p95_ms:.1f} ms (occasional >33ms frame).",
                )
            )

        if rust_p50_ms > 50.0:
            alerts.append(
                BottleneckAlert(
                    code="RUST_COMPUTE_BOUND",
                    severity="WARN",
                    title="Heavy Rust Compute",
                    detail=f"Rust P50={rust_p50_ms:.1f} ms on live drag (limits background preview cadence).",
                )
            )

    if peak_ram_mb > 0.0 and (peak_ram_mb > 6000.0 or peak_ram_mb > expected_ram_mb * 1.4):
        alerts.append(
            BottleneckAlert(
                code="HIGH_MEMORY_PRESSURE",
                severity="WARN",
                title="High Peak RAM Usage",
                detail=f"Peak RAM reached {peak_ram_mb:,.0f} MB (expected <= {expected_ram_mb:,.0f} MB).",
            )
        )

    if not alerts:
        if steady_count > 0 or apply_done_ms is not None:
            alerts.append(
                BottleneckAlert(
                    code="OK",
                    severity="OK",
                    title="Smooth & Balanced",
                    detail="No major bottlenecks detected in this session.",
                )
            )
        else:
            alerts.append(
                BottleneckAlert(
                    code="PENDING",
                    severity="INFO",
                    title="Waiting for Cut Events",
                    detail="Move a cutter or press Apply to generate a full benchmark score.",
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Public Scoring & File Loading API
# ---------------------------------------------------------------------------


def compute_score_from_report(report_dict: dict[str, Any]) -> PerformanceScoreReport:
    """Compute an uncapped Benchmark Performance Score from a profiler report dict."""
    timeline = report_dict.get("timeline") or []
    sys_info = report_dict.get("system") or {}
    eng_info = report_dict.get("engine") or {}

    solver = str(eng_info.get("selected_solver", "UNKNOWN") or "UNKNOWN")
    rust_dll_ok = bool(eng_info.get("rust_dll_available", True))

    if not timeline:
        return PerformanceScoreReport(
            solver=solver,
            rust_dll_available=rust_dll_ok,
            alerts=_detect_bottlenecks(
                steady_count=0,
                wb_p50_ms=0.0,
                wb_p95_ms=0.0,
                rust_p50_ms=0.0,
                active_dt_p50_ms=0.0,
                cadence_eff=1.0,
                enter_ms=None,
                apply_done_ms=None,
                apply_post_ms=None,
                peak_ram_mb=float(sys_info.get("process_peak_ram_mb") or 0.0),
                expected_ram_mb=1200.0,
                solver=solver,
                rust_dll_available=rust_dll_ok,
            ),
        )

    # Extract peak target workload across all EVAL events
    eval_all = [ev for ev in timeline if _get_field(ev, "kind", "") == "EVAL"]
    polys = max((int(_get_field(ev, "target_polys", 0) or 0) for ev in eval_all), default=0)
    cutters = max((int(_get_field(ev, "num_cutters", 0) or 0) for ev in eval_all), default=0)

    # Fallback: check `latest` if timeline had markers only
    latest = report_dict.get("latest")
    if polys <= 0 and isinstance(latest, dict):
        polys = int(latest.get("target_polys") or 0)
        cutters = max(cutters, int(latest.get("num_cutters") or 0))

    # Separate live drag/preview evals from APPLY_EVAL
    live_evals = [ev for ev in eval_all if str(_get_field(ev, "action", "")) != "APPLY_EVAL"]
    # Exclude initial cold target upload frame when multiple live evals exist
    if len(live_evals) > 1:
        first_ev = live_evals[0]
        first_up = float(_get_field(first_ev, "upload_ms", 0.0) or 0.0)
        first_wb = float(_get_field(first_ev, "writeback_ms", 0.0) or 0.0)
        rest_wbs = [float(_get_field(ev, "writeback_ms", 0.0) or 0.0) for ev in live_evals[1:]]
        med_rest_wb = statistics.median(rest_wbs) if rest_wbs else 0.0
        if first_up > 0.0 or (med_rest_wb > 0.0 and first_wb > med_rest_wb * 2.5) or len(live_evals) >= 3:
            steady_evals = live_evals[1:]
        else:
            steady_evals = live_evals
    else:
        steady_evals = live_evals

    # Use main-thread cost: prefer total_ms when present, fallback to writeback_ms
    wb_vals: list[float] = []
    rust_vals: list[float] = []
    for ev in steady_evals:
        wb = float(_get_field(ev, "writeback_ms", 0.0) or 0.0)
        tot = float(_get_field(ev, "total_ms", 0.0) or 0.0)
        ru = float(_get_field(ev, "rust_ms", 0.0) or 0.0)
        # In async GPU delta mode `total_ms == writeback_ms`; in sync mode `total_ms` is full main-thread block
        wb_vals.append(tot if tot > 0.0 else wb)
        rust_vals.append(ru)

    wb_p50 = _percentile(wb_vals, 0.50)
    wb_p95 = _percentile(wb_vals, 0.95)
    wb_avg = (sum(wb_vals) / len(wb_vals)) if wb_vals else 0.0
    main_fps_p50 = (1000.0 / max(wb_p50, 0.1)) if wb_vals else 0.0

    rust_p50 = _percentile(rust_vals, 0.50)
    rust_p95 = _percentile(rust_vals, 0.95)
    rust_avg = (sum(rust_vals) / len(rust_vals)) if rust_vals else 0.0

    # Active drag inter-event cadence (ignore idle pauses > 2.5s between separate drag gestures)
    active_dts: list[float] = []
    for i in range(1, len(steady_evals)):
        t_prev = float(_get_field(steady_evals[i - 1], "elapsed_sec", 0.0) or 0.0)
        t_cur = float(_get_field(steady_evals[i], "elapsed_sec", 0.0) or 0.0)
        dt_ms = (t_cur - t_prev) * 1000.0
        if 0.0 < dt_ms <= ACTIVE_DRAG_MAX_GAP_MS:
            active_dts.append(dt_ms)

    active_dt_p50 = _percentile(active_dts, 0.50) if active_dts else 0.0
    effective_hz = (1000.0 / active_dt_p50) if active_dt_p50 > 0.0 else (
        1000.0 / max(rust_p50 + wb_p50, 1.0) if steady_evals else 0.0
    )

    expected_dt = max(33.3, (rust_p50 + wb_p50) * 1.25)
    cadence_eff = min(1.0, expected_dt / max(active_dt_p50, 1.0)) if active_dt_p50 > 0.0 else 1.0

    # Lifecycle markers: ENTER_SESSION, APPLY_EVAL, APPLY_DONE
    enter_vals = [
        float(_get_field(ev, "total_ms", 0.0) or 0.0)
        for ev in timeline
        if str(_get_field(ev, "action", "")) == "ENTER_SESSION"
    ]
    enter_ms = max(enter_vals) if enter_vals else None

    apply_eval_vals = [
        float(_get_field(ev, "total_ms", 0.0) or 0.0)
        for ev in timeline
        if str(_get_field(ev, "action", "")) == "APPLY_EVAL"
    ]
    apply_done_vals = [
        float(_get_field(ev, "total_ms", 0.0) or 0.0)
        for ev in timeline
        if str(_get_field(ev, "action", "")) == "APPLY_DONE" and float(_get_field(ev, "total_ms", 0.0) or 0.0) > 0.0
    ]
    # Use the latest Apply measurement (most representative if user re-applied)
    apply_eval_ms = apply_eval_vals[-1] if apply_eval_vals else None
    apply_done_ms = apply_done_vals[-1] if apply_done_vals else None
    apply_post_ms = (
        max(0.0, apply_done_ms - apply_eval_ms)
        if (apply_done_ms is not None and apply_eval_ms is not None)
        else None
    )

    # Peak RAM & spikes
    sys_peak = float(sys_info.get("process_peak_ram_mb") or 0.0)
    ev_peak = max((float(_get_field(ev, "ram_mb", 0.0) or 0.0) for ev in timeline), default=0.0)
    peak_ram_mb = max(sys_peak, ev_peak)

    spike_count = sum(1 for v in wb_vals if v >= SPIKE_MS)
    spike_ratio = (spike_count / len(wb_vals)) if wb_vals else 0.0

    # Compute Workload Weight W (uncapped, grows with polys & cutters)
    eff_polys = max(polys, 1)
    eff_cutters = max(cutters, 1)
    workload_units = (eff_polys / 1000.0) * (1.0 + 0.20 * math.sqrt(eff_cutters)) if polys > 0 else 0.0
    tier = _workload_tier(polys)
    expected_ram = 1500.0 + (eff_polys / 1000.0) * 2.0 + eff_cutters * 1.5

    # Compute the 4 uncapped sub-scores
    if polys > 0 and steady_evals:
        main_eff = max(0.7 * wb_p50 + 0.3 * wb_p95, 0.25)
        rust_eff = max(0.7 * rust_p50 + 0.3 * rust_p95, 0.25)

        viewport_f = workload_units * (1000.0 / main_eff) * 1.5
        rust_f = workload_units * (1000.0 / rust_eff) * (0.25 + 0.75 * cadence_eff) * 1.5

        pacing = min(1.0, max(wb_p50, 0.1) / max(wb_p95, 0.1))
        mem_eff = min(1.0, expected_ram / max(peak_ram_mb, 500.0)) if peak_ram_mb > 0.0 else 1.0
        q_stab = 0.35 * pacing + 0.30 * cadence_eff + 0.20 * (1.0 - 0.5 * spike_ratio) + 0.15 * mem_eff
        fps60_bonus = 1.25 if wb_p50 <= 16.6 else 1.0
        stability_f = workload_units * 40.0 * q_stab * fps60_bonus
    else:
        viewport_f = 0.0
        rust_f = 0.0
        stability_f = 0.0

    apply_f: float | None = None
    if polys > 0 and apply_done_ms is not None and apply_done_ms > 0.0:
        apply_f = workload_units * (1000.0 / max(apply_done_ms, 10.0)) * 250.0

    # Combine into uncapped Total Benchmark Score
    if steady_evals and apply_f is not None:
        total_f = 0.30 * viewport_f + 0.30 * rust_f + 0.25 * apply_f + 0.15 * stability_f
    elif steady_evals:
        total_f = 0.40 * viewport_f + 0.40 * rust_f + 0.20 * stability_f
    elif apply_f is not None:
        total_f = apply_f
    else:
        total_f = 0.0

    grade = _compute_smoothness_grade(len(steady_evals), wb_p50, wb_p95, cadence_eff)
    alerts = _detect_bottlenecks(
        steady_count=len(steady_evals),
        wb_p50_ms=wb_p50,
        wb_p95_ms=wb_p95,
        rust_p50_ms=rust_p50,
        active_dt_p50_ms=active_dt_p50,
        cadence_eff=cadence_eff,
        enter_ms=enter_ms,
        apply_done_ms=apply_done_ms,
        apply_post_ms=apply_post_ms,
        peak_ram_mb=peak_ram_mb,
        expected_ram_mb=expected_ram,
        solver=solver,
        rust_dll_available=rust_dll_ok,
    )

    return PerformanceScoreReport(
        total_score=int(round(total_f)),
        viewport_score=int(round(viewport_f)),
        rust_score=int(round(rust_f)),
        apply_score=int(round(apply_f)) if apply_f is not None else None,
        stability_score=int(round(stability_f)),
        workload_units=workload_units,
        workload_tier=tier,
        target_polys=polys,
        num_cutters=cutters,
        smoothness_grade=grade,
        total_events=len(timeline),
        steady_eval_count=len(steady_evals),
        wb_p50_ms=wb_p50,
        wb_p95_ms=wb_p95,
        wb_avg_ms=wb_avg,
        main_fps_p50=main_fps_p50,
        rust_p50_ms=rust_p50,
        rust_p95_ms=rust_p95,
        rust_avg_ms=rust_avg,
        active_dt_p50_ms=active_dt_p50,
        effective_preview_hz=effective_hz,
        cadence_efficiency=cadence_eff,
        enter_ms=enter_ms,
        apply_eval_ms=apply_eval_ms,
        apply_done_ms=apply_done_ms,
        apply_postprocess_ms=apply_post_ms,
        peak_ram_mb=peak_ram_mb,
        spike_count=spike_count,
        spike_ratio=spike_ratio,
        solver=solver,
        rust_dll_available=rust_dll_ok,
        alerts=alerts,
    )


def load_report_file(file_path: str) -> dict[str, Any]:
    """Load either a saved `timeline_*.json` report or a crash-safe `timeline_latest.jsonl` file."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {"system": {}, "engine": {}, "timeline": []}

    if file_path.lower().endswith(".jsonl") or (raw.startswith("{") and "\n{" in raw and not raw.endswith("]}")):
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"system": {}, "engine": {}, "timeline": events}

    data = json.loads(raw)
    if isinstance(data, list):
        return {"system": {}, "engine": {}, "timeline": data}
    if isinstance(data, dict):
        return data
    return {"system": {}, "engine": {}, "timeline": []}


def compute_score_from_file(file_path: str) -> tuple[PerformanceScoreReport, dict[str, Any]]:
    """Load a `.json` or `.jsonl` timeline file and compute its Benchmark Score."""
    rep = load_report_file(file_path)
    return compute_score_from_report(rep), rep


def _build_cmp_metric(
    key: str,
    label: str,
    before: float | None,
    after: float | None,
    unit: str,
    higher_is_better: bool,
) -> ComparisonMetric:
    delta: float | None = None
    pct_change: float | None = None
    improved: bool | None = None
    if before is not None and after is not None:
        delta = after - before
        if abs(before) > 1e-6:
            pct_change = (delta / abs(before)) * 100.0
        elif abs(after) > 1e-6:
            pct_change = 100.0 if after > 0 else -100.0
        else:
            pct_change = 0.0
        if abs(delta) > 1e-6:
            improved = (delta > 0) if higher_is_better else (delta < 0)
    return ComparisonMetric(
        key=key,
        label=label,
        before=before,
        after=after,
        unit=unit,
        higher_is_better=higher_is_better,
        delta=delta,
        pct_change=pct_change,
        improved=improved,
    )


def compare_reports(
    baseline_dict: dict[str, Any],
    current_dict: dict[str, Any],
    baseline_label: str = "Baseline",
    current_label: str = "Current",
) -> ComparisonReport:
    """Compare two timeline reports (Baseline A vs Current B) across Benchmark Score and latencies."""
    base = compute_score_from_report(baseline_dict)
    curr = compute_score_from_report(current_dict)

    delta_total = curr.total_score - base.total_score
    pct_total: float | None = None
    if base.total_score > 0:
        pct_total = (delta_total / base.total_score) * 100.0
    elif curr.total_score > 0:
        pct_total = 100.0

    metrics = [
        _build_cmp_metric("total_score", "Total Benchmark Score", float(base.total_score), float(curr.total_score), "pts", True),
        _build_cmp_metric("viewport_score", "Viewport / Writeback Score", float(base.viewport_score), float(curr.viewport_score), "pts", True),
        _build_cmp_metric("rust_score", "Rust Engine Score", float(base.rust_score), float(curr.rust_score), "pts", True),
        _build_cmp_metric(
            "apply_score",
            "Apply Throughput Score",
            float(base.apply_score) if base.apply_score is not None else None,
            float(curr.apply_score) if curr.apply_score is not None else None,
            "pts",
            True,
        ),
        _build_cmp_metric("stability_score", "Stability & RAM Score", float(base.stability_score), float(curr.stability_score), "pts", True),
        _build_cmp_metric("wb_p50_ms", "Writeback P50 (Main)", base.wb_p50_ms if base.steady_eval_count else None, curr.wb_p50_ms if curr.steady_eval_count else None, "ms", False),
        _build_cmp_metric("wb_p95_ms", "Writeback P95 (Main)", base.wb_p95_ms if base.steady_eval_count else None, curr.wb_p95_ms if curr.steady_eval_count else None, "ms", False),
        _build_cmp_metric("rust_p50_ms", "Rust Compute P50", base.rust_p50_ms if base.steady_eval_count else None, curr.rust_p50_ms if curr.steady_eval_count else None, "ms", False),
        _build_cmp_metric("active_dt_p50_ms", "Active Drag Cadence P50", base.active_dt_p50_ms if base.active_dt_p50_ms > 0 else None, curr.active_dt_p50_ms if curr.active_dt_p50_ms > 0 else None, "ms", False),
        _build_cmp_metric("effective_preview_hz", "Effective Preview Rate", base.effective_preview_hz if base.effective_preview_hz > 0 else None, curr.effective_preview_hz if curr.effective_preview_hz > 0 else None, "Hz", True),
        _build_cmp_metric("apply_done_ms", "Apply Total Time", base.apply_done_ms, curr.apply_done_ms, "ms", False),
        _build_cmp_metric("apply_postprocess_ms", "Apply Post-Process Time", base.apply_postprocess_ms, curr.apply_postprocess_ms, "ms", False),
        _build_cmp_metric("peak_ram_mb", "Peak RAM", base.peak_ram_mb if base.peak_ram_mb > 0 else None, curr.peak_ram_mb if curr.peak_ram_mb > 0 else None, "MB", False),
    ]

    return ComparisonReport(
        baseline_label=baseline_label,
        current_label=current_label,
        baseline=base,
        current=curr,
        delta_total_score=delta_total,
        pct_total_score=pct_total,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Human / AI Summary Formatters
# ---------------------------------------------------------------------------


def format_score_summary_text(score: PerformanceScoreReport) -> str:
    """Format a compact, copy-pasteable Benchmark Score & Bottleneck summary."""
    apply_str = f"{score.apply_score:,} pts" if score.apply_score is not None else "Pending"
    lines = [
        "--- Benchmark Performance Score ---",
        (
            f"Total Score: {score.total_score:,} pts | Smoothness Grade: {score.smoothness_grade} | "
            f"Workload: {score.workload_tier} ({score.target_polys:,} polys, {score.num_cutters}c, W={score.workload_units:,.1f})"
        ),
        (
            f"Sub-Scores: Viewport={score.viewport_score:,} | Rust={score.rust_score:,} | "
            f"Apply={apply_str} | Stability={score.stability_score:,}"
        ),
    ]
    if score.steady_eval_count > 0:
        lines.append(
            f"Live Drag ({score.steady_eval_count}f): Writeback P50={score.wb_p50_ms:.2f}ms "
            f"(P95={score.wb_p95_ms:.2f}ms, ~{score.main_fps_p50:.0f} FPS) | "
            f"Rust P50={score.rust_p50_ms:.2f}ms (P95={score.rust_p95_ms:.2f}ms) | "
            f"Preview={score.effective_preview_hz:.1f} Hz (cadence={score.cadence_efficiency * 100:.0f}%)"
        )
    if score.apply_done_ms is not None:
        post_part = f" (Post={score.apply_postprocess_ms:.0f}ms)" if score.apply_postprocess_ms is not None else ""
        lines.append(f"Apply: Total={score.apply_done_ms:.1f}ms{post_part} | Peak RAM={score.peak_ram_mb:,.0f} MB")
    elif score.peak_ram_mb > 0.0:
        lines.append(f"Peak RAM: {score.peak_ram_mb:,.0f} MB | Spikes(>50ms): {score.spike_count}/{score.steady_eval_count}")

    if score.alerts:
        lines.append("Diagnostics:")
        for al in score.alerts:
            lines.append(f"  [{al.severity}] {al.title}: {al.detail}")
    return "\n".join(lines)


def _fmt_val(val: float | None, unit: str) -> str:
    if val is None:
        return "-"
    if unit == "pts":
        return f"{int(round(val)):,} pts"
    if unit == "MB":
        return f"{val:,.0f} MB"
    if unit == "Hz":
        return f"{val:.2f} Hz"
    return f"{val:,.2f} {unit}"


def format_comparison_text(cmp: ComparisonReport) -> str:
    """Format a copy-pasteable A/B Benchmark Comparison summary."""
    pct_str = f"{cmp.pct_total_score:+.1f}%" if cmp.pct_total_score is not None else "N/A"
    lines = [
        "=== FullDestruc_Tool Benchmark A/B Comparison ===",
        f"Baseline (A): {cmp.baseline_label}  ({cmp.baseline.target_polys:,} polys, {cmp.baseline.num_cutters}c, Grade {cmp.baseline.smoothness_grade})",
        f"Current  (B): {cmp.current_label}  ({cmp.current.target_polys:,} polys, {cmp.current.num_cutters}c, Grade {cmp.current.smoothness_grade})",
        f"Total Score : {cmp.baseline.total_score:,} -> {cmp.current.total_score:,} pts ({cmp.delta_total_score:+,} pts / {pct_str})",
        "--- Breakdown ---",
    ]
    for m in cmp.metrics:
        if m.before is None and m.after is None:
            continue
        b_s = _fmt_val(m.before, m.unit)
        a_s = _fmt_val(m.after, m.unit)
        if m.pct_change is not None:
            tag = "BETTER" if m.improved is True else ("SLOWER" if m.improved is False else "SAME")
            ch_s = f"{m.pct_change:+.1f}% [{tag}]"
        else:
            ch_s = "-"
        lines.append(f"  {m.label:26s}: {b_s:>14s} -> {a_s:>14s}  ({ch_s})")

    if cmp.current.alerts:
        lines.append("--- Current Diagnostics ---")
        for al in cmp.current.alerts:
            lines.append(f"  [{al.severity}] {al.title}: {al.detail}")
    return "\n".join(lines)


def list_saved_timeline_files(logs_dir: str) -> list[str]:
    """Return saved timeline `.json` / `.jsonl` file paths sorted newest-first."""
    if not logs_dir or not os.path.isdir(logs_dir):
        return []
    out: list[str] = []
    for name in os.listdir(logs_dir):
        low = name.lower()
        if low.endswith(".json") or low.endswith(".jsonl"):
            out.append(os.path.join(logs_dir, name))
    out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out
