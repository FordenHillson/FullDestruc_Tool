"""Automated tests for FullDestruc_Tool uncapped Benchmark Performance Score & A/B Comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_synthetic_report(
    *,
    polys: int,
    cutters: int,
    steady_frames: int,
    wb_ms: float,
    rust_ms: float,
    dt_ms: float,
    apply_eval_ms: float | None = None,
    apply_done_ms: float | None = None,
    enter_ms: float = 0.0,
    peak_ram_mb: float = 1500.0,
    solver: str = "RUST",
    dll_ok: bool = True,
) -> dict:
    tl = []
    seq = 0
    t = 1.0
    seq += 1
    tl.append(
        {
            "seq": seq,
            "elapsed_sec": round(t, 3),
            "clock_str": "12:00:01.000",
            "kind": "MARKER",
            "action": "ENTER_SESSION",
            "note": "TestTarget",
            "solver": solver,
            "target_name": "",
            "target_polys": 0,
            "num_cutters": 0,
            "out_polys": 0,
            "upload_ms": 0.0,
            "extract_ms": 0.0,
            "rust_ms": 0.0,
            "writeback_ms": 0.0,
            "total_ms": enter_ms,
            "ram_mb": peak_ram_mb * 0.5,
            "cpu_pct": 10.0,
            "is_spike": enter_ms >= 50.0,
        }
    )

    # Initial cold upload frame
    t += 0.2
    seq += 1
    tl.append(
        {
            "seq": seq,
            "elapsed_sec": round(t, 3),
            "clock_str": "12:00:01.200",
            "kind": "EVAL",
            "action": "EVAL",
            "note": "",
            "solver": solver,
            "target_name": "Target",
            "target_polys": polys,
            "num_cutters": cutters,
            "out_polys": polys,
            "upload_ms": 15.0,
            "extract_ms": 0.0,
            "rust_ms": rust_ms * 2.0,
            "writeback_ms": wb_ms * 8.0,
            "total_ms": wb_ms * 8.0,
            "ram_mb": peak_ram_mb * 0.8,
            "cpu_pct": 25.0,
            "is_spike": True,
        }
    )

    # Steady live drag frames
    for _ in range(steady_frames):
        t += dt_ms / 1000.0
        seq += 1
        tl.append(
            {
                "seq": seq,
                "elapsed_sec": round(t, 3),
                "clock_str": "12:00:02.000",
                "kind": "EVAL",
                "action": "EVAL",
                "note": "",
                "solver": solver,
                "target_name": "Target",
                "target_polys": polys,
                "num_cutters": cutters,
                "out_polys": polys,
                "upload_ms": 0.0,
                "extract_ms": 0.0,
                "rust_ms": rust_ms,
                "writeback_ms": wb_ms,
                "total_ms": wb_ms,
                "ram_mb": peak_ram_mb * 0.9,
                "cpu_pct": 30.0,
                "is_spike": wb_ms >= 50.0,
            }
        )

    if apply_eval_ms is not None:
        t += 0.5
        seq += 1
        tl.append(
            {
                "seq": seq,
                "elapsed_sec": round(t, 3),
                "clock_str": "12:00:05.000",
                "kind": "EVAL",
                "action": "APPLY_EVAL",
                "note": "Target",
                "solver": solver,
                "target_name": "Target",
                "target_polys": polys,
                "num_cutters": cutters,
                "out_polys": polys,
                "upload_ms": 0.0,
                "extract_ms": 0.0,
                "rust_ms": apply_eval_ms * 0.15,
                "writeback_ms": apply_eval_ms * 0.85,
                "total_ms": apply_eval_ms,
                "ram_mb": peak_ram_mb,
                "cpu_pct": 40.0,
                "is_spike": True,
            }
        )

    if apply_done_ms is not None:
        t += 0.5
        seq += 1
        tl.append(
            {
                "seq": seq,
                "elapsed_sec": round(t, 3),
                "clock_str": "12:00:06.000",
                "kind": "MARKER",
                "action": "APPLY_DONE",
                "note": "2 baked pieces",
                "solver": solver,
                "target_name": "",
                "target_polys": 0,
                "num_cutters": 0,
                "out_polys": 0,
                "upload_ms": 0.0,
                "extract_ms": 0.0,
                "rust_ms": 0.0,
                "writeback_ms": 0.0,
                "total_ms": apply_done_ms,
                "ram_mb": peak_ram_mb * 0.7,
                "cpu_pct": 20.0,
                "is_spike": True,
            }
        )

    return {
        "system": {
            "os": "Windows 11",
            "blender_version": "5.2.0 LTS",
            "cpu_cores": 24,
            "process_cpu_pct": 12.0,
            "process_ram_mb": peak_ram_mb * 0.7,
            "process_peak_ram_mb": peak_ram_mb,
        },
        "engine": {
            "selected_solver": solver,
            "rust_dll_available": dll_ok,
        },
        "timeline": tl,
    }


def main() -> None:
    # Register lightweight package stubs (avoids triggering Blender-only addon/__init__.py)
    addon_pkg = types.ModuleType("addon")
    addon_pkg.__path__ = [str(REPO_ROOT / "addon")]
    core_pkg = types.ModuleType("addon.core")
    core_pkg.__path__ = [str(REPO_ROOT / "addon" / "core")]
    sys.modules["addon"] = addon_pkg
    sys.modules["addon.core"] = core_pkg

    # Load perf_score directly (zero bpy dependency!)
    from addon.core import perf_score

    print("=" * 72)
    print("  FullDestruc_Tool Benchmark Performance Score & Compare Unit Tests")
    print("=" * 72)

    # 1. Edge Case: Empty timeline
    empty_s = perf_score.compute_score_from_report({"system": {}, "engine": {}, "timeline": []})
    assert empty_s.total_score == 0
    assert empty_s.smoothness_grade == "-"
    assert empty_s.apply_score is None
    assert empty_s.alerts and empty_s.alerts[0].code == "PENDING"
    print("[PASS] 1. Empty timeline handled cleanly (0 pts, Grade '-', PENDING alert)")

    # 2. Uncapped Benchmark Scaling (10s -> 1,000,000+ pts)
    r_tiny = _make_synthetic_report(polys=6, cutters=1, steady_frames=20, wb_ms=1.0, rust_ms=0.5, dt_ms=16.6, apply_eval_ms=8.0, apply_done_ms=15.0, peak_ram_mb=400.0)
    r_low = _make_synthetic_report(polys=10_000, cutters=5, steady_frames=20, wb_ms=2.0, rust_ms=1.5, dt_ms=16.6, apply_eval_ms=50.0, apply_done_ms=80.0, peak_ram_mb=800.0)
    r_mid = _make_synthetic_report(polys=100_000, cutters=20, steady_frames=20, wb_ms=5.0, rust_ms=4.0, dt_ms=16.6, apply_eval_ms=250.0, apply_done_ms=400.0, peak_ram_mb=1800.0)
    r_stress = _make_synthetic_report(polys=998_784, cutters=500, steady_frames=30, wb_ms=3.18, rust_ms=2.03, dt_ms=16.6, apply_eval_ms=1200.0, apply_done_ms=1800.0, peak_ram_mb=4200.0)

    s_tiny = perf_score.compute_score_from_report(r_tiny)
    s_low = perf_score.compute_score_from_report(r_low)
    s_mid = perf_score.compute_score_from_report(r_mid)
    s_stress = perf_score.compute_score_from_report(r_stress)

    assert 10 <= s_tiny.total_score < 200, f"Unexpected tiny score: {s_tiny.total_score}"
    assert 5_000 <= s_low.total_score < 50_000, f"Unexpected low score: {s_low.total_score}"
    assert 30_000 <= s_mid.total_score < 200_000, f"Unexpected mid score: {s_mid.total_score}"
    assert s_stress.total_score >= 1_000_000, f"Expected 1M+ stress score, got {s_stress.total_score}"
    assert s_tiny.smoothness_grade == "S" and s_stress.smoothness_grade == "S"
    assert s_stress.alerts[0].code == "OK"
    print(
        f"[PASS] 2. Uncapped scale verified: Tiny={s_tiny.total_score:,} | Low={s_low.total_score:,} | "
        f"Mid={s_mid.total_score:,} | 1M+500 Stress={s_stress.total_score:,} pts (Grade S)"
    )

    # 3. Live-only session (no Apply yet) re-normalizes cleanly
    r_live_only = _make_synthetic_report(polys=100_000, cutters=10, steady_frames=15, wb_ms=4.0, rust_ms=3.0, dt_ms=16.6)
    s_live = perf_score.compute_score_from_report(r_live_only)
    assert s_live.apply_score is None
    assert s_live.total_score > 0
    assert "Apply=Pending" in perf_score.format_score_summary_text(s_live)
    print(f"[PASS] 3. Live-only (no Apply) session verified: {s_live.total_score:,} pts, Apply=Pending")

    # 4. Synthetic v2.1 vs v2.1.1 4M-tri Suzanne Diagnostics & A/B Compare
    r_v21 = _make_synthetic_report(
        polys=2_015_232, cutters=1, steady_frames=38, wb_ms=26.0, rust_ms=52.0, dt_ms=1105.0,
        apply_eval_ms=3896.0, apply_done_ms=15731.0, peak_ram_mb=6716.6,
    )
    r_v211 = _make_synthetic_report(
        polys=2_015_232, cutters=1, steady_frames=96, wb_ms=29.0, rust_ms=105.0, dt_ms=105.0,
        apply_eval_ms=2978.0, apply_done_ms=4471.0, peak_ram_mb=5372.3,
    )
    cmp_syn = perf_score.compare_reports(r_v21, r_v211, "v2.1", "v2.1.1")
    codes_v21 = {a.code for a in cmp_syn.baseline.alerts}
    codes_v211 = {a.code for a in cmp_syn.current.alerts}
    assert "POLL_STARVATION" in codes_v21, f"Expected POLL_STARVATION in v2.1, got {codes_v21}"
    assert "SLOW_APPLY_POSTPROCESS" in codes_v21, f"Expected SLOW_APPLY_POSTPROCESS in v2.1, got {codes_v21}"
    assert "POLL_STARVATION" not in codes_v211, f"v2.1.1 should NOT trigger POLL_STARVATION: {codes_v211}"
    assert "SLOW_APPLY_POSTPROCESS" not in codes_v211, f"v2.1.1 should NOT trigger SLOW_APPLY_POSTPROCESS: {codes_v211}"
    assert cmp_syn.delta_total_score > 0 and (cmp_syn.pct_total_score or 0.0) > 30.0
    assert cmp_syn.baseline.smoothness_grade == "F"
    assert cmp_syn.current.smoothness_grade in ("A", "B")
    print(
        f"[PASS] 4. Bottleneck diagnostics & A/B compare verified: "
        f"v2.1 ({cmp_syn.baseline.total_score:,}, Grade {cmp_syn.baseline.smoothness_grade}) -> "
        f"v2.1.1 ({cmp_syn.current.total_score:,}, Grade {cmp_syn.current.smoothness_grade}) "
        f"[{cmp_syn.delta_total_score:+,} pts / {cmp_syn.pct_total_score:+.1f}%]"
    )

    # 5. JSON & JSONL file loading round-trip
    with tempfile.TemporaryDirectory() as td:
        p_json = os.path.join(td, "timeline_test.json")
        p_jsonl = os.path.join(td, "timeline_latest.jsonl")
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(r_mid, f)
        with open(p_jsonl, "w", encoding="utf-8") as f:
            for ev in r_mid["timeline"]:
                f.write(json.dumps(ev) + "\n")
        sc_j, _ = perf_score.compute_score_from_file(p_json)
        sc_jl, _ = perf_score.compute_score_from_file(p_jsonl)
        assert sc_j.total_score == s_mid.total_score
        assert sc_jl.total_score > 0 and sc_jl.steady_eval_count == s_mid.steady_eval_count
        listed = perf_score.list_saved_timeline_files(td)
        assert len(listed) == 2
    print("[PASS] 5. JSON & JSONL file loading + directory listing verified")

    # 6. Profiler integration check (mock minimal bpy)
    mock_bpy = types.ModuleType("bpy")
    mock_bpy.app = types.SimpleNamespace(version_string="5.2.0 LTS")
    mock_bpy.context = types.SimpleNamespace(scene=None)
    sys.modules["bpy"] = mock_bpy
    mock_gp = types.ModuleType("addon.core.gpu_preview")
    mock_gp.get_material_debug_info = lambda: []
    mock_rb = types.ModuleType("addon.core.rust_bridge")
    mock_rb.is_available = lambda: True
    mock_rb.get_last_error = lambda: ""
    sys.modules["addon.core.gpu_preview"] = mock_gp
    sys.modules["addon.core.rust_bridge"] = mock_rb

    from addon.core import profiler

    profiler.start_new_session_log()
    for i in range(6):
        profiler.record_eval(
            profiler.EvalSample(
                timestamp=10.0 + i * 0.02,
                session_id="T1",
                target_name="Wall",
                solver="RUST",
                target_verts=10_000,
                target_polys=10_000,
                num_cutters=4,
                cutter_total_polys=24,
                out_rem_verts=9_500,
                out_rem_polys=9_500,
                out_pieces_verts=500,
                out_pieces_polys=500,
                target_uploaded=(i == 0),
                upload_ms=5.0 if i == 0 else 0.0,
                extract_ms=0.0,
                rust_ms=2.0,
                writeback_ms=2.5,
                total_ms=2.5,
            )
        )
    rep_live = profiler.build_report_dict()
    assert "benchmark_score" in rep_live and rep_live["benchmark_score"]["total_score"] > 0
    ai_txt = profiler.format_timeline_summary_for_ai()
    assert "--- Benchmark Performance Score ---" in ai_txt
    assert "Total Score:" in ai_txt
    print(f"[PASS] 6. profiler.py integration verified (live score={rep_live['benchmark_score']['total_score']:,} pts)")

    # 7. Real local user logs (if present in workspace)
    real_a = Path("/home/antigravity/timeline_20261127_155214.json")
    real_b = Path("/home/antigravity/timeline_20261127_161832.json")
    if real_a.is_file() and real_b.is_file():
        _, rep_a = perf_score.compute_score_from_file(str(real_a))
        _, rep_b = perf_score.compute_score_from_file(str(real_b))
        cmp_real = perf_score.compare_reports(rep_a, rep_b, real_a.name, real_b.name)
        assert cmp_real.delta_total_score > 0
        print("[PASS] 7. Real user v2.1 vs v2.1.1 timeline comparison:")
        print(perf_score.format_comparison_text(cmp_real))

    print("\n" + "=" * 72)
    print("  ALL PERF_SCORE TESTS PASSED!")
    print("=" * 72)


if __name__ == "__main__":
    main()
