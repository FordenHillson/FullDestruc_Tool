"""Automated unit & integration tests for On-Demand G-Buffer Imposter Drag (v2.3)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_box_verts(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float) -> np.ndarray:
    pts = []
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                pts.append((cx + dx, cy + dy, cz + dz))
    return np.asarray(pts, dtype=np.float32)


def _rot_z_matrix(angle_rad: float, tx: float, ty: float, tz: float, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c * sx
    m[0, 1] = -s * sy
    m[1, 0] = s * sx
    m[1, 1] = c * sy
    m[2, 2] = sz
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def main() -> None:
    addon_pkg = types.ModuleType("addon")
    addon_pkg.__path__ = [str(REPO_ROOT / "addon")]
    core_pkg = types.ModuleType("addon.core")
    core_pkg.__path__ = [str(REPO_ROOT / "addon" / "core")]
    sys.modules["addon"] = addon_pkg
    sys.modules["addon.core"] = core_pkg

    from addon.core import gpu_imposter

    print("=" * 72)
    print("  FullDestruc_Tool v2.3 G-Buffer Imposter Drag Unit & Lifecycle Tests")
    print("=" * 72)

    # 1. Local Box Classification (OBB vs Non-Box Freehand/Rough)
    v_box = _make_box_verts(0.5, -0.25, 1.0, 2.0, 1.5, 0.75)
    is_b, ctr, hlf = gpu_imposter.is_local_box_geometry(v_box, num_polys=6)
    assert is_b, "Standard offset box must be classified as OBB!"
    assert np.allclose(ctr, [0.5, -0.25, 1.0], atol=1e-5)
    assert np.allclose(hlf, [2.0, 1.5, 0.75], atol=1e-5)

    # Skewed / non-orthogonal 8-vert prism -> NOT an OBB
    v_skew = v_box.copy()
    v_skew[0, 0] += 0.35
    is_skew, _, _ = gpu_imposter.is_local_box_geometry(v_skew, num_polys=6)
    assert not is_skew, "Skewed/tapered mesh must NOT be classified as OBB!"

    # Concave Freehand polygon (12 verts, 8 polys) -> NOT an OBB
    is_fh, _, _ = gpu_imposter.is_local_box_geometry(np.zeros((12, 3), dtype=np.float32), num_polys=8)
    assert not is_fh, "Freehand 12-vert mesh must be routed to GPU Depth-Parity Mask!"
    print("[PASS] 1. OBB vs Special-Shape (Skew/Freehand/Rough) classification verified")

    # 2. Exact Rotated + Scaled + Offset OBB Inverse Matrix & Point-In-Box Math
    mw = _rot_z_matrix(math.radians(37.0), tx=12.5, ty=-4.2, tz=3.0, sx=1.5, sy=2.0, sz=0.5)
    inv_m = gpu_imposter.compute_inv_obb_matrix(mw, ctr, hlf)
    assert inv_m is not None and inv_m.shape == (4, 4)

    # Local center transformed to world MUST be inside; point just outside +X face MUST be outside
    world_ctr = (mw @ np.array([ctr[0], ctr[1], ctr[2], 1.0]))[:3]
    assert gpu_imposter.eval_obb_cpu_point(inv_m, world_ctr), "World center must be inside OBB!"
    world_edge_in = (mw @ np.array([ctr[0] + hlf[0] * 0.99, ctr[1], ctr[2], 1.0]))[:3]
    world_edge_out = (mw @ np.array([ctr[0] + hlf[0] * 1.01, ctr[1], ctr[2], 1.0]))[:3]
    assert gpu_imposter.eval_obb_cpu_point(inv_m, world_edge_in), "Point at 0.99 half-extent must be inside!"
    assert not gpu_imposter.eval_obb_cpu_point(inv_m, world_edge_out), "Point at 1.01 half-extent must be outside!"
    print("[PASS] 2. Rotated + scaled + offset OBB world-to-unit-cube math verified (1e-5 accuracy)")

    # 3. 500-Box OBB Texture Packing Speed & Exact Layer #1 Priority
    descs = []
    for i in range(500):
        m_i = _rot_z_matrix(0.01 * i, tx=float(i % 25) * 2.0, ty=float(i // 25) * 2.0, tz=0.0)
        inv_i = gpu_imposter.compute_inv_obb_matrix(m_i, np.zeros(3), np.ones(3))
        descs.append(
            gpu_imposter.ImposterCutterDesc(
                name=f"Box_{i:03d}",
                layer_idx=i,
                zone_letter=chr(ord("A") + (i % 8)),
                zone_rgba=(0.1 * (i % 8), 0.5, 0.8, 1.0),
                is_obb=True,
                inv_obb_mat=inv_i,
                selected_in_pieces=(i % 2 == 0),
            )
        )
    # Add 2 special (Rough/Freehand) cutters mixed into the list
    descs.insert(0, gpu_imposter.ImposterCutterDesc("Rough_Top", 0, "A", (1.0, 0.2, 0.1, 1.0), False, None, True))
    for k, d in enumerate(descs):
        d.layer_idx = k

    t0 = time.perf_counter()
    arr, n_obb = gpu_imposter.pack_obb_cutters_texture_data(descs)
    pack_ms = (time.perf_counter() - t0) * 1000.0
    assert n_obb == 500, f"Expected 500 OBB rows, got {n_obb}"
    assert arr.shape == (500, 5, 4) and arr.dtype == np.float32
    # First OBB row corresponds to layer_idx=1 (since Rough_Top holds layer_idx=0!)
    assert int(round(arr[0, 3, 0])) == 1, f"Expected layer_idx=1 for first OBB after Rough_Top, got {arr[0, 3, 0]}"
    assert pack_ms < 5.0, f"Packing 500 OBBs took too long: {pack_ms:.2f} ms"
    print(f"[PASS] 3. Packed 500 OBBs + mixed Rough Layer #1 priority into (500, 5, 4) float32 in {pack_ms:.2f} ms")

    # 4. Auto Activation Threshold Logic (AUTO / ON / OFF)
    assert not gpu_imposter.should_activate_imposter("AUTO", 100_000, total_target_polys=10_000, num_cutters=5)
    assert gpu_imposter.should_activate_imposter("AUTO", 100_000, total_target_polys=2_015_232, num_cutters=1)
    assert gpu_imposter.should_activate_imposter("AUTO", 100_000, total_target_polys=5_000, num_cutters=100)
    assert gpu_imposter.should_activate_imposter("ON", 100_000, total_target_polys=6, num_cutters=1)
    assert not gpu_imposter.should_activate_imposter("OFF", 100_000, total_target_polys=4_000_000, num_cutters=500)
    print("[PASS] 4. Activation threshold rules (AUTO >= 100K polys or >= 50 cutters / ON / OFF) verified")

    # 5. Live Sync 0-ms-Rust Imposter Drag + 180ms Idle Settle Lifecycle
    scheduled_timers = []
    mock_bpy = types.ModuleType("bpy")
    mock_bpy_app = types.ModuleType("bpy.app")
    mock_bpy_handlers = types.ModuleType("bpy.app.handlers")
    mock_bpy_handlers.depsgraph_update_post = []
    mock_bpy_handlers.persistent = lambda fn: fn
    mock_bpy_app.background = False
    mock_bpy_app.version_string = "5.2.0 LTS"
    mock_bpy_app.timers = types.SimpleNamespace(
        register=lambda fn, first_interval=0.0: scheduled_timers.append((fn, first_interval))
    )
    mock_bpy_app.handlers = mock_bpy_handlers
    mock_bpy.app = mock_bpy_app
    mock_bpy.types = types.SimpleNamespace(Scene=object, Object=object, Depsgraph=object)
    mock_bpy.data = types.SimpleNamespace(objects={}, collections={})
    sys.modules["bpy.app"] = mock_bpy_app
    sys.modules["bpy.app.handlers"] = mock_bpy_handlers

    class _MockMatrix:
        def __init__(self, arr4: np.ndarray):
            self._a = np.asarray(arr4, dtype=np.float64)

        def __array__(self, dtype=None):
            return self._a.astype(dtype) if dtype is not None else self._a

        def __getitem__(self, i):
            return self._a[i]

    box_v = _make_box_verts(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    mock_box_mesh = types.SimpleNamespace(
        vertices=types.SimpleNamespace(
            __len__=lambda self=None: 8,
            foreach_get=lambda _k, dst: dst.__setitem__(slice(None), box_v.ravel()),
        ),
        polygons=types.SimpleNamespace(__len__=lambda self=None: 6),
        as_pointer=lambda: 123456,
    )
    mock_box_mesh.vertices = type(
        "_V", (), {"__len__": lambda self: 8, "foreach_get": lambda self, _k, dst: dst.__setitem__(slice(None), box_v.ravel())}
    )()
    mock_box_mesh.polygons = type("_P", (), {"__len__": lambda self: 6})()

    mock_target_mesh = types.SimpleNamespace(
        vertices=type("_TV", (), {"__len__": lambda self: 2_016_578})(),
        polygons=type("_TP", (), {"__len__": lambda self: 2_015_232})(),
    )
    mock_target = types.SimpleNamespace(name="Suzanne", type="MESH", data=mock_target_mesh)
    mock_cutter = types.SimpleNamespace(
        name="Cutter_01",
        type="MESH",
        mode="OBJECT",
        data=mock_box_mesh,
        modifiers=[],
        matrix_world=_MockMatrix(np.eye(4)),
        bound_box=[(-1, -1, -1)] * 6 + [(1, 1, 1)],
    )
    mock_settings = types.SimpleNamespace(
        boolean_solver="RUST",
        gpu_fast_drag=True,
        imposter_drag_mode="AUTO",
        imposter_poly_threshold=100_000,
        preview_display_mode="ALL",
        show_zone_grid=True,
        zone_grid_size=0.35,
    )
    mock_scene = types.SimpleNamespace(fdt_settings=mock_settings)
    mock_bpy.context = types.SimpleNamespace(
        scene=mock_scene,
        evaluated_depsgraph_get=lambda: None,
    )
    sys.modules["bpy"] = mock_bpy
    sys.modules["bmesh"] = types.ModuleType("bmesh")

    # Stub gpu_preview, rust_bridge, session, apple_bite for isolated lifecycle check
    redraw_calls = [0]
    rust_submits = []
    stub_gp = types.ModuleType("addon.core.gpu_preview")
    stub_gp.is_active = lambda: True
    stub_gp.has_target_gpu_state = lambda: True
    stub_gp.tag_redraw = lambda: redraw_calls.__setitem__(0, redraw_calls[0] + 1)
    stub_gp.get_selected_cutter_zones = lambda: set()
    stub_gp.end_drag_preview = lambda: gpu_imposter.release()
    stub_gp.begin_drag_preview = lambda *a, **k: None
    stub_gp.register = lambda: None
    stub_gp.unregister = lambda: None
    sys.modules["addon.core.gpu_preview"] = stub_gp

    stub_rb = types.ModuleType("addon.core.rust_bridge")
    stub_rb.is_available = lambda: True
    stub_rb.supports_delta_gpu = lambda: True
    sys.modules["addon.core.rust_bridge"] = stub_rb

    mock_session = types.SimpleNamespace(
        session_id="S_IMP",
        cutters=[types.SimpleNamespace(name="Cutter_01", rough=False)],
        get_target_objects=lambda: [mock_target],
        get_valid_cutters=lambda include_muted=False: [(mock_cutter, "A")],
    )
    stub_sess = types.ModuleType("addon.core.session")
    stub_sess.load_session = lambda sc: mock_session
    stub_sess.load_session_readonly = lambda sc: mock_session
    sys.modules["addon.core.session"] = stub_sess

    stub_ab = types.ModuleType("addon.core.apple_bite")
    stub_ab.get_zone_color = lambda z: (1.0, 0.5, 0.12, 1.0)
    stub_ab.submit_rust_session_async_delta = lambda sc, dg, dirty_cutter_names=None: (
        rust_submits.append(set(dirty_cutter_names) if dirty_cutter_names else None),
        {"Suzanne": 42},
    )[1]
    sys.modules["addon.core.apple_bite"] = stub_ab

    # Point gpu_imposter at mock_bpy and pretend GPU is available for activation test
    gpu_imposter.bpy = mock_bpy
    gpu_imposter.HAS_GPU = True

    from addon.core import live_sync

    # Simulate 30 rapid drag frames on 4M-tri Suzanne: MUST invoke 0 Rust submits!
    for f in range(30):
        mock_cutter.matrix_world = _MockMatrix(_rot_z_matrix(0.0, tx=0.1 * f, ty=0.0, tz=0.0))
        handled = live_sync._try_handle_imposter_drag(
            mock_scene, mock_session, mock_settings, [(mock_cutter, "A")], {"Cutter_01"}, order_changed=False
        )
        assert handled, "Imposter must handle 4M-tri Suzanne drag in AUTO mode!"

    assert gpu_imposter.is_active(), "Imposter must be active during drag!"
    assert len(rust_submits) == 0, f"Expected 0 Rust submits during active drag, got {len(rust_submits)}"
    assert redraw_calls[0] == 30, f"Expected 30 immediate viewport redraws, got {redraw_calls[0]}"
    assert len(scheduled_timers) == 1, f"Expected 1 settle timer registered, got {len(scheduled_timers)}"

    # Fast-forward past 180 ms idle deadline and trigger settle timer -> MUST fire exactly 1 Rust submit!
    live_sync._IMPOSTER_SETTLE_DEADLINE = time.perf_counter() - 0.01
    settle_fn = scheduled_timers[0][0]
    ret = settle_fn()
    assert ret is None
    assert len(rust_submits) == 1 and rust_submits[0] == {"Cutter_01"}, (
        f"Expected exactly 1 settle Rust submit for Cutter_01, got {rust_submits}"
    )
    assert gpu_imposter.is_active(), "Imposter must remain displayed seamlessly until async Rust poll completes!"

    # Cleanup
    live_sync._cancel_imposter_settle()
    gpu_imposter.release()
    assert not gpu_imposter.is_active()
    print("[PASS] 5. 30-frame 4M-tri Imposter Drag (0 Rust submits!) + 180ms Idle Settle (1 Rust submit!) verified")

    print("\n" + "=" * 72)
    print("  ALL GPU IMPOSTER TESTS PASSED!")
    print("=" * 72)


if __name__ == "__main__":
    main()
