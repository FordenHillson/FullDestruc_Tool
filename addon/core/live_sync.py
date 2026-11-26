"""Real-time depsgraph sync handler for the Rust Boolean Engine.

Supports two modes controlled by `scene.fdt_settings.gpu_fast_drag`:
1. Zero-Switch Textured GPU Mode (Default ON — Locked 60-144+ FPS):
   - Renders the building's REAL Image Textures, Cut Face Material, and Zone Grid
     directly on the GPU (`POST_VIEW` GLSL shader) throughout the entire editing session.
   - Zero flickering, zero switching back to gray clay, zero 20ms commit pauses!
   - Non-blocking Rust background worker (`Latest-Wins` queue) computes in parallel.
   - Poll timer automatically sleeps (0% idle CPU) once the latest frame is uploaded,
     and wakes immediately on the next cutter move/edit.
   - Commits real Blender meshes cleanly on Apply, Exit/Pause, or when disabling GPU Fast Drag.
2. Classic Direct Mesh Sync (when `gpu_fast_drag == False`):
   - Updates `FDT_Res_*` & `FDT_Piece_*` directly at up to 30 FPS with trailing sync.
"""

from __future__ import annotations

from contextlib import contextmanager
import time

import bpy
import bmesh
import numpy as np
from bpy.app.handlers import persistent

from . import gpu_preview, profiler, rust_bridge
from .session import load_session

_IS_SYNCING: bool = False
_LAST_CUTTER_STATES: dict[str, dict[str, tuple]] = {}
_LAST_SYNC_TIME: float = 0.0
_MIN_SYNC_INTERVAL_SEC: float = 0.033
_TRAILING_TIMER_SCHEDULED: bool = False

# Zero-Switch Async GPU state
_GPU_POLL_ACTIVE: bool = False
_POLL_INTERVAL_SEC: float = 0.016  # ~60 Hz polling while jobs are in-flight
_LAST_SUBMITTED_JOB_IDS: dict[str, int] = {}
_LAST_SEEN_JOB_IDS: dict[str, int] = {}


@contextmanager
def suspend():
    """Temporarily pause live depsgraph syncing and end any active GPU preview."""
    global _IS_SYNCING, _GPU_POLL_ACTIVE
    prev = _IS_SYNCING
    _IS_SYNCING = True
    _GPU_POLL_ACTIVE = False
    gpu_preview.end_drag_preview()
    try:
        yield
    finally:
        _IS_SYNCING = prev


def _cutter_state_token(obj: bpy.types.Object) -> tuple:
    m = obj.matrix_world
    mat_t = tuple(round(m[r][c], 5) for r in range(4) for c in range(4))
    mesh = obj.data
    mode = obj.mode
    if mode == "EDIT" and obj.type == "MESH" and mesh is not None:
        try:
            bm = bmesh.from_edit_mesh(mesh)
            nv = len(bm.verts)
            nf = len(bm.faces)
            if nv > 0:
                sx = sy = sz = ssq = 0.0
                mnx = mny = mnz = 1e30
                mxx = mxy = mxz = -1e30
                for v in bm.verts:
                    x, y, z = v.co.x, v.co.y, v.co.z
                    sx += x
                    sy += y
                    sz += z
                    ssq += x * x + y * y + z * z
                    if x < mnx: mnx = x
                    if y < mny: mny = y
                    if z < mnz: mnz = z
                    if x > mxx: mxx = x
                    if y > mxy: mxy = y
                    if z > mxz: mxz = z
                geo_sig = (
                    nv,
                    nf,
                    round(sx, 4),
                    round(sy, 4),
                    round(sz, 4),
                    round(ssq, 3),
                    round(mnx, 4),
                    round(mny, 4),
                    round(mnz, 4),
                    round(mxx, 4),
                    round(mxy, 4),
                    round(mxz, 4),
                )
            else:
                geo_sig = (0, 0)
            return (mat_t, "EDIT", geo_sig, len(obj.modifiers))
        except Exception:
            pass

    n_verts = len(mesh.vertices) if mesh is not None else 0
    n_polys = len(mesh.polygons) if mesh is not None else 0
    bb = obj.bound_box
    bb_t = tuple(round(bb[i][j], 4) for i in (0, 6) for j in range(3)) if bb else ()
    # Fast coordinate checksum for small/medium cutters so internal Edit-Mode edits
    # (that don't change outer bound_box) ALWAYS trigger re-eval on Tab exit!
    v_hash = ()
    if mesh is not None and 0 < n_verts <= 4096:
        co = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co3 = co.reshape(-1, 3)
        sums = co3.sum(axis=0, dtype=np.float64)
        ssq = float(np.dot(co.astype(np.float64), co.astype(np.float64)))
        v_hash = (round(float(sums[0]), 4), round(float(sums[1]), 4), round(float(sums[2]), 4), round(ssq, 3))
    return (mat_t, mode, bb_t, n_verts, n_polys, v_hash, len(obj.modifiers))


def reset_sync_cache(session_id: str | None = None) -> None:
    global _GPU_POLL_ACTIVE
    _GPU_POLL_ACTIVE = False
    gpu_preview.end_drag_preview()
    if session_id is None:
        _LAST_CUTTER_STATES.clear()
        _LAST_SUBMITTED_JOB_IDS.clear()
        _LAST_SEEN_JOB_IDS.clear()
    else:
        _LAST_CUTTER_STATES.pop(session_id, None)
        for d in (_LAST_SUBMITTED_JOB_IDS, _LAST_SEEN_JOB_IDS):
            for k in list(d.keys()):
                if k.startswith(f"{session_id}::"):
                    d.pop(k, None)


def record_current_cutter_states(scene: bpy.types.Scene) -> None:
    """Snapshot current cutter transforms so the next depsgraph tick doesn't redundantly re-run."""
    session = load_session(scene)
    if session is None:
        return
    active_cutters = session.get_valid_cutters(include_muted=False)
    _LAST_CUTTER_STATES[session.session_id] = {
        c_obj.name: _cutter_state_token(c_obj) for c_obj, _ in active_cutters
    }


def start_gpu_async_update(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph | None) -> None:
    """Trigger a non-blocking Textured GPU preview update and wake the 60 Hz poll timer."""
    global _GPU_POLL_ACTIVE, _IS_SYNCING
    session = load_session(scene)
    if session is None:
        return
    settings = getattr(scene, "fdt_settings", None)
    grid_sz = float(getattr(settings, "zone_grid_size", 0.35) or 0.35) if settings else 0.35
    show_grid = bool(getattr(settings, "show_zone_grid", True)) if settings else True
    disp_mode = str(getattr(settings, "preview_display_mode", "ALL") or "ALL") if settings else "ALL"

    prev_sync = _IS_SYNCING
    _IS_SYNCING = True
    try:
        from . import apple_bite

        gpu_preview.begin_drag_preview(
            session, grid_size=grid_sz, show_grid=show_grid, display_mode=disp_mode
        )
        job_map = apple_bite.submit_rust_session_async(scene, depsgraph)
        for tname, jid in job_map.items():
            if jid > 0:
                _LAST_SUBMITTED_JOB_IDS[f"{session.session_id}::{tname}"] = jid
    except Exception as exc:
        print(f"[FullDestruc_Tool] Async GPU submit error: {exc}")
    finally:
        _IS_SYNCING = prev_sync

    if not _GPU_POLL_ACTIVE:
        _GPU_POLL_ACTIVE = True
        bpy.app.timers.register(_gpu_poll_timer, first_interval=_POLL_INTERVAL_SEC)


def _gpu_poll_timer() -> float | None:
    """60 Hz lightweight timer that uploads finished textured GPU frames and sleeps when caught up."""
    global _GPU_POLL_ACTIVE
    if not _GPU_POLL_ACTIVE:
        return None
    if _IS_SYNCING:
        return _POLL_INTERVAL_SEC

    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        _GPU_POLL_ACTIVE = False
        gpu_preview.end_drag_preview()
        return None

    settings = getattr(scene, "fdt_settings", None)
    if settings is None or getattr(settings, "boolean_solver", "") != "RUST" or not bool(
        getattr(settings, "gpu_fast_drag", True)
    ):
        _GPU_POLL_ACTIVE = False
        gpu_preview.end_drag_preview()
        return None

    session = load_session(scene)
    if session is None:
        _GPU_POLL_ACTIVE = False
        gpu_preview.end_drag_preview()
        return None

    try:
        from . import apple_bite

        active_cutters = session.get_valid_cutters(include_muted=False)
        zone_letters = [str(zone or "A").strip() or "A" for _, zone in active_cutters]
        zone_colors = [apple_bite.get_zone_color(z) for z in zone_letters]
        cut_mat = apple_bite._get_cut_material(scene)
        targets = session.get_target_objects()

        all_caught_up = True
        for orig in targets:
            jkey = f"{session.session_id}::{orig.name}"
            last_seen = _LAST_SEEN_JOB_IDS.get(jkey, 0)
            last_sub = _LAST_SUBMITTED_JOB_IDS.get(jkey, 0)

            t_p0 = time.perf_counter()
            res = rust_bridge.poll_target_gpu_preview(session.session_id, orig, last_seen)
            if res is not None:
                _LAST_SEEN_JOB_IDS[jkey] = res["job_id"]
                last_seen = res["job_id"]
                src_materials = [orig.data.materials[i] for i in range(len(orig.data.materials))]
                p_list = []
                for idx, p_subs in enumerate(res["pieces"]):
                    z_col = zone_colors[idx] if idx < len(zone_colors) else (1.0, 0.5, 0.12, 1.0)
                    z_let = zone_letters[idx] if idx < len(zone_letters) else "A"
                    p_list.append((p_subs, z_col, z_let))

                gpu_preview.update_target_gpu_batches(
                    orig.name,
                    src_materials,
                    cut_mat,
                    res["remaining"],
                    p_list,
                )
                gpu_upload_ms = (time.perf_counter() - t_p0) * 1000.0
                rem_tris = sum(int(item[1].shape[0] // 3) for item in res["remaining"])
                profiler.record_eval(
                    profiler.EvalSample(
                        timestamp=time.perf_counter(),
                        session_id=session.session_id,
                        target_name=orig.name,
                        solver="RUST_GPU",
                        target_verts=len(orig.data.vertices),
                        target_polys=len(orig.data.polygons),
                        num_cutters=len(active_cutters),
                        cutter_total_polys=0,
                        out_rem_verts=rem_tris * 3,
                        out_rem_polys=rem_tris,
                        out_pieces_verts=0,
                        out_pieces_polys=0,
                        target_uploaded=False,
                        upload_ms=0.0,
                        extract_ms=0.0,
                        rust_ms=res["compute_ms"],
                        writeback_ms=gpu_upload_ms,
                        total_ms=gpu_upload_ms,
                    )
                )

            if last_seen < last_sub:
                all_caught_up = False

        if all_caught_up:
            # Sleep cleanly (0% idle CPU) while Textured GPU Preview stays displayed on screen!
            _GPU_POLL_ACTIVE = False
            return None
        return _POLL_INTERVAL_SEC
    except Exception as exc:
        print(f"[FullDestruc_Tool] Textured GPU preview poll error: {exc}")
        _GPU_POLL_ACTIVE = False
        return None


def _trailing_sync_timer() -> float | None:
    global _TRAILING_TIMER_SCHEDULED, _IS_SYNCING, _LAST_SYNC_TIME
    _TRAILING_TIMER_SCHEDULED = False
    if _IS_SYNCING:
        return None
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return None
    settings = getattr(scene, "fdt_settings", None)
    if settings is None or getattr(settings, "boolean_solver", "") != "RUST":
        return None
    if bool(getattr(settings, "gpu_fast_drag", True)):
        return None
    session = load_session(scene)
    if session is None:
        return None
    active_cutters = session.get_valid_cutters(include_muted=False)
    if not active_cutters:
        return None
    prev_map = _LAST_CUTTER_STATES.get(session.session_id, {})
    new_map = {c_obj.name: _cutter_state_token(c_obj) for c_obj, _ in active_cutters}
    if new_map == prev_map:
        return None
    _LAST_CUTTER_STATES[session.session_id] = new_map
    _LAST_SYNC_TIME = time.perf_counter()
    _IS_SYNCING = True
    try:
        from . import apple_bite

        depsgraph = bpy.context.evaluated_depsgraph_get()
        with profiler.action_scope("LIVE_DRAG", "trailing"):
            apple_bite.sync_rust_session_live(scene, depsgraph)
    except Exception as exc:
        print(f"[FullDestruc_Tool] Rust trailing sync error: {exc}")
    finally:
        _IS_SYNCING = False
    return None


@persistent
def _on_depsgraph_update(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    global _IS_SYNCING, _LAST_SYNC_TIME, _TRAILING_TIMER_SCHEDULED
    if _IS_SYNCING or scene is None:
        return

    session = load_session(scene)
    if session is None:
        return

    # Auto-enable show_in_front ONLY while editing a cutter in Edit Mode, and hide noisy rough
    # black wireframes (BOUNDS) whenever a rough cutter is idle (neither Edit Mode nor tuning).
    from . import apple_bite as _ab

    for entry in session.cutters:
        c_obj = bpy.data.objects.get(entry.name)
        if c_obj is not None and c_obj.type == "MESH":
            want_front = c_obj.mode == "EDIT"
            if bool(c_obj.show_in_front) != want_front:
                c_obj.show_in_front = want_front
            want_disp = "WIRE" if _ab.is_rough_wire_visible(c_obj, bool(entry.rough)) else "BOUNDS"
            if c_obj.display_type != want_disp:
                c_obj.display_type = want_disp

    settings = getattr(scene, "fdt_settings", None)
    # In Direct Mesh mode + Piece Only, keep visible pieces synced to the selected cutter zone(s)
    if (
        settings is not None
        and not gpu_preview.is_active()
        and getattr(settings, "preview_display_mode", "ALL") == "PIECES"
    ):
        sel_zones = gpu_preview.get_selected_cutter_zones()
        pcs_col = session.get_collections().get("pieces")
        if pcs_col is not None:
            for o in pcs_col.objects:
                if o.type == "MESH":
                    pz = str(o.get("fdt_piece_zone", "A") or "A").strip() or "A"
                    want_hide = bool(sel_zones) and (pz not in sel_zones)
                    if bool(o.hide_get()) != want_hide:
                        try:
                            o.hide_set(want_hide)
                        except Exception:
                            pass

    if settings is None or getattr(settings, "boolean_solver", "") != "RUST":
        return

    if not rust_bridge.is_available():
        return

    active_cutters = session.get_valid_cutters(include_muted=False)
    if not active_cutters:
        return

    prev_map = _LAST_CUTTER_STATES.get(session.session_id, {})
    new_map: dict[str, tuple] = {}
    any_changed = False

    for c_obj, _ in active_cutters:
        tok = _cutter_state_token(c_obj)
        new_map[c_obj.name] = tok
        if prev_map.get(c_obj.name) != tok:
            any_changed = True

    if not any_changed and len(prev_map) == len(new_map):
        return

    # --- Fast Path: Zero-Switch Textured GPU Mode (Locked 60-144+ FPS!) ---
    if bool(getattr(settings, "gpu_fast_drag", True)):
        _LAST_CUTTER_STATES[session.session_id] = new_map
        start_gpu_async_update(scene, depsgraph)
        return

    # --- Classic Path: Direct Blender Mesh Live Sync (30 FPS capped) ---
    now = time.perf_counter()
    if (now - _LAST_SYNC_TIME) < _MIN_SYNC_INTERVAL_SEC:
        if not _TRAILING_TIMER_SCHEDULED:
            _TRAILING_TIMER_SCHEDULED = True
            bpy.app.timers.register(_trailing_sync_timer, first_interval=_MIN_SYNC_INTERVAL_SEC)
        return

    _LAST_CUTTER_STATES[session.session_id] = new_map
    _LAST_SYNC_TIME = now

    _IS_SYNCING = True
    try:
        from . import apple_bite

        with profiler.action_scope("LIVE_DRAG"):
            apple_bite.sync_rust_session_live(scene, depsgraph)
    except Exception as exc:
        print(f"[FullDestruc_Tool] Rust live sync error: {exc}")
    finally:
        _IS_SYNCING = False


def register() -> None:
    gpu_preview.register()
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    handlers.append(_on_depsgraph_update)


def unregister() -> None:
    global _GPU_POLL_ACTIVE
    _GPU_POLL_ACTIVE = False
    gpu_preview.unregister()
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    _LAST_CUTTER_STATES.clear()
    _LAST_SUBMITTED_JOB_IDS.clear()
    _LAST_SEEN_JOB_IDS.clear()
    rust_bridge.destroy_all_sessions()
