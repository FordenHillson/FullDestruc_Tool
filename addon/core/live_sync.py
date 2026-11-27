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
from .session import load_session, load_session_readonly

_IS_SYNCING: bool = False
_LAST_CUTTER_STATES: dict[str, dict[str, tuple]] = {}
_LAST_CUTTER_ORDER: dict[str, tuple[str, ...]] = {}
_LAST_ADOPT_SIG: tuple[str, int, int, int] | None = None
_LAST_EDIT_CUTTER_NAME: str = ""
_LAST_SYNC_TIME: float = 0.0
_MIN_SYNC_INTERVAL_SEC: float = 0.033
_TRAILING_TIMER_SCHEDULED: bool = False

# Zero-Switch Async GPU state
_GPU_POLL_ACTIVE: bool = False
_POLL_INTERVAL_SEC: float = 0.016  # ~60 Hz polling while jobs are in-flight
_LAST_SUBMITTED_JOB_IDS: dict[str, int] = {}
_LAST_SEEN_JOB_IDS: dict[str, int] = {}


def is_gpu_poll_active() -> bool:
    """Return True while the 60 Hz async GPU poll loop is actively receiving drag frames."""
    return _GPU_POLL_ACTIVE


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


def _cutter_state_token(
    obj: bpy.types.Object,
    geom_dirty: bool = True,
    prev_token: tuple | None = None,
) -> tuple:
    m = obj.matrix_world
    mat_t = (
        round(m[0][0], 5), round(m[0][1], 5), round(m[0][2], 5), round(m[0][3], 5),
        round(m[1][0], 5), round(m[1][1], 5), round(m[1][2], 5), round(m[1][3], 5),
        round(m[2][0], 5), round(m[2][1], 5), round(m[2][2], 5), round(m[2][3], 5),
        round(m[3][0], 5), round(m[3][1], 5), round(m[3][2], 5), round(m[3][3], 5),
    )
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
    n_mods = len(obj.modifiers)

    # Fast transform-only path in Object Mode: skip foreach_get + np.dot when geometry did not change!
    if (
        not geom_dirty
        and prev_token is not None
        and len(prev_token) == 7
        and prev_token[1] == mode
        and prev_token[3] == n_verts
        and prev_token[4] == n_polys
        and prev_token[6] == n_mods
    ):
        return (mat_t, mode, prev_token[2], n_verts, n_polys, prev_token[5], n_mods)

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
        co64 = co.astype(np.float64)
        ssq = float(np.dot(co64, co64))
        v_hash = (round(float(sums[0]), 4), round(float(sums[1]), 4), round(float(sums[2]), 4), round(ssq, 3))
    return (mat_t, mode, bb_t, n_verts, n_polys, v_hash, n_mods)


def reset_sync_cache(session_id: str | None = None) -> None:
    global _GPU_POLL_ACTIVE, _LAST_ADOPT_SIG, _LAST_EDIT_CUTTER_NAME
    _GPU_POLL_ACTIVE = False
    _LAST_ADOPT_SIG = None
    _LAST_EDIT_CUTTER_NAME = ""
    gpu_preview.end_drag_preview()
    if session_id is None:
        _LAST_CUTTER_STATES.clear()
        _LAST_CUTTER_ORDER.clear()
        _LAST_SUBMITTED_JOB_IDS.clear()
        _LAST_SEEN_JOB_IDS.clear()
    else:
        _LAST_CUTTER_STATES.pop(session_id, None)
        _LAST_CUTTER_ORDER.pop(session_id, None)
        for d in (_LAST_SUBMITTED_JOB_IDS, _LAST_SEEN_JOB_IDS):
            for k in list(d.keys()):
                if k.startswith(f"{session_id}::"):
                    d.pop(k, None)


def record_current_cutter_states(
    scene: bpy.types.Scene,
    only_names: set[str] | None = None,
) -> None:
    """Snapshot current cutter transforms so the next depsgraph tick doesn't redundantly re-run.

    When `only_names` is given and the session already has a recorded snapshot, only the
    specified cutters are re-tokenized (O(K) instead of O(N) for 500 boxes).
    """
    session = load_session_readonly(scene)
    if session is None:
        return
    active_cutters = session.get_valid_cutters(include_muted=False)
    order = tuple(c_obj.name for c_obj, _ in active_cutters)
    prev_map = _LAST_CUTTER_STATES.get(session.session_id)
    if only_names is not None and prev_map is not None:
        cur_names = set(order)
        # Drop removed cutters from prev_map
        for stale in list(prev_map.keys()):
            if stale not in cur_names:
                prev_map.pop(stale, None)
        for c_obj, _ in active_cutters:
            if c_obj.name in only_names or c_obj.name not in prev_map:
                prev_map[c_obj.name] = _cutter_state_token(c_obj, geom_dirty=True)
        _LAST_CUTTER_ORDER[session.session_id] = order
        return

    _LAST_CUTTER_STATES[session.session_id] = {
        c_obj.name: _cutter_state_token(c_obj, geom_dirty=True) for c_obj, _ in active_cutters
    }
    _LAST_CUTTER_ORDER[session.session_id] = order


def start_gpu_async_update(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph | None,
    dirty_cutter_names: set[str] | None = None,
) -> None:
    """Trigger a non-blocking Textured GPU preview update and wake the 60 Hz poll timer."""
    global _GPU_POLL_ACTIVE, _IS_SYNCING
    session = load_session_readonly(scene)
    if session is None:
        return
    settings = getattr(scene, "fdt_settings", None)
    grid_sz = float(getattr(settings, "zone_grid_size", 0.35) or 0.35) if settings else 0.35
    show_grid = bool(getattr(settings, "show_zone_grid", True)) if settings else True
    disp_mode = str(getattr(settings, "preview_display_mode", "ALL") or "ALL") if settings else "ALL"

    prev_sync = _IS_SYNCING
    was_active = gpu_preview.is_active()
    _IS_SYNCING = True
    try:
        from . import apple_bite

        if rust_bridge.supports_delta_gpu():
            job_map = apple_bite.submit_rust_session_async_delta(
                scene, depsgraph, dirty_cutter_names=dirty_cutter_names
            )
        else:
            job_map = apple_bite.submit_rust_session_async(scene, depsgraph)
        submitted = False
        for tname, jid in job_map.items():
            if jid > 0:
                _LAST_SUBMITTED_JOB_IDS[f"{session.session_id}::{tname}"] = jid
                submitted = True
        if submitted or was_active:
            gpu_preview.begin_drag_preview(
                session, grid_size=grid_sz, show_grid=show_grid, display_mode=disp_mode
            )
    except Exception as exc:
        print(f"[FullDestruc_Tool] Async GPU submit error: {exc}")
    finally:
        _IS_SYNCING = prev_sync

    if not _GPU_POLL_ACTIVE and gpu_preview.is_active():
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

    session = load_session_readonly(scene)
    if session is None:
        _GPU_POLL_ACTIVE = False
        gpu_preview.end_drag_preview()
        return None

    try:
        from . import apple_bite

        active_cutters = session.get_valid_cutters(include_muted=False)
        targets = session.get_target_objects()
        cut_mat = apple_bite._get_cut_material(scene)
        use_delta = rust_bridge.supports_delta_gpu()

        zone_by_name: dict[str, tuple[tuple[float, float, float, float], str]] | None = None
        zone_letters: list[str] | None = None
        zone_colors: list[tuple[float, float, float, float]] | None = None

        all_caught_up = True
        for orig in targets:
            jkey = f"{session.session_id}::{orig.name}"
            last_seen = _LAST_SEEN_JOB_IDS.get(jkey, 0)
            last_sub = _LAST_SUBMITTED_JOB_IDS.get(jkey, 0)

            t_p0 = time.perf_counter()
            if use_delta:
                res = rust_bridge.poll_target_gpu_delta(session.session_id, orig, last_seen)
                if res is not None:
                    _LAST_SEEN_JOB_IDS[jkey] = res["job_id"]
                    last_seen = res["job_id"]
                    src_materials = [orig.data.materials[i] for i in range(len(orig.data.materials))]

                    if zone_by_name is None:
                        zone_by_name = {}
                        for c_obj, z_raw in active_cutters:
                            z_let = str(z_raw or "A").strip() or "A"
                            zone_by_name[c_obj.name] = (apple_bite.get_zone_color(z_let), z_let)

                    piece_meta_by_id: dict[int, tuple[tuple[float, float, float, float], str]] = {}
                    for cid in res["ordered_piece_ids"]:
                        cname = rust_bridge.get_cutter_name_for_id(session.session_id, cid)
                        meta = zone_by_name.get(cname)
                        piece_meta_by_id[cid] = meta if meta is not None else ((1.0, 0.5, 0.12, 1.0), "A")

                    gpu_preview.apply_target_gpu_delta(
                        orig.name,
                        src_materials,
                        cut_mat,
                        res["updated_chunks"],
                        res["removed_chunk_ids"],
                        res["updated_pieces"],
                        res["removed_piece_ids"],
                        res["ordered_piece_ids"],
                        piece_meta_by_id,
                    )
                    gpu_upload_ms = (time.perf_counter() - t_p0) * 1000.0
                    delta_tris = sum(
                        int(item[1].shape[0] // 3)
                        for _, subs in res["updated_chunks"]
                        for item in subs
                    )
                    profiler.record_eval(
                        profiler.EvalSample(
                            timestamp=time.perf_counter(),
                            session_id=session.session_id,
                            target_name=orig.name,
                            solver="RUST_GPU_DELTA",
                            target_verts=len(orig.data.vertices),
                            target_polys=len(orig.data.polygons),
                            num_cutters=len(active_cutters),
                            cutter_total_polys=0,
                            out_rem_verts=delta_tris * 3,
                            out_rem_polys=delta_tris,
                            out_pieces_verts=0,
                            out_pieces_polys=len(res["updated_pieces"]),
                            target_uploaded=False,
                            upload_ms=0.0,
                            extract_ms=0.0,
                            rust_ms=res["compute_ms"],
                            writeback_ms=gpu_upload_ms,
                            total_ms=gpu_upload_ms,
                        )
                    )
            else:
                res = rust_bridge.poll_target_gpu_preview(session.session_id, orig, last_seen)
                if res is not None:
                    _LAST_SEEN_JOB_IDS[jkey] = res["job_id"]
                    last_seen = res["job_id"]
                    src_materials = [orig.data.materials[i] for i in range(len(orig.data.materials))]
                    if zone_letters is None:
                        zone_letters = [str(zone or "A").strip() or "A" for _, zone in active_cutters]
                        zone_colors = [apple_bite.get_zone_color(z) for z in zone_letters]
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
    session = load_session_readonly(scene)
    if session is None:
        return None
    active_cutters = session.get_valid_cutters(include_muted=False)
    if not active_cutters:
        return None
    prev_map = _LAST_CUTTER_STATES.get(session.session_id, {})
    new_map = {
        c_obj.name: _cutter_state_token(c_obj, geom_dirty=True, prev_token=prev_map.get(c_obj.name))
        for c_obj, _ in active_cutters
    }
    if new_map == prev_map:
        return None
    _LAST_CUTTER_STATES[session.session_id] = new_map
    _LAST_CUTTER_ORDER[session.session_id] = tuple(c_obj.name for c_obj, _ in active_cutters)
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


_ADOPT_REBUILD_PENDING = False


def _deferred_adopted_rebuild() -> float | None:
    global _ADOPT_REBUILD_PENDING
    _ADOPT_REBUILD_PENDING = False
    ctx = getattr(bpy, "context", None)
    scene = getattr(ctx, "scene", None) if ctx is not None else None
    if ctx is None or scene is None or load_session_readonly(scene) is None:
        return None
    try:
        from . import apple_bite

        apple_bite.rebuild_session(ctx)
    except Exception as exc:
        print(f"[FullDestruc_Tool] Adopted cutter rebuild error: {exc}")
    return None


def _schedule_adopted_rebuild() -> None:
    global _ADOPT_REBUILD_PENDING
    if _ADOPT_REBUILD_PENDING:
        return
    _ADOPT_REBUILD_PENDING = True
    try:
        bpy.app.timers.register(_deferred_adopted_rebuild, first_interval=0.0)
    except Exception:
        _ADOPT_REBUILD_PENDING = False


@persistent
def _on_depsgraph_update(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    global _IS_SYNCING, _LAST_SYNC_TIME, _TRAILING_TIMER_SCHEDULED
    global _LAST_ADOPT_SIG, _LAST_EDIT_CUTTER_NAME
    if _IS_SYNCING or scene is None:
        return

    session = load_session_readonly(scene)
    if session is None:
        return

    from . import apple_bite as _ab

    # O(1) guard before scanning scene.objects for Shift+D / Alt+D / Ctrl+C+V duplicates!
    cols = session.get_collections()
    cutters_col = cols.get("cutters")
    n_cut_col = len(cutters_col.objects) if cutters_col is not None else 0
    n_scene_obj = len(scene.objects)
    adopt_sig = (session.session_id, n_cut_col, n_scene_obj, len(session.cutters))

    adopted = []
    if adopt_sig != _LAST_ADOPT_SIG:
        _IS_SYNCING = True
        try:
            adopted = _ab.adopt_duplicated_cutters(scene)
        except Exception as exc:
            print(f"[FullDestruc_Tool] Duplicate cutter adopt error: {exc}")
        finally:
            _IS_SYNCING = False
        if adopted:
            session = load_session_readonly(scene)
            if session is None:
                return
            cols = session.get_collections()
            cutters_col = cols.get("cutters")
            n_cut_col = len(cutters_col.objects) if cutters_col is not None else 0
            n_scene_obj = len(scene.objects)
        _LAST_ADOPT_SIG = (session.session_id, n_cut_col, n_scene_obj, len(session.cutters))

    # Extract dirty object & mesh sets directly from Blender's depsgraph updates (O(K) instead of O(500)!)
    updated_obj_names: set[str] = set()
    geom_updated_names: set[str] = set()
    updated_mesh_ptrs: set[int] = set()
    has_updates = False
    if depsgraph is not None:
        obj_cls = getattr(bpy.types, "Object", None)
        mesh_cls = getattr(bpy.types, "Mesh", None)
        for upd in getattr(depsgraph, "updates", ()):
            has_updates = True
            uid = upd.id
            if obj_cls is not None and isinstance(uid, obj_cls):
                uname = uid.name
                updated_obj_names.add(uname)
                if getattr(upd, "is_updated_geometry", False):
                    geom_updated_names.add(uname)
            elif mesh_cls is not None and isinstance(uid, mesh_cls):
                try:
                    updated_mesh_ptrs.add(int(uid.as_pointer()))
                except Exception:
                    pass

    active_obj = getattr(bpy.context, "active_object", None)
    active_name = active_obj.name if active_obj is not None else ""
    prev_map = _LAST_CUTTER_STATES.get(session.session_id)
    prev_order = _LAST_CUTTER_ORDER.get(session.session_id)

    # Auto-enable show_in_front ONLY while editing a cutter in Edit Mode, and hide noisy rough
    # black wireframes (BOUNDS) whenever a rough cutter is idle (neither Edit Mode nor tuning).
    if prev_map is None or adopted or not has_updates:
        # Initial pass or structural change: sync all cutters once
        for entry in session.cutters:
            c_obj = bpy.data.objects.get(entry.name)
            if c_obj is not None and c_obj.type == "MESH":
                want_front = c_obj.mode == "EDIT"
                if want_front:
                    _LAST_EDIT_CUTTER_NAME = entry.name
                if bool(c_obj.show_in_front) != want_front:
                    c_obj.show_in_front = want_front
                want_disp = "WIRE" if _ab.is_rough_wire_visible(c_obj, bool(entry.rough)) else "BOUNDS"
                if c_obj.display_type != want_disp:
                    c_obj.display_type = want_disp
    else:
        check_vis = set(updated_obj_names)
        if active_name:
            check_vis.add(active_name)
        if _LAST_EDIT_CUTTER_NAME:
            check_vis.add(_LAST_EDIT_CUTTER_NAME)
        if check_vis:
            for entry in session.cutters:
                if entry.name not in check_vis:
                    continue
                c_obj = bpy.data.objects.get(entry.name)
                if c_obj is not None and c_obj.type == "MESH":
                    want_front = c_obj.mode == "EDIT"
                    if want_front:
                        _LAST_EDIT_CUTTER_NAME = entry.name
                    elif _LAST_EDIT_CUTTER_NAME == entry.name:
                        _LAST_EDIT_CUTTER_NAME = ""
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
        pcs_col = cols.get("pieces")
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
        if adopted:
            _schedule_adopted_rebuild()
        return

    if not rust_bridge.is_available():
        if adopted:
            _schedule_adopted_rebuild()
        return

    active_cutters = session.get_valid_cutters(include_muted=False)
    if not active_cutters:
        return

    cur_order = tuple(c_obj.name for c_obj, _ in active_cutters)
    dirty_cutter_names: set[str] = set()
    order_changed = (prev_order != cur_order) or adopted

    if prev_map is not None and not order_changed and has_updates:
        # O(K) incremental check: only re-tokenize cutters touched by depsgraph or in Edit Mode!
        check_names = set(updated_obj_names)
        if active_name:
            check_names.add(active_name)
        if _LAST_EDIT_CUTTER_NAME:
            check_names.add(_LAST_EDIT_CUTTER_NAME)

        new_map = prev_map
        for c_obj, _ in active_cutters:
            cname = c_obj.name
            c_mesh = c_obj.data
            mesh_dirty = bool(
                updated_mesh_ptrs
                and c_mesh is not None
                and int(c_mesh.as_pointer()) in updated_mesh_ptrs
            )
            if cname not in check_names and not mesh_dirty and c_obj.mode != "EDIT":
                continue
            prev_tok = prev_map.get(cname)
            geom_dirty = (cname in geom_updated_names) or mesh_dirty or (c_obj.mode == "EDIT") or (prev_tok is None)
            tok = _cutter_state_token(c_obj, geom_dirty=geom_dirty, prev_token=prev_tok)
            if prev_tok != tok:
                new_map[cname] = tok
                dirty_cutter_names.add(cname)
                if geom_dirty and c_mesh is not None:
                    try:
                        gpu_preview.invalidate_cutter_wire_cache(int(c_mesh.as_pointer()))
                    except Exception:
                        pass

        if not dirty_cutter_names:
            return
    else:
        # Full snapshot (first tick, cutter added/removed/reordered, or depsgraph updates unavailable)
        prev_dict = prev_map or {}
        new_map = {}
        for c_obj, _ in active_cutters:
            cname = c_obj.name
            prev_tok = prev_dict.get(cname)
            c_mesh = c_obj.data
            mesh_dirty = bool(
                updated_mesh_ptrs
                and c_mesh is not None
                and int(c_mesh.as_pointer()) in updated_mesh_ptrs
            )
            geom_dirty = (
                (not has_updates)
                or (cname in geom_updated_names)
                or mesh_dirty
                or (c_obj.mode == "EDIT")
                or (prev_tok is None)
            )
            tok = _cutter_state_token(c_obj, geom_dirty=geom_dirty, prev_token=prev_tok)
            new_map[cname] = tok
            if prev_tok != tok:
                dirty_cutter_names.add(cname)
                if geom_dirty and c_mesh is not None:
                    try:
                        gpu_preview.invalidate_cutter_wire_cache(int(c_mesh.as_pointer()))
                    except Exception:
                        pass

        if not dirty_cutter_names and not order_changed and len(prev_dict) == len(new_map):
            _LAST_CUTTER_STATES[session.session_id] = new_map
            _LAST_CUTTER_ORDER[session.session_id] = cur_order
            return

    # --- Fast Path: Zero-Switch Textured GPU Mode (Locked 60-144+ FPS!) ---
    if bool(getattr(settings, "gpu_fast_drag", True)):
        _LAST_CUTTER_STATES[session.session_id] = new_map
        _LAST_CUTTER_ORDER[session.session_id] = cur_order
        if order_changed or (dirty_cutter_names and (len(dirty_cutter_names) > 1 or active_name not in dirty_cutter_names)):
            gpu_preview.mark_cutter_wireframes_dirty()
        start_gpu_async_update(scene, depsgraph, dirty_cutter_names=dirty_cutter_names)
        return

    # --- Classic Path: Direct Blender Mesh Live Sync (30 FPS capped) ---
    now = time.perf_counter()
    if (now - _LAST_SYNC_TIME) < _MIN_SYNC_INTERVAL_SEC:
        if not _TRAILING_TIMER_SCHEDULED:
            _TRAILING_TIMER_SCHEDULED = True
            bpy.app.timers.register(_trailing_sync_timer, first_interval=_MIN_SYNC_INTERVAL_SEC)
        return

    _LAST_CUTTER_STATES[session.session_id] = new_map
    _LAST_CUTTER_ORDER[session.session_id] = cur_order
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
    global _GPU_POLL_ACTIVE, _LAST_ADOPT_SIG, _LAST_EDIT_CUTTER_NAME
    _GPU_POLL_ACTIVE = False
    _LAST_ADOPT_SIG = None
    _LAST_EDIT_CUTTER_NAME = ""
    gpu_preview.unregister()
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    _LAST_CUTTER_STATES.clear()
    _LAST_CUTTER_ORDER.clear()
    _LAST_SUBMITTED_JOB_IDS.clear()
    _LAST_SEEN_JOB_IDS.clear()
    rust_bridge.destroy_all_sessions()

