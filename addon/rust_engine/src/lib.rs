//! C-ABI entry points for `fdt_bool_engine` (v2.0.0 — High-Scale Spatial Chunks + Islands).
//!
//! Features:
//! - Spatial Chunking (`chunks.rs`) for 1,000,000+ polygon meshes: untouched chunks stay
//!   resident in GPU VRAM (0 ms recompute, 0 bytes re-uploaded).
//! - Spatial Island Evaluator (`islands.rs`) for 100–500 cutters: Union-Find clusters ONLY
//!   overlapping cutters while preserving exact Layer #1 > #2 > ... priority. Moving Box #1
//!   or Box #500 computes ONLY 1 local island in O(1) time.
//! - Lossless Latest-Wins Delta Async Worker + Non-Blocking `try_lock()` Delta Poll:
//!   Main thread NEVER stalls waiting for background worker jobs.
//! - Full backwards compatibility with v1.x C-ABI functions (`eval_cutters`, `submit_async`,
//!   `poll_gpu_preview`, `fetch_cached_eval`).
//! - Panic-safe (`std::panic::catch_unwind`) on every exported function.

pub mod boolean;
pub mod box_uv;
pub mod bvh;
pub mod chunks;
pub mod classify;
pub mod gpu_pack;
pub mod islands;
pub mod mesh;
pub mod predicates;
pub mod retriangulate;

use crate::gpu_pack::GpuPackedMesh;
use crate::islands::{DeltaGpuSnapshot, IslandTargetState};
use crate::mesh::HybridMesh;
use glam::DMat4;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Global Session Registry (Per-Target Arc<Mutex<IslandTargetState>>)
// ---------------------------------------------------------------------------

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

struct SessionState {
    targets: HashMap<u32, Arc<Mutex<IslandTargetState>>>,
}

static SESSIONS: Mutex<Option<HashMap<u64, SessionState>>> = Mutex::new(None);

fn with_sessions<R>(f: impl FnOnce(&mut HashMap<u64, SessionState>) -> R) -> R {
    let mut guard = SESSIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

fn get_target_arc(handle: u64, target_id: u32) -> Option<Arc<Mutex<IslandTargetState>>> {
    with_sessions(|map| map.get(&handle)?.targets.get(&target_id).cloned())
}

// ---------------------------------------------------------------------------
// Async Background Worker (Lossless Latest-Wins Queue)
// ---------------------------------------------------------------------------

enum PendingJob {
    /// Legacy full-list async submit (positional IDs 1..=N).
    FullLegacy {
        job_id: u64,
        raw_cutters: Vec<(HybridMesh, DMat4)>,
        cut_mat_idx: i32,
        uv_scale: f32,
    },
    /// v200 Delta submit: only changed cutters + ordered active IDs.
    Delta {
        job_id: u64,
        updated_cutters: HashMap<u32, (HybridMesh, DMat4)>,
        ordered_active_ids: Vec<u32>,
        cut_mat_idx: i32,
        uv_scale: f32,
    },
}

struct CompletedMeta {
    job_id: u64,
    compute_ms: f32,
    recomputed_islands: u32,
}

struct CompletedLegacyFrame {
    job_id: u64,
    compute_ms: f32,
    dirty_from_idx: u32,
    remaining_gpu: GpuPackedMesh,
    pieces_gpu: Vec<GpuPackedMesh>,
}

struct WorkerShared {
    pending_targets: HashMap<(u64, u32), (HybridMesh, DMat4)>,
    pending: HashMap<(u64, u32), PendingJob>,
    in_flight: HashSet<(u64, u32)>,
    completed_delta_meta: HashMap<(u64, u32), CompletedMeta>,
    completed_legacy: HashMap<(u64, u32), CompletedLegacyFrame>,
}

struct AsyncWorker {
    state: Mutex<WorkerShared>,
    work_cvar: Condvar,
    done_cvar: Condvar,
}

static WORKER: OnceLock<AsyncWorker> = OnceLock::new();

fn get_worker() -> &'static AsyncWorker {
    WORKER.get_or_init(|| {
        let worker = AsyncWorker {
            state: Mutex::new(WorkerShared {
                pending_targets: HashMap::new(),
                pending: HashMap::new(),
                in_flight: HashSet::new(),
                completed_delta_meta: HashMap::new(),
                completed_legacy: HashMap::new(),
            }),
            work_cvar: Condvar::new(),
            done_cvar: Condvar::new(),
        };
        std::thread::Builder::new()
            .name("fdt-rust-async-worker".to_string())
            .spawn(worker_thread_loop)
            .expect("failed to spawn fdt async worker thread");
        worker
    })
}

fn wait_for_target_idle(handle: u64, target_id: u32) {
    let key = (handle, target_id);
    if let Some(worker) = WORKER.get() {
        let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.pending_targets.contains_key(&key)
            || st.pending.contains_key(&key)
            || st.in_flight.contains(&key)
        {
            st = worker.done_cvar.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
}

enum WorkerTask {
    SetTarget {
        base_mesh: HybridMesh,
        world_mat: DMat4,
    },
    Job(PendingJob),
}

fn worker_thread_loop() {
    let worker = get_worker();
    loop {
        let (key, task) = {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.pending_targets.is_empty() && st.pending.is_empty() {
                st = worker.work_cvar.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if let Some(&first_key) = st.pending_targets.keys().next() {
                let (base_mesh, world_mat) = st.pending_targets.remove(&first_key).unwrap();
                st.in_flight.insert(first_key);
                (first_key, WorkerTask::SetTarget { base_mesh, world_mat })
            } else {
                let first_key = *st.pending.keys().next().unwrap();
                let job = st.pending.remove(&first_key).unwrap();
                st.in_flight.insert(first_key);
                (first_key, WorkerTask::Job(job))
            }
        };

        let (handle, target_id) = key;
        let t0 = Instant::now();

        match task {
            WorkerTask::SetTarget {
                base_mesh,
                world_mat,
            } => {
                let built = catch_unwind(|| {
                    let mut tc = IslandTargetState::new(base_mesh, world_mat);
                    tc.prewarm_base_gpu();
                    Arc::new(Mutex::new(tc))
                });
                if let Ok(state_arc) = built {
                    with_sessions(|map| {
                        if let Some(sess) = map.get_mut(&handle) {
                            sess.targets.insert(target_id, state_arc);
                        }
                    });
                }
                let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
                st.completed_delta_meta.remove(&key);
                st.completed_legacy.remove(&key);
                st.in_flight.remove(&key);
                worker.done_cvar.notify_all();
            }
            WorkerTask::Job(PendingJob::Delta {
                job_id,
                updated_cutters,
                ordered_active_ids,
                cut_mat_idx,
                uv_scale,
            }) => {
                let res = catch_unwind(|| {
                    let target_arc = get_target_arc(handle, target_id)?;
                    let mut tc = target_arc.lock().unwrap_or_else(|e| e.into_inner());
                    let updates: Vec<(u32, HybridMesh, DMat4)> = updated_cutters
                        .into_iter()
                        .map(|(cid, (m, mat))| (cid, m, mat))
                        .collect();
                    tc.upsert_cutters(updates);
                    let recomputed =
                        tc.evaluate_active(&ordered_active_ids, cut_mat_idx, uv_scale, true);
                    tc.retain_cutters(&ordered_active_ids);
                    Some(recomputed as u32)
                });
                let compute_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;
                let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Ok(Some(recomputed_islands)) = res {
                    st.completed_delta_meta.insert(
                        key,
                        CompletedMeta {
                            job_id,
                            compute_ms,
                            recomputed_islands,
                        },
                    );
                }
                st.in_flight.remove(&key);
                worker.done_cvar.notify_all();
            }
            WorkerTask::Job(PendingJob::FullLegacy {
                job_id,
                raw_cutters,
                cut_mat_idx,
                uv_scale,
            }) => {
                let res = catch_unwind(|| {
                    let target_arc = get_target_arc(handle, target_id)?;
                    let mut tc = target_arc.lock().unwrap_or_else(|e| e.into_inner());
                    let nc = raw_cutters.len();
                    let ordered_ids: Vec<u32> = (1..=(nc as u32)).collect();
                    let updates: Vec<(u32, HybridMesh, DMat4)> = raw_cutters
                        .into_iter()
                        .enumerate()
                        .map(|(i, (m, mat))| ((i + 1) as u32, m, mat))
                        .collect();
                    tc.upsert_cutters(updates);
                    let recomputed =
                        tc.evaluate_active(&ordered_ids, cut_mat_idx, uv_scale, false);
                    let (remaining_gpu, pieces_gpu) = tc.build_legacy_gpu_snapshot();
                    Some((recomputed as u32, remaining_gpu, pieces_gpu))
                });
                let compute_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;
                let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Ok(Some((recomputed, remaining_gpu, pieces_gpu))) = res {
                    st.completed_legacy.insert(
                        key,
                        CompletedLegacyFrame {
                            job_id,
                            compute_ms,
                            dirty_from_idx: if recomputed == 0 { u32::MAX } else { 0 },
                            remaining_gpu,
                            pieces_gpu,
                        },
                    );
                }
                st.in_flight.remove(&key);
                worker.done_cvar.notify_all();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C-ABI Structs (v1.x + v2.0/v2.1 Delta)
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct FdtMeshInput {
    pub num_verts: u32,
    pub verts_xyz: *const f32,
    pub num_loops: u32,
    pub loop_verts: *const u32,
    pub num_polys: u32,
    pub poly_starts: *const u32,
    pub mat_ids: *const i32,
    pub num_uv_layers: u32,
    pub uv_data: *const f32,
    pub world_matrix: *const f32,
    /// Optional per-loop custom/split normals (`num_loops * 3` f32). Null = compute geometric normals.
    pub loop_normals: *const f32,
}

#[repr(C)]
pub struct FdtMeshOutput {
    pub num_verts: u32,
    pub verts_xyz: *mut f32,
    pub num_loops: u32,
    pub loop_verts: *mut u32,
    pub num_polys: u32,
    pub poly_starts: *mut u32,
    pub mat_ids: *mut i32,
    pub is_cut_face: *mut u8,
    pub num_uv_layers: u32,
    pub uv_data: *mut f32,
    pub num_edges: u32,
    pub edge_verts: *mut u32,
    pub loop_edges: *mut u32,
    /// Per-loop preserved/interpolated split normals (`num_loops * 3` f32).
    pub loop_normals: *mut f32,
}

#[repr(C)]
pub struct FdtEvalOutput {
    pub status: i32,
    pub dirty_from_idx: u32,
    pub remaining: FdtMeshOutput,
    pub num_pieces: u32,
    pub pieces: *mut FdtMeshOutput,
}

#[repr(C)]
pub struct FdtGpuSubMesh {
    pub mat_id: i32,
    pub num_tris: u32,
    pub pos_xyz: *mut f32, // length = num_tris * 9
    pub nor_xyz: *mut f32, // length = num_tris * 9
    pub uv_xy: *mut f32,   // length = num_tris * 6 (UVMap0)
    pub uv1_xy: *mut f32,  // length = num_tris * 6 (UVMap1)
}

#[repr(C)]
pub struct FdtGpuMeshData {
    pub num_submeshes: u32,
    pub submeshes: *mut FdtGpuSubMesh,
}

#[repr(C)]
pub struct FdtGpuPreviewOutput {
    pub job_id: u64,
    pub compute_ms: f32,
    pub dirty_from_idx: u32,
    pub remaining: FdtGpuMeshData,
    pub num_pieces: u32,
    pub pieces: *mut FdtGpuMeshData,
}

#[repr(C)]
pub struct FdtGpuChunkDelta {
    pub chunk_id: u32,
    pub mesh: FdtGpuMeshData,
}

#[repr(C)]
pub struct FdtGpuChunkOwner {
    pub chunk_id: u32,
    pub owner_cutter_id: u32,
}

#[repr(C)]
pub struct FdtGpuPieceDelta {
    pub cutter_id: u32,
    pub mesh: FdtGpuMeshData,
}

#[repr(C)]
pub struct FdtGpuDeltaOutput {
    pub job_id: u64,
    pub compute_ms: f32,
    pub recomputed_islands: u32,
    pub num_updated_chunks: u32,
    pub updated_chunks: *mut FdtGpuChunkDelta,
    pub num_removed_chunks: u32,
    pub removed_chunk_ids: *mut u32,
    pub num_updated_pieces: u32,
    pub updated_pieces: *mut FdtGpuPieceDelta,
    pub num_removed_pieces: u32,
    pub removed_piece_ids: *mut u32,
    pub num_ordered_pieces: u32,
    pub ordered_piece_ids: *mut u32,
    /// v210: whole base chunks whose owner changed (`0` = Remaining, `> 0` = Piece of cutter_id).
    pub num_chunk_owners: u32,
    pub chunk_owners: *mut FdtGpuChunkOwner,
}

// ---------------------------------------------------------------------------
// Helpers: C <-> HybridMesh & GPU Buffers
// ---------------------------------------------------------------------------

unsafe fn read_mat4(ptr: *const f32) -> DMat4 {
    if ptr.is_null() {
        return DMat4::IDENTITY;
    }
    let slice = std::slice::from_raw_parts(ptr, 16);
    let mut arr = [0.0f64; 16];
    for i in 0..16 {
        arr[i] = slice[i] as f64;
    }
    DMat4::from_cols_array(&arr)
}

unsafe fn read_mesh_input(inp: &FdtMeshInput) -> (HybridMesh, DMat4) {
    let nv = inp.num_verts as usize;
    let nl = inp.num_loops as usize;
    let np = inp.num_polys as usize;
    let nuv = inp.num_uv_layers as usize;

    let mut positions = Vec::with_capacity(nv);
    if nv > 0 && !inp.verts_xyz.is_null() {
        let v_slice = std::slice::from_raw_parts(inp.verts_xyz, nv * 3);
        for i in 0..nv {
            positions.push([
                v_slice[i * 3] as f64,
                v_slice[i * 3 + 1] as f64,
                v_slice[i * 3 + 2] as f64,
            ]);
        }
    }

    let loop_verts = if nl > 0 && !inp.loop_verts.is_null() {
        std::slice::from_raw_parts(inp.loop_verts, nl).to_vec()
    } else {
        Vec::new()
    };

    let poly_starts = if np > 0 && !inp.poly_starts.is_null() {
        std::slice::from_raw_parts(inp.poly_starts, np + 1).to_vec()
    } else {
        vec![0]
    };

    let mat_ids = if np > 0 && !inp.mat_ids.is_null() {
        std::slice::from_raw_parts(inp.mat_ids, np).to_vec()
    } else {
        vec![0; np]
    };

    let mut uv_layers = Vec::with_capacity(nuv);
    if nuv > 0 && nl > 0 && !inp.uv_data.is_null() {
        let all_uv = std::slice::from_raw_parts(inp.uv_data, nuv * nl * 2);
        for layer_idx in 0..nuv {
            let base = layer_idx * nl * 2;
            let mut layer = Vec::with_capacity(nl);
            for li in 0..nl {
                layer.push([all_uv[base + li * 2], all_uv[base + li * 2 + 1]]);
            }
            uv_layers.push(layer);
        }
    } else {
        for _ in 0..nuv {
            uv_layers.push(vec![[0.0, 0.0]; nl]);
        }
    }

    let loop_normals = if nl > 0 && !inp.loop_normals.is_null() {
        let raw_n = std::slice::from_raw_parts(inp.loop_normals, nl * 3);
        let mut out_n = Vec::with_capacity(nl);
        for li in 0..nl {
            out_n.push([raw_n[li * 3], raw_n[li * 3 + 1], raw_n[li * 3 + 2]]);
        }
        out_n
    } else {
        Vec::new()
    };

    let mat = read_mat4(inp.world_matrix);
    let mut mesh = HybridMesh {
        positions,
        loop_verts,
        poly_starts,
        uv_layers,
        loop_normals,
        mat_ids,
        is_cut_face: vec![false; np],
    };
    mesh.ensure_loop_normals();
    (mesh, mat)
}

struct SendMeshOut(FdtMeshOutput);
unsafe impl Send for SendMeshOut {}

fn to_mesh_output(mesh: &HybridMesh, include_edges: bool) -> FdtMeshOutput {
    let nv = mesh.positions.len();
    let nl = mesh.loop_verts.len();
    let np = mesh.num_polys();
    let nuv = mesh.num_uv_layers();

    let mut verts_xyz = Vec::with_capacity(nv * 3);
    for p in &mesh.positions {
        verts_xyz.push(p[0] as f32);
        verts_xyz.push(p[1] as f32);
        verts_xyz.push(p[2] as f32);
    }
    let loop_verts = mesh.loop_verts.clone();
    let poly_starts = mesh.poly_starts.clone();
    let mat_ids = mesh.mat_ids.clone();
    let is_cut_face: Vec<u8> = mesh.is_cut_face.iter().map(|&b| u8::from(b)).collect();

    let mut uv_data = Vec::with_capacity(nuv * nl * 2);
    for layer in &mesh.uv_layers {
        for uv in layer {
            uv_data.push(uv[0]);
            uv_data.push(uv[1]);
        }
    }

    let mut loop_nrm_flat = Vec::with_capacity(nl * 3);
    if mesh.loop_normals.len() == nl {
        for n in &mesh.loop_normals {
            loop_nrm_flat.push(n[0]);
            loop_nrm_flat.push(n[1]);
            loop_nrm_flat.push(n[2]);
        }
    } else {
        for pi in 0..np {
            let pn = mesh.poly_normal(pi);
            let cnt = mesh.poly_loop_range(pi).len();
            for _ in 0..cnt {
                loop_nrm_flat.push(pn[0] as f32);
                loop_nrm_flat.push(pn[1] as f32);
                loop_nrm_flat.push(pn[2] as f32);
            }
        }
    }

    let (num_edges, edge_verts_ptr, loop_edges_ptr) = if include_edges && nl > 0 {
        let (ev, le) = mesh.compute_edges();
        let ne = (ev.len() / 2) as u32;
        (
            ne,
            Box::into_raw(ev.into_boxed_slice()) as *mut u32,
            Box::into_raw(le.into_boxed_slice()) as *mut u32,
        )
    } else {
        (0u32, std::ptr::null_mut(), std::ptr::null_mut())
    };

    FdtMeshOutput {
        num_verts: nv as u32,
        verts_xyz: Box::into_raw(verts_xyz.into_boxed_slice()) as *mut f32,
        num_loops: nl as u32,
        loop_verts: Box::into_raw(loop_verts.into_boxed_slice()) as *mut u32,
        num_polys: np as u32,
        poly_starts: Box::into_raw(poly_starts.into_boxed_slice()) as *mut u32,
        mat_ids: Box::into_raw(mat_ids.into_boxed_slice()) as *mut i32,
        is_cut_face: Box::into_raw(is_cut_face.into_boxed_slice()) as *mut u8,
        num_uv_layers: nuv as u32,
        uv_data: Box::into_raw(uv_data.into_boxed_slice()) as *mut f32,
        num_edges,
        edge_verts: edge_verts_ptr,
        loop_edges: loop_edges_ptr,
        loop_normals: Box::into_raw(loop_nrm_flat.into_boxed_slice()) as *mut f32,
    }
}

fn to_gpu_mesh_data_owned(g: GpuPackedMesh) -> FdtGpuMeshData {
    if g.submeshes.is_empty() {
        return FdtGpuMeshData {
            num_submeshes: 0,
            submeshes: std::ptr::null_mut(),
        };
    }
    let subs: Vec<FdtGpuSubMesh> = g
        .submeshes
        .into_iter()
        .map(|s| FdtGpuSubMesh {
            mat_id: s.mat_id,
            num_tris: s.num_tris,
            pos_xyz: Box::into_raw(s.pos_xyz.into_boxed_slice()) as *mut f32,
            nor_xyz: Box::into_raw(s.nor_xyz.into_boxed_slice()) as *mut f32,
            uv_xy: Box::into_raw(s.uv_xy.into_boxed_slice()) as *mut f32,
            uv1_xy: Box::into_raw(s.uv1_xy.into_boxed_slice()) as *mut f32,
        })
        .collect();
    let n = subs.len() as u32;
    FdtGpuMeshData {
        num_submeshes: n,
        submeshes: Box::into_raw(subs.into_boxed_slice()) as *mut FdtGpuSubMesh,
    }
}

unsafe fn free_mesh_output(m: &mut FdtMeshOutput) {
    let nv = m.num_verts as usize;
    let nl = m.num_loops as usize;
    let np = m.num_polys as usize;
    let nuv = m.num_uv_layers as usize;
    let ne = m.num_edges as usize;

    if !m.verts_xyz.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.verts_xyz, nv * 3));
        m.verts_xyz = std::ptr::null_mut();
    }
    if !m.loop_verts.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.loop_verts, nl));
        m.loop_verts = std::ptr::null_mut();
    }
    if !m.poly_starts.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.poly_starts, np + 1));
        m.poly_starts = std::ptr::null_mut();
    }
    if !m.mat_ids.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.mat_ids, np));
        m.mat_ids = std::ptr::null_mut();
    }
    if !m.is_cut_face.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.is_cut_face, np));
        m.is_cut_face = std::ptr::null_mut();
    }
    if !m.uv_data.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.uv_data, nuv * nl * 2));
        m.uv_data = std::ptr::null_mut();
    }
    if !m.edge_verts.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.edge_verts, ne * 2));
        m.edge_verts = std::ptr::null_mut();
    }
    if !m.loop_edges.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.loop_edges, nl));
        m.loop_edges = std::ptr::null_mut();
    }
    if !m.loop_normals.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.loop_normals, nl * 3));
        m.loop_normals = std::ptr::null_mut();
    }
}

unsafe fn free_gpu_mesh_data(m: &mut FdtGpuMeshData) {
    if m.submeshes.is_null() {
        return;
    }
    let ns = m.num_submeshes as usize;
    let mut subs = Box::from_raw(std::ptr::slice_from_raw_parts_mut(m.submeshes, ns));
    for s in subs.iter_mut() {
        let nt = s.num_tris as usize;
        if !s.pos_xyz.is_null() {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(s.pos_xyz, nt * 9));
            s.pos_xyz = std::ptr::null_mut();
        }
        if !s.nor_xyz.is_null() {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(s.nor_xyz, nt * 9));
            s.nor_xyz = std::ptr::null_mut();
        }
        if !s.uv_xy.is_null() {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(s.uv_xy, nt * 6));
            s.uv_xy = std::ptr::null_mut();
        }
        if !s.uv1_xy.is_null() {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(s.uv1_xy, nt * 6));
            s.uv1_xy = std::ptr::null_mut();
        }
    }
    m.submeshes = std::ptr::null_mut();
}

fn pack_full_eval_output(tc: &IslandTargetState, dirty_idx: usize, gen_edges: bool) -> *mut FdtEvalOutput {
    let (final_rem, pieces) = tc.assemble_full_eval();
    let (rem_wrapped, piece_wrapped): (SendMeshOut, Vec<SendMeshOut>) = rayon::join(
        || SendMeshOut(to_mesh_output(&final_rem, gen_edges)),
        || {
            pieces
                .par_iter()
                .map(|p| SendMeshOut(to_mesh_output(p, gen_edges)))
                .collect()
        },
    );

    let rem_out = rem_wrapped.0;
    let piece_outs: Vec<FdtMeshOutput> = piece_wrapped.into_iter().map(|w| w.0).collect();
    let num_pieces = piece_outs.len() as u32;
    let pieces_ptr = Box::into_raw(piece_outs.into_boxed_slice()) as *mut FdtMeshOutput;

    Box::into_raw(Box::new(FdtEvalOutput {
        status: 0,
        dirty_from_idx: dirty_idx as u32,
        remaining: rem_out,
        num_pieces,
        pieces: pieces_ptr,
    }))
}

fn pack_delta_output(
    job_id: u64,
    compute_ms: f32,
    recomputed_islands: u32,
    snap: DeltaGpuSnapshot,
) -> *mut FdtGpuDeltaOutput {
    let updated_chunks: Vec<FdtGpuChunkDelta> = snap
        .updated_chunks
        .into_iter()
        .map(|(chunk_id, mesh)| FdtGpuChunkDelta {
            chunk_id,
            mesh: to_gpu_mesh_data_owned(mesh),
        })
        .collect();
    let num_updated_chunks = updated_chunks.len() as u32;
    let updated_chunks_ptr = if num_updated_chunks > 0 {
        Box::into_raw(updated_chunks.into_boxed_slice()) as *mut FdtGpuChunkDelta
    } else {
        std::ptr::null_mut()
    };

    let num_removed_chunks = snap.removed_chunk_ids.len() as u32;
    let removed_chunk_ids_ptr = if num_removed_chunks > 0 {
        Box::into_raw(snap.removed_chunk_ids.into_boxed_slice()) as *mut u32
    } else {
        std::ptr::null_mut()
    };

    let updated_pieces: Vec<FdtGpuPieceDelta> = snap
        .updated_pieces
        .into_iter()
        .map(|(cutter_id, mesh)| FdtGpuPieceDelta {
            cutter_id,
            mesh: to_gpu_mesh_data_owned(mesh),
        })
        .collect();
    let num_updated_pieces = updated_pieces.len() as u32;
    let updated_pieces_ptr = if num_updated_pieces > 0 {
        Box::into_raw(updated_pieces.into_boxed_slice()) as *mut FdtGpuPieceDelta
    } else {
        std::ptr::null_mut()
    };

    let num_removed_pieces = snap.removed_piece_ids.len() as u32;
    let removed_piece_ids_ptr = if num_removed_pieces > 0 {
        Box::into_raw(snap.removed_piece_ids.into_boxed_slice()) as *mut u32
    } else {
        std::ptr::null_mut()
    };

    let num_ordered_pieces = snap.ordered_piece_ids.len() as u32;
    let ordered_piece_ids_ptr = if num_ordered_pieces > 0 {
        Box::into_raw(snap.ordered_piece_ids.into_boxed_slice()) as *mut u32
    } else {
        std::ptr::null_mut()
    };

    let chunk_owners: Vec<FdtGpuChunkOwner> = snap
        .chunk_owners
        .into_iter()
        .map(|(chunk_id, owner_cutter_id)| FdtGpuChunkOwner {
            chunk_id,
            owner_cutter_id,
        })
        .collect();
    let num_chunk_owners = chunk_owners.len() as u32;
    let chunk_owners_ptr = if num_chunk_owners > 0 {
        Box::into_raw(chunk_owners.into_boxed_slice()) as *mut FdtGpuChunkOwner
    } else {
        std::ptr::null_mut()
    };

    Box::into_raw(Box::new(FdtGpuDeltaOutput {
        job_id,
        compute_ms,
        recomputed_islands,
        num_updated_chunks,
        updated_chunks: updated_chunks_ptr,
        num_removed_chunks,
        removed_chunk_ids: removed_chunk_ids_ptr,
        num_updated_pieces,
        updated_pieces: updated_pieces_ptr,
        num_removed_pieces,
        removed_piece_ids: removed_piece_ids_ptr,
        num_ordered_pieces,
        ordered_piece_ids: ordered_piece_ids_ptr,
        num_chunk_owners,
        chunk_owners: chunk_owners_ptr,
    }))
}

// ---------------------------------------------------------------------------
// Exported C-ABI Functions (v2.1.0)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn fdt_engine_version() -> u32 {
    210 // v2.1.0 (Chunk-Level Interior Ownership + Zero-BVH-Rebuild + Async Target Upload)
}

#[no_mangle]
pub extern "C" fn fdt_session_create() -> u64 {
    catch_unwind(|| {
        let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        with_sessions(|map| {
            map.insert(
                id,
                SessionState {
                    targets: HashMap::new(),
                },
            );
        });
        id
    })
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn fdt_session_destroy(handle: u64) -> i32 {
    catch_unwind(|| {
        if let Some(worker) = WORKER.get() {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            st.pending_targets.retain(|&(h, _), _| h != handle);
            st.pending.retain(|&(h, _), _| h != handle);
            st.completed_delta_meta.retain(|&(h, _), _| h != handle);
            st.completed_legacy.retain(|&(h, _), _| h != handle);
        }
        with_sessions(|map| {
            map.remove(&handle);
        });
        0
    })
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn fdt_session_set_target(
    handle: u64,
    target_id: u32,
    input: *const FdtMeshInput,
) -> i32 {
    catch_unwind(|| {
        if input.is_null() {
            return -2;
        }
        let (base_mesh, world_mat) = read_mesh_input(&*input);
        let state = Arc::new(Mutex::new(IslandTargetState::new(base_mesh, world_mat)));
        if let Some(worker) = WORKER.get() {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            let key = (handle, target_id);
            st.pending_targets.remove(&key);
            st.pending.remove(&key);
            st.completed_delta_meta.remove(&key);
            st.completed_legacy.remove(&key);
            while st.in_flight.contains(&key) {
                st = worker.done_cvar.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
        with_sessions(|map| {
            if let Some(sess) = map.get_mut(&handle) {
                sess.targets.insert(target_id, state);
                0
            } else {
                -1
            }
        })
    })
    .unwrap_or(-99)
}

/// v210 Non-blocking async target upload: copies raw mesh arrays on Main Thread,
/// then builds the SpatialChunks + Dipole BVH + pre-warmed base GPU buffers on
/// the background worker thread so Blender NEVER freezes ("Not Responding") on
/// 4M-triangle targets!
#[no_mangle]
pub unsafe extern "C" fn fdt_session_set_target_async(
    handle: u64,
    target_id: u32,
    input: *const FdtMeshInput,
) -> i32 {
    catch_unwind(|| {
        if input.is_null() {
            return -2;
        }
        let exists = with_sessions(|map| map.contains_key(&handle));
        if !exists {
            return -1;
        }
        let (base_mesh, world_mat) = read_mesh_input(&*input);
        let worker = get_worker();
        let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        let key = (handle, target_id);
        st.pending_targets.insert(key, (base_mesh, world_mat));
        st.pending.remove(&key);
        st.completed_delta_meta.remove(&key);
        st.completed_legacy.remove(&key);
        worker.work_cvar.notify_one();
        0
    })
    .unwrap_or(-99)
}

/// Synchronous evaluation (used on Apply, Rebuild, or when GPU Fast Drag is disabled).
#[no_mangle]
pub unsafe extern "C" fn fdt_session_eval_cutters(
    handle: u64,
    target_id: u32,
    cutters: *const FdtMeshInput,
    num_cutters: u32,
    cut_mat_idx: i32,
    uv_scale: f32,
    include_edges: u8,
) -> *mut FdtEvalOutput {
    let res = catch_unwind(|| {
        let nc = num_cutters as usize;
        let gen_edges = include_edges != 0;
        let cutter_inputs = if nc > 0 && !cutters.is_null() {
            std::slice::from_raw_parts(cutters, nc)
        } else {
            &[]
        };

        let raw_cutters: Vec<(HybridMesh, DMat4)> = cutter_inputs
            .iter()
            .map(|c| read_mesh_input(c))
            .collect();

        wait_for_target_idle(handle, target_id);

        let target_arc = match get_target_arc(handle, target_id) {
            Some(a) => a,
            None => return std::ptr::null_mut(),
        };
        let mut tc = target_arc.lock().unwrap_or_else(|e| e.into_inner());
        let ordered_ids: Vec<u32> = (1..=(nc as u32)).collect();
        let updates: Vec<(u32, HybridMesh, DMat4)> = raw_cutters
            .into_iter()
            .enumerate()
            .map(|(i, (m, mat))| ((i + 1) as u32, m, mat))
            .collect();
        tc.upsert_cutters(updates);
        let recomputed = tc.evaluate_active(&ordered_ids, cut_mat_idx, uv_scale, false);
        tc.retain_cutters(&ordered_ids);
        let dirty_idx = if recomputed == 0 { nc } else { 0 };
        pack_full_eval_output(&tc, dirty_idx, gen_edges)
    });

    res.unwrap_or(std::ptr::null_mut())
}

/// Legacy non-blocking full-list async submission.
#[no_mangle]
pub unsafe extern "C" fn fdt_session_submit_async(
    handle: u64,
    target_id: u32,
    cutters: *const FdtMeshInput,
    num_cutters: u32,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> u64 {
    catch_unwind(|| {
        let nc = num_cutters as usize;
        let cutter_inputs = if nc > 0 && !cutters.is_null() {
            std::slice::from_raw_parts(cutters, nc)
        } else {
            &[]
        };
        let raw_cutters: Vec<(HybridMesh, DMat4)> = cutter_inputs
            .iter()
            .map(|c| read_mesh_input(c))
            .collect();

        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        let worker = get_worker();
        let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        st.pending.insert(
            (handle, target_id),
            PendingJob::FullLegacy {
                job_id,
                raw_cutters,
                cut_mat_idx,
                uv_scale,
            },
        );
        worker.work_cvar.notify_one();
        job_id
    })
    .unwrap_or(0)
}

/// v200 Non-blocking Delta Async Submission (< 0.1 ms on Main Thread even with 500 boxes!).
/// Only changed cutters are transferred across FFI; pending delta updates merge losslessly.
#[no_mangle]
pub unsafe extern "C" fn fdt_session_submit_async_delta(
    handle: u64,
    target_id: u32,
    updated_ids: *const u32,
    updated_cutters: *const FdtMeshInput,
    num_updated: u32,
    ordered_active_ids: *const u32,
    num_active: u32,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> u64 {
    catch_unwind(|| {
        let nu = num_updated as usize;
        let na = num_active as usize;

        let mut updates: HashMap<u32, (HybridMesh, DMat4)> = HashMap::with_capacity(nu);
        if nu > 0 && !updated_ids.is_null() && !updated_cutters.is_null() {
            let ids = std::slice::from_raw_parts(updated_ids, nu);
            let inputs = std::slice::from_raw_parts(updated_cutters, nu);
            for i in 0..nu {
                updates.insert(ids[i], read_mesh_input(&inputs[i]));
            }
        }

        let active_vec: Vec<u32> = if na > 0 && !ordered_active_ids.is_null() {
            std::slice::from_raw_parts(ordered_active_ids, na).to_vec()
        } else {
            Vec::new()
        };

        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        let worker = get_worker();
        let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        let key = (handle, target_id);

        // Lossless Latest-Wins merge: preserve any not-yet-processed updates for OTHER cutter IDs!
        let merged_updates = match st.pending.remove(&key) {
            Some(PendingJob::Delta {
                mut updated_cutters,
                ..
            }) => {
                for (cid, val) in updates {
                    updated_cutters.insert(cid, val);
                }
                updated_cutters
            }
            _ => updates,
        };

        st.pending.insert(
            key,
            PendingJob::Delta {
                job_id,
                updated_cutters: merged_updates,
                ordered_active_ids: active_vec,
                cut_mat_idx,
                uv_scale,
            },
        );
        worker.work_cvar.notify_one();
        job_id
    })
    .unwrap_or(0)
}

/// v200 Non-blocking Delta GPU Poll (`try_lock` — NEVER blocks Blender's Main Thread!).
/// Returns NULL if no newer frame is ready or if the worker is mid-compute on the next frame.
#[no_mangle]
pub extern "C" fn fdt_session_poll_gpu_delta(
    handle: u64,
    target_id: u32,
    last_seen_job_id: u64,
) -> *mut FdtGpuDeltaOutput {
    catch_unwind(|| {
        let worker = match WORKER.get() {
            Some(w) => w,
            None => return std::ptr::null_mut(),
        };
        let meta = {
            let st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            match st.completed_delta_meta.get(&(handle, target_id)) {
                Some(m) if m.job_id > last_seen_job_id => (m.job_id, m.compute_ms, m.recomputed_islands),
                _ => return std::ptr::null_mut(),
            }
        };

        let target_arc = match get_target_arc(handle, target_id) {
            Some(a) => a,
            None => return std::ptr::null_mut(),
        };

        // Non-blocking try_lock: if worker is mid-compute on next frame, skip this poll tick cleanly!
        let mut tc = match target_arc.try_lock() {
            Ok(g) => g,
            Err(_) => return std::ptr::null_mut(),
        };
        let snap = tc.take_gpu_delta();
        pack_delta_output(meta.0, meta.1, meta.2, snap)
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn fdt_gpu_delta_free(ptr: *mut FdtGpuDeltaOutput) {
    let _ = catch_unwind(|| {
        if ptr.is_null() {
            return;
        }
        let boxed = Box::from_raw(ptr);
        if !boxed.updated_chunks.is_null() {
            let n = boxed.num_updated_chunks as usize;
            let mut slice =
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(boxed.updated_chunks, n));
            for c in slice.iter_mut() {
                free_gpu_mesh_data(&mut c.mesh);
            }
        }
        if !boxed.removed_chunk_ids.is_null() {
            let n = boxed.num_removed_chunks as usize;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                boxed.removed_chunk_ids,
                n,
            ));
        }
        if !boxed.updated_pieces.is_null() {
            let n = boxed.num_updated_pieces as usize;
            let mut slice =
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(boxed.updated_pieces, n));
            for p in slice.iter_mut() {
                free_gpu_mesh_data(&mut p.mesh);
            }
        }
        if !boxed.removed_piece_ids.is_null() {
            let n = boxed.num_removed_pieces as usize;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                boxed.removed_piece_ids,
                n,
            ));
        }
        if !boxed.ordered_piece_ids.is_null() {
            let n = boxed.num_ordered_pieces as usize;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                boxed.ordered_piece_ids,
                n,
            ));
        }
        if !boxed.chunk_owners.is_null() {
            let n = boxed.num_chunk_owners as usize;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                boxed.chunk_owners,
                n,
            ));
        }
    });
}

/// Legacy non-blocking poll for full GPU preview frame (> `last_seen_job_id`).
#[no_mangle]
pub extern "C" fn fdt_session_poll_gpu_preview(
    handle: u64,
    target_id: u32,
    last_seen_job_id: u64,
) -> *mut FdtGpuPreviewOutput {
    catch_unwind(|| {
        let worker = match WORKER.get() {
            Some(w) => w,
            None => return std::ptr::null_mut(),
        };
        let frame = {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            let ready = st
                .completed_legacy
                .get(&(handle, target_id))
                .map_or(false, |f| f.job_id > last_seen_job_id);
            if !ready {
                return std::ptr::null_mut();
            }
            st.completed_legacy.remove(&(handle, target_id)).unwrap()
        };

        let rem_gpu = to_gpu_mesh_data_owned(frame.remaining_gpu);
        let piece_gpus: Vec<FdtGpuMeshData> = frame
            .pieces_gpu
            .into_iter()
            .map(to_gpu_mesh_data_owned)
            .collect();
        let num_pieces = piece_gpus.len() as u32;
        let pieces_ptr = Box::into_raw(piece_gpus.into_boxed_slice()) as *mut FdtGpuMeshData;

        Box::into_raw(Box::new(FdtGpuPreviewOutput {
            job_id: frame.job_id,
            compute_ms: frame.compute_ms,
            dirty_from_idx: frame.dirty_from_idx,
            remaining: rem_gpu,
            num_pieces,
            pieces: pieces_ptr,
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn fdt_gpu_preview_free(ptr: *mut FdtGpuPreviewOutput) {
    let _ = catch_unwind(|| {
        if ptr.is_null() {
            return;
        }
        let mut boxed = Box::from_raw(ptr);
        free_gpu_mesh_data(&mut boxed.remaining);
        if !boxed.pieces.is_null() {
            let n = boxed.num_pieces as usize;
            let mut pieces_box =
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(boxed.pieces, n));
            for p in pieces_box.iter_mut() {
                free_gpu_mesh_data(p);
            }
        }
    });
}

/// Wait for any in-flight async job for `(handle, target_id)` to finish, then return
/// the already-computed `FdtEvalOutput` from cache with ZERO boolean recomputation!
#[no_mangle]
pub extern "C" fn fdt_session_fetch_cached_eval(
    handle: u64,
    target_id: u32,
    include_edges: u8,
) -> *mut FdtEvalOutput {
    catch_unwind(|| {
        wait_for_target_idle(handle, target_id);

        let gen_edges = include_edges != 0;
        let target_arc = match get_target_arc(handle, target_id) {
            Some(a) => a,
            None => return std::ptr::null_mut(),
        };
        let tc = target_arc.lock().unwrap_or_else(|e| e.into_inner());
        pack_full_eval_output(&tc, 0, gen_edges)
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn fdt_eval_free(ptr: *mut FdtEvalOutput) {
    let _ = catch_unwind(|| {
        if ptr.is_null() {
            return;
        }
        let mut boxed = Box::from_raw(ptr);
        free_mesh_output(&mut boxed.remaining);
        if !boxed.pieces.is_null() {
            let n = boxed.num_pieces as usize;
            let mut pieces_box =
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(boxed.pieces, n));
            for p in pieces_box.iter_mut() {
                free_mesh_output(p);
            }
        }
    });
}

#[cfg(test)]
mod async_tests {
    use super::*;

    fn box_input(cx: f32) -> (Vec<f32>, Vec<u32>, Vec<u32>, Vec<i32>, Vec<f32>, [f32; 16]) {
        box_input_at(cx, 0.0, 0.0, 1.0, 1.0, 1.0)
    }

    fn box_input_at(
        cx: f32,
        cy: f32,
        cz: f32,
        hx: f32,
        hy: f32,
        hz: f32,
    ) -> (Vec<f32>, Vec<u32>, Vec<u32>, Vec<i32>, Vec<f32>, [f32; 16]) {
        let verts = vec![
            cx - hx, cy - hy, cz - hz,
            cx + hx, cy - hy, cz - hz,
            cx + hx, cy + hy, cz - hz,
            cx - hx, cy + hy, cz - hz,
            cx - hx, cy - hy, cz + hz,
            cx + hx, cy - hy, cz + hz,
            cx + hx, cy + hy, cz + hz,
            cx - hx, cy + hy, cz + hz,
        ];
        let loops = vec![
            4, 5, 6, 7,
            0, 1, 5, 4,
            2, 3, 7, 6,
            3, 0, 4, 7,
            1, 2, 6, 5,
            0, 3, 2, 1,
        ];
        let starts = vec![0, 4, 8, 12, 16, 20, 24];
        let mats = vec![0; 6];
        let uvs = vec![0.0; 48];
        let mat = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        (verts, loops, starts, mats, uvs, mat)
    }

    #[test]
    fn test_async_latest_wins_and_fetch_cached() {
        let h = fdt_session_create();
        assert!(h > 0);

        let (tv, tl, ts, tm, tuv, tmat) = box_input(0.0);
        let mut custom_nrms = vec![0.0f32; 24 * 3];
        for li in 0..4 {
            custom_nrms[li * 3] = 0.0;
            custom_nrms[li * 3 + 1] = 0.6;
            custom_nrms[li * 3 + 2] = 0.8;
        }
        for li in 4..24 {
            custom_nrms[li * 3 + 2] = 1.0;
        }
        let t_inp = FdtMeshInput {
            num_verts: 8,
            verts_xyz: tv.as_ptr(),
            num_loops: 24,
            loop_verts: tl.as_ptr(),
            num_polys: 6,
            poly_starts: ts.as_ptr(),
            mat_ids: tm.as_ptr(),
            num_uv_layers: 1,
            uv_data: tuv.as_ptr(),
            world_matrix: tmat.as_ptr(),
            loop_normals: custom_nrms.as_ptr(),
        };
        unsafe {
            assert_eq!(fdt_session_set_target(h, 1, &t_inp), 0);
        }

        let t_sub = Instant::now();
        let mut last_jid = 0u64;
        for step in 0..10 {
            let (cv, cl, cs, cm, cuv, cmat) = box_input(0.5 + (step as f32) * 0.05);
            let c_inp = FdtMeshInput {
                num_verts: 8,
                verts_xyz: cv.as_ptr(),
                num_loops: 24,
                loop_verts: cl.as_ptr(),
                num_polys: 6,
                poly_starts: cs.as_ptr(),
                mat_ids: cm.as_ptr(),
                num_uv_layers: 0,
                uv_data: cuv.as_ptr(),
                world_matrix: cmat.as_ptr(),
                loop_normals: std::ptr::null(),
            };
            unsafe {
                last_jid = fdt_session_submit_async(h, 1, &c_inp, 1, 1, 0.5);
            }
        }
        let submit_ms = t_sub.elapsed().as_secs_f64() * 1000.0;
        println!("10 async submits took {:.3} ms (last_jid={})", submit_ms, last_jid);
        assert!(last_jid > 0);

        let eval_ptr = fdt_session_fetch_cached_eval(h, 1, 1);
        assert!(!eval_ptr.is_null());
        unsafe {
            let ev = &*eval_ptr;
            assert_eq!(ev.status, 0);
            assert!(ev.remaining.num_polys > 0);
            assert!(!ev.remaining.loop_normals.is_null());
            let rn = std::slice::from_raw_parts(
                ev.remaining.loop_normals,
                (ev.remaining.num_loops as usize) * 3,
            );
            let found_tilted = (0..ev.remaining.num_loops as usize).any(|i| {
                (rn[i * 3 + 1] - 0.6).abs() < 1e-4 && (rn[i * 3 + 2] - 0.8).abs() < 1e-4
            });
            assert!(
                found_tilted,
                "Custom tilted split normal should survive retriangulation!"
            );
            assert_eq!(ev.num_pieces, 1);
            fdt_eval_free(eval_ptr);
        }

        let gpu_ptr = fdt_session_poll_gpu_preview(h, 1, 0);
        assert!(!gpu_ptr.is_null());
        unsafe {
            let gp = &*gpu_ptr;
            assert_eq!(gp.job_id, last_jid);
            assert!(gp.remaining.num_submeshes >= 2);
            assert_eq!(gp.num_pieces, 1);
            assert!((*gp.pieces).num_submeshes >= 2);
            fdt_gpu_preview_free(gpu_ptr);
        }

        assert_eq!(fdt_session_destroy(h), 0);
    }

    #[test]
    fn test_v200_delta_islands_and_layer_priority() {
        // Verify:
        // 1. Non-overlapping cutters live in separate islands -> moving cutter #1 (index 0!)
        //    recomputes ONLY 1 island (not all cutters) and sends minimal delta chunks/pieces!
        // 2. Overlapping cutters obey exact Layer #1 > Layer #2 priority!
        let h = fdt_session_create();
        let (tv, tl, ts, tm, tuv, tmat) = box_input_at(0.0, 0.0, 0.0, 10.0, 10.0, 2.0);
        let t_inp = FdtMeshInput {
            num_verts: 8,
            verts_xyz: tv.as_ptr(),
            num_loops: 24,
            loop_verts: tl.as_ptr(),
            num_polys: 6,
            poly_starts: ts.as_ptr(),
            mat_ids: tm.as_ptr(),
            num_uv_layers: 1,
            uv_data: tuv.as_ptr(),
            world_matrix: tmat.as_ptr(),
            loop_normals: std::ptr::null(),
        };
        unsafe {
            assert_eq!(fdt_session_set_target(h, 1, &t_inp), 0);
        }

        // Cutter 10 (top layer, index 0) and Cutter 20 (layer 1, index 1) overlapping on right corner
        let (c10v, c10l, c10s, c10m, c10uv, c10mat) = box_input_at(9.0, 0.0, 0.0, 1.5, 2.0, 3.0);
        let (c20v, c20l, c20s, c20m, c20uv, c20mat) = box_input_at(8.5, 0.0, 0.0, 1.5, 2.0, 3.0);
        let c_inps = [
            FdtMeshInput {
                num_verts: 8,
                verts_xyz: c10v.as_ptr(),
                num_loops: 24,
                loop_verts: c10l.as_ptr(),
                num_polys: 6,
                poly_starts: c10s.as_ptr(),
                mat_ids: c10m.as_ptr(),
                num_uv_layers: 0,
                uv_data: c10uv.as_ptr(),
                world_matrix: c10mat.as_ptr(),
                loop_normals: std::ptr::null(),
            },
            FdtMeshInput {
                num_verts: 8,
                verts_xyz: c20v.as_ptr(),
                num_loops: 24,
                loop_verts: c20l.as_ptr(),
                num_polys: 6,
                poly_starts: c20s.as_ptr(),
                mat_ids: c20m.as_ptr(),
                num_uv_layers: 0,
                uv_data: c20uv.as_ptr(),
                world_matrix: c20mat.as_ptr(),
                loop_normals: std::ptr::null(),
            },
        ];
        let ids = [10u32, 20u32];
        let jid1 = unsafe {
            fdt_session_submit_async_delta(
                h,
                1,
                ids.as_ptr(),
                c_inps.as_ptr(),
                2,
                ids.as_ptr(),
                2,
                1,
                0.5,
            )
        };
        assert!(jid1 > 0);
        // Wait via fetch_cached_eval and verify piece 10 & piece 20 are non-empty and non-overlapping
        let ev_ptr = fdt_session_fetch_cached_eval(h, 1, 0);
        assert!(!ev_ptr.is_null());
        unsafe {
            let ev = &*ev_ptr;
            assert_eq!(ev.status, 0);
            assert_eq!(ev.num_pieces, 2);
            let p0 = &*ev.pieces.add(0);
            let p1 = &*ev.pieces.add(1);
            assert!(p0.num_polys > 0, "Layer #1 cutter must produce piece");
            assert!(p1.num_polys > 0, "Layer #2 overlapping cutter must produce remainder piece");
            // Layer #2 piece (x in [7.0, 7.5]) must have NO vertex strictly > 7.51 (Layer #1 claimed x >= 7.5!)
            let p1_verts = std::slice::from_raw_parts(p1.verts_xyz, (p1.num_verts as usize) * 3);
            let max_x = p1_verts
                .chunks_exact(3)
                .map(|v| v[0])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max_x <= 7.52,
                "Layer #1 priority violated: Layer #2 piece reached max_x={}",
                max_x
            );
            fdt_eval_free(ev_ptr);
        }

        // Poll delta should deliver chunks & 2 pieces
        let d1 = fdt_session_poll_gpu_delta(h, 1, 0);
        assert!(!d1.is_null());
        unsafe {
            assert_eq!((*d1).job_id, jid1);
            assert!((*d1).num_updated_chunks >= 1);
            assert_eq!((*d1).num_updated_pieces, 2);
            fdt_gpu_delta_free(d1);
        }

        // Second poll with same job_id returns NULL
        assert!(fdt_session_poll_gpu_delta(h, 1, jid1).is_null());

        assert_eq!(fdt_session_destroy(h), 0);
    }

    #[test]
    fn test_v210_async_target_and_chunk_owners() {
        assert_eq!(fdt_engine_version(), 210);
        let h = fdt_session_create();
        assert!(h > 0);

        // Build a 60x60x6 = 21,600-quad (43,200-tri) 2m x 2m x 2m subdivided cube (many spatial chunks!)
        let n = 60usize;
        let stride = n + 1;
        let mut verts = Vec::with_capacity(6 * stride * stride * 3);
        let mut loops = Vec::with_capacity(6 * n * n * 4);
        let mut starts = Vec::with_capacity(6 * n * n + 1);
        starts.push(0u32);
        let mut mats = Vec::with_capacity(6 * n * n);
        let mut uvs = Vec::with_capacity(6 * n * n * 8);
        let inv_n = 1.0f32 / (n as f32);

        for side in 0..6usize {
            let base_v = (verts.len() / 3) as u32;
            for iy in 0..=n {
                let tv = (iy as f32) * inv_n;
                for ix in 0..=n {
                    let tu = (ix as f32) * inv_n;
                    let u = -1.0 + 2.0 * tu;
                    let v = -1.0 + 2.0 * tv;
                    let (x, y, z) = match side {
                        0 => (u, v, 1.0),
                        1 => (u, -v, -1.0),
                        2 => (u, -1.0, v),
                        3 => (-u, 1.0, v),
                        4 => (1.0, u, v),
                        _ => (-1.0, -u, v),
                    };
                    verts.extend_from_slice(&[x, y, z]);
                }
            }
            for iy in 0..n {
                for ix in 0..n {
                    let i00 = base_v + (iy * stride + ix) as u32;
                    let i10 = base_v + (iy * stride + ix + 1) as u32;
                    let i11 = base_v + ((iy + 1) * stride + ix + 1) as u32;
                    let i01 = base_v + ((iy + 1) * stride + ix) as u32;
                    loops.extend_from_slice(&[i00, i10, i11, i01]);
                    starts.push(loops.len() as u32);
                    mats.push(0);
                    uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
                }
            }
        }
        let ident = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let t_inp = FdtMeshInput {
            num_verts: (verts.len() / 3) as u32,
            verts_xyz: verts.as_ptr(),
            num_loops: loops.len() as u32,
            loop_verts: loops.as_ptr(),
            num_polys: mats.len() as u32,
            poly_starts: starts.as_ptr(),
            mat_ids: mats.as_ptr(),
            num_uv_layers: 1,
            uv_data: uvs.as_ptr(),
            world_matrix: ident.as_ptr(),
            loop_normals: std::ptr::null(),
        };

        // 1. Async target upload returns immediately
        unsafe {
            assert_eq!(fdt_session_set_target_async(h, 1, &t_inp), 0);
        }

        // 2. Submit a large cutter covering the ENTIRE +X half (x in [0.0, 2.0]) -> multiple strictly-interior chunks!
        let (cv, cl, cs, cm, cuv, cmat) = box_input_at(1.0, 0.0, 0.0, 1.0, 1.5, 1.5);
        let c_inp = FdtMeshInput {
            num_verts: 8,
            verts_xyz: cv.as_ptr(),
            num_loops: 24,
            loop_verts: cl.as_ptr(),
            num_polys: 6,
            poly_starts: cs.as_ptr(),
            mat_ids: cm.as_ptr(),
            num_uv_layers: 0,
            uv_data: cuv.as_ptr(),
            world_matrix: cmat.as_ptr(),
            loop_normals: std::ptr::null(),
        };
        let cid = 42u32;
        let jid1 = unsafe {
            fdt_session_submit_async_delta(h, 1, &cid, &c_inp, 1, &cid, 1, 1, 0.5)
        };
        assert!(jid1 > 0);

        // Wait for worker completion via fetch_cached_eval, verify full assembly has BOTH halves!
        let ev_ptr = fdt_session_fetch_cached_eval(h, 1, 0);
        assert!(!ev_ptr.is_null());
        unsafe {
            let ev = &*ev_ptr;
            assert_eq!(ev.status, 0);
            assert_eq!(ev.num_pieces, 1);
            let p0 = &*ev.pieces.add(0);
            // Total base quads = 21,600 -> half is ~10,800 quads each (piece + remaining)!
            assert!(
                p0.num_polys >= 9_000,
                "Full assembled piece must include owned interior chunks! got {} polys",
                p0.num_polys
            );
            assert!(
                ev.remaining.num_polys >= 9_000,
                "Full assembled remaining must include outside chunks! got {} polys",
                ev.remaining.num_polys
            );
            fdt_eval_free(ev_ptr);
        }

        // Poll delta: must report chunk_owners flipping interior chunks to owner_cutter_id == 42!
        let d1 = fdt_session_poll_gpu_delta(h, 1, 0);
        assert!(!d1.is_null());
        unsafe {
            let delta = &*d1;
            assert_eq!(delta.job_id, jid1);
            assert!(
                delta.num_chunk_owners > 0 && !delta.chunk_owners.is_null(),
                "Expected interior chunks to flip via chunk_owners (0-copy GPU ownership)!"
            );
            let owners = std::slice::from_raw_parts(delta.chunk_owners, delta.num_chunk_owners as usize);
            let owned_by_42 = owners.iter().filter(|o| o.owner_cutter_id == 42).count();
            assert!(
                owned_by_42 > 0,
                "Expected at least 1 interior chunk owned by cutter 42, got 0"
            );
            fdt_gpu_delta_free(d1);
        }

        // Move cutter away (outside target at x = 10.0) -> chunk_owners must release all back to 0!
        let (cv2, cl2, cs2, cm2, cuv2, cmat2) = box_input_at(10.0, 0.0, 0.0, 1.0, 1.5, 1.5);
        let c_inp2 = FdtMeshInput {
            num_verts: 8,
            verts_xyz: cv2.as_ptr(),
            num_loops: 24,
            loop_verts: cl2.as_ptr(),
            num_polys: 6,
            poly_starts: cs2.as_ptr(),
            mat_ids: cm2.as_ptr(),
            num_uv_layers: 0,
            uv_data: cuv2.as_ptr(),
            world_matrix: cmat2.as_ptr(),
            loop_normals: std::ptr::null(),
        };
        let jid2 = unsafe {
            fdt_session_submit_async_delta(h, 1, &cid, &c_inp2, 1, &cid, 1, 1, 0.5)
        };
        let ev2 = fdt_session_fetch_cached_eval(h, 1, 0);
        assert!(!ev2.is_null());
        unsafe {
            assert_eq!((*ev2).remaining.num_polys, 21_600);
            assert_eq!((*(*ev2).pieces.add(0)).num_polys, 0);
            fdt_eval_free(ev2);
        }
        let d2 = fdt_session_poll_gpu_delta(h, 1, jid1);
        assert!(!d2.is_null());
        unsafe {
            let delta = &*d2;
            assert_eq!(delta.job_id, jid2);
            let owners = std::slice::from_raw_parts(delta.chunk_owners, delta.num_chunk_owners as usize);
            let released = owners.iter().filter(|o| o.owner_cutter_id == 0).count();
            assert!(released > 0, "Expected interior chunks released back to Remaining (0)");
            fdt_gpu_delta_free(d2);
        }

        assert_eq!(fdt_session_destroy(h), 0);
    }
}
