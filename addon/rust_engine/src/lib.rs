//! C-ABI entry points for `fdt_bool_engine.dll` (v1.2.0, loaded by Blender via `ctypes`).
//!
//! Features:
//! - Parallel Dipole BVH (`rayon::join`) + Partial Vertex Weld + Dirty-Piece Mask
//! - Non-blocking Async Background Worker with Latest-Wins queue for locked 60-144+ FPS
//! - World-space GPU Triangle Packer (`gpu_pack.rs`) for direct Blender `POST_VIEW` rendering
//! - Zero-recompute `fdt_session_fetch_cached_eval` when committing back to real Blender meshes
//! - Panic-safe (`std::panic::catch_unwind`) on every exported function.

pub mod boolean;
pub mod box_uv;
pub mod bvh;
pub mod classify;
pub mod gpu_pack;
pub mod mesh;
pub mod predicates;
pub mod retriangulate;

use crate::boolean::split_mesh_cached;
use crate::gpu_pack::GpuPackedMesh;
use crate::mesh::{CachedMeshGeo, HybridMesh};
use glam::DMat4;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Global Session Registry
// ---------------------------------------------------------------------------

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

struct CutterCacheStep {
    hash: u64,
    remaining_after: HybridMesh,
    /// Lazily populated only if a subsequent cutter (`i + 1`) splits `remaining_after`.
    remaining_geo: Option<CachedMeshGeo>,
    piece: HybridMesh,
}

struct TargetCache {
    base_geo: CachedMeshGeo,
    world_mat: DMat4,
    steps: Vec<CutterCacheStep>,
    last_dirty_idx: usize,
}

struct SessionState {
    targets: HashMap<u32, TargetCache>,
}

static SESSIONS: Mutex<Option<HashMap<u64, SessionState>>> = Mutex::new(None);

fn with_sessions<R>(f: impl FnOnce(&mut HashMap<u64, SessionState>) -> R) -> R {
    let mut guard = SESSIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

// ---------------------------------------------------------------------------
// Async Background Worker (Latest-Wins Queue)
// ---------------------------------------------------------------------------

struct PendingJob {
    job_id: u64,
    raw_cutters: Vec<(HybridMesh, DMat4)>,
    cut_mat_idx: i32,
    uv_scale: f32,
}

struct CompletedGpuFrame {
    job_id: u64,
    compute_ms: f32,
    dirty_from_idx: u32,
    remaining_gpu: GpuPackedMesh,
    pieces_gpu: Vec<GpuPackedMesh>,
}

struct WorkerShared {
    pending: HashMap<(u64, u32), PendingJob>,
    in_flight: HashSet<(u64, u32)>,
    completed: HashMap<(u64, u32), CompletedGpuFrame>,
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
                pending: HashMap::new(),
                in_flight: HashSet::new(),
                completed: HashMap::new(),
            }),
            work_cvar: Condvar::new(),
            done_cvar: Condvar::new(),
        };
        // Spawn dedicated background thread
        std::thread::Builder::new()
            .name("fdt-rust-async-worker".to_string())
            .spawn(worker_thread_loop)
            .expect("failed to spawn fdt async worker thread");
        worker
    })
}

fn worker_thread_loop() {
    let worker = get_worker();
    loop {
        // Wait for at least one pending job (Latest-Wins per (handle, target_id))
        let (key, job) = {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.pending.is_empty() {
                st = worker.work_cvar.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            let first_key = *st.pending.keys().next().unwrap();
            let job = st.pending.remove(&first_key).unwrap();
            st.in_flight.insert(first_key);
            (first_key, job)
        };

        let (handle, target_id) = key;
        let t0 = Instant::now();

        let gpu_result = catch_unwind(|| {
            with_sessions(|map| {
                let sess = map.get_mut(&handle)?;
                let tc = sess.targets.get_mut(&target_id)?;
                let dirty_idx = evaluate_target_cache_in_place(
                    tc,
                    job.raw_cutters,
                    job.cut_mat_idx,
                    job.uv_scale,
                );
                let final_rem = if tc.steps.is_empty() {
                    &tc.base_geo.mesh
                } else {
                    &tc.steps[tc.steps.len() - 1].remaining_after
                };
                let world_mat = tc.world_mat;
                let (remaining_gpu, pieces_gpu) = rayon::join(
                    || GpuPackedMesh::from_hybrid_mesh(final_rem, &world_mat),
                    || {
                        tc.steps
                            .par_iter()
                            .map(|s| GpuPackedMesh::from_hybrid_mesh(&s.piece, &world_mat))
                            .collect::<Vec<_>>()
                    },
                );
                Some((dirty_idx as u32, remaining_gpu, pieces_gpu))
            })
        });

        let compute_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;

        let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(Some((dirty_from_idx, remaining_gpu, pieces_gpu))) = gpu_result {
            st.completed.insert(
                key,
                CompletedGpuFrame {
                    job_id: job.job_id,
                    compute_ms,
                    dirty_from_idx,
                    remaining_gpu,
                    pieces_gpu,
                },
            );
        }
        st.in_flight.remove(&key);
        worker.done_cvar.notify_all();
    }
}

fn evaluate_target_cache_in_place(
    tc: &mut TargetCache,
    raw_cutters: Vec<(HybridMesh, DMat4)>,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> usize {
    let nc = raw_cutters.len();
    let target_world_inv = tc.world_mat.inverse();

    let mut local_cutters: Vec<(HybridMesh, u64)> = Vec::with_capacity(nc);
    for (mut c_mesh, c_world) in raw_cutters {
        let rel_mat = target_world_inv * c_world;
        c_mesh.transform_in_place(&rel_mat);
        let h = hash_cutter_input(&c_mesh, cut_mat_idx, uv_scale);
        local_cutters.push((c_mesh, h));
    }

    let mut dirty_idx = 0usize;
    while dirty_idx < nc
        && dirty_idx < tc.steps.len()
        && tc.steps[dirty_idx].hash == local_cutters[dirty_idx].1
    {
        dirty_idx += 1;
    }
    tc.steps.truncate(dirty_idx);

    for i in dirty_idx..nc {
        let (next_rem, piece) = if i == 0 {
            split_mesh_cached(
                &tc.base_geo,
                &local_cutters[i].0,
                &tc.world_mat,
                cut_mat_idx,
                uv_scale,
            )
        } else {
            if tc.steps[i - 1].remaining_geo.is_none() {
                let geo = CachedMeshGeo::from_mesh(tc.steps[i - 1].remaining_after.clone());
                tc.steps[i - 1].remaining_geo = Some(geo);
            }
            let prev_geo = tc.steps[i - 1].remaining_geo.as_ref().unwrap();
            split_mesh_cached(
                prev_geo,
                &local_cutters[i].0,
                &tc.world_mat,
                cut_mat_idx,
                uv_scale,
            )
        };
        tc.steps.push(CutterCacheStep {
            hash: local_cutters[i].1,
            remaining_after: next_rem,
            remaining_geo: None,
            piece,
        });
    }

    tc.last_dirty_idx = dirty_idx;
    dirty_idx
}

// ---------------------------------------------------------------------------
// C-ABI Structs (v1.2.0)
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

fn hash_cutter_input(mesh: &HybridMesh, cut_mat_idx: i32, uv_scale: f32) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    mesh.positions.len().hash(&mut hasher);
    mesh.loop_verts.len().hash(&mut hasher);
    for p in &mesh.positions {
        p[0].to_bits().hash(&mut hasher);
        p[1].to_bits().hash(&mut hasher);
        p[2].to_bits().hash(&mut hasher);
    }
    mesh.loop_verts.hash(&mut hasher);
    mesh.poly_starts.hash(&mut hasher);
    cut_mat_idx.hash(&mut hasher);
    uv_scale.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn empty_mesh_output_stub(num_polys_hint: u32) -> FdtMeshOutput {
    FdtMeshOutput {
        num_verts: 0,
        verts_xyz: std::ptr::null_mut(),
        num_loops: 0,
        loop_verts: std::ptr::null_mut(),
        num_polys: num_polys_hint,
        poly_starts: std::ptr::null_mut(),
        mat_ids: std::ptr::null_mut(),
        is_cut_face: std::ptr::null_mut(),
        num_uv_layers: 0,
        uv_data: std::ptr::null_mut(),
        num_edges: 0,
        edge_verts: std::ptr::null_mut(),
        loop_edges: std::ptr::null_mut(),
        loop_normals: std::ptr::null_mut(),
    }
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

fn to_gpu_mesh_data(g: &GpuPackedMesh) -> FdtGpuMeshData {
    if g.submeshes.is_empty() {
        return FdtGpuMeshData {
            num_submeshes: 0,
            submeshes: std::ptr::null_mut(),
        };
    }
    let subs: Vec<FdtGpuSubMesh> = g
        .submeshes
        .iter()
        .map(|s| FdtGpuSubMesh {
            mat_id: s.mat_id,
            num_tris: s.num_tris,
            pos_xyz: Box::into_raw(s.pos_xyz.clone().into_boxed_slice()) as *mut f32,
            nor_xyz: Box::into_raw(s.nor_xyz.clone().into_boxed_slice()) as *mut f32,
            uv_xy: Box::into_raw(s.uv_xy.clone().into_boxed_slice()) as *mut f32,
            uv1_xy: Box::into_raw(s.uv1_xy.clone().into_boxed_slice()) as *mut f32,
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

fn pack_eval_output(tc: &TargetCache, dirty_idx: usize, gen_edges: bool, stub_clean: bool) -> *mut FdtEvalOutput {
    let nc = tc.steps.len();
    let final_rem = if tc.steps.is_empty() {
        &tc.base_geo.mesh
    } else {
        &tc.steps[nc - 1].remaining_after
    };

    let (rem_wrapped, piece_wrapped): (SendMeshOut, Vec<SendMeshOut>) = rayon::join(
        || {
            if stub_clean && gen_edges && nc > 0 && dirty_idx >= nc {
                SendMeshOut(empty_mesh_output_stub(final_rem.num_polys() as u32))
            } else {
                SendMeshOut(to_mesh_output(final_rem, gen_edges))
            }
        },
        || {
            tc.steps
                .par_iter()
                .enumerate()
                .map(|(idx, s)| {
                    if stub_clean && gen_edges && idx < dirty_idx {
                        SendMeshOut(empty_mesh_output_stub(s.piece.num_polys() as u32))
                    } else {
                        SendMeshOut(to_mesh_output(&s.piece, gen_edges))
                    }
                })
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

// ---------------------------------------------------------------------------
// Exported C-ABI Functions (v1.2.0)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn fdt_engine_version() -> u32 {
    151 // v1.5.1 (CDT + Dangling-Fin Pruning + Trapped Pinhole Closing)
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
        // Clear any pending/completed async state for this session
        if let Some(worker) = WORKER.get() {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            st.pending.retain(|&(h, _), _| h != handle);
            st.completed.retain(|&(h, _), _| h != handle);
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
        let base_geo = CachedMeshGeo::from_mesh(base_mesh);
        with_sessions(|map| {
            if let Some(sess) = map.get_mut(&handle) {
                sess.targets.insert(
                    target_id,
                    TargetCache {
                        base_geo,
                        world_mat,
                        steps: Vec::new(),
                        last_dirty_idx: 0,
                    },
                );
                0
            } else {
                -1
            }
        })
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

        with_sessions(|map| {
            let sess = match map.get_mut(&handle) {
                Some(s) => s,
                None => return std::ptr::null_mut(),
            };
            let tc = match sess.targets.get_mut(&target_id) {
                Some(t) => t,
                None => return std::ptr::null_mut(),
            };

            let dirty_idx = evaluate_target_cache_in_place(tc, raw_cutters, cut_mat_idx, uv_scale);
            pack_eval_output(tc, dirty_idx, gen_edges, true)
        })
    });

    res.unwrap_or(std::ptr::null_mut())
}

/// Non-blocking submission into the background worker (Latest-Wins queue).
/// Returns monotonic `job_id` (> 0) in < 0.1 ms without blocking Blender's UI thread.
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
            PendingJob {
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

/// Non-blocking poll for the newest finished GPU preview frame (> `last_seen_job_id`).
/// Returns NULL if no newer frame is ready yet. Caller frees with `fdt_gpu_preview_free`.
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
        let st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
        let frame = match st.completed.get(&(handle, target_id)) {
            Some(f) if f.job_id > last_seen_job_id => f,
            _ => return std::ptr::null_mut(),
        };

        let rem_gpu = to_gpu_mesh_data(&frame.remaining_gpu);
        let piece_gpus: Vec<FdtGpuMeshData> =
            frame.pieces_gpu.iter().map(to_gpu_mesh_data).collect();
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
        let key = (handle, target_id);
        if let Some(worker) = WORKER.get() {
            let mut st = worker.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.pending.contains_key(&key) || st.in_flight.contains(&key) {
                st = worker.done_cvar.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }

        let gen_edges = include_edges != 0;
        with_sessions(|map| {
            let sess = match map.get(&handle) {
                Some(s) => s,
                None => return std::ptr::null_mut(),
            };
            let tc = match sess.targets.get(&target_id) {
                Some(t) => t,
                None => return std::ptr::null_mut(),
            };
            // Always send full mesh data (stub_clean = false) so commit writes cleanly into Blender
            pack_eval_output(tc, 0, gen_edges, false)
        })
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
        let verts = vec![
            cx - 1.0, -1.0, -1.0,
            cx + 1.0, -1.0, -1.0,
            cx + 1.0,  1.0, -1.0,
            cx - 1.0,  1.0, -1.0,
            cx - 1.0, -1.0,  1.0,
            cx + 1.0, -1.0,  1.0,
            cx + 1.0,  1.0,  1.0,
            cx - 1.0,  1.0,  1.0,
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
        // Custom tilted split normals on face 0 (+Z top face) to verify exact preservation through split!
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

        // Rapidly submit 10 async jobs (simulating fast mouse drag)
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

        // Fetch cached eval (waits for latest job and returns full mesh with 0 recompute)
        let eval_ptr = fdt_session_fetch_cached_eval(h, 1, 1);
        assert!(!eval_ptr.is_null());
        unsafe {
            let ev = &*eval_ptr;
            assert_eq!(ev.status, 0);
            assert!(ev.remaining.num_polys > 0);
            assert!(!ev.remaining.loop_normals.is_null());
            let rn = std::slice::from_raw_parts(ev.remaining.loop_normals, (ev.remaining.num_loops as usize) * 3);
            let found_tilted = (0..ev.remaining.num_loops as usize).any(|i| {
                (rn[i * 3 + 1] - 0.6).abs() < 1e-4 && (rn[i * 3 + 2] - 0.8).abs() < 1e-4
            });
            assert!(found_tilted, "Custom tilted split normal should survive retriangulation!");
            assert_eq!(ev.num_pieces, 1);
            fdt_eval_free(eval_ptr);
        }

        // Poll GPU preview should return the exact latest job_id with multi-material submeshes
        let gpu_ptr = fdt_session_poll_gpu_preview(h, 1, 0);
        assert!(!gpu_ptr.is_null());
        unsafe {
            let gp = &*gpu_ptr;
            assert_eq!(gp.job_id, last_jid);
            assert!(gp.remaining.num_submeshes >= 2, "should have orig mat + cut mat submeshes");
            assert_eq!(gp.num_pieces, 1);
            assert!((*gp.pieces).num_submeshes >= 2);
            fdt_gpu_preview_free(gpu_ptr);
        }

        assert_eq!(fdt_session_destroy(h), 0);
    }
}
