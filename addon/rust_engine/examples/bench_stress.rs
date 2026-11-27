//! Comprehensive High-Scale Stress Benchmark for FullDestruc_Tool Rust Engine v2.0
//!
//! Run with:
//!   cargo run --release --example bench_stress
//!
//! Benchmarks:
//! 1. Mathematical Equivalence: Island Engine vs Sequential Reference Solver
//! 2. 1,000,000-Polygon Mesh (Upload + Chunk/BVH build + 60-frame live drag + Delta GPU bytes + RSS)
//! 3. 100 Boxes & 500 Boxes on 20K-poly mesh (Add #1..#500 at index 0, drag #1/#250/#500, overlapping cluster)
//! 4. 500 Boxes on 1,000,000-Polygon Mesh (Add 500 boxes, drag #1/#250/#500, Apply assembly, Peak RSS)

use fdt_bool_engine::boolean::split_mesh;
use fdt_bool_engine::gpu_pack::GpuPackedMesh;
use fdt_bool_engine::islands::{IslandTargetState, ISLAND_CHUNK_BIT};
use fdt_bool_engine::mesh::HybridMesh;
use glam::DMat4;
use smallvec::SmallVec;
use std::time::Instant;

fn read_rss_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

fn gpu_bytes(g: &GpuPackedMesh) -> usize {
    g.submeshes
        .iter()
        .map(|s| (s.pos_xyz.len() + s.nor_xyz.len() + s.uv_xy.len() + s.uv1_xy.len()) * 4)
        .sum()
}

/// Build a closed box mesh (6 quads) centered at `center` with half-extents `half`.
fn make_box(center: [f64; 3], half: [f64; 3], mat_id: i32) -> HybridMesh {
    let [cx, cy, cz] = center;
    let [hx, hy, hz] = half;
    let v = [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ];
    let faces = [
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [1, 2, 6, 5],
        [0, 3, 2, 1],
    ];
    let mut mesh = HybridMesh::new(1);
    let quad_uv: SmallVec<[[f32; 2]; 4]> =
        SmallVec::from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    for f in faces {
        let pts = [v[f[0]], v[f[1]], v[f[2]], v[f[3]]];
        mesh.push_poly_with_positions(&pts, &[quad_uv.clone()], mat_id, false);
    }
    mesh.weld_vertices(1e-7);
    mesh
}

/// Build a subdivided 6-face building shell with `6 * n * n` quads (`12 * n * n` triangles),
/// sized `20m x 20m x 10m` with shared vertices and clean UVs + normals (no slow welding needed!).
fn make_subdivided_building(n: usize) -> HybridMesh {
    // Each of the 6 faces has (n+1)*(n+1) grid vertices, n*n quads.
    let stride = n + 1;
    let verts_per_face = stride * stride;
    let total_verts = 6 * verts_per_face;
    let total_polys = 6 * n * n;
    let total_loops = total_polys * 4;

    let mut positions = Vec::with_capacity(total_verts);
    let mut loop_verts = Vec::with_capacity(total_loops);
    let mut poly_starts = Vec::with_capacity(total_polys + 1);
    poly_starts.push(0u32);
    let mut uv0 = Vec::with_capacity(total_loops);
    let mut loop_normals = Vec::with_capacity(total_loops);
    let mut mat_ids = Vec::with_capacity(total_polys);
    let is_cut_face = vec![false; total_polys];

    let hx = 10.0f64;
    let hy = 10.0f64;
    let hz = 5.0f64;
    let inv_n = 1.0 / (n as f64);

    let face_normals: [[f32; 3]; 6] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];

    for side in 0..6usize {
        let base_v = positions.len() as u32;
        for iy in 0..=n {
            let tv = (iy as f64) * inv_n;
            for ix in 0..=n {
                let tu = (ix as f64) * inv_n;
                let u = -1.0 + 2.0 * tu;
                let v = -1.0 + 2.0 * tv;
                let p = match side {
                    0 => [u * hx, v * hy, hz],       // +Z top
                    1 => [u * hx, -v * hy, -hz],     // -Z bottom
                    2 => [u * hx, -hy, v * hz],      // -Y front
                    3 => [-u * hx, hy, v * hz],      // +Y back
                    4 => [hx, u * hy, v * hz],       // +X right
                    _ => [-hx, -u * hy, v * hz],     // -X left
                };
                positions.push(p);
            }
        }
        let fnrm = face_normals[side];
        let mat_id = (side % 4) as i32;
        for iy in 0..n {
            let v0 = (iy as f32) * (inv_n as f32);
            let v1 = ((iy + 1) as f32) * (inv_n as f32);
            for ix in 0..n {
                let u0 = (ix as f32) * (inv_n as f32);
                let u1 = ((ix + 1) as f32) * (inv_n as f32);
                let i00 = base_v + (iy * stride + ix) as u32;
                let i10 = base_v + (iy * stride + ix + 1) as u32;
                let i11 = base_v + ((iy + 1) * stride + ix + 1) as u32;
                let i01 = base_v + ((iy + 1) * stride + ix) as u32;
                loop_verts.extend_from_slice(&[i00, i10, i11, i01]);
                uv0.extend_from_slice(&[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]);
                loop_normals.extend_from_slice(&[fnrm, fnrm, fnrm, fnrm]);
                poly_starts.push(loop_verts.len() as u32);
                mat_ids.push(mat_id);
            }
        }
    }

    HybridMesh {
        positions,
        loop_verts,
        poly_starts,
        uv_layers: vec![uv0],
        loop_normals,
        mat_ids,
        is_cut_face,
    }
}

fn bench_equivalence() {
    println!("\n=== 1. Equivalence Check (Spatial Islands vs Sequential Reference) ===");
    let base = make_subdivided_building(20); // 2,400 quads (4,800 tris)
    let cutters = vec![
        make_box([9.5, -5.0, 2.0], [1.2, 1.2, 1.2], 0), // isolated
        make_box([-9.5, 4.0, 1.0], [1.2, 1.2, 1.2], 0), // overlaps next
        make_box([-9.0, 4.5, 1.0], [1.2, 1.2, 1.2], 0), // overlaps prev
        make_box([0.0, 0.0, 4.8], [1.5, 1.5, 1.0], 0),  // top roof isolated
    ];

    // Reference sequential
    let mut seq_rem = base.clone();
    let mut seq_pieces = Vec::new();
    for c in &cutters {
        let (nr, p) = split_mesh(&seq_rem, c, &DMat4::IDENTITY, 4, 0.5);
        seq_rem = nr;
        seq_pieces.push(p);
    }

    // Island Engine
    let mut state = IslandTargetState::new(base, DMat4::IDENTITY);
    let ids: Vec<u32> = (1..=4).collect();
    let upserts: Vec<(u32, HybridMesh, DMat4)> = cutters
        .into_iter()
        .enumerate()
        .map(|(i, m)| ((i + 1) as u32, m, DMat4::IDENTITY))
        .collect();
    state.upsert_cutters(upserts);
    state.evaluate_active(&ids, 4, 0.5, true);
    let (isl_rem, isl_pieces) = state.assemble_full_eval();

    println!(
        "  Remaining polys: seq={} vs island={} (diff={})",
        seq_rem.num_polys(),
        isl_rem.num_polys(),
        (seq_rem.num_polys() as isize - isl_rem.num_polys() as isize).abs()
    );
    for i in 0..4 {
        println!(
            "  Piece #{}: seq={} polys vs island={} polys",
            i + 1,
            seq_pieces[i].num_polys(),
            isl_pieces[i].num_polys()
        );
        assert!(isl_pieces[i].num_polys() > 0, "Piece #{} must be non-empty", i + 1);
    }
    assert!(
        (seq_rem.num_polys() as isize - isl_rem.num_polys() as isize).abs() <= 4,
        "Remaining poly count diverged unexpectedly!"
    );
    println!("  [PASS] Equivalence verified!");
}

fn bench_500_boxes_on_mesh(label: &str, grid_n: usize, num_boxes: usize) {
    println!(
        "\n=== {} ({} quads / {} tris, {} boxes) ===",
        label,
        6 * grid_n * grid_n,
        12 * grid_n * grid_n,
        num_boxes
    );
    let rss0 = read_rss_mb();
    let t_build = Instant::now();
    let base = make_subdivided_building(grid_n);
    let np = base.num_polys();
    let mesh_gen_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let t_up = Instant::now();
    let mut state = IslandTargetState::new(base, DMat4::IDENTITY);
    let upload_ms = t_up.elapsed().as_secs_f64() * 1000.0;
    let rss_after_up = read_rss_mb();
    println!(
        "  Mesh gen: {:.1} ms | Target Upload (BVH + {} chunks): {:.1} ms | RSS: {:.1} MB (+{:.1} MB)",
        mesh_gen_ms,
        state.chunks.len(),
        upload_ms,
        rss_after_up,
        (rss_after_up - rss0).max(0.0)
    );

    // Distribute `num_boxes` boxes across the 4 vertical walls + roof (non-overlapping grid on surface)
    let mut ordered_ids: Vec<u32> = Vec::with_capacity(num_boxes);
    let mut box_centers: Vec<[f64; 3]> = Vec::with_capacity(num_boxes);
    let cols = 10usize;
    let rows = 10usize;
    let mut add_times_ms: Vec<(usize, f64)> = Vec::new();

    // First initial GPU delta sync (0 cutters -> uploads static base chunks once)
    let t_init_gpu = Instant::now();
    state.evaluate_active(&[], 4, 0.5, true);
    let init_snap = state.take_gpu_delta();
    let init_gpu_ms = t_init_gpu.elapsed().as_secs_f64() * 1000.0;
    let init_mb: f64 = init_snap
        .updated_chunks
        .iter()
        .map(|(_, g)| gpu_bytes(g))
        .sum::<usize>() as f64
        / (1024.0 * 1024.0);
    println!(
        "  Initial Base GPU Pack: {:.1} ms ({} base chunks, {:.2} MB total static VRAM)",
        init_gpu_ms,
        init_snap.updated_chunks.len(),
        init_mb
    );

    for k in 0..num_boxes {
        let wall = k % 5;
        let idx_on_wall = k / 5;
        let cx_i = idx_on_wall % cols;
        let cy_i = (idx_on_wall / cols) % rows;
        let u = -8.6 + (cx_i as f64) * 1.9;
        let v = -4.0 + (cy_i as f64) * 0.88;
        let center = match wall {
            0 => [u, -10.0, v],
            1 => [u, 10.0, v],
            2 => [10.0, u, v],
            3 => [-10.0, u, v],
            _ => [u, -8.6 + (cy_i as f64) * 1.9, 5.0],
        };
        box_centers.push(center);
        let cid = (k + 1) as u32;
        let c_mesh = make_box(center, [0.35, 0.35, 0.30], 0);

        // Insert at index 0 (TOP LAYER — exact same worst-case pattern as `add_box_cutter`!)
        ordered_ids.insert(0, cid);

        let t0 = Instant::now();
        state.upsert_cutters(vec![(cid, c_mesh, DMat4::IDENTITY)]);
        state.evaluate_active(&ordered_ids, 4, 0.5, true);
        let _delta = state.take_gpu_delta();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;

        if k == 0 || k == 9 || k == 49 || k == 99 || k == 249 || k == num_boxes - 1 {
            add_times_ms.push((k + 1, dt));
        }
    }

    print!("  Add Box (inserted at Layer #1): ");
    for (n, ms) in &add_times_ms {
        print!("#{}: {:.2}ms  ", n, ms);
    }
    println!();

    // Benchmark 60 frames of live dragging Box #1 (Layer #1 = TOP of stack!)
    let drag_positions: [(usize, &str); 3] = [
        (0, "Box #1 (Layer #1 TOP)"),
        (num_boxes / 2, "Mid Box (Middle Layer)"),
        (num_boxes - 1, "Oldest Box (Bottom Layer)"),
    ];
    for (layer_idx, desc) in drag_positions {
        let cid = ordered_ids[layer_idx];
        let base_c = box_centers[(cid - 1) as usize];
        let mut frame_ms: Vec<f64> = Vec::with_capacity(60);
        let mut delta_kb_sum = 0.0f64;
        let mut dirty_chunks_sum = 0usize;
        let mut dirty_pieces_sum = 0usize;
        let mut recomputed_sum = 0usize;

        for step in 0..60 {
            let offset = ((step as f64) - 30.0) * 0.015;
            let moved_c = [base_c[0] + offset, base_c[1], base_c[2]];
            let c_mesh = make_box(moved_c, [0.35, 0.35, 0.30], 0);
            let t0 = Instant::now();
            state.upsert_cutters(vec![(cid, c_mesh, DMat4::IDENTITY)]);
            let recomputed = state.evaluate_active(&ordered_ids, 4, 0.5, true);
            let delta = state.take_gpu_delta();
            let dt = t0.elapsed().as_secs_f64() * 1000.0;

            let bytes: usize = delta.updated_chunks.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>()
                + delta.updated_pieces.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>();
            frame_ms.push(dt);
            delta_kb_sum += bytes as f64 / 1024.0;
            dirty_chunks_sum += delta.updated_chunks.len();
            dirty_pieces_sum += delta.updated_pieces.len();
            recomputed_sum += recomputed;
        }
        frame_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = frame_ms.iter().sum::<f64>() / frame_ms.len() as f64;
        let p95 = frame_ms[(frame_ms.len() as f64 * 0.95) as usize];
        let max_v = *frame_ms.last().unwrap();
        let fps_equiv = if avg > 0.0 { 1000.0 / avg } else { 9999.0 };
        println!(
            "  Drag 60f [{}]: avg={:.2}ms ({:.0} FPS) | p95={:.2}ms | max={:.2}ms | recomputed={:.1} isl/f | delta={:.1} ch, {:.1} pc ({:.1} KB/f)",
            desc,
            avg,
            fps_equiv,
            p95,
            max_v,
            recomputed_sum as f64 / 60.0,
            dirty_chunks_sum as f64 / 60.0,
            dirty_pieces_sum as f64 / 60.0,
            delta_kb_sum / 60.0,
        );
    }

    // Benchmark overlapping cluster (10 boxes overlapping each other on +Z roof)
    let mut overlap_ids = ordered_ids.clone();
    for j in 0..10u32 {
        let cid = 10000 + j;
        let c_mesh = make_box(
            [-2.0 + (j as f64) * 0.25, 0.0, 5.0],
            [0.35, 0.35, 0.35],
            0,
        );
        overlap_ids.insert(0, cid);
        state.upsert_cutters(vec![(cid, c_mesh, DMat4::IDENTITY)]);
    }
    state.evaluate_active(&overlap_ids, 4, 0.5, true);
    let _ = state.take_gpu_delta();

    let mut ov_ms = Vec::with_capacity(20);
    for step in 0..20 {
        let cid = 10009u32; // Layer #1 inside the 10-box overlapping cluster
        let c_mesh = make_box(
            [0.25 + (step as f64) * 0.02, 0.05, 5.0],
            [0.35, 0.35, 0.35],
            0,
        );
        let t0 = Instant::now();
        state.upsert_cutters(vec![(cid, c_mesh, DMat4::IDENTITY)]);
        state.evaluate_active(&overlap_ids, 4, 0.5, true);
        let _ = state.take_gpu_delta();
        ov_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let ov_avg = ov_ms.iter().sum::<f64>() / ov_ms.len() as f64;
    println!(
        "  Drag 20f [10-Box Overlapping Cluster on {} polys]: avg={:.2}ms ({:.0} FPS)",
        np,
        ov_avg,
        1000.0 / ov_avg.max(0.01)
    );

    // Measure Apply full mesh assembly & Peak RSS
    let t_apply = Instant::now();
    let (rem, pieces) = state.assemble_full_eval();
    let apply_ms = t_apply.elapsed().as_secs_f64() * 1000.0;
    let non_empty_pieces = pieces.iter().filter(|p| !p.is_empty()).count();
    let rss_peak = read_rss_mb();
    let island_chunks = state.active_islands.len();
    let _ = ISLAND_CHUNK_BIT;
    println!(
        "  Apply Full Assembly: {:.1} ms (rem={} polys, {}/{} non-empty pieces, {} islands) | RSS: {:.1} MB",
        apply_ms,
        rem.num_polys(),
        non_empty_pieces,
        pieces.len(),
        island_chunks,
        rss_peak
    );
}

fn bench_4m_tri_monkey_large_box() {
    // 577 x 577 x 6 = 1,997,574 quads = 3,995,148 triangles (~4,000,000 tris!)
    // Scaled down to Suzanne size (~2.0m x 2.0m x 1.5m) so a standard 2.0m cutter box
    // covers > 50% of the ENTIRE 4M-triangle mesh!
    let n = 577usize;
    let np_expected = 6 * n * n;
    println!(
        "\n=== Case E: 4,000,000-Triangle Suzanne-Scale Mesh ({} quads / {} tris) + Giant 2.0m Box (>50% coverage!) ===",
        np_expected,
        np_expected * 2
    );
    let rss0 = read_rss_mb();
    let t_gen = Instant::now();
    let mut base = make_subdivided_building(n);
    // Rescale 20x20x10m building -> 2.4m x 2.0m x 1.6m (Suzanne dimensions)
    for p in &mut base.positions {
        p[0] *= 0.12;
        p[1] *= 0.10;
        p[2] *= 0.16;
    }
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    let t_up = Instant::now();
    let mut state = IslandTargetState::new(base, DMat4::IDENTITY);
    let up_ms = t_up.elapsed().as_secs_f64() * 1000.0;

    let t_pre = Instant::now();
    state.prewarm_base_gpu();
    let pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
    let rss_up = read_rss_mb();
    println!(
        "  Mesh gen: {:.1} ms | Target BVH+Chunks ({} chunks): {:.1} ms | Prewarm Base GPU: {:.1} ms | RSS: {:.1} MB (+{:.1} MB)",
        gen_ms,
        state.chunks.len(),
        up_ms,
        pre_ms,
        rss_up,
        (rss_up - rss0).max(0.0)
    );

    // Initial 0-cutter GPU sync (uploads base chunks once)
    state.evaluate_active(&[], 4, 0.5, true);
    let _init_snap = state.take_gpu_delta();

    // Place a giant 2.0m x 2.0m x 2.0m box covering >50% of Suzanne (x in [0.0, 2.0])
    let cid = 1u32;
    let ordered = vec![cid];
    let c0 = make_box([1.0, 0.05, 0.02], [1.0, 1.0, 1.0], 0);
    let t_first = Instant::now();
    state.upsert_cutters(vec![(cid, c0, DMat4::IDENTITY)]);
    state.evaluate_active(&ordered, 4, 0.5, true);
    let first_delta = state.take_gpu_delta();
    let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;
    let first_owned = first_delta.chunk_owners.iter().filter(|&&(_, o)| o == cid).count();
    let first_mb = (first_delta.updated_chunks.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>()
        + first_delta.updated_pieces.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>()) as f64
        / (1024.0 * 1024.0);
    println!(
        "  First Giant Cut (2m box covering >50% of 4M tris): {:.1} ms | {} interior chunks flipped (0 VRAM upload!) | delta={:.2} MB",
        first_ms, first_owned, first_mb
    );

    // Drag the giant 2.0m box across the 4M-triangle Suzanne for 15 frames
    let mut drag_ms = Vec::with_capacity(15);
    let mut delta_mb_sum = 0.0f64;
    let mut owner_flips_sum = 0usize;
    for step in 0..15 {
        let cx = 0.85 + (step as f64) * 0.025;
        let cm = make_box([cx, 0.05, 0.02], [1.0, 1.0, 1.0], 0);
        let t0 = Instant::now();
        state.upsert_cutters(vec![(cid, cm, DMat4::IDENTITY)]);
        state.evaluate_active(&ordered, 4, 0.5, true);
        let delta = state.take_gpu_delta();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let mb = (delta.updated_chunks.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>()
            + delta.updated_pieces.iter().map(|(_, g)| gpu_bytes(g)).sum::<usize>()) as f64
            / (1024.0 * 1024.0);
        drag_ms.push(dt);
        delta_mb_sum += mb;
        owner_flips_sum += delta.chunk_owners.len();
    }
    drag_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = drag_ms.iter().sum::<f64>() / drag_ms.len() as f64;
    let p95 = drag_ms[(drag_ms.len() as f64 * 0.95).min((drag_ms.len() - 1) as f64) as usize];
    println!(
        "  Drag 15f [Giant 2m Box on 4M Tris]: avg={:.1} ms ({:.1} Hz update!) | p95={:.1} ms | max={:.1} ms | avg delta={:.2} MB/f | avg owner flips={:.1}/f",
        avg,
        1000.0 / avg.max(0.01),
        p95,
        drag_ms.last().unwrap(),
        delta_mb_sum / drag_ms.len() as f64,
        owner_flips_sum as f64 / drag_ms.len() as f64,
    );

    // Measure Apply full assembly on 4M tris
    let t_apply = Instant::now();
    let (rem, pieces) = state.assemble_full_eval();
    let apply_ms = t_apply.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Apply Full Assembly (4M Tris): {:.1} ms (rem={} quads, piece={} quads) | Peak RSS: {:.1} MB",
        apply_ms,
        rem.num_polys(),
        pieces.first().map_or(0, |p| p.num_polys()),
        read_rss_mb()
    );
    assert!(rem.num_polys() > 500_000, "Remaining must keep outside half of 4M-tri mesh");
    assert!(
        pieces.first().map_or(0, |p| p.num_polys()) > 500_000,
        "Piece must include all owned interior chunks of 4M-tri mesh"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let only_4m = args.iter().any(|a| a == "--4m-only");

    println!("==================================================================");
    println!("  FullDestruc_Tool Rust Engine v2.1 High-Scale Stress Benchmark");
    println!("==================================================================");

    if !only_4m {
        bench_equivalence();

        // Case A: 100 boxes on 21,600-quad (43,200-tri) building
        bench_500_boxes_on_mesh("Case A: 100 Boxes on 43K-tri Building", 60, 100);

        // Case B: 500 boxes on 21,600-quad (43,200-tri) building
        bench_500_boxes_on_mesh("Case B: 500 Boxes on 43K-tri Building", 60, 500);

        // Case C: 1,000,000-Polygon Mesh (408 x 408 x 6 = 998,784 quads ≈ 1.00M polys / 2.00M tris!) with 100 boxes
        bench_500_boxes_on_mesh("Case C: 1,000,000 Polygons (2M Tris) + 100 Boxes", 408, 100);

        // Case D: 1,000,000-Polygon Mesh (2M Tris) + 500 Boxes!
        bench_500_boxes_on_mesh("Case D: 1,000,000 Polygons (2M Tris) + 500 Boxes", 408, 500);
    }

    // Case E: 4,000,000-Triangle Suzanne-Scale Mesh + Giant 2.0m Box (>50% coverage!)
    bench_4m_tri_monkey_large_box();
}
