//! Single-pass hybrid Boolean engine (`split_mesh`, `difference`, `intersect`, `union`)
//! with multi-threaded execution (`rayon`) and prefix caching for real-time multi-cutter workflows.

use crate::box_uv::apply_cut_face_box_uv_from;
use crate::bvh::Bvh;
use crate::classify::{
    check_coplanar_surface, classify_point, classify_point_convex, extract_convex_planes, Location,
};
use crate::mesh::{Aabb, CachedMeshGeo, HybridMesh, TriRef};
use crate::predicates::{tri_tri_intersect, TriTriHit};
use crate::retriangulate::{split_triangle, triangulate_cutter_poly_regions, OrientedCutSeg};
use glam::{DMat4, DVec3};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

const PAIR_EPS: f64 = 1e-6;
const WELD_EPS: f64 = 1e-6;

#[inline]
fn tri_aabb(positions: &[[f64; 3]], t: &TriRef) -> Aabb {
    let mut b = Aabb::from_point(positions[t.v[0] as usize]);
    b.expand_point(positions[t.v[1] as usize]);
    b.expand_point(positions[t.v[2] as usize]);
    b
}

#[inline]
fn tri_centroid_and_normal(pts: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
    let c = [
        (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0,
        (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0,
        (pts[0][2] + pts[1][2] + pts[2][2]) / 3.0,
    ];
    let p0 = DVec3::from_array(pts[0]);
    let p1 = DVec3::from_array(pts[1]);
    let p2 = DVec3::from_array(pts[2]);
    let n = (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array();
    (c, n)
}

/// Perform a single-pass split of `target` by `cutter` (both in target-local space).
///
/// Builds target BVH on the fly. For real-time dragging, prefer `split_mesh_cached` or `split_mesh_local`.
pub fn split_mesh(
    target: &HybridMesh,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> (HybridMesh, HybridMesh) {
    let num_uv_layers = target.num_uv_layers();
    if target.is_empty() || cutter.is_empty() {
        return (target.clone(), HybridMesh::new(num_uv_layers));
    }
    let target_geo = CachedMeshGeo::from_mesh(target.clone());
    split_mesh_local(&target_geo, cutter, target_world, cut_mat_idx, uv_scale, None)
}

/// High-speed single-pass split using a precomputed `CachedMeshGeo` for `target`.
///
/// Reuses `target_geo.tris` and `target_geo.bvh` (zero target triangulation/BVH rebuild)
/// and uses Partial Vertex Welding so untouched polygons never enter a vertex HashMap.
pub fn split_mesh_cached(
    target_geo: &CachedMeshGeo,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> (HybridMesh, HybridMesh) {
    split_mesh_local(target_geo, cutter, target_world, cut_mat_idx, uv_scale, None)
}

/// Precomputed earlier-layer cutter volume inside an overlapping Spatial Island.
/// Points inside any excluded volume were already removed by a higher-priority cutter.
pub struct ExcludedVolume<'a> {
    pub aabb: Aabb,
    pub positions: &'a [[f64; 3]],
    pub tris: &'a [TriRef],
    pub bvh: &'a Bvh,
}

#[inline]
fn classify_point_in_remaining(
    p: [f64; 3],
    n: [f64; 3],
    local_geo: &CachedMeshGeo,
    w_aabb: &Aabb,
    w_positions: &[[f64; 3]],
    w_tris: &[TriRef],
    w_bvh: &Bvh,
    excluded: &[ExcludedVolume<'_>],
) -> Location {
    // 1. Check coplanar Z-fighting against the ACTUAL current remaining mesh (including prior cut caps!)
    if local_geo.aabb.expanded(PAIR_EPS * 10.0).contains_point(p) {
        if let Some(dot) = check_coplanar_surface(
            p,
            n,
            &local_geo.mesh.positions,
            &local_geo.tris,
            &local_geo.bvh,
        ) {
            return if dot >= 0.0 {
                Location::CoplanarSame
            } else {
                Location::CoplanarOpposite
            };
        }
    }
    // 2. Reject any point strictly inside (or coplanar-same with) a higher-priority excluded cutter
    for ex in excluded {
        if ex.aabb.expanded(PAIR_EPS).contains_point(p) {
            let loc = classify_point(p, n, &ex.aabb, ex.positions, ex.tris, ex.bvh);
            if matches!(loc, Location::Inside | Location::CoplanarSame) {
                return Location::Outside;
            }
        }
    }
    // 3. Classify against the global base winding BVH
    classify_point(p, n, w_aabb, w_positions, w_tris, w_bvh)
}

/// Spatial-island local split: intersects & splits ONLY `target_geo` (a small local subset of
/// base polygons overlapping the cutter/island), while optionally using `winding_geo` (the full
/// base mesh Dipole BVH) for exact 3D Fast Winding Number cap classification on open/hollow meshes.
pub fn split_mesh_local(
    target_geo: &CachedMeshGeo,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
    winding_geo: Option<&CachedMeshGeo>,
) -> (HybridMesh, HybridMesh) {
    split_mesh_local_with_exclusions(
        target_geo,
        cutter,
        target_world,
        cut_mat_idx,
        uv_scale,
        winding_geo,
        &[],
    )
}

/// Same as `split_mesh_local`, additionally subtracting `excluded_volumes` (higher-priority
/// overlapping cutters already applied in this island) from `winding_geo` inside/outside tests.
pub fn split_mesh_local_with_exclusions(
    target_geo: &CachedMeshGeo,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
    winding_geo: Option<&CachedMeshGeo>,
    excluded_volumes: &[ExcludedVolume<'_>],
) -> (HybridMesh, HybridMesh) {
    split_mesh_core(
        target_geo,
        None,
        cutter,
        None,
        target_world,
        cut_mat_idx,
        uv_scale,
        winding_geo,
        excluded_volumes,
    )
}

/// Zero-BVH-rebuild boundary split for single-cutter islands on ultra-high-poly (1M–4M+ tri) meshes.
///
/// Reuses `base_geo.bvh` AND precomputed `(cutter_tris, cutter_bvh)` directly — ZERO `extract_subset`,
/// ZERO local BVH rebuild, and ONLY classifies `subset_polys` (boundary chunks, never interior chunks!).
#[allow(clippy::too_many_arguments)]
pub fn split_base_subset_single_cutter(
    base_geo: &CachedMeshGeo,
    subset_polys: &[u32],
    cutter: &HybridMesh,
    cutter_tris: &[TriRef],
    cutter_bvh: &Bvh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> (HybridMesh, HybridMesh) {
    split_mesh_core(
        base_geo,
        Some(subset_polys),
        cutter,
        Some((cutter_tris, cutter_bvh)),
        target_world,
        cut_mat_idx,
        uv_scale,
        Some(base_geo),
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn split_mesh_core(
    target_geo: &CachedMeshGeo,
    subset_polys: Option<&[u32]>,
    cutter: &HybridMesh,
    precomputed_cutter: Option<(&[TriRef], &Bvh)>,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
    winding_geo: Option<&CachedMeshGeo>,
    excluded_volumes: &[ExcludedVolume<'_>],
) -> (HybridMesh, HybridMesh) {
    let target = &target_geo.mesh;
    let num_uv_layers = target.num_uv_layers();

    if target.is_empty() || cutter.is_empty() || target_geo.tris.is_empty() {
        return (
            if subset_polys.is_some() {
                HybridMesh::new(num_uv_layers)
            } else {
                target.clone()
            },
            HybridMesh::new(num_uv_layers),
        );
    }

    let target_aabb = target_geo.aabb;
    let cutter_aabb = cutter.mesh_aabb();
    if !target_aabb.expanded(PAIR_EPS).overlaps(&cutter_aabb) {
        return (
            if let Some(sp) = subset_polys {
                target.extract_subset(sp)
            } else {
                target.clone()
            },
            HybridMesh::new(num_uv_layers),
        );
    }

    // Winding reference for cut-cap inside/outside tests (full base mesh when splitting a local subset!)
    let w_geo = winding_geo.unwrap_or(target_geo);
    let w_aabb = w_geo.aabb;
    let w_positions = &w_geo.mesh.positions;
    let w_tris = &w_geo.tris;
    let w_bvh = &w_geo.bvh;

    // 1. Use precomputed cutter BVH when provided (0 ms!), otherwise build on the fly
    let owned_cutter_tris: Vec<TriRef>;
    let owned_cutter_bvh: Bvh;
    let (cutter_tris, cutter_bvh): (&[TriRef], &Bvh) = if let Some(pre) = precomputed_cutter {
        if pre.0.is_empty() {
            return (
                if let Some(sp) = subset_polys {
                    target.extract_subset(sp)
                } else {
                    target.clone()
                },
                HybridMesh::new(num_uv_layers),
            );
        }
        pre
    } else {
        owned_cutter_tris = cutter.extract_triangles();
        if owned_cutter_tris.is_empty() {
            return (
                if let Some(sp) = subset_polys {
                    target.extract_subset(sp)
                } else {
                    target.clone()
                },
                HybridMesh::new(num_uv_layers),
            );
        }
        let cutter_tri_aabbs: Vec<Aabb> = owned_cutter_tris
            .iter()
            .map(|t| tri_aabb(&cutter.positions, t))
            .collect();
        owned_cutter_bvh =
            Bvh::build_with_dipoles(&cutter_tri_aabbs, &cutter.positions, &owned_cutter_tris);
        (&owned_cutter_tris, &owned_cutter_bvh)
    };

    let target_tris = &target_geo.tris;
    let target_bvh = &target_geo.bvh;

    // 2. Broadphase to find candidate intersecting triangle pairs.
    // For low/medium-poly cutters (boxes, sculpted, draw, rough <= 2048 tris), query each
    // cutter triangle's tight AABB + supporting plane directly against `target_bvh` in parallel!
    // This eliminates BVH leaf inflation (e.g. a 12-tri box's 2 leaves spanning 2M interior tris!).
    let hits: Vec<(u32, u32, TriTriHit)> = if cutter_tris.len() <= 2048 {
        (0..cutter_tris.len())
            .into_par_iter()
            .flat_map_iter(|tb_idx| {
                let tb = cutter_tris[tb_idx];
                let b_pts = [
                    cutter.positions[tb.v[0] as usize],
                    cutter.positions[tb.v[1] as usize],
                    cutter.positions[tb.v[2] as usize],
                ];
                let mut local_ta: Vec<u32> = Vec::new();
                target_bvh.query_tri_plane(b_pts[0], b_pts[1], b_pts[2], PAIR_EPS, &mut local_ta);
                let mut local_hits: Vec<(u32, u32, TriTriHit)> = Vec::new();
                for ta_idx in local_ta {
                    let ta = target_tris[ta_idx as usize];
                    if let Some(sp) = subset_polys {
                        if sp.binary_search(&ta.poly_idx).is_err() {
                            continue;
                        }
                    }
                    let a_pts = [
                        target.positions[ta.v[0] as usize],
                        target.positions[ta.v[1] as usize],
                        target.positions[ta.v[2] as usize],
                    ];
                    match tri_tri_intersect(a_pts, b_pts) {
                        TriTriHit::None => {}
                        other => local_hits.push((ta_idx, tb_idx as u32, other)),
                    }
                }
                local_hits
            })
            .collect()
    } else {
        let mut candidate_pairs = Vec::new();
        target_bvh.query_bvh_pairs(cutter_bvh, PAIR_EPS, &mut candidate_pairs);
        candidate_pairs
            .par_iter()
            .filter_map(|&(ta_idx, tb_idx)| {
                let ta = target_tris[ta_idx as usize];
                if let Some(sp) = subset_polys {
                    if sp.binary_search(&ta.poly_idx).is_err() {
                        return None;
                    }
                }
                let tb = cutter_tris[tb_idx as usize];
                let a_pts = [
                    target.positions[ta.v[0] as usize],
                    target.positions[ta.v[1] as usize],
                    target.positions[ta.v[2] as usize],
                ];
                let b_pts = [
                    cutter.positions[tb.v[0] as usize],
                    cutter.positions[tb.v[1] as usize],
                    cutter.positions[tb.v[2] as usize],
                ];
                let hit = tri_tri_intersect(a_pts, b_pts);
                match hit {
                    TriTriHit::None => None,
                    other => Some((ta_idx, tb_idx, other)),
                }
            })
            .collect()
    };

    // Group cut segments per target triangle and per WHOLE cutter polygon (tagged with target normal!)
    let mut segs_per_target_tri: HashMap<u32, Vec<[[f64; 3]; 2]>> = HashMap::new();
    let mut segs_per_cutter_poly: HashMap<u32, Vec<OrientedCutSeg>> = HashMap::new();
    let mut touched_poly_set: HashSet<u32> = HashSet::with_capacity(hits.len().min(65536));

    for (ta_idx, tb_idx, hit) in hits {
        let ta = target_tris[ta_idx as usize];
        let poly_a = ta.poly_idx;
        let poly_b = cutter_tris[tb_idx as usize].poly_idx;
        let a_pts = [
            target.positions[ta.v[0] as usize],
            target.positions[ta.v[1] as usize],
            target.positions[ta.v[2] as usize],
        ];
        let (_, ta_norm) = tri_centroid_and_normal(&a_pts);
        match hit {
            TriTriHit::Segment(seg) => {
                touched_poly_set.insert(poly_a);
                segs_per_target_tri.entry(ta_idx).or_default().push(seg);
                segs_per_cutter_poly
                    .entry(poly_b)
                    .or_default()
                    .push(OrientedCutSeg {
                        seg,
                        src_normal: ta_norm,
                    });
            }
            TriTriHit::Coplanar {
                segs_on_a,
                segs_on_b,
            } => {
                touched_poly_set.insert(poly_a);
                if !segs_on_a.is_empty() {
                    segs_per_target_tri
                        .entry(ta_idx)
                        .or_default()
                        .extend(segs_on_a);
                }
                if !segs_on_b.is_empty() {
                    let entry = segs_per_cutter_poly.entry(poly_b).or_default();
                    for seg in segs_on_b {
                        entry.push(OrientedCutSeg {
                            seg,
                            src_normal: ta_norm,
                        });
                    }
                }
            }
            TriTriHit::None => {}
        }
    }

    // 4. Classify UNTOUCHED target polygons in parallel (keeping exact original Quads/N-gons!)
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PolyDest {
        Remaining,
        Piece,
        Split,
    }

    let convex_planes = extract_convex_planes(&cutter.positions, cutter_tris);

    let classify_poly = |poly_idx: usize| -> PolyDest {
        if touched_poly_set.contains(&(poly_idx as u32)) {
            return PolyDest::Split;
        }
        let paabb = target_geo.poly_aabbs[poly_idx];
        if !paabb.expanded(PAIR_EPS).overlaps(&cutter_aabb) {
            return PolyDest::Remaining;
        }
        let c = target.poly_centroid(poly_idx);
        let loc = if let Some(ref planes) = convex_planes {
            // Ultra-fast 6-dot-product convex classification (~125x faster than FWN for Box Cutters!)
            let mut near_plane = false;
            let mut outside = false;
            for pl in planes {
                let s = pl.n[0] * c[0] + pl.n[1] * c[1] + pl.n[2] * c[2] - pl.d;
                if s > 1e-6 {
                    outside = true;
                    break;
                }
                if s >= -1e-6 {
                    near_plane = true;
                }
            }
            if outside {
                Location::Outside
            } else if !near_plane {
                Location::Inside
            } else {
                let n = target.poly_normal(poly_idx);
                classify_point_convex(c, n, planes)
            }
        } else {
            let n = target.poly_normal(poly_idx);
            classify_point(
                c,
                n,
                &cutter_aabb,
                &cutter.positions,
                cutter_tris,
                cutter_bvh,
            )
        };
        match loc {
            Location::Inside | Location::CoplanarSame => PolyDest::Piece,
            Location::Outside | Location::CoplanarOpposite => PolyDest::Remaining,
        }
    };

    let mut remaining = HybridMesh::new(num_uv_layers);
    let mut piece = HybridMesh::new(num_uv_layers);
    let mut rem_seam_verts: Vec<u32> = Vec::new();
    let mut piece_seam_verts: Vec<u32> = Vec::new();

    if let Some(sp) = subset_polys {
        let poly_dests: Vec<PolyDest> = sp
            .par_iter()
            .map(|&pi| classify_poly(pi as usize))
            .collect();
        let est_v = (sp.len()).min(target.positions.len());
        let ((rem_built, rem_seams), (piece_built, piece_seams)) = rayon::join(
            || {
                let mut m = HybridMesh::new(num_uv_layers);
                let mut remap: HashMap<u32, u32> = HashMap::with_capacity(est_v);
                for (&pi, &dest) in sp.iter().zip(poly_dests.iter()) {
                    if dest == PolyDest::Remaining {
                        m.push_poly_from_shared_map(target, pi as usize, &mut remap);
                    }
                }
                let mut seams = Vec::new();
                for &pi in &touched_poly_set {
                    for &orig_vi in target.poly_verts(pi as usize) {
                        if let Some(&rv) = remap.get(&orig_vi) {
                            seams.push(rv);
                        }
                    }
                }
                (m, seams)
            },
            || {
                let mut m = HybridMesh::new(num_uv_layers);
                let mut remap: HashMap<u32, u32> = HashMap::with_capacity(est_v);
                for (&pi, &dest) in sp.iter().zip(poly_dests.iter()) {
                    if dest == PolyDest::Piece {
                        m.push_poly_from_shared_map(target, pi as usize, &mut remap);
                    }
                }
                let mut seams = Vec::new();
                for &pi in &touched_poly_set {
                    for &orig_vi in target.poly_verts(pi as usize) {
                        if let Some(&pv) = remap.get(&orig_vi) {
                            seams.push(pv);
                        }
                    }
                }
                (m, seams)
            },
        );
        remaining = rem_built;
        piece = piece_built;
        rem_seam_verts = rem_seams;
        piece_seam_verts = piece_seams;
    } else {
        let poly_dests: Vec<PolyDest> = (0..target.num_polys())
            .into_par_iter()
            .map(classify_poly)
            .collect();
        let mut rem_remap = vec![u32::MAX; target.positions.len()];
        let mut piece_remap = vec![u32::MAX; target.positions.len()];
        for (poly_idx, &dest) in poly_dests.iter().enumerate() {
            match dest {
                PolyDest::Remaining => remaining.push_poly_from_shared(target, poly_idx, &mut rem_remap),
                PolyDest::Piece => piece.push_poly_from_shared(target, poly_idx, &mut piece_remap),
                PolyDest::Split => {}
            }
        }
        for &pi in &touched_poly_set {
            for &orig_vi in target.poly_verts(pi as usize) {
                let rv = rem_remap[orig_vi as usize];
                if rv != u32::MAX {
                    rem_seam_verts.push(rv);
                }
                let pv = piece_remap[orig_vi as usize];
                if pv != u32::MAX {
                    piece_seam_verts.push(pv);
                }
            }
        }
    }

    let rem_fixed_verts = remaining.positions.len();
    let rem_fixed_polys = remaining.num_polys();
    let piece_fixed_verts = piece.positions.len();
    let piece_fixed_polys = piece.num_polys();

    // 5. Split & classify TOUCHED target triangles in parallel using O(1) `poly_tri_starts`
    //    (NEVER scans 4,000,000 triangles!)
    let mut touched_polys_sorted: Vec<u32> = touched_poly_set.into_iter().collect();
    touched_polys_sorted.sort_unstable();
    let mut touched_target_tri_indices: Vec<usize> = Vec::with_capacity(touched_polys_sorted.len() * 2);
    for &pi in &touched_polys_sorted {
        let ts = target_geo.poly_tri_starts[pi as usize] as usize;
        let te = target_geo.poly_tri_starts[pi as usize + 1] as usize;
        for ti in ts..te {
            touched_target_tri_indices.push(ti);
        }
    }

    struct ClassifiedTargetTri {
        pts: [[f64; 3]; 3],
        uvs: Vec<[[f32; 2]; 3]>,
        nrms: [[f32; 3]; 3],
        mat_id: i32,
        is_cut: bool,
        to_piece: bool,
    }

    let target_has_nrms = target.loop_normals.len() == target.loop_verts.len();

    let split_target_results: Vec<ClassifiedTargetTri> = touched_target_tri_indices
        .par_iter()
        .flat_map(|&ti| {
            let t = target_tris[ti];
            let poly_idx = t.poly_idx as usize;
            let mat_id = target.mat_ids[poly_idx];
            let is_cut = target.is_cut_face[poly_idx];

            let tri_pts = [
                target.positions[t.v[0] as usize],
                target.positions[t.v[1] as usize],
                target.positions[t.v[2] as usize],
            ];
            let mut tri_uvs = Vec::with_capacity(num_uv_layers);
            for layer in &target.uv_layers {
                tri_uvs.push([
                    layer[t.l[0] as usize],
                    layer[t.l[1] as usize],
                    layer[t.l[2] as usize],
                ]);
            }
            let tri_nrms = if target_has_nrms {
                [
                    target.loop_normals[t.l[0] as usize],
                    target.loop_normals[t.l[1] as usize],
                    target.loop_normals[t.l[2] as usize],
                ]
            } else {
                let (_, gn) = tri_centroid_and_normal(&tri_pts);
                let gnf = [gn[0] as f32, gn[1] as f32, gn[2] as f32];
                [gnf, gnf, gnf]
            };

            let segs = segs_per_target_tri
                .get(&(ti as u32))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let sub_tris = split_triangle(tri_pts, &tri_uvs, tri_nrms, segs);

            let mut out = Vec::with_capacity(sub_tris.len());
            for st in sub_tris {
                let (c, n) = tri_centroid_and_normal(&st.pts);
                let loc = if let Some(ref planes) = convex_planes {
                    classify_point_convex(c, n, planes)
                } else {
                    classify_point(
                        c,
                        n,
                        &cutter_aabb,
                        &cutter.positions,
                        cutter_tris,
                        cutter_bvh,
                    )
                };
                let to_piece = matches!(loc, Location::Inside | Location::CoplanarSame);
                out.push(ClassifiedTargetTri {
                    pts: st.pts,
                    uvs: st.uvs,
                    nrms: st.nrms,
                    mat_id,
                    is_cut,
                    to_piece,
                });
            }
            out
        })
        .collect();

    for st in split_target_results {
        let uvs_sv: Vec<SmallVec<[[f32; 2]; 4]>> = st
            .uvs
            .iter()
            .map(|u| SmallVec::from_slice(u))
            .collect();
        if st.to_piece {
            piece.push_poly_with_positions_and_normals(&st.pts, &uvs_sv, &st.nrms, st.mat_id, st.is_cut);
        } else {
            remaining.push_poly_with_positions_and_normals(&st.pts, &uvs_sv, &st.nrms, st.mat_id, st.is_cut);
        }
    }

    // 6. Generate clean, watertight cut cap faces per WHOLE cutter polygon using
    //    2D Constrained Delaunay Triangulation + Dual-Graph Region Oriented Normal Voting.
    let cut_caps: Vec<[[f64; 3]; 3]> = (0..cutter.num_polys())
        .into_par_iter()
        .flat_map(|poly_idx| {
            let paabb = cutter.poly_aabb(poly_idx);
            if !paabb.expanded(PAIR_EPS).overlaps(&w_aabb) {
                return Vec::new();
            }

            let segs = segs_per_cutter_poly
                .get(&(poly_idx as u32))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let poly_n = cutter.poly_normal(poly_idx);

            // Fast path: untouched cutter polygon (either fully inside solid or fully outside in sky)
            if segs.is_empty() {
                let c = cutter.poly_centroid(poly_idx);
                let loc = classify_point_in_remaining(
                    c,
                    poly_n,
                    target_geo,
                    &w_aabb,
                    w_positions,
                    w_tris,
                    w_bvh,
                    excluded_volumes,
                );
                if loc != Location::Inside {
                    return Vec::new();
                }
                let tris = cutter.triangulate_poly(poly_idx);
                let mut out = Vec::with_capacity(tris.len());
                for (vs, _) in tris {
                    out.push([
                        cutter.positions[vs[0] as usize],
                        cutter.positions[vs[1] as usize],
                        cutter.positions[vs[2] as usize],
                    ]);
                }
                return out;
            }

            // CDT + Connected 2D Region Partitioning on the whole cutter polygon
            let poly_pts: Vec<[f64; 3]> = cutter
                .poly_verts(poly_idx)
                .iter()
                .map(|&vi| cutter.positions[vi as usize])
                .collect();

            let regions = triangulate_cutter_poly_regions(&poly_pts, poly_n, segs);
            let mut out = Vec::new();

            for region in regions {
                if region.tris.is_empty() || region.rep_points.is_empty() {
                    continue;
                }
                // Check coplanar Z-fighting & excluded prior cutter volumes on largest triangle centroid first
                let primary_loc = classify_point_in_remaining(
                    region.rep_points[0].0,
                    poly_n,
                    target_geo,
                    &w_aabb,
                    w_positions,
                    w_tris,
                    w_bvh,
                    excluded_volumes,
                );
                if matches!(primary_loc, Location::CoplanarSame | Location::CoplanarOpposite) {
                    continue;
                }

                // If the region's representative point lies strictly inside (or coplanar-same with)
                // a higher-priority excluded cutter, discard it (already claimed by Layer #1..k-1).
                if !excluded_volumes.is_empty() && primary_loc == Location::Outside {
                    let in_ex = excluded_volumes.iter().any(|ex| {
                        ex.aabb.expanded(PAIR_EPS).contains_point(region.rep_points[0].0)
                            && matches!(
                                classify_point(
                                    region.rep_points[0].0,
                                    poly_n,
                                    &ex.aabb,
                                    ex.positions,
                                    ex.tris,
                                    ex.bvh
                                ),
                                Location::Inside | Location::CoplanarSame
                            )
                    });
                    if in_ex {
                        continue;
                    }
                }

                // 2-level classifier:
                // 1) Unanimous / cleanly bounded region (|net| >= 0.70): one consistent decision for
                //    the entire region -> zero holes inside walls, zero spikes inside enclosed rooms!
                // 2) Leaked / mixed region (an open gap in a game mesh connected a solid slab to empty air):
                //    classify each clean Delaunay triangle individually using its direct cut-edge normal
                //    vote + 3D Fast Winding Number!
                let total_vote = region.solid_vote + region.air_vote;
                if total_vote > 1e-5 {
                    let net = (region.solid_vote - region.air_vote) / total_vote;
                    if net >= 0.70 {
                        out.extend(region.tris);
                    } else if net <= -0.70 {
                        // Clean air region -> discard all
                    } else {
                        // Mixed / leaked region -> 3-pass clean resolution:
                        // Pass 1: Initial per-triangle classification (2D harmonic field + 3D winding)
                        let m = region.tris.len();
                        let mut kept = vec![false; m];
                        let mut is_coplanar = vec![false; m];
                        let mut wn_cache = vec![-1.0f64; m];

                        for i in 0..m {
                            let tri_pts = &region.tris[i];
                            let hf = region.tri_field[i];
                            let (c, n) = tri_centroid_and_normal(tri_pts);
                            let loc = classify_point_in_remaining(
                                c,
                                n,
                                target_geo,
                                &w_aabb,
                                w_positions,
                                w_tris,
                                w_bvh,
                                excluded_volumes,
                            );
                            if matches!(loc, Location::CoplanarSame | Location::CoplanarOpposite) {
                                is_coplanar[i] = true;
                                continue;
                            }
                            kept[i] = if hf > 0.12 {
                                true
                            } else if hf < -0.12 {
                                false
                            } else {
                                loc == Location::Inside
                            };
                        }

                        // Pass 2: Iterative Dangling-Fin Pruning (removes floating triangles sticking out
                        // into empty air from single-sided interior walls without touching solid slabs)
                        loop {
                            let mut changed = false;
                            for i in 0..m {
                                if !kept[i] {
                                    continue;
                                }
                                let mut open_air_edges = 0u32;
                                let mut open_air_len = 0.0f64;
                                let mut perim = 0.0f64;
                                for e in &region.tri_edges[i] {
                                    perim += e.edge_len;
                                    if !e.is_cut_edge
                                        && !e.is_poly_rim
                                        && e.neighbor.map_or(false, |nb| !kept[nb])
                                    {
                                        open_air_edges += 1;
                                        open_air_len += e.edge_len;
                                    }
                                }
                                if open_air_edges >= 2 {
                                    kept[i] = false;
                                    changed = true;
                                } else if open_air_edges == 1 {
                                    if wn_cache[i] < 0.0 {
                                        let (c, _) = tri_centroid_and_normal(&region.tris[i]);
                                        wn_cache[i] = w_bvh.fast_winding_number(
                                            c,
                                            w_positions,
                                            w_tris,
                                        );
                                    }
                                    let long_open = perim > 1e-8 && (open_air_len / perim) > 0.55;
                                    if wn_cache[i] < 0.50 || (long_open && wn_cache[i] < 0.65) {
                                        kept[i] = false;
                                        changed = true;
                                    }
                                }
                            }
                            if !changed {
                                break;
                            }
                        }

                        // Pass 3: Trapped Pinhole Closing (fills 1-triangle pinholes completely
                        // surrounded by orig walls, box rim, and kept cap faces)
                        for _ in 0..3 {
                            let mut filled = false;
                            for i in 0..m {
                                if kept[i] || is_coplanar[i] {
                                    continue;
                                }
                                let surrounded = region.tri_edges[i].iter().all(|e| {
                                    e.is_cut_edge
                                        || e.is_poly_rim
                                        || e.neighbor.map_or(false, |nb| kept[nb])
                                });
                                if surrounded {
                                    kept[i] = true;
                                    filled = true;
                                }
                            }
                            if !filled {
                                break;
                            }
                        }

                        for i in 0..m {
                            if kept[i] {
                                out.push(region.tris[i]);
                            }
                        }
                    }
                } else if primary_loc == Location::Inside {
                    out.extend(region.tris);
                }
            }
            out
        })
        .collect();

    let empty_uvs: Vec<SmallVec<[[f32; 2]; 4]>> =
        vec![SmallVec::from_slice(&[[0.0, 0.0]; 3]); num_uv_layers];

    let rem_cap_start = remaining.num_polys();
    let piece_cap_start = piece.num_polys();
    for cap in cut_caps {
        // For `remaining = target \ cutter`, flip cap normal inward so it faces out of the cavity
        let flipped = [cap[0], cap[2], cap[1]];
        remaining.push_poly_with_positions(&flipped, &empty_uvs, cut_mat_idx, true);
        // For `piece = target ∩ cutter`, keep cutter's outward normal
        piece.push_poly_with_positions(&cap, &empty_uvs, cut_mat_idx, true);
    }

    // 7. Apply world-space Box UV to ONLY newly added cut cap faces (O(caps) instead of O(1M)),
    //    then partially weld new cut vertices in parallel
    rayon::join(
        || {
            apply_cut_face_box_uv_from(&mut remaining, rem_cap_start, target_world, uv_scale);
            remaining.weld_partial(rem_fixed_verts, rem_fixed_polys, &rem_seam_verts, WELD_EPS);
        },
        || {
            apply_cut_face_box_uv_from(&mut piece, piece_cap_start, target_world, uv_scale);
            piece.weld_partial(piece_fixed_verts, piece_fixed_polys, &piece_seam_verts, WELD_EPS);
        },
    );

    (remaining, piece)
}

/// Difference convenience wrapper (`target \ cutter`).
pub fn difference(
    target: &HybridMesh,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> HybridMesh {
    split_mesh(target, cutter, target_world, cut_mat_idx, uv_scale).0
}

/// Intersect convenience wrapper (`target ∩ cutter`).
pub fn intersect(
    target: &HybridMesh,
    cutter: &HybridMesh,
    target_world: &DMat4,
    cut_mat_idx: i32,
    uv_scale: f32,
) -> HybridMesh {
    split_mesh(target, cutter, target_world, cut_mat_idx, uv_scale).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Helper: build a unit/sized box centered at `center` with half-extents `half`,
    /// using 6 Quads with clean [0..1] UVs and optional open bottom.
    fn make_box_mesh(center: [f64; 3], half: [f64; 3], mat_id: i32, open_bottom: bool) -> HybridMesh {
        let [cx, cy, cz] = center;
        let [hx, hy, hz] = half;
        let v = [
            [cx - hx, cy - hy, cz - hz], // 0
            [cx + hx, cy - hy, cz - hz], // 1
            [cx + hx, cy + hy, cz - hz], // 2
            [cx - hx, cy + hy, cz - hz], // 3
            [cx - hx, cy - hy, cz + hz], // 4
            [cx + hx, cy - hy, cz + hz], // 5
            [cx + hx, cy + hy, cz + hz], // 6
            [cx - hx, cy + hy, cz + hz], // 7
        ];
        // Outward CCW quads
        let mut faces: Vec<[usize; 4]> = vec![
            [4, 5, 6, 7], // +Z top
            [0, 1, 5, 4], // -Y front
            [2, 3, 7, 6], // +Y back
            [3, 0, 4, 7], // -X left
            [1, 2, 6, 5], // +X right
        ];
        if !open_bottom {
            faces.push([0, 3, 2, 1]); // -Z bottom
        }

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

    #[test]
    fn test_basic_closed_split_and_hybrid_quads() {
        let target = make_box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0, false);
        // Cutter cuts only the +X half (x in [0.2, 2.0])
        let cutter = make_box_mesh([1.1, 0.0, 0.0], [0.9, 1.5, 1.5], 0, false);

        let (rem, piece) = split_mesh(&target, &cutter, &DMat4::IDENTITY, 1, 0.5);

        assert!(!rem.is_empty(), "remaining should not be empty");
        assert!(!piece.is_empty(), "piece should not be empty");

        // Untouched -X face of target must remain a pure 4-vertex Quad with mat_id 0!
        let has_untouched_quad = (0..rem.num_polys())
            .any(|i| rem.poly_verts(i).len() == 4 && rem.mat_ids[i] == 0 && !rem.is_cut_face[i]);
        assert!(has_untouched_quad, "untouched quad must stay a Quad");

        // Cut faces must have cut_mat_idx = 1 and non-zero Box UV spread
        let rem_cut_faces = rem.is_cut_face.iter().filter(|&&b| b).count();
        let piece_cut_faces = piece.is_cut_face.iter().filter(|&&b| b).count();
        assert!(rem_cut_faces > 0, "remaining must have cut cap faces");
        assert!(piece_cut_faces > 0, "piece must have cut cap faces");
        for i in 0..rem.num_polys() {
            if rem.is_cut_face[i] {
                assert_eq!(rem.mat_ids[i], 1);
            }
        }
    }

    #[test]
    fn test_open_bottom_game_mesh_split() {
        // Open-bottom building/box (non-manifold!)
        let target = make_box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0, true);
        let cutter = make_box_mesh([0.8, 0.8, 0.8], [0.6, 0.6, 0.6], 0, false);

        let (rem, piece) = split_mesh(&target, &cutter, &DMat4::IDENTITY, 2, 0.5);

        assert!(rem.num_polys() >= 5, "open-bottom mesh should keep its walls");
        assert!(!piece.is_empty(), "bitten corner piece should not be empty");
    }

    #[test]
    fn test_coplanar_no_z_fighting() {
        // Cutter shares exact Z=+1 and Y=+1 planes with target!
        let target = make_box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0, false);
        let cutter = make_box_mesh([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0, false);

        let (rem, piece) = split_mesh(&target, &cutter, &DMat4::IDENTITY, 1, 0.5);

        assert!(!rem.is_empty());
        assert!(!piece.is_empty());

        // Check no two polygons in `rem` share the exact same centroid (no double-drawn coplanar faces)
        let mut centroids: Vec<[f64; 3]> = (0..rem.num_polys())
            .map(|i| rem.poly_centroid(i))
            .collect();
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let d2 = DVec3::from_array(centroids[i])
                    .distance_squared(DVec3::from_array(centroids[j]));
                assert!(
                    d2 > 1e-8,
                    "duplicate/coplanar Z-fighting face detected at {:?}",
                    centroids[i]
                );
            }
        }
        centroids.clear();
    }

    #[test]
    fn test_uv_preservation_on_split_face() {
        let target = make_box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0, false);
        // Cut right down the middle X = 0
        let cutter = make_box_mesh([1.0, 0.0, 0.0], [1.0, 2.0, 2.0], 0, false);
        let (rem, _piece) = split_mesh(&target, &cutter, &DMat4::IDENTITY, 1, 0.5);

        // On the +Z top face (z == 1.0), U corresponds to (x + 1)/2, V corresponds to (y + 1)/2.
        // Every split vertex on the top face must still satisfy u ~= (x + 1)/2 and v ~= (y + 1)/2!
        let mut checked = 0usize;
        for poly_idx in 0..rem.num_polys() {
            if rem.is_cut_face[poly_idx] {
                continue;
            }
            let n = rem.poly_normal(poly_idx);
            if n[2] > 0.99 {
                for li in rem.poly_loop_range(poly_idx) {
                    let p = rem.positions[rem.loop_verts[li] as usize];
                    let uv = rem.uv_layers[0][li];
                    let expected_u = ((p[0] + 1.0) * 0.5) as f32;
                    let expected_v = ((p[1] + 1.0) * 0.5) as f32;
                    assert!(
                        (uv[0] - expected_u).abs() < 1e-4 && (uv[1] - expected_v).abs() < 1e-4,
                        "UV drifted at {:?}: got {:?}, expected ({}, {})",
                        p,
                        uv,
                        expected_u,
                        expected_v
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 3, "should have verified top face UVs");
    }

    #[test]
    fn test_sequential_20_cutters_speed() {
        let target = make_box_mesh([0.0, 0.0, 0.0], [5.0, 5.0, 2.0], 0, false);
        let cutters: Vec<HybridMesh> = (0..20)
            .map(|i| {
                let x = -4.5 + (i as f64) * 0.45;
                make_box_mesh([x, 4.5, 0.0], [0.35, 1.0, 3.0], 0, false)
            })
            .collect();

        let t0 = Instant::now();
        let mut rem = target;
        let mut pieces = Vec::new();
        for c in &cutters {
            let (next_rem, piece) = split_mesh(&rem, c, &DMat4::IDENTITY, 1, 0.5);
            rem = next_rem;
            pieces.push(piece);
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(!rem.is_empty());
        assert_eq!(pieces.len(), 20);
        assert!(pieces.iter().all(|p| !p.is_empty()));
        println!("20 sequential splits took {:.2} ms", elapsed_ms);
    }

    #[test]
    fn test_heavy_mesh_20k_tris_speed() {
        // Subdivided 6-face building box: 40x40 quads per face = 9,600 quads = 19,200 triangles!
        let n = 40usize;
        let mut target = HybridMesh::new(1);
        let quad_uv: SmallVec<[[f32; 2]; 4]> =
            SmallVec::from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let step = 2.0 / (n as f64);
        for side in 0..6 {
            for iy in 0..n {
                for ix in 0..n {
                    let u0 = -1.0 + (ix as f64) * step;
                    let u1 = u0 + step;
                    let v0 = -1.0 + (iy as f64) * step;
                    let v1 = v0 + step;
                    let pts = match side {
                        0 => [[u0, v0, 1.0], [u1, v0, 1.0], [u1, v1, 1.0], [u0, v1, 1.0]],
                        1 => [[u0, v1, -1.0], [u1, v1, -1.0], [u1, v0, -1.0], [u0, v0, -1.0]],
                        2 => [[u0, -1.0, v0], [u1, -1.0, v0], [u1, -1.0, v1], [u0, -1.0, v1]],
                        3 => [[u1, 1.0, v0], [u0, 1.0, v0], [u0, 1.0, v1], [u1, 1.0, v1]],
                        4 => [[1.0, u0, v0], [1.0, u1, v0], [1.0, u1, v1], [1.0, u0, v1]],
                        _ => [[-1.0, u1, v0], [-1.0, u0, v0], [-1.0, u0, v1], [-1.0, u1, v1]],
                    };
                    target.push_poly_with_positions(&pts, &[quad_uv.clone()], 0, false);
                }
            }
        }
        target.weld_vertices(1e-7);

        let cutter = make_box_mesh([0.7, 0.3, 0.0], [0.9, 1.2, 1.5], 0, false);

        // Measure cached live-drag speed (target BVH pre-built once)
        let target_geo = CachedMeshGeo::from_mesh(target);
        let t0 = Instant::now();
        let (rem, piece) = split_mesh_cached(&target_geo, &cutter, &DMat4::IDENTITY, 1, 0.5);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Heavy 19,200-tri CACHED split took {:.2} ms (rem={} polys, piece={} polys)",
            elapsed_ms,
            rem.num_polys(),
            piece.num_polys()
        );
        assert!(!rem.is_empty());
        assert!(!piece.is_empty());
        assert!(elapsed_ms < 500.0, "heavy mesh split too slow: {:.2} ms", elapsed_ms);

        // Verify pre-computed edges match every single polygon loop 100%
        for mesh in [&rem, &piece] {
            let (edge_verts, loop_edges) = mesh.compute_edges();
            let num_edges = edge_verts.len() / 2;
            assert_eq!(loop_edges.len(), mesh.loop_verts.len());
            for pi in 0..mesh.num_polys() {
                let r = mesh.poly_loop_range(pi);
                let m = r.len();
                for k in 0..m {
                    let v0 = mesh.loop_verts[r.start + k];
                    let v1 = mesh.loop_verts[r.start + ((k + 1) % m)];
                    let ei = loop_edges[r.start + k] as usize;
                    assert!(ei < num_edges);
                    let ev0 = edge_verts[ei * 2];
                    let ev1 = edge_verts[ei * 2 + 1];
                    assert!(
                        (ev0 == v0 && ev1 == v1) || (ev0 == v1 && ev1 == v0),
                        "edge mismatch on poly {} loop {}",
                        pi,
                        k
                    );
                }
            }
        }
    }

    #[test]
    fn test_single_sided_internal_wall_no_fins() {
        // Replicates an Enfusion hollow attic: solid floor slab (z = 0..0.5) + single-sided
        // vertical interior partition wall at x = 0.0 (z = 0.5..2.0, normal = +X, no back face!).
        let mut target = make_box_mesh([0.0, 0.0, 0.25], [2.0, 2.0, 0.25], 0, false);
        let quad_uv: SmallVec<[[f32; 2]; 4]> =
            SmallVec::from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        // Single-sided vertical wall at x = 0.0, y in [-2, 2], z in [0.5, 2.0], normal = (+1, 0, 0)
        let wall_pts = [
            [0.0, -2.0, 0.5],
            [0.0, 2.0, 0.5],
            [0.0, 2.0, 2.0],
            [0.0, -2.0, 2.0],
        ];
        target.push_poly_with_positions(&wall_pts, &[quad_uv], 0, false);
        target.weld_vertices(1e-7);

        // Cut through y = 0.0 across both the floor slab AND the single-sided attic wall
        let cutter = make_box_mesh([0.0, -1.5, 1.0], [3.0, 1.5, 2.0], 0, false);
        let (_rem, piece) = split_mesh(&target, &cutter, &DMat4::IDENTITY, 1, 1.0);

        let mut floor_caps = 0usize;
        let mut attic_fin_caps = 0usize;
        for pi in 0..piece.num_polys() {
            if !piece.is_cut_face[pi] {
                continue;
            }
            let c = piece.poly_centroid(pi);
            if c[2] > 0.6 {
                attic_fin_caps += 1;
            } else {
                floor_caps += 1;
            }
        }
        assert!(floor_caps > 0, "Solid floor slab must still be capped!");
        assert_eq!(
            attic_fin_caps, 0,
            "Single-sided attic wall must not produce floating triangular fin caps (got {})",
            attic_fin_caps
        );
    }
}
