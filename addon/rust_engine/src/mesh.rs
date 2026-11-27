//! Hybrid polygon/triangle mesh with multi-layer per-loop UVs and material IDs.
//!
//! Layout mirrors Blender's Mesh data model (CSR polygons + per-loop UVs) so
//! transferring data across the C-ABI via `foreach_get`/`foreach_set` is direct
//! and untouched polygons remain 100% original Quads/N-gons.

use crate::bvh::Bvh;
use glam::{DMat4, DVec3};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    #[inline]
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    #[inline]
    pub fn from_point(p: [f64; 3]) -> Self {
        Self { min: p, max: p }
    }

    #[inline]
    pub fn expand_point(&mut self, p: [f64; 3]) {
        for i in 0..3 {
            if p[i] < self.min[i] {
                self.min[i] = p[i];
            }
            if p[i] > self.max[i] {
                self.max[i] = p[i];
            }
        }
    }

    #[inline]
    pub fn expand_aabb(&mut self, other: &Aabb) {
        for i in 0..3 {
            if other.min[i] < self.min[i] {
                self.min[i] = other.min[i];
            }
            if other.max[i] > self.max[i] {
                self.max[i] = other.max[i];
            }
        }
    }

    #[inline]
    pub fn expanded(&self, eps: f64) -> Self {
        Self {
            min: [self.min[0] - eps, self.min[1] - eps, self.min[2] - eps],
            max: [self.max[0] + eps, self.max[1] + eps, self.max[2] + eps],
        }
    }

    #[inline]
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    #[inline]
    pub fn contains_point(&self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    #[inline]
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    #[inline]
    pub fn longest_axis(&self) -> usize {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz {
            0
        } else if dy >= dz {
            1
        } else {
            2
        }
    }
}

/// A hybrid mesh that stores arbitrary polygons (triangles, quads, n-gons)
/// alongside multi-layer per-loop UVs, per-loop split normals, and per-polygon material indices.
#[derive(Clone, Debug)]
pub struct HybridMesh {
    pub positions: Vec<[f64; 3]>,
    pub loop_verts: Vec<u32>,
    /// CSR offsets into `loop_verts`, length = `num_polys + 1`.
    pub poly_starts: Vec<u32>,
    /// `uv_layers[layer_idx][loop_idx] = [u, v]`.
    pub uv_layers: Vec<Vec<[f32; 2]>>,
    /// Per-loop custom/split unit normals (length = `loop_verts.len()`).
    pub loop_normals: Vec<[f32; 3]>,
    /// Per-polygon material slot index.
    pub mat_ids: Vec<i32>,
    /// Per-polygon flag: `true` if generated as a new cut cap face.
    pub is_cut_face: Vec<bool>,
}

impl HybridMesh {
    pub fn new(num_uv_layers: usize) -> Self {
        Self {
            positions: Vec::new(),
            loop_verts: Vec::new(),
            poly_starts: vec![0],
            uv_layers: vec![Vec::new(); num_uv_layers],
            loop_normals: Vec::new(),
            mat_ids: Vec::new(),
            is_cut_face: Vec::new(),
        }
    }

    /// Ensure `loop_normals` has length `loop_verts.len()` (filling missing entries with geometric face normals).
    pub fn ensure_loop_normals(&mut self) {
        let nl = self.loop_verts.len();
        if self.loop_normals.len() == nl {
            return;
        }
        self.loop_normals.clear();
        self.loop_normals.reserve(nl);
        let np = self.num_polys();
        for pi in 0..np {
            let n = self.poly_normal(pi);
            let nf = [n[0] as f32, n[1] as f32, n[2] as f32];
            let cnt = self.poly_loop_range(pi).len();
            for _ in 0..cnt {
                self.loop_normals.push(nf);
            }
        }
    }

    #[inline]
    pub fn num_polys(&self) -> usize {
        self.mat_ids.len()
    }

    #[inline]
    pub fn num_uv_layers(&self) -> usize {
        self.uv_layers.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mat_ids.is_empty()
    }

    #[inline]
    pub fn poly_loop_range(&self, poly_idx: usize) -> std::ops::Range<usize> {
        let start = self.poly_starts[poly_idx] as usize;
        let end = self.poly_starts[poly_idx + 1] as usize;
        start..end
    }

    #[inline]
    pub fn poly_verts(&self, poly_idx: usize) -> &[u32] {
        let r = self.poly_loop_range(poly_idx);
        &self.loop_verts[r]
    }

    pub fn poly_aabb(&self, poly_idx: usize) -> Aabb {
        let mut aabb = Aabb::empty();
        for &vi in self.poly_verts(poly_idx) {
            aabb.expand_point(self.positions[vi as usize]);
        }
        aabb
    }

    pub fn mesh_aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        for &p in &self.positions {
            aabb.expand_point(p);
        }
        aabb
    }

    pub fn poly_centroid(&self, poly_idx: usize) -> [f64; 3] {
        let verts = self.poly_verts(poly_idx);
        if verts.is_empty() {
            return [0.0; 3];
        }
        let mut sum = [0.0; 3];
        for &vi in verts {
            let p = self.positions[vi as usize];
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
        }
        let inv = 1.0 / (verts.len() as f64);
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    }

    /// Newell's method polygon normal (unit length, or [0,0,1] if degenerate).
    pub fn poly_normal(&self, poly_idx: usize) -> [f64; 3] {
        let verts = self.poly_verts(poly_idx);
        let n = verts.len();
        if n < 3 {
            return [0.0, 0.0, 1.0];
        }
        let mut nx = 0.0;
        let mut ny = 0.0;
        let mut nz = 0.0;
        for i in 0..n {
            let p0 = self.positions[verts[i] as usize];
            let p1 = self.positions[verts[(i + 1) % n] as usize];
            nx += (p0[1] - p1[1]) * (p0[2] + p1[2]);
            ny += (p0[2] - p1[2]) * (p0[0] + p1[0]);
            nz += (p0[0] - p1[0]) * (p0[1] + p1[1]);
        }
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-15 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Push a polygon with explicit vertex positions (welded later), per-layer UVs, and geometric normals.
    pub fn push_poly_with_positions(
        &mut self,
        pts: &[[f64; 3]],
        uvs_per_layer: &[SmallVec<[[f32; 2]; 4]>],
        mat_id: i32,
        is_cut: bool,
    ) {
        if pts.len() < 3 {
            return;
        }
        let p0 = DVec3::from_array(pts[0]);
        let p1 = DVec3::from_array(pts[1]);
        let p2 = DVec3::from_array(pts[2]);
        let gn = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let nf = if gn.length_squared() > 1e-12 {
            [gn.x as f32, gn.y as f32, gn.z as f32]
        } else {
            [0.0, 0.0, 1.0]
        };
        let mut nrms: SmallVec<[[f32; 3]; 4]> = SmallVec::with_capacity(pts.len());
        for _ in 0..pts.len() {
            nrms.push(nf);
        }
        self.push_poly_with_positions_and_normals(pts, uvs_per_layer, &nrms, mat_id, is_cut);
    }

    /// Push a polygon with explicit vertex positions, per-layer UVs, and per-loop split normals.
    pub fn push_poly_with_positions_and_normals(
        &mut self,
        pts: &[[f64; 3]],
        uvs_per_layer: &[SmallVec<[[f32; 2]; 4]>],
        normals: &[[f32; 3]],
        mat_id: i32,
        is_cut: bool,
    ) {
        if pts.len() < 3 {
            return;
        }
        let base_v = self.positions.len() as u32;
        self.positions.extend_from_slice(pts);
        for i in 0..pts.len() {
            self.loop_verts.push(base_v + i as u32);
        }
        for (layer_idx, layer) in self.uv_layers.iter_mut().enumerate() {
            if let Some(src) = uvs_per_layer.get(layer_idx) {
                layer.extend_from_slice(src);
            } else {
                for _ in 0..pts.len() {
                    layer.push([0.0, 0.0]);
                }
            }
        }
        if normals.len() == pts.len() {
            self.loop_normals.extend_from_slice(normals);
        } else {
            for _ in 0..pts.len() {
                self.loop_normals.push([0.0, 0.0, 1.0]);
            }
        }
        self.poly_starts.push(self.loop_verts.len() as u32);
        self.mat_ids.push(mat_id);
        self.is_cut_face.push(is_cut);
    }

    /// Copy an untouched polygon directly from `src`, preserving its exact
    /// Quad/N-gon topology, vertex positions, all UV layers, split normals, and material ID.
    pub fn push_poly_from(&mut self, src: &HybridMesh, poly_idx: usize) {
        let range = src.poly_loop_range(poly_idx);
        let n = range.len();
        if n < 3 {
            return;
        }
        let base_v = self.positions.len() as u32;
        for li in range.clone() {
            let vi = src.loop_verts[li] as usize;
            self.positions.push(src.positions[vi]);
        }
        for i in 0..n {
            self.loop_verts.push(base_v + i as u32);
        }
        for (layer_idx, dst_layer) in self.uv_layers.iter_mut().enumerate() {
            if let Some(src_layer) = src.uv_layers.get(layer_idx) {
                dst_layer.extend_from_slice(&src_layer[range.clone()]);
            } else {
                for _ in 0..n {
                    dst_layer.push([0.0, 0.0]);
                }
            }
        }
        if src.loop_normals.len() == src.loop_verts.len() {
            self.loop_normals.extend_from_slice(&src.loop_normals[range.clone()]);
        } else {
            let pn = src.poly_normal(poly_idx);
            let nf = [pn[0] as f32, pn[1] as f32, pn[2] as f32];
            for _ in 0..n {
                self.loop_normals.push(nf);
            }
        }
        self.poly_starts.push(self.loop_verts.len() as u32);
        self.mat_ids.push(src.mat_ids[poly_idx]);
        self.is_cut_face.push(src.is_cut_face[poly_idx]);
    }

    /// Triangulate a single polygon into `(v0, v1, v2, l0, l1, l2)` tuples.
    /// Uses 2D ear-clipping for N-gons (N > 3) so concave polygons stay accurate.
    pub fn triangulate_poly(&self, poly_idx: usize) -> SmallVec<[([u32; 3], [usize; 3]); 2]> {
        let range = self.poly_loop_range(poly_idx);
        let n = range.len();
        let mut out = SmallVec::new();
        if n < 3 {
            return out;
        }
        let l_start = range.start;
        if n == 3 {
            let v0 = self.loop_verts[l_start];
            let v1 = self.loop_verts[l_start + 1];
            let v2 = self.loop_verts[l_start + 2];
            out.push(([v0, v1, v2], [l_start, l_start + 1, l_start + 2]));
            return out;
        }
        if n == 4 {
            // Split quad along shorter / better diagonal
            let v = [
                self.loop_verts[l_start],
                self.loop_verts[l_start + 1],
                self.loop_verts[l_start + 2],
                self.loop_verts[l_start + 3],
            ];
            let p = [
                DVec3::from_array(self.positions[v[0] as usize]),
                DVec3::from_array(self.positions[v[1] as usize]),
                DVec3::from_array(self.positions[v[2] as usize]),
                DVec3::from_array(self.positions[v[3] as usize]),
            ];
            let d02 = p[0].distance_squared(p[2]);
            let d13 = p[1].distance_squared(p[3]);
            if d02 <= d13 {
                out.push(([v[0], v[1], v[2]], [l_start, l_start + 1, l_start + 2]));
                out.push(([v[0], v[2], v[3]], [l_start, l_start + 2, l_start + 3]));
            } else {
                out.push(([v[0], v[1], v[3]], [l_start, l_start + 1, l_start + 3]));
                out.push(([v[1], v[2], v[3]], [l_start + 1, l_start + 2, l_start + 3]));
            }
            return out;
        }

        // General N-gon ear-clipping projected to dominant plane
        let normal = self.poly_normal(poly_idx);
        let (ax0, ax1, flip) = dominant_2d_axes(normal);
        let mut indices: Vec<usize> = (0..n).collect();
        let pts2d: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let p = self.positions[self.loop_verts[l_start + i] as usize];
                if flip {
                    [p[ax1], p[ax0]]
                } else {
                    [p[ax0], p[ax1]]
                }
            })
            .collect();

        // Ensure CCW in 2D
        let mut signed_area = 0.0;
        for i in 0..n {
            let a = pts2d[i];
            let b = pts2d[(i + 1) % n];
            signed_area += a[0] * b[1] - b[0] * a[1];
        }
        let ccw_sign = if signed_area >= 0.0 { 1.0 } else { -1.0 };

        let mut guard = n * n;
        while indices.len() > 3 && guard > 0 {
            guard -= 1;
            let m = indices.len();
            let mut clipped = false;
            for i in 0..m {
                let prev = indices[(i + m - 1) % m];
                let curr = indices[i];
                let next = indices[(i + 1) % m];
                let a = pts2d[prev];
                let b = pts2d[curr];
                let c = pts2d[next];
                let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                if cross * ccw_sign <= 1e-12 {
                    continue;
                }
                let mut any_inside = false;
                for &k in &indices {
                    if k == prev || k == curr || k == next {
                        continue;
                    }
                    if point_in_tri_2d(pts2d[k], a, b, c, ccw_sign) {
                        any_inside = true;
                        break;
                    }
                }
                if !any_inside {
                    out.push((
                        [
                            self.loop_verts[l_start + prev],
                            self.loop_verts[l_start + curr],
                            self.loop_verts[l_start + next],
                        ],
                        [l_start + prev, l_start + curr, l_start + next],
                    ));
                    indices.remove(i);
                    clipped = true;
                    break;
                }
            }
            if !clipped {
                // Fallback fan for nearly degenerate n-gons
                let i0 = indices[0];
                for j in 1..indices.len() - 1 {
                    let i1 = indices[j];
                    let i2 = indices[j + 1];
                    out.push((
                        [
                            self.loop_verts[l_start + i0],
                            self.loop_verts[l_start + i1],
                            self.loop_verts[l_start + i2],
                        ],
                        [l_start + i0, l_start + i1, l_start + i2],
                    ));
                }
                return out;
            }
        }
        if indices.len() == 3 {
            out.push((
                [
                    self.loop_verts[l_start + indices[0]],
                    self.loop_verts[l_start + indices[1]],
                    self.loop_verts[l_start + indices[2]],
                ],
                [l_start + indices[0], l_start + indices[1], l_start + indices[2]],
            ));
        }
        out
    }

    /// Transform all vertices in-place by a 4x4 matrix.
    pub fn transform_in_place(&mut self, mat: &DMat4) {
        for p in &mut self.positions {
            let v = mat.transform_point3(DVec3::from_array(*p));
            *p = v.to_array();
        }
    }

    /// Weld spatially coincident vertices within `eps` without altering per-loop UVs,
    /// and sanitize polygons so no polygon ever contains duplicate vertex indices
    /// (which would crash Blender's BMesh on Apply).
    pub fn weld_vertices(&mut self, eps: f64) {
        if self.positions.is_empty() {
            return;
        }
        let inv_eps = 1.0 / eps.max(1e-12);
        let mut cell_map: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(self.positions.len());
        let mut remap: Vec<u32> = Vec::with_capacity(self.positions.len());
        let mut new_positions: Vec<[f64; 3]> = Vec::with_capacity(self.positions.len());

        for &p in &self.positions {
            let key = (
                (p[0] * inv_eps).round() as i64,
                (p[1] * inv_eps).round() as i64,
                (p[2] * inv_eps).round() as i64,
            );
            if let Some(&existing) = cell_map.get(&key) {
                remap.push(existing);
            } else {
                let new_idx = new_positions.len() as u32;
                new_positions.push(p);
                cell_map.insert(key, new_idx);
                remap.push(new_idx);
            }
        }

        let num_polys = self.num_polys();
        let num_uv = self.uv_layers.len();
        let has_nrms = self.loop_normals.len() == self.loop_verts.len();
        let mut clean_loop_verts: Vec<u32> = Vec::with_capacity(self.loop_verts.len());
        let mut clean_poly_starts: Vec<u32> = Vec::with_capacity(num_polys + 1);
        clean_poly_starts.push(0);
        let mut clean_uvs: Vec<Vec<[f32; 2]>> = (0..num_uv)
            .map(|_| Vec::with_capacity(self.loop_verts.len()))
            .collect();
        let mut clean_normals: Vec<[f32; 3]> = if has_nrms {
            Vec::with_capacity(self.loop_verts.len())
        } else {
            Vec::new()
        };
        let mut clean_mat_ids: Vec<i32> = Vec::with_capacity(num_polys);
        let mut clean_is_cut: Vec<bool> = Vec::with_capacity(num_polys);

        let mut kept_verts: SmallVec<[u32; 8]> = SmallVec::new();
        let mut kept_loops: SmallVec<[usize; 8]> = SmallVec::new();

        for poly_idx in 0..num_polys {
            let range = self.poly_loop_range(poly_idx);
            kept_verts.clear();
            kept_loops.clear();

            for li in range {
                let v = remap[self.loop_verts[li] as usize];
                if !kept_verts.contains(&v) {
                    kept_verts.push(v);
                    kept_loops.push(li);
                }
            }

            // Drop degenerate polygons with < 3 unique vertices or zero area
            if kept_verts.len() < 3 {
                continue;
            }
            let p0 = DVec3::from_array(new_positions[kept_verts[0] as usize]);
            let mut area2 = DVec3::ZERO;
            for k in 1..(kept_verts.len() - 1) {
                let p1 = DVec3::from_array(new_positions[kept_verts[k] as usize]);
                let p2 = DVec3::from_array(new_positions[kept_verts[k + 1] as usize]);
                area2 += (p1 - p0).cross(p2 - p0);
            }
            if area2.length_squared() < 1e-16 {
                continue;
            }

            clean_loop_verts.extend_from_slice(&kept_verts);
            for (layer_idx, dst_layer) in clean_uvs.iter_mut().enumerate() {
                let src_layer = &self.uv_layers[layer_idx];
                for &li in &kept_loops {
                    dst_layer.push(src_layer[li]);
                }
            }
            if has_nrms {
                for &li in &kept_loops {
                    clean_normals.push(self.loop_normals[li]);
                }
            }
            clean_poly_starts.push(clean_loop_verts.len() as u32);
            clean_mat_ids.push(self.mat_ids[poly_idx]);
            clean_is_cut.push(self.is_cut_face[poly_idx]);
        }

        self.positions = new_positions;
        self.loop_verts = clean_loop_verts;
        self.poly_starts = clean_poly_starts;
        self.uv_layers = clean_uvs;
        self.loop_normals = clean_normals;
        self.mat_ids = clean_mat_ids;
        self.is_cut_face = clean_is_cut;
    }

    /// Copy an untouched polygon directly from `src`, sharing original vertex indices
    /// via `vert_remap` (O(1) per loop, avoiding full-mesh vertex hashing later).
    pub fn push_poly_from_shared(
        &mut self,
        src: &HybridMesh,
        poly_idx: usize,
        vert_remap: &mut [u32],
    ) {
        let range = src.poly_loop_range(poly_idx);
        let n = range.len();
        if n < 3 {
            return;
        }
        for li in range.clone() {
            let orig_vi = src.loop_verts[li] as usize;
            let mut mapped = vert_remap[orig_vi];
            if mapped == u32::MAX {
                mapped = self.positions.len() as u32;
                self.positions.push(src.positions[orig_vi]);
                vert_remap[orig_vi] = mapped;
            }
            self.loop_verts.push(mapped);
        }
        for (layer_idx, dst_layer) in self.uv_layers.iter_mut().enumerate() {
            if let Some(src_layer) = src.uv_layers.get(layer_idx) {
                dst_layer.extend_from_slice(&src_layer[range.clone()]);
            } else {
                for _ in 0..n {
                    dst_layer.push([0.0, 0.0]);
                }
            }
        }
        if src.loop_normals.len() == src.loop_verts.len() {
            self.loop_normals.extend_from_slice(&src.loop_normals[range.clone()]);
        } else {
            let pn = src.poly_normal(poly_idx);
            let nf = [pn[0] as f32, pn[1] as f32, pn[2] as f32];
            for _ in 0..n {
                self.loop_normals.push(nf);
            }
        }
        self.poly_starts.push(self.loop_verts.len() as u32);
        self.mat_ids.push(src.mat_ids[poly_idx]);
        self.is_cut_face.push(src.is_cut_face[poly_idx]);
    }

    /// Weld only newly added vertices (`fixed_vert_count..`) and sanitize only newly added
    /// polygons (`fixed_poly_count..`), while seeding the spatial hash with `seam_verts`
    /// (boundary vertices from untouched polygons adjacent to the cut) so seams weld cleanly.
    pub fn weld_partial(
        &mut self,
        fixed_vert_count: usize,
        fixed_poly_count: usize,
        seam_verts: &[u32],
        eps: f64,
    ) {
        if fixed_poly_count >= self.num_polys() {
            return;
        }
        let inv_eps = 1.0 / eps.max(1e-12);
        let new_v_count = self.positions.len().saturating_sub(fixed_vert_count);
        let mut cell_map: HashMap<(i64, i64, i64), u32> =
            HashMap::with_capacity(seam_verts.len() + new_v_count);

        let key_of = |p: [f64; 3]| -> (i64, i64, i64) {
            (
                (p[0] * inv_eps).round() as i64,
                (p[1] * inv_eps).round() as i64,
                (p[2] * inv_eps).round() as i64,
            )
        };

        // Seed spatial hash with untouched boundary vertices so split triangles weld seamlessly
        for &vi in seam_verts {
            let p = self.positions[vi as usize];
            cell_map.entry(key_of(p)).or_insert(vi);
        }

        let mut new_remap: Vec<u32> = Vec::with_capacity(new_v_count);
        let mut welded_tail_positions: Vec<[f64; 3]> = Vec::with_capacity(new_v_count);

        for vi in fixed_vert_count..self.positions.len() {
            let p = self.positions[vi];
            let k = key_of(p);
            if let Some(&existing) = cell_map.get(&k) {
                new_remap.push(existing);
            } else {
                let dst_idx = (fixed_vert_count + welded_tail_positions.len()) as u32;
                welded_tail_positions.push(p);
                cell_map.insert(k, dst_idx);
                new_remap.push(dst_idx);
            }
        }

        // Truncate positions back to fixed_vert_count and append only unique welded new positions
        self.positions.truncate(fixed_vert_count);
        self.positions.extend_from_slice(&welded_tail_positions);

        // Sanitize ONLY the newly added polygons (`fixed_poly_count..`)
        let fixed_loop_count = self.poly_starts[fixed_poly_count] as usize;
        let orig_num_polys = self.num_polys();

        let has_nrms = self.loop_normals.len() == self.loop_verts.len();
        let tail_loops = self.loop_verts.split_off(fixed_loop_count);
        let tail_starts = self.poly_starts.split_off(fixed_poly_count + 1);
        let tail_mats = self.mat_ids.split_off(fixed_poly_count);
        let tail_cuts = self.is_cut_face.split_off(fixed_poly_count);
        let tail_uvs: Vec<Vec<[f32; 2]>> = self
            .uv_layers
            .iter_mut()
            .map(|l| l.split_off(fixed_loop_count))
            .collect();
        let tail_nrms: Vec<[f32; 3]> = if has_nrms {
            self.loop_normals.split_off(fixed_loop_count)
        } else {
            Vec::new()
        };

        let mut kept_verts: SmallVec<[u32; 8]> = SmallVec::new();
        let mut kept_local_loops: SmallVec<[usize; 8]> = SmallVec::new();
        let mut seen_tail_faces: std::collections::HashSet<SmallVec<[u32; 4]>> =
            std::collections::HashSet::with_capacity(orig_num_polys - fixed_poly_count);

        let mut prev_start = fixed_loop_count;
        for local_pi in 0..(orig_num_polys - fixed_poly_count) {
            let end = tail_starts[local_pi] as usize;
            let rel_start = prev_start - fixed_loop_count;
            let rel_end = end - fixed_loop_count;
            prev_start = end;

            kept_verts.clear();
            kept_local_loops.clear();

            for rel_li in rel_start..rel_end {
                let raw_v = tail_loops[rel_li] as usize;
                let v = if raw_v < fixed_vert_count {
                    raw_v as u32
                } else {
                    new_remap[raw_v - fixed_vert_count]
                };
                if !kept_verts.contains(&v) {
                    kept_verts.push(v);
                    kept_local_loops.push(rel_li);
                }
            }

            if kept_verts.len() < 3 {
                continue;
            }

            // Drop zero-area degenerate slivers created by welding near-collinear cut triangles
            let p0 = DVec3::from_array(self.positions[kept_verts[0] as usize]);
            let mut area2 = DVec3::ZERO;
            for k in 1..(kept_verts.len() - 1) {
                let p1 = DVec3::from_array(self.positions[kept_verts[k] as usize]);
                let p2 = DVec3::from_array(self.positions[kept_verts[k + 1] as usize]);
                area2 += (p1 - p0).cross(p2 - p0);
            }
            if area2.length_squared() < 1e-15 {
                continue;
            }

            // Drop duplicate faces (same unordered vertex set) so Blender's validator stays 100% clean
            let mut face_key: SmallVec<[u32; 4]> = kept_verts.iter().copied().collect();
            face_key.sort_unstable();
            if !seen_tail_faces.insert(face_key) {
                continue;
            }

            self.loop_verts.extend_from_slice(&kept_verts);
            for (layer_idx, dst_layer) in self.uv_layers.iter_mut().enumerate() {
                let src_layer = &tail_uvs[layer_idx];
                for &rel_li in &kept_local_loops {
                    dst_layer.push(src_layer[rel_li]);
                }
            }
            if has_nrms {
                for &rel_li in &kept_local_loops {
                    self.loop_normals.push(tail_nrms[rel_li]);
                }
            }
            self.poly_starts.push(self.loop_verts.len() as u32);
            self.mat_ids.push(tail_mats[local_pi]);
            self.is_cut_face.push(tail_cuts[local_pi]);
        }
    }

    /// Pre-compute Blender-compatible unique undirected edges (`edge_verts`: `[v0, v1, ...]`)
    /// and per-loop edge indices (`loop_edges`: length = `loop_verts.len()`).
    pub fn compute_edges(&self) -> (Vec<u32>, Vec<u32>) {
        let nl = self.loop_verts.len();
        if nl == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut edge_map: HashMap<u64, u32> = HashMap::with_capacity(nl);
        let mut edge_verts: Vec<u32> = Vec::with_capacity(nl * 2);
        let mut loop_edges: Vec<u32> = vec![0u32; nl];

        let num_polys = self.num_polys();
        for poly_idx in 0..num_polys {
            let start = self.poly_starts[poly_idx] as usize;
            let end = self.poly_starts[poly_idx + 1] as usize;
            let n = end - start;
            if n < 2 {
                continue;
            }
            for k in 0..n {
                let v0 = self.loop_verts[start + k];
                let v1 = self.loop_verts[start + ((k + 1) % n)];
                let (lo, hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                let key = ((lo as u64) << 32) | (hi as u64);
                let edge_idx = match edge_map.get(&key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = (edge_verts.len() / 2) as u32;
                        edge_verts.push(lo);
                        edge_verts.push(hi);
                        edge_map.insert(key, idx);
                        idx
                    }
                };
                loop_edges[start + k] = edge_idx;
            }
        }

        (edge_verts, loop_edges)
    }

    /// Extract a subset of polygons by index in O(K) time (never allocates O(V_total)
    /// when `poly_indices` is a small fraction of a 1M-vertex base mesh).
    pub fn extract_subset(&self, poly_indices: &[u32]) -> HybridMesh {
        let num_uv = self.num_uv_layers();
        let mut out = HybridMesh::new(num_uv);
        if poly_indices.is_empty() || self.is_empty() {
            return out;
        }
        let k = poly_indices.len();
        out.poly_starts.reserve(k);
        out.mat_ids.reserve(k);
        out.is_cut_face.reserve(k);
        out.loop_verts.reserve(k * 4);
        out.loop_normals.reserve(k * 4);
        for l in &mut out.uv_layers {
            l.reserve(k * 4);
        }

        let has_nrms = self.loop_normals.len() == self.loop_verts.len();
        if k * 16 < self.positions.len() {
            let mut vmap: HashMap<u32, u32> = HashMap::with_capacity(k * 2);
            for &pi_u32 in poly_indices {
                let pi = pi_u32 as usize;
                let range = self.poly_loop_range(pi);
                let n = range.len();
                if n < 3 {
                    continue;
                }
                for li in range.clone() {
                    let orig_vi = self.loop_verts[li];
                    let mapped = match vmap.get(&orig_vi) {
                        Some(&m) => m,
                        None => {
                            let m = out.positions.len() as u32;
                            out.positions.push(self.positions[orig_vi as usize]);
                            vmap.insert(orig_vi, m);
                            m
                        }
                    };
                    out.loop_verts.push(mapped);
                }
                for (layer_idx, dst_layer) in out.uv_layers.iter_mut().enumerate() {
                    if let Some(src_layer) = self.uv_layers.get(layer_idx) {
                        dst_layer.extend_from_slice(&src_layer[range.clone()]);
                    } else {
                        for _ in 0..n {
                            dst_layer.push([0.0, 0.0]);
                        }
                    }
                }
                if has_nrms {
                    out.loop_normals.extend_from_slice(&self.loop_normals[range]);
                } else {
                    let pn = self.poly_normal(pi);
                    let nf = [pn[0] as f32, pn[1] as f32, pn[2] as f32];
                    for _ in 0..n {
                        out.loop_normals.push(nf);
                    }
                }
                out.poly_starts.push(out.loop_verts.len() as u32);
                out.mat_ids.push(self.mat_ids[pi]);
                out.is_cut_face.push(self.is_cut_face[pi]);
            }
        } else {
            let mut remap = vec![u32::MAX; self.positions.len()];
            for &pi_u32 in poly_indices {
                out.push_poly_from_shared(self, pi_u32 as usize, &mut remap);
            }
        }
        out
    }

    /// Bulk-append all geometry from `other` into `self` with vertex/loop offset remapping.
    pub fn append_mesh(&mut self, other: &HybridMesh) {
        if other.is_empty() {
            return;
        }
        let v_off = self.positions.len() as u32;
        let l_off = self.loop_verts.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.loop_verts.reserve(other.loop_verts.len());
        for &vi in &other.loop_verts {
            self.loop_verts.push(vi + v_off);
        }
        self.poly_starts.reserve(other.num_polys());
        for &s in &other.poly_starts[1..] {
            self.poly_starts.push(s + l_off);
        }
        for (layer_idx, dst_layer) in self.uv_layers.iter_mut().enumerate() {
            if let Some(src_layer) = other.uv_layers.get(layer_idx) {
                dst_layer.extend_from_slice(src_layer);
            } else {
                dst_layer.resize(dst_layer.len() + other.loop_verts.len(), [0.0, 0.0]);
            }
        }
        if other.loop_normals.len() == other.loop_verts.len() {
            self.loop_normals.extend_from_slice(&other.loop_normals);
        } else {
            let np = other.num_polys();
            for pi in 0..np {
                let pn = other.poly_normal(pi);
                let nf = [pn[0] as f32, pn[1] as f32, pn[2] as f32];
                let cnt = other.poly_loop_range(pi).len();
                for _ in 0..cnt {
                    self.loop_normals.push(nf);
                }
            }
        }
        self.mat_ids.extend_from_slice(&other.mat_ids);
        self.is_cut_face.extend_from_slice(&other.is_cut_face);
    }

    /// Build a flat triangle-only representation for fast spatial/winding queries (parallel on large meshes).
    pub fn extract_triangles(&self) -> Vec<TriRef> {
        let np = self.num_polys();
        if np < 4096 {
            let mut tris = Vec::with_capacity(np * 2);
            for poly_idx in 0..np {
                for (vs, ls) in self.triangulate_poly(poly_idx) {
                    tris.push(TriRef {
                        poly_idx: poly_idx as u32,
                        v: vs,
                        l: [ls[0] as u32, ls[1] as u32, ls[2] as u32],
                    });
                }
            }
            return tris;
        }
        let chunk_sz = 4096;
        let parts: Vec<Vec<TriRef>> = (0..np)
            .into_par_iter()
            .step_by(chunk_sz)
            .map(|start| {
                let end = (start + chunk_sz).min(np);
                let mut local = Vec::with_capacity((end - start) * 2);
                for poly_idx in start..end {
                    for (vs, ls) in self.triangulate_poly(poly_idx) {
                        local.push(TriRef {
                            poly_idx: poly_idx as u32,
                            v: vs,
                            l: [ls[0] as u32, ls[1] as u32, ls[2] as u32],
                        });
                    }
                }
                local
            })
            .collect();
        let total: usize = parts.iter().map(|v| v.len()).sum();
        let mut tris = Vec::with_capacity(total);
        for p in parts {
            tris.extend(p);
        }
        tris
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriRef {
    pub poly_idx: u32,
    pub v: [u32; 3],
    pub l: [u32; 3],
}

/// Precomputed triangle + Dipole BVH + per-polygon AABB acceleration bundle cached across frames.
#[derive(Clone, Debug)]
pub struct CachedMeshGeo {
    pub mesh: HybridMesh,
    pub aabb: Aabb,
    pub poly_aabbs: Vec<Aabb>,
    pub tris: Vec<TriRef>,
    pub tri_aabbs: Vec<Aabb>,
    pub bvh: Bvh,
}

impl CachedMeshGeo {
    pub fn from_mesh(mesh: HybridMesh) -> Self {
        let np = mesh.num_polys();
        let poly_aabbs: Vec<Aabb> = if np >= 4096 {
            (0..np)
                .into_par_iter()
                .map(|pi| mesh.poly_aabb(pi))
                .collect()
        } else {
            (0..np).map(|pi| mesh.poly_aabb(pi)).collect()
        };

        let aabb = if np >= 4096 {
            poly_aabbs
                .par_iter()
                .copied()
                .reduce(Aabb::empty, |mut acc, b| {
                    acc.expand_aabb(&b);
                    acc
                })
        } else {
            let mut b = Aabb::empty();
            for pa in &poly_aabbs {
                b.expand_aabb(pa);
            }
            b
        };

        let tris = mesh.extract_triangles();
        let tri_aabbs: Vec<Aabb> = if tris.len() >= 4096 {
            tris.par_iter()
                .map(|t| {
                    let mut b = Aabb::from_point(mesh.positions[t.v[0] as usize]);
                    b.expand_point(mesh.positions[t.v[1] as usize]);
                    b.expand_point(mesh.positions[t.v[2] as usize]);
                    b
                })
                .collect()
        } else {
            tris.iter()
                .map(|t| {
                    let mut b = Aabb::from_point(mesh.positions[t.v[0] as usize]);
                    b.expand_point(mesh.positions[t.v[1] as usize]);
                    b.expand_point(mesh.positions[t.v[2] as usize]);
                    b
                })
                .collect()
        };
        let bvh = Bvh::build_with_dipoles(&tri_aabbs, &mesh.positions, &tris);
        Self {
            mesh,
            aabb,
            poly_aabbs,
            tris,
            tri_aabbs,
            bvh,
        }
    }

    /// Lightweight `CachedMeshGeo` using AABB-only BVH (skips winding dipole computation).
    /// Used for local spatial islands where the full base mesh provides the global winding BVH.
    pub fn from_mesh_aabb_only(mesh: HybridMesh) -> Self {
        let np = mesh.num_polys();
        let poly_aabbs: Vec<Aabb> = (0..np).map(|pi| mesh.poly_aabb(pi)).collect();
        let mut aabb = Aabb::empty();
        for pa in &poly_aabbs {
            aabb.expand_aabb(pa);
        }
        let tris = mesh.extract_triangles();
        let tri_aabbs: Vec<Aabb> = tris
            .iter()
            .map(|t| {
                let mut b = Aabb::from_point(mesh.positions[t.v[0] as usize]);
                b.expand_point(mesh.positions[t.v[1] as usize]);
                b.expand_point(mesh.positions[t.v[2] as usize]);
                b
            })
            .collect();
        let bvh = Bvh::build_aabbs(&tri_aabbs);
        Self {
            mesh,
            aabb,
            poly_aabbs,
            tris,
            tri_aabbs,
            bvh,
        }
    }
}

#[inline]
pub fn dominant_2d_axes(normal: [f64; 3]) -> (usize, usize, bool) {
    let ax = normal[0].abs();
    let ay = normal[1].abs();
    let az = normal[2].abs();
    if ax >= ay && ax >= az {
        (1, 2, normal[0] < 0.0)
    } else if ay >= az {
        (2, 0, normal[1] < 0.0)
    } else {
        (0, 1, normal[2] < 0.0)
    }
}

#[inline]
fn point_in_tri_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2], sign: f64) -> bool {
    let c1 = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) * sign;
    let c2 = ((c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])) * sign;
    let c3 = ((a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])) * sign;
    c1 >= -1e-12 && c2 >= -1e-12 && c3 >= -1e-12
}
