//! Parallel world-space multi-material triangle packer for direct Blender GPU `POST_VIEW` preview.
//!
//! Groups triangles by `mat_id` and packs flat contiguous `f32` vertex buffers
//! (`pos_xyz`, `nor_xyz`, `uv_xy`) so Blender can render the building's REAL textures
//! and cut-face material directly on the GPU at 60-144+ FPS with zero switching!

use crate::mesh::HybridMesh;
use glam::{DMat4, DVec3};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct GpuSubMesh {
    pub mat_id: i32,
    pub num_tris: u32,
    /// Length = `num_tris * 9` (`[x, y, z]` for each of 3 triangle corners).
    pub pos_xyz: Vec<f32>,
    /// Length = `num_tris * 9` (flat unit world-space normal repeated for 3 corners).
    pub nor_xyz: Vec<f32>,
    /// Length = `num_tris * 6` (`[u, v]` for each of 3 triangle corners from UV layer 0).
    pub uv_xy: Vec<f32>,
    /// Length = `num_tris * 6` (`[u, v]` for each of 3 triangle corners from UV layer 1, or UV0 fallback).
    pub uv1_xy: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct GpuPackedMesh {
    pub submeshes: Vec<GpuSubMesh>,
}

impl GpuPackedMesh {
    pub fn total_tris(&self) -> u32 {
        self.submeshes.iter().map(|s| s.num_tris).sum()
    }

    /// Pack a `HybridMesh` into material-grouped world-space GPU triangle buffers in parallel.
    /// Uses thread-local material buckets (`fold`/`reduce`) to avoid per-polygon heap allocations.
    pub fn from_hybrid_mesh(mesh: &HybridMesh, world_mat: &DMat4) -> Self {
        let num_polys = mesh.num_polys();
        if num_polys == 0 {
            return Self::default();
        }
        Self::pack_poly_slice(mesh, None, world_mat)
    }

    /// Pack a subset of polygons (e.g. untouched polygons of a single spatial chunk) directly
    /// into world-space GPU triangle buffers without allocating an intermediate `HybridMesh`.
    pub fn from_hybrid_mesh_subset(
        mesh: &HybridMesh,
        poly_indices: &[u32],
        world_mat: &DMat4,
    ) -> Self {
        if poly_indices.is_empty() || mesh.is_empty() {
            return Self::default();
        }
        Self::pack_poly_slice(mesh, Some(poly_indices), world_mat)
    }

    fn pack_poly_slice(
        mesh: &HybridMesh,
        subset: Option<&[u32]>,
        world_mat: &DMat4,
    ) -> Self {
        let count = subset.map_or_else(|| mesh.num_polys(), |s| s.len());
        if count == 0 {
            return Self::default();
        }

        let uv0: Option<&[[f32; 2]]> = mesh.uv_layers.first().map(|v| v.as_slice());
        let uv1: Option<&[[f32; 2]]> = mesh.uv_layers.get(1).map(|v| v.as_slice()).or(uv0);
        let loop_nrms: Option<&[[f32; 3]]> = if mesh.loop_normals.len() == mesh.loop_verts.len() {
            Some(mesh.loop_normals.as_slice())
        } else {
            None
        };
        let normal_mat = world_mat.inverse().transpose();

        let pack_one = |buckets: &mut HashMap<i32, GpuSubMesh>, poly_idx: usize| {
            let mat_id = mesh.mat_ids[poly_idx];
            let tris = mesh.triangulate_poly(poly_idx);
            if tris.is_empty() {
                return;
            }
            let entry = buckets.entry(mat_id).or_insert_with(|| GpuSubMesh {
                mat_id,
                num_tris: 0,
                pos_xyz: Vec::new(),
                nor_xyz: Vec::new(),
                uv_xy: Vec::new(),
                uv1_xy: Vec::new(),
            });
            for (vs, ls) in tris {
                let p0 = world_mat.transform_point3(DVec3::from_array(mesh.positions[vs[0] as usize]));
                let p1 = world_mat.transform_point3(DVec3::from_array(mesh.positions[vs[1] as usize]));
                let p2 = world_mat.transform_point3(DVec3::from_array(mesh.positions[vs[2] as usize]));
                entry.pos_xyz.extend_from_slice(&[
                    p0.x as f32, p0.y as f32, p0.z as f32,
                    p1.x as f32, p1.y as f32, p1.z as f32,
                    p2.x as f32, p2.y as f32, p2.z as f32,
                ]);
                if let Some(nrms) = loop_nrms {
                    let wn0 = normal_mat
                        .transform_vector3(DVec3::new(
                            nrms[ls[0]][0] as f64,
                            nrms[ls[0]][1] as f64,
                            nrms[ls[0]][2] as f64,
                        ))
                        .normalize_or_zero();
                    let wn1 = normal_mat
                        .transform_vector3(DVec3::new(
                            nrms[ls[1]][0] as f64,
                            nrms[ls[1]][1] as f64,
                            nrms[ls[1]][2] as f64,
                        ))
                        .normalize_or_zero();
                    let wn2 = normal_mat
                        .transform_vector3(DVec3::new(
                            nrms[ls[2]][0] as f64,
                            nrms[ls[2]][1] as f64,
                            nrms[ls[2]][2] as f64,
                        ))
                        .normalize_or_zero();
                    entry.nor_xyz.extend_from_slice(&[
                        wn0.x as f32, wn0.y as f32, wn0.z as f32,
                        wn1.x as f32, wn1.y as f32, wn1.z as f32,
                        wn2.x as f32, wn2.y as f32, wn2.z as f32,
                    ]);
                } else {
                    let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                    let nx = n.x as f32;
                    let ny = n.y as f32;
                    let nz = n.z as f32;
                    entry.nor_xyz.extend_from_slice(&[nx, ny, nz, nx, ny, nz, nx, ny, nz]);
                }
                if let Some(uvs) = uv0 {
                    let u0 = uvs[ls[0]];
                    let u1 = uvs[ls[1]];
                    let u2 = uvs[ls[2]];
                    entry
                        .uv_xy
                        .extend_from_slice(&[u0[0], u0[1], u1[0], u1[1], u2[0], u2[1]]);
                } else {
                    entry.uv_xy.extend_from_slice(&[0.0f32; 6]);
                }
                if let Some(uvs) = uv1 {
                    let u0 = uvs[ls[0]];
                    let u1 = uvs[ls[1]];
                    let u2 = uvs[ls[2]];
                    entry
                        .uv1_xy
                        .extend_from_slice(&[u0[0], u0[1], u1[0], u1[1], u2[0], u2[1]]);
                } else {
                    entry.uv1_xy.extend_from_slice(&[0.0f32; 6]);
                }
                entry.num_tris += 1;
            }
        };

        let merge_buckets = |mut a: HashMap<i32, GpuSubMesh>, b: HashMap<i32, GpuSubMesh>| {
            for (mat_id, sub) in b {
                if sub.num_tris == 0 {
                    continue;
                }
                match a.get_mut(&mat_id) {
                    Some(dst) => {
                        dst.pos_xyz.extend_from_slice(&sub.pos_xyz);
                        dst.nor_xyz.extend_from_slice(&sub.nor_xyz);
                        dst.uv_xy.extend_from_slice(&sub.uv_xy);
                        dst.uv1_xy.extend_from_slice(&sub.uv1_xy);
                        dst.num_tris += sub.num_tris;
                    }
                    None => {
                        a.insert(mat_id, sub);
                    }
                }
            }
            a
        };

        let buckets: HashMap<i32, GpuSubMesh> = if count < 1024 {
            let mut local = HashMap::new();
            match subset {
                Some(idxs) => {
                    for &pi in idxs {
                        pack_one(&mut local, pi as usize);
                    }
                }
                None => {
                    for pi in 0..count {
                        pack_one(&mut local, pi);
                    }
                }
            }
            local
        } else {
            match subset {
                Some(idxs) => idxs
                    .par_iter()
                    .fold(HashMap::new, |mut acc, &pi| {
                        pack_one(&mut acc, pi as usize);
                        acc
                    })
                    .reduce(HashMap::new, merge_buckets),
                None => (0..count)
                    .into_par_iter()
                    .fold(HashMap::new, |mut acc, pi| {
                        pack_one(&mut acc, pi);
                        acc
                    })
                    .reduce(HashMap::new, merge_buckets),
            }
        };

        let mut submeshes: Vec<GpuSubMesh> = buckets.into_values().collect();
        submeshes.sort_by_key(|s| s.mat_id);

        Self { submeshes }
    }
}
