//! World-space Triplanar / Box UV projection for newly generated cut faces.
//!
//! Ensures cut faces have consistent, non-stretched texel density matching
//! `FullDestruc_Tool`'s `texel_per_meter / texture_size` settings, while never
//! touching the UVs of original mesh faces.

use crate::mesh::HybridMesh;
use glam::{DMat4, DVec3};

/// Apply world-space Box UV to all polygons marked `is_cut_face == true`.
///
/// `uv_scale` is typically `texel_per_meter / texture_size` (e.g. `512.0 / 1024.0 = 0.5`).
pub fn apply_cut_face_box_uv(mesh: &mut HybridMesh, object_world: &DMat4, uv_scale: f32) {
    apply_cut_face_box_uv_from(mesh, 0, object_world, uv_scale);
}

/// Apply world-space Box UV only to polygons in `start_poly..mesh.num_polys()` marked `is_cut_face == true`.
/// Avoids O(M) full-mesh scans on 1M-polygon meshes when only tail cut cap faces were added.
pub fn apply_cut_face_box_uv_from(
    mesh: &mut HybridMesh,
    start_poly: usize,
    object_world: &DMat4,
    uv_scale: f32,
) {
    if mesh.uv_layers.is_empty() || mesh.is_empty() {
        return;
    }
    let scale = uv_scale as f64;
    let num_polys = mesh.num_polys();
    if start_poly >= num_polys {
        return;
    }

    for poly_idx in start_poly..num_polys {
        if !mesh.is_cut_face[poly_idx] {
            continue;
        }
        let range = mesh.poly_loop_range(poly_idx);
        if range.len() < 3 {
            continue;
        }

        // Compute world-space normal of the cut face
        let v0 = object_world.transform_point3(DVec3::from_array(
            mesh.positions[mesh.loop_verts[range.start] as usize],
        ));
        let v1 = object_world.transform_point3(DVec3::from_array(
            mesh.positions[mesh.loop_verts[range.start + 1] as usize],
        ));
        let v2 = object_world.transform_point3(DVec3::from_array(
            mesh.positions[mesh.loop_verts[range.start + 2] as usize],
        ));
        let n = (v1 - v0).cross(v2 - v0);
        let ax = n.x.abs();
        let ay = n.y.abs();
        let az = n.z.abs();

        for li in range {
            let vi = mesh.loop_verts[li] as usize;
            let wp = object_world.transform_point3(DVec3::from_array(mesh.positions[vi]));
            let uv = if ax >= ay && ax >= az {
                [(wp.y * scale) as f32, (wp.z * scale) as f32]
            } else if ay >= az {
                [(wp.x * scale) as f32, (wp.z * scale) as f32]
            } else {
                [(wp.x * scale) as f32, (wp.y * scale) as f32]
            };
            for layer in &mut mesh.uv_layers {
                layer[li] = uv;
            }
        }
    }
}
