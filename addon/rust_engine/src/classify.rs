//! Inside/Outside classification using Generalized Winding Number (Jacobson et al.)
//! with exact coplanar surface tie-breaking.
//!
//! Generalized Winding Number returns 1.0 inside and 0.0 outside for closed meshes,
//! and degrades smoothly on open-bottom / thin-walled / overlapping game assets
//! without ray-casting edge hits or missing faces.

use crate::bvh::Bvh;
use crate::mesh::{Aabb, TriRef};
use glam::DVec3;
use std::f64::consts::PI;

const SURFACE_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Outside,
    Inside,
    /// Point lies directly on a surface triangle of the reference mesh with matching normal orientation.
    CoplanarSame,
    /// Point lies directly on a surface triangle of the reference mesh with opposite normal orientation.
    CoplanarOpposite,
}

/// Van Oosterom & Strackee exact solid angle of triangle `(a, b, c)` seen from `p`.
#[inline]
fn triangle_solid_angle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let va = a - p;
    let vb = b - p;
    let vc = c - p;
    let la = va.length();
    let lb = vb.length();
    let lc = vc.length();
    if la < 1e-12 || lb < 1e-12 || lc < 1e-12 {
        return 0.0;
    }
    let num = va.dot(vb.cross(vc));
    let den = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;
    2.0 * num.atan2(den)
}

/// Compute the generalized winding number of `p` with respect to a triangle mesh.
pub fn winding_number(p: [f64; 3], positions: &[[f64; 3]], tris: &[TriRef]) -> f64 {
    let pv = DVec3::from_array(p);
    let mut total_omega = 0.0;
    for t in tris {
        let a = DVec3::from_array(positions[t.v[0] as usize]);
        let b = DVec3::from_array(positions[t.v[1] as usize]);
        let c = DVec3::from_array(positions[t.v[2] as usize]);
        total_omega += triangle_solid_angle(pv, a, b, c);
    }
    total_omega / (4.0 * PI)
}

/// Check if point `p` lies on any triangle of the reference mesh (via BVH query).
/// If so, returns `Some(dot(query_normal, ref_normal))`.
fn check_coplanar_surface(
    p: [f64; 3],
    query_normal: [f64; 3],
    positions: &[[f64; 3]],
    tris: &[TriRef],
    bvh: &Bvh,
) -> Option<f64> {
    let query_box = Aabb::from_point(p).expanded(SURFACE_EPS * 10.0);
    let mut candidates = Vec::new();
    bvh.query_aabb(&query_box, &mut candidates);
    if candidates.is_empty() {
        return None;
    }

    let pv = DVec3::from_array(p);
    let qn = DVec3::from_array(query_normal).normalize_or_zero();

    for &ti in &candidates {
        let t = tris[ti as usize];
        let a = DVec3::from_array(positions[t.v[0] as usize]);
        let b = DVec3::from_array(positions[t.v[1] as usize]);
        let c = DVec3::from_array(positions[t.v[2] as usize]);

        let ab = b - a;
        let ac = c - a;
        let n_raw = ab.cross(ac);
        let n_len = n_raw.length();
        if n_len < 1e-14 {
            continue;
        }
        let n = n_raw / n_len;

        // Plane distance check
        let dist = (pv - a).dot(n).abs();
        if dist > SURFACE_EPS {
            continue;
        }

        // Barycentric containment check
        let ap = pv - a;
        let d00 = ab.dot(ab);
        let d01 = ab.dot(ac);
        let d11 = ac.dot(ac);
        let d20 = ap.dot(ab);
        let d21 = ap.dot(ac);
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-16 {
            continue;
        }
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        if u >= -1e-5 && v >= -1e-5 && w >= -1e-5 {
            let dot = qn.dot(n);
            if dot.abs() > 0.99 {
                return Some(dot);
            }
        }
    }

    None
}

/// Classify a face centroid `p` (with unit normal `query_normal`) against a reference mesh.
pub fn classify_point(
    p: [f64; 3],
    query_normal: [f64; 3],
    ref_aabb: &Aabb,
    ref_positions: &[[f64; 3]],
    ref_tris: &[TriRef],
    ref_bvh: &Bvh,
) -> Location {
    if !ref_aabb.expanded(SURFACE_EPS * 10.0).contains_point(p) {
        return Location::Outside;
    }

    if let Some(dot) = check_coplanar_surface(p, query_normal, ref_positions, ref_tris, ref_bvh) {
        return if dot >= 0.0 {
            Location::CoplanarSame
        } else {
            Location::CoplanarOpposite
        };
    }

    let wn = ref_bvh.fast_winding_number(p, ref_positions, ref_tris);
    if wn > 0.5 {
        Location::Inside
    } else {
        Location::Outside
    }
}
