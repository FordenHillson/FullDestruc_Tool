//! Exact geometric predicates and triangle-triangle intersection using Shewchuk's
//! adaptive-precision arithmetic (`robust` crate).

use crate::mesh::dominant_2d_axes;
use glam::DVec3;
use robust::{orient3d, Coord3D};
use smallvec::SmallVec;

const COPLANAR_DIST_EPS: f64 = 1e-7;
const SEG_LEN_EPS: f64 = 1e-9;

#[inline]
fn to_coord3d(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

#[derive(Clone, Debug)]
pub enum TriTriHit {
    None,
    /// Single 3D intersection segment shared by both triangles.
    Segment([[f64; 3]; 2]),
    /// Coplanar overlap: segments of B clipped to A (to split A) and segments of A clipped to B (to split B).
    Coplanar {
        segs_on_a: SmallVec<[[[f64; 3]; 2]; 3]>,
        segs_on_b: SmallVec<[[[f64; 3]; 2]; 3]>,
    },
}

#[inline]
fn tri_normal_raw(p0: DVec3, p1: DVec3, p2: DVec3) -> DVec3 {
    (p1 - p0).cross(p2 - p0)
}

/// Compute signed distances of `pts` to the plane of triangle `(q0, q1, q2)` using
/// `robust::orient3d` for exact sign determination, normalized by twice triangle area.
#[inline]
fn signed_dists_to_plane(pts: [[f64; 3]; 3], q0: [f64; 3], q1: [f64; 3], q2: [f64; 3], area2: f64) -> [f64; 3] {
    let c0 = to_coord3d(q0);
    let c1 = to_coord3d(q1);
    let c2 = to_coord3d(q2);
    let inv = if area2 > 1e-18 { 1.0 / area2 } else { 0.0 };
    // robust::orient3d(a, b, c, d) returns -det([a-d, b-d, c-d]), positive when d is below (a,b,c).
    // Negating gives standard signed volume / area2 = signed perpendicular distance.
    [
        -orient3d(c0, c1, c2, to_coord3d(pts[0])) * inv,
        -orient3d(c0, c1, c2, to_coord3d(pts[1])) * inv,
        -orient3d(c0, c1, c2, to_coord3d(pts[2])) * inv,
    ]
}

/// Extract the 2 endpoints where triangle `pts` crosses the plane (given signed distances `d`).
fn plane_cross_interval(pts: [[f64; 3]; 3], d: [f64; 3]) -> Option<[[f64; 3]; 2]> {
    let mut hits: SmallVec<[DVec3; 4]> = SmallVec::new();

    for i in 0..3 {
        let j = (i + 1) % 3;
        let di = d[i];
        let dj = d[j];
        let pi = DVec3::from_array(pts[i]);
        let pj = DVec3::from_array(pts[j]);

        if di.abs() <= COPLANAR_DIST_EPS {
            hits.push(pi);
        } else if (di > 0.0 && dj < -COPLANAR_DIST_EPS) || (di < 0.0 && dj > COPLANAR_DIST_EPS) {
            let t = di / (di - dj);
            hits.push(pi + (pj - pi) * t);
        }
    }

    if hits.len() < 2 {
        return None;
    }

    // Pick the two farthest points in `hits`
    let mut best_d2 = -1.0;
    let mut best_pair = [hits[0], hits[1]];
    for i in 0..hits.len() {
        for j in (i + 1)..hits.len() {
            let d2 = hits[i].distance_squared(hits[j]);
            if d2 > best_d2 {
                best_d2 = d2;
                best_pair = [hits[i], hits[j]];
            }
        }
    }

    if best_d2 <= SEG_LEN_EPS * SEG_LEN_EPS {
        return None;
    }
    Some([best_pair[0].to_array(), best_pair[1].to_array()])
}

/// Clip a 2D segment `(p0, p1)` against a 2D triangle `(t0, t1, t2)` (returns parametric `[tmin, tmax]`).
fn clip_segment_to_tri_2d(
    p0: [f64; 2],
    p1: [f64; 2],
    tri: [[f64; 2]; 3],
    ccw_sign: f64,
) -> Option<(f64, f64)> {
    let mut t_min = 0.0f64;
    let mut t_max = 1.0f64;
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];

    for i in 0..3 {
        let a = tri[i];
        let b = tri[(i + 1) % 3];
        // Inward edge normal
        let nx = -(b[1] - a[1]) * ccw_sign;
        let ny = (b[0] - a[0]) * ccw_sign;
        let num = nx * (p0[0] - a[0]) + ny * (p0[1] - a[1]);
        let den = nx * dx + ny * dy;

        if den.abs() < 1e-14 {
            if num < -1e-9 {
                return None;
            }
        } else {
            let t = -num / den;
            if den > 0.0 {
                if t > t_min {
                    t_min = t;
                }
            } else if t < t_max {
                t_max = t;
            }
            if t_min > t_max + 1e-9 {
                return None;
            }
        }
    }

    if t_max - t_min <= 1e-8 {
        return None;
    }
    Some((t_min.clamp(0.0, 1.0), t_max.clamp(0.0, 1.0)))
}

fn coplanar_clip_edges(
    src: [[f64; 3]; 3],
    target: [[f64; 3]; 3],
    normal: [f64; 3],
) -> SmallVec<[[[f64; 3]; 2]; 3]> {
    let (ax0, ax1, flip) = dominant_2d_axes(normal);
    let proj = |p: [f64; 3]| -> [f64; 2] {
        if flip {
            [p[ax1], p[ax0]]
        } else {
            [p[ax0], p[ax1]]
        }
    };

    let t2 = [proj(target[0]), proj(target[1]), proj(target[2])];
    let area2 = (t2[1][0] - t2[0][0]) * (t2[2][1] - t2[0][1])
        - (t2[1][1] - t2[0][1]) * (t2[2][0] - t2[0][0]);
    if area2.abs() < 1e-14 {
        return SmallVec::new();
    }
    let ccw_sign = if area2 >= 0.0 { 1.0 } else { -1.0 };

    let mut out = SmallVec::new();
    for i in 0..3 {
        let j = (i + 1) % 3;
        let s0 = proj(src[i]);
        let s1 = proj(src[j]);
        if let Some((t0, t1)) = clip_segment_to_tri_2d(s0, s1, t2, ccw_sign) {
            let p0 = DVec3::from_array(src[i]);
            let p1 = DVec3::from_array(src[j]);
            let c0 = p0 + (p1 - p0) * t0;
            let c1 = p0 + (p1 - p0) * t1;
            if c0.distance_squared(c1) > SEG_LEN_EPS * SEG_LEN_EPS {
                out.push([c0.to_array(), c1.to_array()]);
            }
        }
    }
    out
}

/// Exact triangle-triangle intersection test returning cut segments for both triangles.
pub fn tri_tri_intersect(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> TriTriHit {
    let na = tri_normal_raw(
        DVec3::from_array(a[0]),
        DVec3::from_array(a[1]),
        DVec3::from_array(a[2]),
    );
    let nb = tri_normal_raw(
        DVec3::from_array(b[0]),
        DVec3::from_array(b[1]),
        DVec3::from_array(b[2]),
    );
    let area2_a = na.length();
    let area2_b = nb.length();
    if area2_a < 1e-14 || area2_b < 1e-14 {
        return TriTriHit::None;
    }

    let da = signed_dists_to_plane(a, b[0], b[1], b[2], area2_b);
    if (da[0] > COPLANAR_DIST_EPS && da[1] > COPLANAR_DIST_EPS && da[2] > COPLANAR_DIST_EPS)
        || (da[0] < -COPLANAR_DIST_EPS && da[1] < -COPLANAR_DIST_EPS && da[2] < -COPLANAR_DIST_EPS)
    {
        return TriTriHit::None;
    }

    let db = signed_dists_to_plane(b, a[0], a[1], a[2], area2_a);
    if (db[0] > COPLANAR_DIST_EPS && db[1] > COPLANAR_DIST_EPS && db[2] > COPLANAR_DIST_EPS)
        || (db[0] < -COPLANAR_DIST_EPS && db[1] < -COPLANAR_DIST_EPS && db[2] < -COPLANAR_DIST_EPS)
    {
        return TriTriHit::None;
    }

    // Check coplanar case
    let a_coplanar = da[0].abs() <= COPLANAR_DIST_EPS
        && da[1].abs() <= COPLANAR_DIST_EPS
        && da[2].abs() <= COPLANAR_DIST_EPS;
    let b_coplanar = db[0].abs() <= COPLANAR_DIST_EPS
        && db[1].abs() <= COPLANAR_DIST_EPS
        && db[2].abs() <= COPLANAR_DIST_EPS;

    if a_coplanar || b_coplanar {
        let n_unit = (na / area2_a).to_array();
        let segs_on_a = coplanar_clip_edges(b, a, n_unit);
        let segs_on_b = coplanar_clip_edges(a, b, n_unit);
        if segs_on_a.is_empty() && segs_on_b.is_empty() {
            return TriTriHit::None;
        }
        return TriTriHit::Coplanar {
            segs_on_a,
            segs_on_b,
        };
    }

    // Transverse intersection: find 1D overlap along intersection line L = na x nb
    let seg_a = match plane_cross_interval(a, da) {
        Some(s) => s,
        None => return TriTriHit::None,
    };
    let seg_b = match plane_cross_interval(b, db) {
        Some(s) => s,
        None => return TriTriHit::None,
    };

    let line_dir = na.cross(nb);
    let dir_len = line_dir.length();
    if dir_len < 1e-14 {
        return TriTriHit::None;
    }
    let dir = line_dir / dir_len;
    let origin = DVec3::from_array(seg_a[0]);

    let ta0 = (DVec3::from_array(seg_a[0]) - origin).dot(dir);
    let ta1 = (DVec3::from_array(seg_a[1]) - origin).dot(dir);
    let tb0 = (DVec3::from_array(seg_b[0]) - origin).dot(dir);
    let tb1 = (DVec3::from_array(seg_b[1]) - origin).dot(dir);

    let (a_min, a_max) = if ta0 <= ta1 { (ta0, ta1) } else { (ta1, ta0) };
    let (b_min, b_max) = if tb0 <= tb1 { (tb0, tb1) } else { (tb1, tb0) };

    let o_min = a_min.max(b_min);
    let o_max = a_max.min(b_max);
    if o_max - o_min <= SEG_LEN_EPS {
        return TriTriHit::None;
    }

    // Reconstruct endpoints strictly on A's intersection segment for clean alignment
    let seg_a_vec = DVec3::from_array(seg_a[1]) - DVec3::from_array(seg_a[0]);
    let seg_a_span = ta1 - ta0;
    let p_start = if seg_a_span.abs() > 1e-14 {
        let u = ((o_min - ta0) / seg_a_span).clamp(0.0, 1.0);
        origin + seg_a_vec * u
    } else {
        origin + dir * o_min
    };
    let p_end = if seg_a_span.abs() > 1e-14 {
        let u = ((o_max - ta0) / seg_a_span).clamp(0.0, 1.0);
        origin + seg_a_vec * u
    } else {
        origin + dir * o_max
    };

    if p_start.distance_squared(p_end) <= SEG_LEN_EPS * SEG_LEN_EPS {
        return TriTriHit::None;
    }

    TriTriHit::Segment([p_start.to_array(), p_end.to_array()])
}
