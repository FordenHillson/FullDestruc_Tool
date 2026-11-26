//! Fast flat AABB Bounding Volume Hierarchy (BVH) with parallel construction (`rayon::join`)
//! and precomputed area-weighted dipoles for $O(\log N)$ Fast Winding Number.

use crate::mesh::{Aabb, TriRef};
use glam::DVec3;
use std::f64::consts::PI;

const LEAF_MAX: usize = 8;
const PARALLEL_THRESHOLD: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct BvhNode {
    pub aabb: Aabb,
    /// If `count > 0`: leaf starting at `left_or_first` in `indices`.
    /// If `count == 0`: internal node with left child `left_or_first` and right child `right_child`.
    pub left_or_first: u32,
    pub right_child: u32,
    pub count: u32,
    /// Precomputed Fast Winding Number dipole: sum of area-weighted outward normals
    /// `0.5 * (b - a) x (c - a)` for all triangles in this subtree.
    pub area_normal: [f64; 3],
    /// Area-weighted centroid of all triangles in this subtree.
    pub weighted_center: [f64; 3],
    /// Bounding radius around `weighted_center`.
    pub radius: f64,
}

#[derive(Clone, Debug)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct TriDipole {
    area_normal: DVec3,
    center: DVec3,
    area: f64,
}

impl Bvh {
    /// Build a BVH with Fast Winding Number dipole moments using parallel `rayon::join`.
    pub fn build_with_dipoles(
        aabbs: &[Aabb],
        positions: &[[f64; 3]],
        tris: &[TriRef],
    ) -> Self {
        let n = aabbs.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                indices: Vec::new(),
            };
        }

        let dipoles: Vec<TriDipole> = tris
            .iter()
            .map(|t| {
                let a = DVec3::from_array(positions[t.v[0] as usize]);
                let b = DVec3::from_array(positions[t.v[1] as usize]);
                let c = DVec3::from_array(positions[t.v[2] as usize]);
                let an = 0.5 * (b - a).cross(c - a);
                let area = an.length();
                let center = (a + b + c) * (1.0 / 3.0);
                TriDipole {
                    area_normal: an,
                    center,
                    area,
                }
            })
            .collect();

        let mut indices: Vec<u32> = (0..n as u32).collect();
        let centers: Vec<[f64; 3]> = aabbs.iter().map(|b| b.center()).collect();

        let nodes = Self::build_subtree_parallel(&mut indices, 0, aabbs, &centers, &dipoles);
        Self { nodes, indices }
    }

    fn compute_node_stats(
        slice: &[u32],
        aabbs: &[Aabb],
        centers: &[[f64; 3]],
        dipoles: &[TriDipole],
    ) -> (Aabb, Aabb, [f64; 3], [f64; 3], f64) {
        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        let mut sum_an = DVec3::ZERO;
        let mut sum_wc = DVec3::ZERO;
        let mut sum_area = 0.0;

        for &idx in slice {
            let ui = idx as usize;
            bounds.expand_aabb(&aabbs[ui]);
            centroid_bounds.expand_point(centers[ui]);
            let d = dipoles[ui];
            sum_an += d.area_normal;
            sum_wc += d.center * d.area;
            sum_area += d.area;
        }

        let wc = if sum_area > 1e-18 {
            sum_wc / sum_area
        } else {
            DVec3::from_array(bounds.center())
        };

        let dx = (wc.x - bounds.min[0]).abs().max((bounds.max[0] - wc.x).abs());
        let dy = (wc.y - bounds.min[1]).abs().max((bounds.max[1] - wc.y).abs());
        let dz = (wc.z - bounds.min[2]).abs().max((bounds.max[2] - wc.z).abs());
        let radius = (dx * dx + dy * dy + dz * dz).sqrt();

        (bounds, centroid_bounds, sum_an.to_array(), wc.to_array(), radius)
    }

    fn partition_slice(slice: &mut [u32], centroid_bounds: &Aabb, centers: &[[f64; 3]]) -> usize {
        let count = slice.len();
        let axis = centroid_bounds.longest_axis();
        let split_pos = 0.5 * (centroid_bounds.min[axis] + centroid_bounds.max[axis]);

        let mut i = 0;
        let mut j = count;
        while i < j {
            if centers[slice[i] as usize][axis] < split_pos {
                i += 1;
            } else {
                j -= 1;
                slice.swap(i, j);
            }
        }
        if i == 0 || i == count {
            count / 2
        } else {
            i
        }
    }

    fn build_subtree_parallel(
        slice: &mut [u32],
        global_start: usize,
        aabbs: &[Aabb],
        centers: &[[f64; 3]],
        dipoles: &[TriDipole],
    ) -> Vec<BvhNode> {
        let count = slice.len();
        if count <= PARALLEL_THRESHOLD {
            let mut nodes = Vec::with_capacity(count * 2);
            Self::build_serial_into(&mut nodes, slice, global_start, aabbs, centers, dipoles);
            return nodes;
        }

        let (bounds, centroid_bounds, area_normal, weighted_center, radius) =
            Self::compute_node_stats(slice, aabbs, centers, dipoles);

        let left_count = Self::partition_slice(slice, &centroid_bounds, centers);
        let (left_slice, right_slice) = slice.split_at_mut(left_count);

        let (mut left_nodes, mut right_nodes) = rayon::join(
            || Self::build_subtree_parallel(left_slice, global_start, aabbs, centers, dipoles),
            || {
                Self::build_subtree_parallel(
                    right_slice,
                    global_start + left_count,
                    aabbs,
                    centers,
                    dipoles,
                )
            },
        );

        let left_offset = 1u32;
        let right_offset = 1u32 + left_nodes.len() as u32;

        for node in &mut left_nodes {
            if node.count == 0 {
                node.left_or_first += left_offset;
                node.right_child += left_offset;
            }
        }
        for node in &mut right_nodes {
            if node.count == 0 {
                node.left_or_first += right_offset;
                node.right_child += right_offset;
            }
        }

        let mut out = Vec::with_capacity(1 + left_nodes.len() + right_nodes.len());
        out.push(BvhNode {
            aabb: bounds,
            left_or_first: left_offset,
            right_child: right_offset,
            count: 0,
            area_normal,
            weighted_center,
            radius,
        });
        out.extend_from_slice(&left_nodes);
        out.extend_from_slice(&right_nodes);
        out
    }

    fn build_serial_into(
        nodes: &mut Vec<BvhNode>,
        slice: &mut [u32],
        global_start: usize,
        aabbs: &[Aabb],
        centers: &[[f64; 3]],
        dipoles: &[TriDipole],
    ) -> u32 {
        let count = slice.len();
        let (bounds, centroid_bounds, area_normal, weighted_center, radius) =
            Self::compute_node_stats(slice, aabbs, centers, dipoles);

        let my_idx = nodes.len() as u32;
        if count <= LEAF_MAX {
            nodes.push(BvhNode {
                aabb: bounds,
                left_or_first: global_start as u32,
                right_child: 0,
                count: count as u32,
                area_normal,
                weighted_center,
                radius,
            });
            return my_idx;
        }

        nodes.push(BvhNode {
            aabb: bounds,
            left_or_first: 0,
            right_child: 0,
            count: 0,
            area_normal,
            weighted_center,
            radius,
        });

        let left_count = Self::partition_slice(slice, &centroid_bounds, centers);
        let (left_slice, right_slice) = slice.split_at_mut(left_count);

        let left_child =
            Self::build_serial_into(nodes, left_slice, global_start, aabbs, centers, dipoles);
        let right_child = Self::build_serial_into(
            nodes,
            right_slice,
            global_start + left_count,
            aabbs,
            centers,
            dipoles,
        );

        nodes[my_idx as usize].left_or_first = left_child;
        nodes[my_idx as usize].right_child = right_child;
        my_idx
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn root_aabb(&self) -> Aabb {
        if self.nodes.is_empty() {
            Aabb::empty()
        } else {
            self.nodes[0].aabb
        }
    }

    /// Find all primitive indices whose AABB overlaps `query`.
    pub fn query_aabb(&self, query: &Aabb, out: &mut Vec<u32>) {
        if self.nodes.is_empty() || !self.nodes[0].aabb.overlaps(query) {
            return;
        }
        let mut stack = [0usize; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp]];
            if node.count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.count as usize;
                out.extend_from_slice(&self.indices[first..end]);
            } else {
                let left = node.left_or_first as usize;
                let right = node.right_child as usize;
                if self.nodes[left].aabb.overlaps(query) && sp < 64 {
                    stack[sp] = left;
                    sp += 1;
                }
                if self.nodes[right].aabb.overlaps(query) && sp < 64 {
                    stack[sp] = right;
                    sp += 1;
                }
            }
        }
    }

    /// $O(\log N)$ Fast Winding Number evaluation using Barnes-Hut dipole approximation
    /// (Barill et al., SIGGRAPH 2018). Exact solid angles are used for nearby leaves.
    pub fn fast_winding_number(
        &self,
        p: [f64; 3],
        positions: &[[f64; 3]],
        tris: &[TriRef],
    ) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        const BETA: f64 = 2.0;
        let pv = DVec3::from_array(p);
        let mut total_omega = 0.0;

        let mut stack = [0usize; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp]];
            let wc = DVec3::from_array(node.weighted_center);
            let diff = wc - pv;
            let dist_sq = diff.length_squared();
            let threshold = BETA * node.radius;

            if node.count == 0 && dist_sq > threshold * threshold && dist_sq > 1e-18 {
                let dist = dist_sq.sqrt();
                let an = DVec3::from_array(node.area_normal);
                total_omega += an.dot(diff) / (dist_sq * dist);
                continue;
            }

            if node.count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.count as usize;
                for &ti in &self.indices[first..end] {
                    let t = tris[ti as usize];
                    let a = DVec3::from_array(positions[t.v[0] as usize]);
                    let b = DVec3::from_array(positions[t.v[1] as usize]);
                    let c = DVec3::from_array(positions[t.v[2] as usize]);
                    total_omega += triangle_solid_angle(pv, a, b, c);
                }
            } else {
                let left = node.left_or_first as usize;
                let right = node.right_child as usize;
                if sp + 2 <= 64 {
                    stack[sp] = left;
                    sp += 1;
                    stack[sp] = right;
                    sp += 1;
                }
            }
        }

        total_omega / (4.0 * PI)
    }

    /// Dual-tree traversal finding all overlapping `(idx_a, idx_b)` primitive pairs.
    pub fn query_bvh_pairs(&self, other: &Bvh, eps: f64, pairs: &mut Vec<(u32, u32)>) {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return;
        }
        let mut stack = Vec::with_capacity(128);
        stack.push((0usize, 0usize));

        while let Some((na, nb)) = stack.pop() {
            let node_a = &self.nodes[na];
            let node_b = &other.nodes[nb];
            if !node_a.aabb.expanded(eps).overlaps(&node_b.aabb) {
                continue;
            }
            let a_leaf = node_a.count > 0;
            let b_leaf = node_b.count > 0;
            if a_leaf && b_leaf {
                let fa = node_a.left_or_first as usize;
                let fb = node_b.left_or_first as usize;
                for &ia in &self.indices[fa..fa + node_a.count as usize] {
                    for &ib in &other.indices[fb..fb + node_b.count as usize] {
                        pairs.push((ia, ib));
                    }
                }
            } else if a_leaf {
                let lb = node_b.left_or_first as usize;
                let rb = node_b.right_child as usize;
                stack.push((na, lb));
                stack.push((na, rb));
            } else if b_leaf {
                let la = node_a.left_or_first as usize;
                let ra = node_a.right_child as usize;
                stack.push((la, nb));
                stack.push((ra, nb));
            } else {
                let la = node_a.left_or_first as usize;
                let ra = node_a.right_child as usize;
                let lb = node_b.left_or_first as usize;
                let rb = node_b.right_child as usize;
                stack.push((la, lb));
                stack.push((la, rb));
                stack.push((ra, lb));
                stack.push((ra, rb));
            }
        }
    }
}

/// Van Oosterom & Strackee exact solid angle of triangle `(a, b, c)` seen from `p`.
#[inline]
pub fn triangle_solid_angle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
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
