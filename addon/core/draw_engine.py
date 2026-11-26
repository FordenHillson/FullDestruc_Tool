"""Standalone 2D-to-3D Draw & Extrude Geometry Engine (Pure Python + NumPy, zero Blender imports).

Handles:
- Polyline cleanup & Ramer-Douglas-Peucker simplification (for smooth Freehand + sharp Polygon)
- Concave & convex 2D polygon triangulation (Ear-Clipping)
- Exact Perspective & Orthographic screen-to-world unprojection and world-to-screen projection
- Watertight, outward-normal 3D prism extrusion ready to feed into any 3D host
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CameraSnapshot:
    """Plain NumPy snapshot of a 3D viewport camera and target bounding depth."""

    width: float
    height: float
    view_matrix: np.ndarray  # (4, 4) float64
    perspective_matrix: np.ndarray  # (4, 4) float64
    inv_perspective_matrix: np.ndarray  # (4, 4) float64
    is_perspective: bool
    camera_pos: np.ndarray  # (3,) float64
    view_forward: np.ndarray  # (3,) float64 unit vector pointing INTO the screen
    target_center: np.ndarray  # (3,) float64
    target_depth_min: float  # min dot(corner, view_forward) across target bbox
    target_depth_max: float  # max dot(corner, view_forward) across target bbox

    @property
    def target_thickness(self) -> float:
        return max(float(self.target_depth_max - self.target_depth_min), 0.5)

    def through_building_range(self, margin: float = 0.5) -> tuple[float, float]:
        """Return (front_depth, extrude_length) that cleanly spans all the way through the target."""
        m = max(float(margin), 0.1)
        front = float(self.target_depth_min) - m
        back = float(self.target_depth_max) + m
        if self.is_perspective:
            cam_depth = float(np.dot(self.camera_pos, self.view_forward))
            min_safe_front = cam_depth + 0.15
            if front < min_safe_front:
                front = min_safe_front
            if back < front + 1.0:
                back = front + 1.0
        length = back - front
        return front, max(length, 1.0)


def _point_to_segment_dist_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab[0] * ab[0] + ab[1] * ab[1])
    if denom <= 1e-12:
        d = p - a
        return float(d[0] * d[0] + d[1] * d[1])
    t = float(((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    d = p - proj
    return float(d[0] * d[0] + d[1] * d[1])


def _rdp_indices(pts: np.ndarray, start: int, end: int, eps_sq: float) -> list[int]:
    if end <= start + 1:
        return [start, end]
    a = pts[start]
    b = pts[end]
    max_d = -1.0
    max_idx = -1
    for i in range(start + 1, end):
        d = _point_to_segment_dist_sq(pts[i], a, b)
        if d > max_d:
            max_d = d
            max_idx = i
    if max_d > eps_sq and max_idx > start:
        left = _rdp_indices(pts, start, max_idx, eps_sq)
        right = _rdp_indices(pts, max_idx, end, eps_sq)
        return left[:-1] + right
    return [start, end]


def simplify_polyline(
    points_2d: Sequence[tuple[float, float]] | np.ndarray,
    min_dist_px: float = 3.0,
    rdp_epsilon: float = 2.0,
    closed: bool = True,
) -> np.ndarray:
    """Remove near-duplicate vertices and apply Ramer-Douglas-Peucker simplification."""
    arr = np.asarray(points_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)

    # 1. Deduplicate consecutive points closer than min_dist_px
    min_d_sq = float(min_dist_px) ** 2
    dedup = [arr[0]]
    for i in range(1, len(arr)):
        diff = arr[i] - dedup[-1]
        if float(diff[0] * diff[0] + diff[1] * diff[1]) >= min_d_sq:
            dedup.append(arr[i])
    if closed and len(dedup) >= 2:
        diff = dedup[-1] - dedup[0]
        if float(diff[0] * diff[0] + diff[1] * diff[1]) < min_d_sq:
            dedup.pop()
    pts = np.asarray(dedup, dtype=np.float64)

    if len(pts) <= 3 or rdp_epsilon <= 0.0:
        return pts

    # 2. Apply RDP (split closed loop at farthest vertex from pts[0] so loop shape is preserved)
    eps_sq = float(rdp_epsilon) ** 2
    if closed:
        dists = np.sum((pts - pts[0]) ** 2, axis=1)
        far_idx = int(np.argmax(dists))
        if far_idx == 0 or far_idx == len(pts) - 1:
            keep = _rdp_indices(pts, 0, len(pts) - 1, eps_sq)
            return pts[keep]
        chain1 = pts[: far_idx + 1]
        chain2 = np.vstack([pts[far_idx:], pts[0:1]])
        k1 = _rdp_indices(chain1, 0, len(chain1) - 1, eps_sq)
        k2 = _rdp_indices(chain2, 0, len(chain2) - 1, eps_sq)
        res = np.vstack([chain1[k1], chain2[k2[1:-1]]])
        return res
    else:
        keep = _rdp_indices(pts, 0, len(pts) - 1, eps_sq)
        return pts[keep]


def polygon_signed_area_2d(pts: np.ndarray) -> float:
    """Shoelace signed area in 2D (positive = CCW in standard math XY)."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw_2d(pts: np.ndarray) -> np.ndarray:
    """Return copy of 2D polygon vertices ordered with positive signed area."""
    if len(pts) < 3:
        return pts.copy()
    if polygon_signed_area_2d(pts) < 0.0:
        return pts[::-1].copy()
    return pts.copy()


def adjust_polygon_vertex_count(
    raw_points_2d: Sequence[tuple[float, float]] | np.ndarray,
    target_count: int,
    min_verts: int = 3,
    max_verts: int = 256,
) -> np.ndarray:
    """Adjust a closed 2D polygon to target_count vertices.

    - target_count < len(raw): Visvalingam-Whyatt simplification (removes least-significant vertex first,
      preserving sharp corners cleanly down to min_verts).
    - target_count == len(raw): returns original deduplicated vertices.
    - target_count > len(raw): subdivides the longest edges at their midpoints until target_count.
    """
    pts = simplify_polyline(raw_points_2d, min_dist_px=2.0, rdp_epsilon=0.0, closed=True)
    if len(pts) < 3:
        return pts

    target = max(int(min_verts), min(int(max_verts), int(target_count)))
    work: list[np.ndarray] = [pts[i].copy() for i in range(len(pts))]

    # Reduce vertices (Visvalingam-Whyatt: remove vertex forming smallest triangle with neighbors)
    while len(work) > target and len(work) > min_verts:
        m = len(work)
        best_idx = 0
        best_area = float("inf")
        for i in range(m):
            a = work[(i - 1) % m]
            b = work[i]
            c = work[(i + 1) % m]
            area2 = abs(float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
            if area2 < best_area:
                best_area = area2
                best_idx = i
        work.pop(best_idx)

    # Increase vertices (subdivide longest edge at midpoint)
    while len(work) < target:
        m = len(work)
        longest_idx = 0
        longest_d2 = -1.0
        for i in range(m):
            p0 = work[i]
            p1 = work[(i + 1) % m]
            d = p1 - p0
            d2 = float(d[0] * d[0] + d[1] * d[1])
            if d2 > longest_d2:
                longest_d2 = d2
                longest_idx = i
        if longest_d2 <= 1e-9:
            break
        p0 = work[longest_idx]
        p1 = work[(longest_idx + 1) % m]
        mid = (p0 + p1) * 0.5
        work.insert(longest_idx + 1, mid)

    return np.asarray(work, dtype=np.float64)


def _cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    c1 = _cross_2d(a, b, p)
    c2 = _cross_2d(b, c, p)
    c3 = _cross_2d(c, a, p)
    eps = -1e-9
    return (c1 >= eps and c2 >= eps and c3 >= eps) or (c1 <= -eps and c2 <= -eps and c3 <= -eps)


def triangulate_polygon_2d(points_2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation for arbitrary simple (convex or concave) 2D polygons.

    Expects CCW vertices (positive signed area). Returns list of (i0, i1, i2) triangle index tuples.
    """
    n = len(points_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Ensure positive orientation for ear test
    area = polygon_signed_area_2d(points_2d)
    indices = list(range(n)) if area >= 0.0 else list(reversed(range(n)))
    triangles: list[tuple[int, int, int]] = []

    max_iters = n * n + 10
    iters = 0
    while len(indices) > 3 and iters < max_iters:
        iters += 1
        ear_found = False
        m = len(indices)
        for k in range(m):
            i_prev = indices[(k - 1) % m]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]

            a = points_2d[i_prev]
            b = points_2d[i_curr]
            c = points_2d[i_next]

            # Must be strictly convex corner
            if _cross_2d(a, b, c) <= 1e-10:
                continue

            # Check no other polygon vertex lies inside triangle (a, b, c)
            has_inside = False
            for j in indices:
                if j == i_prev or j == i_curr or j == i_next:
                    continue
                if _point_in_triangle_2d(points_2d[j], a, b, c):
                    has_inside = True
                    break

            if not has_inside:
                triangles.append((i_prev, i_curr, i_next))
                indices.pop(k)
                ear_found = True
                break

        if not ear_found:
            # Remove nearly collinear vertex if any, else break to fallback fan
            removed_collinear = False
            for k in range(m):
                i_prev = indices[(k - 1) % m]
                i_curr = indices[k]
                i_next = indices[(k + 1) % m]
                if abs(_cross_2d(points_2d[i_prev], points_2d[i_curr], points_2d[i_next])) <= 1e-7:
                    indices.pop(k)
                    removed_collinear = True
                    break
            if not removed_collinear:
                break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    elif len(indices) > 3:
        # Degenerate/self-intersecting fallback fan so we never crash
        base = indices[0]
        for k in range(1, len(indices) - 1):
            triangles.append((base, indices[k], indices[k + 1]))

    return triangles


def unproject_qt_to_world_plane(
    points_qt: np.ndarray,
    cam: CameraSnapshot,
    plane_depth: float,
) -> np.ndarray:
    """Unproject Qt-local 2D points (origin top-left, Y down) onto the plane dot(P, view_forward) = plane_depth."""
    pts = np.asarray(points_qt, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    w = max(float(cam.width), 1.0)
    h = max(float(cam.height), 1.0)

    ndc_x = (pts[:, 0] / w) * 2.0 - 1.0
    ndc_y = 1.0 - (pts[:, 1] / h) * 2.0  # Qt Y is top-down, NDC Y is bottom-up

    near_clip = np.column_stack([ndc_x, ndc_y, np.full(n, -1.0), np.ones(n)])
    far_clip = np.column_stack([ndc_x, ndc_y, np.full(n, 1.0), np.ones(n)])

    near_h = near_clip @ cam.inv_perspective_matrix.T
    far_h = far_clip @ cam.inv_perspective_matrix.T

    near_w = np.where(np.abs(near_h[:, 3:4]) > 1e-12, near_h[:, 3:4], 1.0)
    far_w = np.where(np.abs(far_h[:, 3:4]) > 1e-12, far_h[:, 3:4], 1.0)

    p_near = near_h[:, :3] / near_w
    p_far = far_h[:, :3] / far_w

    ray_dir = p_far - p_near
    ray_len = np.linalg.norm(ray_dir, axis=1, keepdims=True)
    ray_dir = ray_dir / np.where(ray_len > 1e-12, ray_len, 1.0)

    fwd = cam.view_forward
    denom = ray_dir @ fwd  # (N,)
    denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
    t = (float(plane_depth) - (p_near @ fwd)) / denom
    return p_near + ray_dir * t[:, np.newaxis]


def project_world_to_qt_2d(
    points_3d: np.ndarray,
    cam: CameraSnapshot,
) -> np.ndarray:
    """Project 3D world points onto Qt-local 2D coordinates (origin top-left, Y down)."""
    pts = np.asarray(points_3d, dtype=np.float64)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    clip = hom @ cam.perspective_matrix.T
    cw = np.where(np.abs(clip[:, 3:4]) > 1e-12, clip[:, 3:4], 1.0)
    ndc = clip[:, :3] / cw

    qx = (ndc[:, 0] + 1.0) * 0.5 * float(cam.width)
    qy = (1.0 - ndc[:, 1]) * 0.5 * float(cam.height)
    return np.column_stack([qx, qy])


def compute_mesh_signed_volume(vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> float:
    """Compute signed volume of a closed mesh (positive = outward-facing normals)."""
    vol = 0.0
    for f in faces:
        if len(f) < 3:
            continue
        v0 = vertices[f[0]]
        for k in range(1, len(f) - 1):
            v1 = vertices[f[k]]
            v2 = vertices[f[k + 1]]
            vol += float(np.dot(v0, np.cross(v1, v2)))
    return vol / 6.0


def extrude_polygon_3d_to_mesh(
    front_3d: np.ndarray,
    cap_tris: Sequence[tuple[int, int, int]],
    extrude_dir: np.ndarray,
    extrude_length: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Extrude an already-unprojected 3D planar polygon along extrude_dir (supports +/- length)."""
    pts = np.asarray(front_3d, dtype=np.float64)
    n = len(pts)
    if n < 3 or not cap_tris:
        raise ValueError("Polygon needs at least 3 vertices and valid cap triangles.")

    d_vec = np.asarray(extrude_dir, dtype=np.float64)
    d_norm = float(np.linalg.norm(d_vec))
    if d_norm <= 1e-12:
        raise ValueError("extrude_dir cannot be zero.")
    d_unit = d_vec / d_norm

    length = float(extrude_length)
    if abs(length) < 0.01:
        length = 0.01 if length >= 0.0 else -0.01

    back_3d = pts + d_unit[np.newaxis, :] * length
    vertices = np.vstack([pts, back_3d])

    faces: list[tuple[int, ...]] = []
    for a, b, c in cap_tris:
        faces.append((a, c, b))
    for a, b, c in cap_tris:
        faces.append((n + a, n + b, n + c))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))

    if compute_mesh_signed_volume(vertices, faces) < 0.0:
        faces = [tuple(reversed(f)) for f in faces]

    return vertices, faces


def extrude_polygon_2d_to_mesh(
    points_qt: Sequence[tuple[float, float]] | np.ndarray,
    cam: CameraSnapshot,
    front_depth: float,
    extrude_length: float,
    simplify: bool = True,
    min_dist_px: float = 3.0,
    rdp_epsilon: float = 2.0,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Convert a closed 2D screen polygon into a watertight 3D extruded prism with outward normals.

    Returns (vertices: np.ndarray[2N, 3], faces: list[tuple[int, ...]]).
    Raises ValueError if the polygon is degenerate (< 3 vertices or near-zero area).
    """
    pts_2d = np.asarray(points_qt, dtype=np.float64)
    if simplify:
        pts_2d = simplify_polyline(pts_2d, min_dist_px=min_dist_px, rdp_epsilon=rdp_epsilon, closed=True)

    if len(pts_2d) < 3 or abs(polygon_signed_area_2d(pts_2d)) < 1.0:
        raise ValueError("Polygon is too small or has fewer than 3 distinct points.")

    pts_2d = ensure_ccw_2d(pts_2d)
    cap_tris = triangulate_polygon_2d(pts_2d)
    if not cap_tris:
        raise ValueError("Failed to triangulate drawn polygon.")

    front_3d = unproject_qt_to_world_plane(pts_2d, cam, front_depth)
    return extrude_polygon_3d_to_mesh(front_3d, cap_tris, cam.view_forward, extrude_length)


def extrude_polygon_2d_frustum_to_mesh(
    pts_ccw_2d: np.ndarray,
    cap_tris: Sequence[tuple[int, int, int]],
    cam: CameraSnapshot,
    front_depth: float,
    back_depth: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Build a watertight 3D prism/frustum matching screen rays pixel-for-pixel in Perspective & Ortho views."""
    pts = np.asarray(pts_ccw_2d, dtype=np.float64)
    n = len(pts)
    if n < 3 or not cap_tris:
        raise ValueError("Polygon needs at least 3 vertices and valid cap triangles.")

    safe_front = float(front_depth)
    safe_back = float(back_depth)
    if cam.is_perspective:
        cam_depth = float(np.dot(cam.camera_pos, cam.view_forward))
        safe_front = max(safe_front, cam_depth + 0.15)
    if safe_back <= safe_front + 0.05:
        safe_back = safe_front + 1.0

    front_3d = unproject_qt_to_world_plane(pts, cam, safe_front)
    back_3d = unproject_qt_to_world_plane(pts, cam, safe_back)
    vertices = np.vstack([front_3d, back_3d])

    faces: list[tuple[int, ...]] = []
    for a, b, c in cap_tris:
        faces.append((a, c, b))
    for a, b, c in cap_tris:
        faces.append((n + a, n + b, n + c))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))

    if compute_mesh_signed_volume(vertices, faces) < 0.0:
        faces = [tuple(reversed(f)) for f in faces]

    return vertices, faces
