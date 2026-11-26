"""Concrete debris chunks clinging to generated rebars.

The look is ported from Red Panda's ``RP_Rebar_Gernerator/debris.py``: a
level-0 icosphere is scattered along the exposed part of each bar, biased
towards the concrete exit point, then randomly rotated, stretched and dented.

Two deliberate improvements over the original:

1. Chunk centres and shapes come from separate deterministic streams, so a
   given seed always reproduces the same debris.
2. Chunks are appended to the generator's existing mesh instead of creating a
   separate ``Rebars_Debris`` object.  That keeps the ``sid`` identity, bake,
   delete guard and collection ownership intact, and avoids Red Panda's
   prefix-based cleanup bugs.

This module is pure data: no Blender objects, materials or operators.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from mathutils import Matrix, Vector

from . import rebar_spline as rs

# Safety cap: a 5,000-bar set at full density would otherwise create ~30,000
# chunks (600,000 faces) in one commit.
MAX_DEBRIS_CHUNKS = 5000

# Debris faces use the second material slot of the rebar object.
DEBRIS_MATERIAL_INDEX = 1

_PHI = (1.0 + math.sqrt(5.0)) * 0.5

_ICO_VERTICES = tuple(
    Vector(v).normalized()
    for v in (
        (-1.0, _PHI, 0.0), (1.0, _PHI, 0.0),
        (-1.0, -_PHI, 0.0), (1.0, -_PHI, 0.0),
        (0.0, -1.0, _PHI), (0.0, 1.0, _PHI),
        (0.0, -1.0, -_PHI), (0.0, 1.0, -_PHI),
        (_PHI, 0.0, -1.0), (_PHI, 0.0, 1.0),
        (-_PHI, 0.0, -1.0), (-_PHI, 0.0, 1.0),
    )
)

_ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


@dataclass
class DebrisResult:
    """Payload ready to merge into a :class:`rebar_spline.SplineMesh`."""

    payload: rs.SplineMesh
    chunk_count: int = 0
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return self.chunk_count == 0 or self.payload.is_empty()


# ---------------------------------------------------------------------------
# UV helpers
# ---------------------------------------------------------------------------


def compute_spherical_uvs(
    v0: Vector, v1: Vector, v2: Vector
) -> list[tuple[float, float]]:
    """Spherical UVs for one triangular icosphere face, in [0, 1]."""
    phis = [math.atan2(v.y, v.x) for v in (v0, v1, v2)]
    thetas = [math.asin(max(-1.0, min(1.0, v.z))) for v in (v0, v1, v2)]

    # A face crossing the -X seam would otherwise stretch across the texture.
    if max(phis) - min(phis) > math.pi:
        phis = [p + math.tau if p < 0.0 else p for p in phis]

    phi_span = max(phis) - min(phis)
    theta_span = max(thetas) - min(thetas)
    return [
        (
            (phi - min(phis)) / phi_span if phi_span > 1e-6 else 0.0,
            (theta - min(thetas)) / theta_span if theta_span > 1e-6 else 0.5,
        )
        for phi, theta in zip(phis, thetas)
    ]


# ---------------------------------------------------------------------------
# Placement (shared by preview and baked geometry)
# ---------------------------------------------------------------------------


def _segment_lengths(points: list[Vector]) -> list[float]:
    return [
        max((points[i + 1] - points[i]).length, 0.0)
        for i in range(len(points) - 1)
    ]


def _point_at_distance(
    points: list[Vector], lengths: list[float], target: float
) -> Vector:
    """Walk a polyline by arc length, clamped to its ends."""
    if len(points) < 2:
        return Vector(points[0]) if points else Vector((0.0, 0.0, 0.0))
    travelled = 0.0
    for index, length in enumerate(lengths):
        if travelled + length >= target:
            if length <= 1e-9:
                return Vector(points[index])
            frac = max(0.0, min(1.0, (target - travelled) / length))
            return Vector(points[index]).lerp(Vector(points[index + 1]), frac)
        travelled += length
    return Vector(points[-1])


def debris_centers(splines, params) -> list[tuple[Vector, float]]:
    """Local-space ``(centre, size)`` pairs, deterministic for a given seed."""
    if not params.debris_enabled or not splines:
        return []

    rng = random.Random(int(params.seed) + 12345)
    centers: list[tuple[Vector, float]] = []
    density = max(float(params.debris_density), 0.0)
    if density <= 0.0:
        return []

    bias_exp = 1.0 + max(0.0, min(1.0, float(params.debris_concentration))) * 3.0
    min_size = max(float(params.debris_min_size), 1e-5)
    max_size = max(min_size, float(params.debris_max_size))
    sink = max(float(params.sink_depth), 0.0)

    # Splines are (points, subdivisions, twist) triples; debris only needs the
    # final points, so a bent bar automatically pulls its chunks along.
    for spline in splines:
        points = spline[0]
        if len(centers) >= MAX_DEBRIS_CHUNKS:
            break
        if len(points) < 2:
            continue
        lengths = _segment_lengths(points)
        total_len = sum(lengths)
        if total_len <= 1e-6:
            continue

        # Red Panda clamps the embedded fraction so a very short bar still has
        # an exposed section where debris can sit.
        t_embedded = min(sink / total_len, 0.95)
        exposed = total_len * (1.0 - t_embedded)
        count = int(exposed * density * 15.0)
        if count <= 0:
            continue

        for _ in range(count):
            if len(centers) >= MAX_DEBRIS_CHUNKS:
                break
            # Biasing t_raw towards 0 clusters chunks at the surface exit.
            t_raw = rng.random()
            t_biased = t_raw**bias_exp
            t = t_embedded + t_biased * (1.0 - t_embedded)

            target_dist = t * total_len
            position = _point_at_distance(points, lengths, target_dist)

            size = rng.uniform(min_size, max_size)
            distance_from_surface = (
                (t - t_embedded) / (1.0 - t_embedded)
                if (1.0 - t_embedded) > 1e-6
                else 1.0
            )
            size *= max(0.3, 1.0 - distance_from_surface * 0.7)
            centers.append((position, size))

    return centers


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def build_debris(splines, params, uv_scale: float) -> DebrisResult:
    """Build irregular low-poly chunks for every exposed rebar section."""
    centers = debris_centers(splines, params)
    if not centers:
        return DebrisResult(rs.SplineMesh())

    shape_rng = random.Random(int(params.seed) + 54321)
    irregularity = max(0.0, min(1.0, float(params.debris_irregularity)))
    scale = max(float(uv_scale), 1e-9)
    payload = rs.SplineMesh()

    for position, size in centers:
        rot = (
            Matrix.Rotation(shape_rng.uniform(0.0, math.tau), 3, "X")
            @ Matrix.Rotation(shape_rng.uniform(0.0, math.tau), 3, "Y")
            @ Matrix.Rotation(shape_rng.uniform(0.0, math.tau), 3, "Z")
        )
        sx, sy, sz = (shape_rng.uniform(0.5, 1.5) for _ in range(3))

        vertex_offset = len(payload.vertices)
        for vertex in _ICO_VERTICES:
            displacement = 1.0 + shape_rng.uniform(-irregularity, irregularity)
            scaled = Vector(
                (
                    vertex.x * sx * displacement,
                    vertex.y * sy * displacement,
                    vertex.z * sz * displacement,
                )
            ) * size
            payload.vertices.append(tuple(position + (rot @ scaled)))

        u_offset = shape_rng.uniform(0.0, 10.0)
        v_offset = shape_rng.uniform(0.0, 10.0)
        uv_u = (math.tau * size) * scale
        uv_v = (math.pi * size) * scale

        for face in _ICO_FACES:
            payload.faces.append(tuple(vertex_offset + index for index in face))
            payload.smooth_faces.append(True)
            payload.face_materials.append(DEBRIS_MATERIAL_INDEX)
            for u, v in compute_spherical_uvs(
                _ICO_VERTICES[face[0]], _ICO_VERTICES[face[1]], _ICO_VERTICES[face[2]]
            ):
                payload.uvs.append((u_offset + u * uv_u, v_offset + v * uv_v))

    return DebrisResult(
        payload=payload,
        chunk_count=len(centers),
        truncated=len(centers) >= MAX_DEBRIS_CHUNKS,
    )


def merge_payload(target: rs.SplineMesh, debris: rs.SplineMesh) -> None:
    """Append debris geometry to a rebar payload in place."""
    if debris.is_empty():
        return
    # The bar builder leaves face_materials empty (everything uses slot 0).
    # Pad it first so only the appended debris faces are tagged with slot 1.
    while len(target.face_materials) < len(target.faces):
        target.face_materials.append(0)

    offset = len(target.vertices)
    target.vertices.extend(debris.vertices)
    target.faces.extend(tuple(index + offset for index in face) for face in debris.faces)
    target.uvs.extend(debris.uvs)
    target.smooth_faces.extend(debris.smooth_faces)
    target.face_materials.extend(debris.face_materials)
    for a, b in debris.seam_edges:
        target.seam_edges.add((a + offset, b + offset))
