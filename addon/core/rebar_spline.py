"""Pure spline + mesh geometry for the Concrete Rebars tool.

This module is deliberately free of Blender operators, collections and sessions so
it can be unit tested with plain data.  It is ported from Red Panda's
``RP_Rebar_Gernerator/core.py`` (grid lattice, bend composition, proportional
subdivisions, parallel-transport square-profile mesher, T-strip UVs, tip cap and
seam marking) with three deliberate changes:

1. The POLY-curve branch is gone — FDT only ever builds a real mesh.
2. ``bevel_depth`` is expressed as a bar *radius* derived from an mm diameter.
3. Ownership/texel parameters are injected instead of read from Red Panda's
   ``scene.rp_texel``.

Nothing here writes to ``bpy.data``; :func:`apply_spline_mesh` is the only
function that touches a mesh datablock and it takes one as an argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mathutils import Vector

# Face attribute carrying the rebar group index on the host mesh.
GROUP_ATTR_NAME = "fdt_rebar_group"

# cos(45°): square profile corner offset from the tube axis.
_PROFILE_COS = 0.7071067811865476

# Horizontal gap (UV units) between rebar islands.
_UV_ISLAND_GAP = 0.02

# Default texel density used when the scene settings are unavailable.
DEFAULT_TEXEL_PER_METER = 512.0
DEFAULT_TEXTURE_SIZE = 1024.0


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def mm_to_m(value: float) -> float:
    """Convert millimetres (UI) to metres (Blender)."""
    return float(value) * 0.001


def m_to_mm(value: float) -> float:
    """Convert metres (Blender) to millimetres (UI)."""
    return float(value) * 1000.0


def diameter_to_radius(diameter_mm: float) -> float:
    """Bar radius in metres from an mm diameter, floored so it can never be 0."""
    return max(mm_to_m(diameter_mm) * 0.5, 1e-5)


def texel_scale(texel_per_meter: float, texture_size: float) -> float:
    """UV units per metre, matching FullDestruc's texel density convention."""
    tpm = float(texel_per_meter) if texel_per_meter else DEFAULT_TEXEL_PER_METER
    size = float(texture_size) if texture_size else DEFAULT_TEXTURE_SIZE
    return tpm / max(size, 1.0)


# ---------------------------------------------------------------------------
# Grid lattice
# ---------------------------------------------------------------------------


def get_grid_coords(min_val: float, max_val: float, spacing: float, offset_val: float = 0.0) -> list[float]:
    """Grid line positions inside ``[min_val, max_val]``.

    Mirrors Red Panda: lines are ``offset_val + i * spacing`` clamped to the
    range, so the lattice is anchored to the offset and not to the bbox.
    """
    spacing = float(spacing)
    if spacing <= 1e-9:
        return []
    lo = float(min_val)
    hi = float(max_val)
    if hi < lo:
        lo, hi = hi, lo
    offset_val = float(offset_val)
    start_idx = math.ceil((lo - offset_val) / spacing)
    end_idx = math.floor((hi - offset_val) / spacing)
    if end_idx < start_idx:
        return []
    return [offset_val + i * spacing for i in range(start_idx, end_idx + 1)]


def estimate_bar_count(
    min_co: Vector,
    max_co: Vector,
    spacings: tuple[float, float, float],
    align: tuple[bool, bool, bool],
    offset: Vector | None = None,
) -> int:
    """Cheap estimate of how many grid rays will be cast for a face cluster.

    Deliberately optimistic (it assumes every ray hits), which is what the UI
    needs for a "will this freeze Blender?" warning before the user commits.
    """
    offset = offset or Vector((0.0, 0.0, 0.0))
    total = 1
    for axis in range(3):
        if not align[axis]:
            continue
        lines = get_grid_coords(min_co[axis], max_co[axis], spacings[axis], offset[axis])
        total *= max(len(lines), 1)
    return total


# ---------------------------------------------------------------------------
# Bend + sampling
# ---------------------------------------------------------------------------


def proportional_subdivision(total_len: float, max_len: float, base_subdivisions: int) -> int:
    """Shorter bars get fewer segments so the longest bar keeps ``base_subdivisions``."""
    base = max(int(base_subdivisions), 1)
    if max_len <= 1e-6:
        return base
    ratio = max(total_len, 0.0) / max_len
    return max(1, round(ratio * base))


# --- Bend / twist model constants -----------------------------------------

# The twist law: a bar may twist at most ``TWIST_MAX_RAD`` around its axis,
# reached when Bend Amount hits ``TWIST_REF_BEND`` (320 mm).  Below that the
# allowed twist grows linearly with the bend amount, so a straight bar never
# twists.  Twist deliberately has no parameter of its own — it is derived from
# Bend Amount (see docs/FullDestruc/02_Tools/Rebar.md).
TWIST_REF_BEND = 0.32
TWIST_MAX_RAD = math.pi


def twist_limit(bending_strength: float) -> float:
    """Maximum allowed |total twist| in radians for a given bend amount (m)."""
    strength = max(float(bending_strength), 0.0)
    return min(TWIST_MAX_RAD, strength / TWIST_REF_BEND * TWIST_MAX_RAD)


@dataclass
class BendSpec:
    """Per-bar random shape of the exposed rebar section (metres / radians).

    Every field comes from the bar's seeded RNG stream, so a fixed seed always
    reproduces the exact same shapes on Re-apply.

    * ``direction`` — unit, dominant bend direction (bias + random).
    * ``lateral``   — unit, secondary out-of-plane wobble direction.
    * ``amplitude`` — Bend Amount in metres (UI "Bend (mm)").
    * ``wavelength``— Bend Scale in metres (UI "Bend Scale (mm)").
    * ``phase1/2/3`` and ``wave_scale1/2`` — random wave shape.
    * ``twist``     — signed total twist at the tip; the frame starts untwisted
      at the bar base and ramps linearly to this angle.
    """

    direction: Vector
    lateral: Vector
    amplitude: float
    wavelength: float
    phase1: float = 0.0
    phase2: float = 0.0
    phase3: float = 0.0
    wave_scale1: float = 1.0
    wave_scale2: float = 0.4
    twist: float = 0.0


def compose_bend_spec(
    normal_perp: Vector,
    gravity_perp: Vector,
    avoidance_perp: Vector,
    random_b1: Vector,
    random_b2: Vector,
    normal_bias: float,
    gravity_bias: float,
    avoidance_strength: float,
    bending_strength: float,
    bend_scale: float,
    phase1: float,
    phase2: float,
    phase3: float,
    wave_scale1: float,
    wave_scale2: float,
    twist: float,
) -> BendSpec:
    """Compose the :class:`BendSpec` for one bar.

    The direction law stays faithful to the Red Panda port: shared
    normal/gravity/avoidance biases plus one independent random direction.
    The amplitude is exactly ``bending_strength``; the wave shape (phases and
    relative wavelengths) and the twist come from per-bar random values, so
    every bar looks different while a fixed seed reproduces all of them.
    All input directions are expected to be already perpendicular to the bar
    axis.
    """
    bias = Vector(normal_perp) * float(normal_bias) + Vector(gravity_perp) * float(gravity_bias)
    main = Vector(random_b1) + bias + Vector(avoidance_perp) * float(avoidance_strength)
    if main.length > 1e-4:
        direction = main.normalized()
    elif Vector(random_b1).length > 1e-4:
        direction = Vector(random_b1).normalized()
    else:
        direction = Vector((0.0, 0.0, 0.0))

    return BendSpec(
        direction=direction,
        lateral=perpendicular_component(random_b2, direction),
        amplitude=max(float(bending_strength), 0.0),
        wavelength=max(float(bend_scale), 1e-3),
        phase1=float(phase1) % math.tau,
        phase2=float(phase2) % math.tau,
        phase3=float(phase3) % math.tau,
        wave_scale1=max(float(wave_scale1), 0.05),
        wave_scale2=max(float(wave_scale2), 0.05),
        twist=float(twist),
    )


def bend_offset(spec: BendSpec, s: float, free_len: float) -> Vector:
    """Offset from the straight axis at distance ``s`` into the exposed part.

    The shape matches rebar kinked by construction debris: the shaft stays
    (nearly) straight and only the top curls gently to one side.  The arc
    exponent (3.0-3.8, from the random ``phase1``) keeps most of the bend in
    the top ~30% of the exposed length.  A tiny out-of-plane wobble keeps
    bars out of one shared plane.  A smoothstep ramp over the first ~15% of
    the exposed length keeps the tangent at the surface aligned with the
    original bar axis.  The result is clamped to 1.5 x amplitude.
    """
    if spec.amplitude <= 1e-9 or free_len <= 1e-9 or s <= 0.0:
        return Vector((0.0, 0.0, 0.0))

    base = max(spec.wavelength, 1e-4)

    def wave(scale: float, phase: float) -> float:
        return math.sin(math.tau * s / max(base * scale, 1e-4) + phase) - math.sin(phase)

    ramp = min(max(s / max(0.15 * free_len, 1e-6), 0.0), 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)

    # Real demolished rebar is straight for most of its length and curves
    # once near the tip — so the arc is the whole in-plane shape (a sine
    # ripple reads as hard kinks at its inflection points).  High exponent:
    # the curl concentrates in the top ~30% of the bar.
    u = min(s / free_len, 1.0)
    power = 3.0 + 0.8 * (0.5 + 0.5 * math.sin(spec.phase1))
    main = u**power
    # Tiny out-of-plane wobble so bars are not all in one plane (±3% of A).
    side = 0.03 * u * 0.5 * wave(spec.wave_scale2, spec.phase3)

    offset = spec.direction * (spec.amplitude * ramp * main) + spec.lateral * (
        spec.amplitude * ramp * side
    )
    limit = 1.5 * spec.amplitude
    if offset.length > limit:
        offset = offset.normalized() * limit
    return offset


def build_rebar_points(
    start: Vector,
    direction: Vector,
    total_len: float,
    sink_depth: float,
    spec: BendSpec,
    subdivisions: int,
) -> list[Vector]:
    """Sample one bar: a straight embedded section, then the wave bend.

    ``start`` is already sunk into the concrete, so the first ``sink_depth`` of
    travel is straight and the bend only applies to the exposed part.
    """
    subdivisions = max(int(subdivisions), 1)
    if total_len <= 1e-6:
        return [Vector(start)]

    # A bent bar is a straight-segment sweep, so its smoothness is set by the
    # sample count: densify the exposed part until (a) the fastest ripple has
    # ~7 samples per wavelength and (b) the arc's tip curvature keeps each
    # segment turn under ~8 deg (capped, so long bars stay cheap).
    if spec.amplitude > 1e-9:
        exposed = max(float(total_len) - float(sink_depth), 0.0)
        if exposed > 1e-6:
            shortest = max(spec.wavelength * min(spec.wave_scale1, spec.wave_scale2), 1e-3)
            ripple = int(math.ceil(exposed / (shortest / 7.0)))
            power = 3.0 + 0.8 * (0.5 + 0.5 * abs(math.sin(spec.phase1)))
            arc = int(math.ceil(math.atan(spec.amplitude * power / exposed) / math.radians(8.0)))
            subdivisions = max(subdivisions, min(max(ripple, arc), 40))

    d = Vector(direction)
    if d.length < 1e-9:
        return [Vector(start)]
    d.normalize()

    free_len = max(float(total_len) - float(sink_depth), 0.0)
    t_embedded = min(max(float(sink_depth) / total_len, 0.0), 1.0)
    span = max(1.0 - t_embedded, 1e-9)

    points: list[Vector] = []
    for i in range(subdivisions + 1):
        t = i / subdivisions
        base = Vector(start) + d * (t * total_len)
        if t <= t_embedded:
            points.append(base)
            continue
        s = (t - t_embedded) / span * free_len
        points.append(base + bend_offset(spec, s, free_len))
    return points


def perpendicular_component(vec: Vector, axis: Vector) -> Vector:
    """Remove the ``axis`` component from ``vec`` and normalise (0 if degenerate)."""
    v = Vector(vec)
    a = Vector(axis)
    if a.length < 1e-9:
        return Vector((0.0, 0.0, 0.0))
    a.normalize()
    perp = v - v.dot(a) * a
    return perp.normalized() if perp.length > 1e-4 else Vector((0.0, 0.0, 0.0))


def clothoid_frames(points: list[Vector]) -> list[tuple[Vector, Vector, Vector]]:
    """Parallel-transport frames along a polyline (no twisting around the axis)."""
    if len(points) < 2:
        return []

    d0 = points[1] - points[0]
    if d0.length < 1e-9:
        return []
    d0.normalize()

    if abs(d0.x) > 0.9:
        u0 = Vector((0.0, 1.0, 0.0))
    elif abs(d0.y) > 0.9:
        u0 = Vector((1.0, 0.0, 0.0))
    else:
        u0 = Vector((1.0, 0.0, 0.0))
    u0 = (u0 - u0.dot(d0) * d0).normalized()
    v0 = d0.cross(u0).normalized()

    frames: list[tuple[Vector, Vector, Vector]] = [(u0, v0, d0)]
    u_prev, v_prev = u0, v0

    for i in range(1, len(points)):
        if i == len(points) - 1:
            d_i = points[i] - points[i - 1]
        else:
            d_i = points[i + 1] - points[i - 1]
        if d_i.length < 1e-9:
            d_i = d0
        d_i.normalize()

        u_i = u_prev - u_prev.dot(d_i) * d_i
        if u_i.length < 1e-4:
            u_i = v_prev - v_prev.dot(d_i) * d_i
        if u_i.length < 1e-9:
            u_i = Vector((1.0, 0.0, 0.0))
        u_i.normalize()
        v_i = d_i.cross(u_i).normalized()

        frames.append((u_i, v_i, d_i))
        u_prev, v_prev = u_i, v_i

    return frames


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------


@dataclass
class SplineMesh:
    """Plain mesh payload: no Blender datablock references."""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    seam_edges: set[tuple[int, int]] = field(default_factory=set)
    smooth_faces: list[bool] = field(default_factory=list)
    # Per-face material slot index. Empty means "use slot 0 for everything".
    # Debris appends faces tagged with slot 1 so it can use its own material.
    face_materials: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.vertices or not self.faces


def build_spline_mesh(
    splines: list[tuple[list[Vector], int, float]],
    radius: float,
    uv_scale: float,
) -> SplineMesh:
    """Build the square-profile tube mesh for every bar.

    ``splines`` is a list of ``(points, subdivisions, twist)`` where
    ``twist`` is the total axial rotation at the tip in radians.  The mesh
    spans every point in ``points`` (bent bars are sampled denser than the
    requested ``subdivisions``), and the profile starts untwisted at the bar
    base and ramps linearly to ``twist`` at the tip, so the square
    cross-section visibly corkscrews along the exposed length.  (Legacy
    2-tuples without twist are accepted and treated as twist 0.)  Each bar
    contributes ``4 * len(points)`` vertices, ``4 * (len(points) - 1)`` side
    quads and one tip cap — the base is left open because it sits inside the
    concrete.
    """
    result = SplineMesh()
    r = max(float(radius), 1e-6)
    scale = max(float(uv_scale), 1e-9)
    # Exact length of one face of the square profile once rotated 45°.
    side_width = math.sqrt(2.0) * r * scale

    for island_idx, entry in enumerate(splines):
        points, _requested_sub = entry[0], entry[1]
        twist = float(entry[2]) if len(entry) > 2 else 0.0
        if len(points) < 2:
            continue
        # Bent bars are sampled denser than the requested subdivisions
        # (rebar_spline.build_rebar_points); the mesh must span every point,
        # so the ring count follows the actual sample count.
        subdivisions = len(points) - 1

        frames = clothoid_frames(points)
        if len(frames) != len(points):
            continue

        start_idx = len(result.vertices)
        last = len(points) - 1

        for index, (point, (u_axis, v_axis, _d_axis)) in enumerate(zip(points, frames)):
            if twist:
                # Rotate the profile around the bar axis: 0 at the base,
                # ``twist`` at the tip.
                angle = twist * (index / last)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                u_rot = u_axis * cos_a + v_axis * sin_a
                v_rot = v_axis * cos_a - u_axis * sin_a
            else:
                u_rot, v_rot = u_axis, v_axis
            result.vertices.append(tuple(point + r * (_PROFILE_COS * u_rot + _PROFILE_COS * v_rot)))
            result.vertices.append(tuple(point + r * (-_PROFILE_COS * u_rot + _PROFILE_COS * v_rot)))
            result.vertices.append(tuple(point + r * (-_PROFILE_COS * u_rot - _PROFILE_COS * v_rot)))
            result.vertices.append(tuple(point + r * (_PROFILE_COS * u_rot - _PROFILE_COS * v_rot)))

        v_coords = [0.0]
        acc = 0.0
        for i in range(1, len(points)):
            acc += (points[i] - points[i - 1]).length
            v_coords.append(acc * scale)

        u_offset = island_idx * (4.0 * side_width + _UV_ISLAND_GAP)

        for i in range(subdivisions):
            for k in range(4):
                k_next = (k + 1) % 4
                idx0 = start_idx + i * 4 + k
                idx1 = start_idx + i * 4 + k_next
                idx2 = start_idx + (i + 1) * 4 + k_next
                idx3 = start_idx + (i + 1) * 4 + k

                result.faces.append((idx0, idx1, idx2, idx3))
                result.smooth_faces.append(True)
                result.uvs.extend(
                    [
                        (u_offset + k * side_width, v_coords[i]),
                        (u_offset + (k + 1) * side_width, v_coords[i]),
                        (u_offset + (k + 1) * side_width, v_coords[i + 1]),
                        (u_offset + k * side_width, v_coords[i + 1]),
                    ]
                )

        # Tip cap only.
        tip = [start_idx + subdivisions * 4 + k for k in range(4)]
        result.faces.append(tuple(tip))
        result.smooth_faces.append(False)
        tip_v = v_coords[-1]
        result.uvs.extend(
            [
                (u_offset, tip_v),
                (u_offset + side_width, tip_v),
                (u_offset + side_width, tip_v + side_width),
                (u_offset, tip_v + side_width),
            ]
        )

        # Seam along the +U corner plus the three cap edges.
        for i in range(subdivisions):
            result.seam_edges.add(
                tuple(sorted((start_idx + i * 4, start_idx + (i + 1) * 4)))
            )
        for a, b in ((1, 2), (2, 3), (3, 0)):
            result.seam_edges.add(tuple(sorted((tip[a], tip[b]))))

    return result


def spline_vertex_count(subdivisions: int) -> int:
    """Vertices contributed by one bar (test helper and UI estimate)."""
    return 4 * (max(int(subdivisions), 1) + 1)


def spline_face_count(subdivisions: int) -> int:
    """Faces contributed by one bar: 4 side quads per segment plus one tip cap."""
    return 4 * max(int(subdivisions), 1) + 1


# ---------------------------------------------------------------------------
# Blender write-out
# ---------------------------------------------------------------------------


def apply_spline_mesh(mesh, payload: SplineMesh, uv_layer_name: str = "UVMap0") -> None:
    """Write a :class:`SplineMesh` into an existing mesh datablock."""
    if payload.is_empty():
        return

    mesh.clear_geometry()
    mesh.from_pydata(payload.vertices, [], payload.faces)

    for index, polygon in enumerate(mesh.polygons):
        if index < len(payload.smooth_faces):
            polygon.use_smooth = payload.smooth_faces[index]
        if index < len(payload.face_materials):
            # The matching slot may be appended by the caller just after this
            # call; Blender stores the index independently of slot count.
            polygon.material_index = max(0, int(payload.face_materials[index]))

    mesh.update()

    uv_layer = mesh.uv_layers.get(uv_layer_name) or mesh.uv_layers.new(name=uv_layer_name)
    for loop_index, uv in enumerate(payload.uvs):
        if loop_index < len(uv_layer.data):
            uv_layer.data[loop_index].uv = uv

    for edge in mesh.edges:
        key = tuple(sorted((edge.vertices[0], edge.vertices[1])))
        if key in payload.seam_edges:
            edge.use_seam = True
