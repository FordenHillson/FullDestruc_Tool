"""Face tagging and rebar coverage generation for the Concrete Rebars tool.

Two responsibilities:

1. **Tagging** — remember which faces the user selected, so the bar set can be
   regenerated or extended later.  The tag is a FACE-domain INT attribute
   (``fdt_rebar_group``) holding the group index inside the host mesh, with
   ``-1`` meaning "no group".  Face attributes survive modifiers in this
   codebase (``core/geonodes.py`` relies on the same trick for ``fdt_orig``), so
   this is more robust than storing polygon indices.

2. **Generation** — port of Red Panda's grid raycast + bar growth, but bounded
   by the tagged face set instead of "every face with the target material".

Nothing here creates Blender objects; it returns plain spline data for
``rebar_spline.build_spline_mesh``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import rebar_spline

GROUP_ATTR_NAME = rebar_spline.GROUP_ATTR_NAME
NO_GROUP = -1

# Rays are extended slightly past the face-cluster bounds so a bar can still
# emerge from a face sitting exactly on the boundary.
RAY_MARGIN = 0.05

# Distance stepped past each hit so the walk does not re-hit the same face.
RAY_STEP = 0.001

# Exposed bar length (m) at which the Bend Amount applies at full strength.
# Shorter exposed sections bend proportionally less (stiff short lever).
_BEND_REF_FREE_LEN = 0.35


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class RebarParams:
    """Plain-data snapshot of everything generation needs (all values in metres)."""

    spacing: tuple[float, float, float] = (0.15, 0.15, 0.15)
    align: tuple[bool, bool, bool] = (True, True, True)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.006
    sink_depth: float = 0.03
    min_length: float = 0.15
    max_length: float = 0.6
    bending_strength: float = 0.03
    bend_scale: float = 0.4
    subdivisions: int = 6
    seed: int = 0
    angle_threshold: float = 0.5
    normal_bias: float = 0.5
    gravity_bias: float = 0.5
    smooth_radius: float = 0.3
    avoidance_strength: float = 0.5
    max_bars: int = 5000
    debris_enabled: bool = False
    debris_density: float = 0.5
    debris_min_size: float = 0.005
    debris_max_size: float = 0.04
    debris_irregularity: float = 0.4
    debris_concentration: float = 0.7


@dataclass
class GenerationResult:
    # Each spline is ``(points, subdivisions, twist_rad)``; the twist is the
    # total axial rotation at the tip (see rebar_spline.build_spline_mesh).
    splines: list[tuple[list[Vector], int, float]] = field(default_factory=list)
    bar_count: int = 0
    truncated: bool = False
    skipped_grazing: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.splines


# ---------------------------------------------------------------------------
# Face tags
# ---------------------------------------------------------------------------


def ensure_group_attribute(mesh) -> bpy.types.Attribute:
    """Return the FACE INT group attribute, creating it at ``NO_GROUP`` if absent."""
    attr = mesh.attributes.get(GROUP_ATTR_NAME)
    if attr is not None and attr.domain == "FACE" and attr.data_type == "INT":
        return attr
    if attr is not None:
        mesh.attributes.remove(attr)
    attr = mesh.attributes.new(GROUP_ATTR_NAME, "INT", "FACE")
    for item in attr.data:
        item.value = NO_GROUP
    return attr


def _group_attribute(mesh) -> bpy.types.Attribute | None:
    attr = mesh.attributes.get(GROUP_ATTR_NAME)
    if attr is None or attr.domain != "FACE" or attr.data_type != "INT":
        return None
    return attr


def tag_faces(mesh, face_indices, group_index: int) -> int:
    """Assign ``group_index`` to the given polygons. Returns how many changed."""
    group_index = int(group_index)
    if group_index < 0:
        raise ValueError("group_index must be >= 0")
    attr = ensure_group_attribute(mesh)
    changed = 0
    count = len(attr.data)
    for index in face_indices:
        if 0 <= int(index) < count:
            attr.data[int(index)].value = group_index
            changed += 1
    return changed


def untag_faces(mesh, face_indices) -> int:
    """Clear the group tag on the given polygons. Returns how many changed."""
    attr = _group_attribute(mesh)
    if attr is None:
        return 0
    changed = 0
    count = len(attr.data)
    for index in face_indices:
        if 0 <= int(index) < count and attr.data[int(index)].value != NO_GROUP:
            attr.data[int(index)].value = NO_GROUP
            changed += 1
    return changed


def tagged_face_indices(mesh, group_index: int) -> list[int]:
    """All polygons currently carrying ``group_index``."""
    attr = _group_attribute(mesh)
    if attr is None:
        return []
    group_index = int(group_index)
    return [i for i, item in enumerate(attr.data) if item.value == group_index]


def tagged_face_count(mesh, group_index: int) -> int:
    return len(tagged_face_indices(mesh, group_index))


def count_any_tagged(mesh) -> int:
    """How many faces carry any group — used by the delete guard."""
    attr = _group_attribute(mesh)
    if attr is None:
        return 0
    return sum(1 for item in attr.data if item.value != NO_GROUP)


def next_group_index(mesh) -> int:
    """Smallest unused group index on this mesh (``0`` when empty)."""
    attr = _group_attribute(mesh)
    if attr is None:
        return 0
    used = {item.value for item in attr.data if item.value != NO_GROUP}
    index = 0
    while index in used:
        index += 1
    return index


# ---------------------------------------------------------------------------
# Selection + material resolution
# ---------------------------------------------------------------------------


def material_index_for(obj, material) -> int:
    """Slot index of ``material`` on ``obj``, or ``-1``."""
    if obj is None or material is None or not getattr(obj, "material_slots", None):
        return -1
    for index, slot in enumerate(obj.material_slots):
        if slot.material is not None and slot.material.name == material.name:
            return index
    return -1


def selected_face_indices(context, obj, material_index: int = -1) -> list[int]:
    """Selected polygons on ``obj``, optionally filtered by material slot.

    Works in both Edit Mode (via bmesh) and Object Mode (via polygon flags).
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return []

    indices: list[int] = []
    if obj.mode == "EDIT":
        import bmesh

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            if not face.select:
                continue
            if material_index >= 0 and face.material_index != material_index:
                continue
            indices.append(face.index)
        return indices

    for polygon in obj.data.polygons:
        if not polygon.select:
            continue
        if material_index >= 0 and polygon.material_index != material_index:
            continue
        indices.append(polygon.index)
    return indices


def source_mesh(context, obj, group_index: int | None = None) -> tuple[object, bool, int]:
    """Pick the mesh to sample and report whether it is the evaluated one.

    Prefers the evaluated mesh so booleans are honoured, but only when the group
    attribute actually survived modifiers and still carries tags.  Returns
    ``(mesh, is_evaluated, eval_obj_or_None)`` so the caller can clear it.
    """
    if obj is None or obj.data is None:
        return obj.data, False, None

    depsgraph = None
    if context is not None:
        try:
            depsgraph = context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None

    if depsgraph is not None:
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
        except Exception:
            eval_mesh = None
            eval_obj = None
        if eval_mesh is not None:
            if group_index is None or tagged_face_count(eval_mesh, group_index) > 0:
                return eval_mesh, True, eval_obj
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass

    return obj.data, False, None


def release_source(context, is_evaluated: bool, eval_obj) -> None:
    """Clear an evaluated mesh obtained from :func:`source_mesh`."""
    if is_evaluated and eval_obj is not None:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# BVH + bounds
# ---------------------------------------------------------------------------


def build_bvh(mesh, face_indices) -> BVHTree | None:
    """BVH over only the given faces (the "broken" surface)."""
    polygons = [tuple(mesh.polygons[i].vertices) for i in face_indices if 0 <= i < len(mesh.polygons)]
    if not polygons:
        return None
    vertices = [v.co.copy() for v in mesh.vertices]
    try:
        return BVHTree.FromPolygons(vertices, polygons)
    except Exception:
        return None


def build_complement_bvh(mesh, face_indices) -> BVHTree | None:
    """BVH over every face *not* in the set (intact structure, used for avoidance)."""
    exclude = set(face_indices)
    polygons = [tuple(p.vertices) for i, p in enumerate(mesh.polygons) if i not in exclude]
    if not polygons:
        return None
    vertices = [v.co.copy() for v in mesh.vertices]
    try:
        return BVHTree.FromPolygons(vertices, polygons)
    except Exception:
        return None


def face_bounds(mesh, face_indices) -> tuple[Vector, Vector] | None:
    """Local-space bounding box of the vertices touched by the given faces."""
    vertex_ids: set[int] = set()
    for index in face_indices:
        if 0 <= index < len(mesh.polygons):
            vertex_ids.update(mesh.polygons[index].vertices)
    if not vertex_ids:
        return None
    coords = [mesh.vertices[i].co for i in vertex_ids]
    min_co = Vector(
        (
            min(c.x for c in coords),
            min(c.y for c in coords),
            min(c.z for c in coords),
        )
    )
    max_co = Vector(
        (
            max(c.x for c in coords),
            max(c.y for c in coords),
            max(c.z for c in coords),
        )
    )
    return min_co, max_co


def gravity_local(obj) -> Vector:
    """World -Z expressed in the object's local space."""
    try:
        w2l = obj.matrix_world.to_3x3().inverted()
        return (w2l @ Vector((0.0, 0.0, -1.0))).normalized()
    except Exception:
        return Vector((0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# Grid raycast
# ---------------------------------------------------------------------------


def cast_grid_hits(
    bvh: BVHTree, min_co: Vector, max_co: Vector, params: RebarParams
) -> list[tuple[Vector, Vector, Vector]]:
    """Walk every grid line and collect surface hits as ``(position, normal, axis)``."""
    axes = []
    if params.align[0]:
        axes.append(("X", Vector((1.0, 0.0, 0.0)), min_co.x - RAY_MARGIN, max_co.x + RAY_MARGIN))
    if params.align[1]:
        axes.append(("Y", Vector((0.0, 1.0, 0.0)), min_co.y - RAY_MARGIN, max_co.y + RAY_MARGIN))
    if params.align[2]:
        axes.append(("Z", Vector((0.0, 0.0, 1.0)), min_co.z - RAY_MARGIN, max_co.z + RAY_MARGIN))

    # Y-direction bars are lifted by a full diameter so they rest on the X bars.
    y_axis_lift = 2.0 * params.radius
    hits: list[tuple[Vector, Vector, Vector]] = []

    for axis_name, direction, min_val, max_val in axes:
        if axis_name == "X":
            c1 = rebar_spline.get_grid_coords(min_co.y, max_co.y, params.spacing[1], params.offset[1])
            c2 = rebar_spline.get_grid_coords(min_co.z, max_co.z, params.spacing[2], params.offset[2])
        elif axis_name == "Y":
            c1 = rebar_spline.get_grid_coords(min_co.x, max_co.x, params.spacing[0], params.offset[0])
            c2 = rebar_spline.get_grid_coords(min_co.z, max_co.z, params.spacing[2], params.offset[2] + y_axis_lift)
        else:
            c1 = rebar_spline.get_grid_coords(min_co.x, max_co.x, params.spacing[0], params.offset[0])
            c2 = rebar_spline.get_grid_coords(min_co.y, max_co.y, params.spacing[1], params.offset[1])

        max_dist = max_val - min_val
        if max_dist <= 0.0:
            continue

        for g1 in c1:
            for g2 in c2:
                if axis_name == "X":
                    origin = Vector((min_val, g1, g2))
                elif axis_name == "Y":
                    origin = Vector((g1, min_val, g2))
                else:
                    origin = Vector((g1, g2, min_val))

                current = origin.copy()
                remaining = max_dist
                guard = 0
                while remaining > 0.0 and guard < 4096:
                    guard += 1
                    hit_co, hit_normal, _hit_index, hit_dist = bvh.ray_cast(current, direction, remaining)
                    if hit_co is None:
                        break
                    hits.append((hit_co.copy(), hit_normal.copy(), direction))
                    step = RAY_STEP
                    current = hit_co + direction * step
                    remaining -= hit_dist + step

    return hits


def grow_bars(
    hits: list[tuple[Vector, Vector, Vector]],
    bvh: BVHTree,
    complement_bvh: BVHTree | None,
    gravity: Vector,
    params: RebarParams,
) -> GenerationResult:
    """Turn surface hits into (points, subdivisions, twist) triples.

    Every bar draws from its own seeded stream keyed by ``(seed, index)``, so a
    fixed seed reproduces the exact same set — and the shape of bar *n* never
    depends on how many bars before it were skipped.  A *string* seed
    (``"fdt-rebar-{seed}:{index}"``) is used because string seeds are mixed
    with SHA-512 inside the random module — deterministic across processes and
    Python versions — while arbitrary hashables such as int tuples are
    rejected by newer CPython builds.
    """
    result = GenerationResult()

    pending: list[dict] = []
    for index, (hit_co, hit_normal, ray_dir) in enumerate(hits):
        if len(pending) >= max(int(params.max_bars), 1):
            result.truncated = True
            break
        # String seed: supported by every CPython version and mixed with
        # SHA-512 internally, so (seed, index) maps to a stable stream.
        rng = random.Random(f"fdt-rebar-{int(params.seed)}:{index}")

        near = bvh.find_nearest_range(hit_co, params.smooth_radius)
        if near:
            summed = Vector((0.0, 0.0, 0.0))
            for _loc, norm, _idx, _dist in near:
                summed += norm
            smoothed = summed.normalized() if summed.length > 1e-6 else hit_normal
        else:
            smoothed = hit_normal

        # Grazing hits would produce bars hugging the surface.
        dot = smoothed.dot(ray_dir)
        if abs(dot) < params.angle_threshold:
            result.skipped_grazing += 1
            continue

        d_exit = ray_dir if dot > 0.0 else -ray_dir
        p_start = hit_co - params.sink_depth * d_exit

        length = rng.uniform(params.min_length, params.max_length)
        total_len = params.sink_depth + length

        u_axis, v_axis = _basis_for(d_exit)

        normal_perp = rebar_spline.perpendicular_component(smoothed, d_exit)
        grav_perp = rebar_spline.perpendicular_component(gravity, d_exit)

        avoid_perp = Vector((0.0, 0.0, 0.0))
        if complement_bvh is not None:
            near_co, _near_norm, _near_idx, _near_dist = complement_bvh.find_nearest(hit_co)
            if near_co is not None:
                avoid_perp = rebar_spline.perpendicular_component(hit_co - near_co, d_exit)

        angle1 = rng.uniform(0.0, math.tau)
        angle2 = rng.uniform(0.0, math.tau)
        rand1 = u_axis * math.cos(angle1) + v_axis * math.sin(angle1)
        rand2 = u_axis * math.cos(angle2) + v_axis * math.sin(angle2)

        # Wave shape + twist: all per-bar randoms, drawn in a fixed order so
        # the (seed, index) stream stays reproducible across versions.
        phase1 = rng.uniform(0.0, math.tau)
        phase2 = rng.uniform(0.0, math.tau)
        phase3 = rng.uniform(0.0, math.tau)
        wave1 = rng.uniform(0.55, 0.80)
        wave2 = rng.uniform(0.55, 0.85)
        # Not every bar bends the same: in a demolished slab most bars are
        # barely kinked and a few bend hard.  Skew the factor toward the low
        # end so straight-ish bars are the majority.
        bend_factor = 0.35 + 0.65 * (rng.uniform(0.0, 1.0) ** 1.5)
        twist_u = bend_factor * rng.uniform(-1.0, 1.0)

        spec = rebar_spline.compose_bend_spec(
            normal_perp=normal_perp,
            gravity_perp=grav_perp,
            avoidance_perp=avoid_perp,
            random_b1=rand1,
            random_b2=rand2,
            normal_bias=params.normal_bias,
            gravity_bias=params.gravity_bias,
            avoidance_strength=params.avoidance_strength,
            bending_strength=params.bending_strength,
            bend_scale=params.bend_scale,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            wave_scale1=wave1,
            wave_scale2=wave2,
            twist=0.0,
        )

        # A short exposed section is a stiff, short lever: it bends less.
        # Scale the amplitude down below the reference exposed length so the
        # default values read as gently kinked rebar at every bar length.
        if length > 1e-6 and spec.amplitude > 1e-9:
            spec.amplitude *= min(1.0, max(length, 0.0) / _BEND_REF_FREE_LEN)
            spec.amplitude *= bend_factor

        # Twist tracks the bar's *actual* bend (signed fraction x the limit
        # for its real amplitude): barely-kinked bars stay nearly untwisted.
        spec.twist = twist_u * rebar_spline.twist_limit(spec.amplitude)

        pending.append(
            {
                "start": p_start,
                "direction": d_exit,
                "total_len": total_len,
                "spec": spec,
            }
        )

    if not pending:
        return result

    longest = max(item["total_len"] for item in pending)
    for item in pending:
        per_sub = rebar_spline.proportional_subdivision(item["total_len"], longest, params.subdivisions)
        points = rebar_spline.build_rebar_points(
            start=item["start"],
            direction=item["direction"],
            total_len=item["total_len"],
            sink_depth=params.sink_depth,
            spec=item["spec"],
            subdivisions=per_sub,
        )
        result.splines.append((points, per_sub, item["spec"].twist))

    result.bar_count = len(result.splines)
    return result


def _basis_for(direction: Vector) -> tuple[Vector, Vector]:
    """Two unit vectors perpendicular to ``direction``."""
    d = Vector(direction)
    if d.length < 1e-9:
        d = Vector((0.0, 0.0, 1.0))
    d.normalize()
    if abs(d.x) > 0.9:
        u = Vector((0.0, 1.0, 0.0))
        v = Vector((0.0, 0.0, 1.0))
    elif abs(d.y) > 0.9:
        u = Vector((1.0, 0.0, 0.0))
        v = Vector((0.0, 0.0, 1.0))
    else:
        u = Vector((1.0, 0.0, 0.0))
        v = Vector((0.0, 1.0, 0.0))
    u = (u - u.dot(d) * d).normalized()
    v = d.cross(u).normalized()
    return u, v


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def generate(
    mesh,
    face_indices,
    params: RebarParams,
    *,
    matrix_world=None,
    include_complement: bool = True,
) -> GenerationResult:
    """Generate rebar splines for the given face set of ``mesh`` (local space)."""
    indices = [int(i) for i in face_indices]
    if not indices:
        return GenerationResult()

    bvh = build_bvh(mesh, indices)
    if bvh is None:
        return GenerationResult()

    bounds = face_bounds(mesh, indices)
    if bounds is None:
        return GenerationResult()
    min_co, max_co = bounds

    complement = build_complement_bvh(mesh, indices) if include_complement else None

    if matrix_world is not None:
        try:
            w2l = matrix_world.to_3x3().inverted()
            gravity = (w2l @ Vector((0.0, 0.0, -1.0))).normalized()
        except Exception:
            gravity = Vector((0.0, 0.0, -1.0))
    else:
        gravity = Vector((0.0, 0.0, -1.0))

    hits = cast_grid_hits(bvh, min_co, max_co, params)
    return grow_bars(
        hits=hits,
        bvh=bvh,
        complement_bvh=complement,
        gravity=gravity,
        params=params,
    )
