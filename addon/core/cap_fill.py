"""Safe Cut-Plane Auto-Fill & 1-Click Manual Fill Cut Cap for FullDestruc_Tool.

Fills unfilled solid cavities strictly on cut planes (never closing windows, doors,
or hollow rooms/attics) AND provides a 1-click Manual Fill tool in both Edit Mode
and Object Mode that automatically assigns:
- Cut Material
- World-Space Box UV (matching Rust `box_uv.rs` texel density)
- `fdt_orig = 0.0`, `sharp_edge = True`, `freestyle_edge = True` on the cut rim
- Preserved Custom Split Normals on every untouched original face
"""

from __future__ import annotations

from typing import Iterable, Sequence

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

from . import rust_bridge
from .geonodes import ORIG_ATTR_NAME


def _plane_basis(normal: Vector) -> tuple[Vector, Vector, Vector]:
    n = normal.normalized()
    ref = Vector((1.0, 0.0, 0.0)) if abs(n.x) < 0.8 else Vector((0.0, 1.0, 0.0))
    u = ref.cross(n).normalized()
    v = n.cross(u).normalized()
    return u, v, n


def _dedup_planes(
    raw_planes: Sequence[tuple[Vector, Vector]],
    dist_tol: float,
    dot_tol: float = 0.995,
) -> list[tuple[Vector, Vector]]:
    unique: list[tuple[Vector, Vector]] = []
    for pt, nrm in raw_planes:
        if nrm.length < 1e-6:
            continue
        nu = nrm.normalized()
        d = pt.dot(nu)
        found = False
        for upt, un in unique:
            dot = nu.dot(un)
            if abs(dot) >= dot_tol:
                ud = upt.dot(un)
                cmp_d = d if dot > 0.0 else -d
                if abs(cmp_d - ud) <= dist_tol:
                    found = True
                    break
        if not found:
            unique.append((pt.copy(), nu))
    return unique


def extract_cutter_planes_local(
    cutter_objs: Iterable[bpy.types.Object],
    target_matrix_world: Matrix,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> list[tuple[Vector, Vector]]:
    """Extract unique cutter face planes `(point, outward_normal)` in `target`'s local space."""
    inv_target = target_matrix_world.inverted_safe()
    normal_mat = inv_target.to_3x3().inverted_safe().transposed()
    raw: list[tuple[Vector, Vector]] = []
    max_span = 1.0

    for c_obj in cutter_objs:
        if c_obj is None or c_obj.type != "MESH" or c_obj.data is None:
            continue
        eval_mesh = None
        if depsgraph is not None and len(c_obj.modifiers) > 0:
            try:
                eval_obj = c_obj.evaluated_get(depsgraph)
                eval_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
                me = eval_mesh
            except Exception:
                me = c_obj.data
        else:
            me = c_obj.data

        try:
            # Skip ultra-dense displaced cutters here (only planar/faceted planes up to 256 polys)
            if len(me.polygons) == 0 or len(me.polygons) > 512:
                continue
            c_mw = c_obj.matrix_world
            c_nmat = c_mw.to_3x3().inverted_safe().transposed()
            for poly in me.polygons:
                if poly.area < 1e-7:
                    continue
                pw = c_mw @ poly.center
                nw = (c_nmat @ poly.normal).normalized()
                pl = inv_target @ pw
                nl = (normal_mat @ nw).normalized()
                max_span = max(max_span, pl.length)
                raw.append((pl, nl))
        finally:
            if eval_mesh is not None:
                try:
                    bpy.data.meshes.remove(eval_mesh)
                except Exception:
                    pass

    dist_tol = max(0.005, max_span * 1e-4)
    return _dedup_planes(raw, dist_tol=dist_tol)


def detect_existing_cap_planes(
    bm: bmesh.types.BMesh,
    orig_layer,
    mesh_diag: float,
) -> list[tuple[Vector, Vector]]:
    """Infer cut planes from existing cut cap faces (`fdt_orig < 0.5`) on an already-applied mesh."""
    raw: list[tuple[Vector, Vector]] = []
    if orig_layer is not None:
        for f in bm.faces:
            if f[orig_layer] < 0.5 and f.calc_area() > 1e-6:
                raw.append((f.calc_center_median(), f.normal.copy()))
    dist_tol = max(0.01, mesh_diag * 5e-4)
    return _dedup_planes(raw, dist_tol=dist_tol)


def _apply_world_box_uv_to_face(
    face: bmesh.types.BMFace,
    uv_layers: Sequence,
    matrix_world: Matrix,
    uv_scale: float,
) -> None:
    if not uv_layers:
        return
    nmat = matrix_world.to_3x3().inverted_safe().transposed()
    wn = (nmat @ face.normal).normalized()
    ax, ay, az = abs(wn.x), abs(wn.y), abs(wn.z)
    scale = float(uv_scale) if uv_scale > 1e-6 else 0.5

    for lp in face.loops:
        wp = matrix_world @ lp.vert.co
        if az >= ax and az >= ay:
            u, v = wp.x * scale, wp.y * scale
        elif ax >= ay:
            u, v = wp.y * scale, wp.z * scale
        else:
            u, v = wp.x * scale, wp.z * scale
        for uv_l in uv_layers:
            lp[uv_l].uv = (u, v)


def _ensure_cut_material_slot(mesh: bpy.types.Mesh, cut_material: bpy.types.Material | None) -> int:
    if cut_material is None:
        return 0
    for idx, m in enumerate(mesh.materials):
        if m == cut_material:
            return idx
    mesh.materials.append(cut_material)
    return len(mesh.materials) - 1


def _trace_closed_loops_from_edges(edges: Sequence[bmesh.types.BMEdge]) -> list[list[bmesh.types.BMVert]]:
    """Extract simple closed vertex cycles from a list of boundary edges."""
    adj: dict[bmesh.types.BMVert, list[bmesh.types.BMVert]] = {}
    edge_lookup: dict[tuple[int, int], bmesh.types.BMEdge] = {}
    for e in edges:
        v0, v1 = e.verts[0], e.verts[1]
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)
        k = (min(v0.index, v1.index), max(v0.index, v1.index))
        edge_lookup[k] = e

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[bmesh.types.BMVert]] = []

    for start_v, nbs in adj.items():
        if len(nbs) != 2:
            continue
        for first_nb in nbs:
            ek = (min(start_v.index, first_nb.index), max(start_v.index, first_nb.index))
            if ek in visited_edges:
                continue

            cycle = [start_v]
            prev = start_v
            curr = first_nb
            visited_edges.add(ek)
            closed = False

            for _ in range(len(edges) + 2):
                if curr == start_v:
                    closed = True
                    break
                cycle.append(curr)
                curr_nbs = adj.get(curr, ())
                if len(curr_nbs) != 2:
                    break
                nxt = curr_nbs[0] if curr_nbs[1] == prev else curr_nbs[1]
                nek = (min(curr.index, nxt.index), max(curr.index, nxt.index))
                if nek in visited_edges:
                    if nxt == start_v:
                        closed = True
                    break
                visited_edges.add(nek)
                prev, curr = curr, nxt

            if closed and len(cycle) >= 3:
                loops.append(cycle)

    return loops


def _loop_signed_area_2d(pts2: Sequence[Vector]) -> float:
    n = len(pts2)
    area = 0.0
    for i in range(n):
        a = pts2[i]
        b = pts2[(i + 1) % n]
        area += a.x * b.y - b.x * a.y
    return 0.5 * area


def _vote_loop_is_solid_cavity(
    cycle: list[bmesh.types.BMVert],
    u_axis: Vector,
    v_axis: Vector,
    origin: Vector,
    orig_layer,
    edge_lookup: dict[tuple[int, int], bmesh.types.BMEdge],
) -> tuple[bool, list[bmesh.types.BMVert]]:
    """Return `(is_solid_cavity, ccw_cycle)` using surrounding building face normals.

    Solid wall/floor/roof cavities have adjacent outward normals pointing AWAY from the loop
    (`-adj_face.normal` points INTO the loop). Hollow rooms, attics, and windows have adjacent
    wall normals pointing INTO the loop (`+adj_face.normal` points INTO the loop) and are skipped!
    """
    pts2 = [Vector(((v.co - origin).dot(u_axis), (v.co - origin).dot(v_axis), 0.0)) for v in cycle]
    area = _loop_signed_area_2d(pts2)
    if abs(area) < 1e-7:
        return False, cycle
    if area < 0.0:
        cycle = list(reversed(cycle))
        pts2 = list(reversed(pts2))

    solid_vote = 0.0
    air_vote = 0.0
    n = len(cycle)

    for i in range(n):
        va = cycle[i]
        vb = cycle[(i + 1) % n]
        k = (min(va.index, vb.index), max(va.index, vb.index))
        e = edge_lookup.get(k)
        if e is None or len(e.link_faces) != 1:
            continue
        adj_f = e.link_faces[0]
        pa = pts2[i]
        pb = pts2[(i + 1) % n]
        ed = pb - pa
        elen = ed.length
        if elen < 1e-7:
            continue
        # Left-hand inward normal for CCW 2D polygon points strictly INTO the hole interior
        into_hole = Vector((-ed.y / elen, ed.x / elen, 0.0))

        is_orig = (orig_layer is None) or (adj_f[orig_layer] >= 0.5)
        if not is_orig:
            # Adjacent face is ALREADY a cut cap on this plane: if cap centroid lies OUTSIDE the loop,
            # this loop is a missing pinhole/pocket inside the cut cap!
            fc = adj_f.calc_center_median()
            fc2 = Vector(((fc - origin).dot(u_axis), (fc - origin).dot(v_axis), 0.0))
            mid2 = (pa + pb) * 0.5
            to_cap = fc2 - mid2
            if to_cap.length > 1e-6:
                if to_cap.normalized().dot(into_hole) < -0.1:
                    solid_vote += elen
                else:
                    air_vote += elen
            continue

        # Original building face: `-adj_f.normal` points INTO the solid wall interior
        fn = adj_f.normal
        in_solid_2d = Vector((-fn.dot(u_axis), -fn.dot(v_axis), 0.0))
        if in_solid_2d.length < 0.15:
            continue
        in_solid_2d.normalize()
        dot = in_solid_2d.dot(into_hole)
        if dot > 0.15:
            solid_vote += elen * dot
        elif dot < -0.15:
            air_vote += elen * (-dot)

    is_solid = solid_vote > 1e-4 and solid_vote > air_vote * 1.15
    return is_solid, cycle


def _orient_desired_normal_for_cycle(
    bm: bmesh.types.BMesh,
    cycle: Sequence[bmesh.types.BMVert],
    p0: Vector,
    n_unit: Vector,
    orig_layer,
    fallback_n: Vector,
) -> Vector:
    """Determine true outward normal for a planar cap cycle from its adjacent faces.

    1) If adjacent to an existing coplanar cap face, match its normal.
    2) Otherwise, check which side of the plane the adjacent building faces sit on:
       if wall faces sit on the `-n_unit` side, the cap must face `+n_unit` (outward),
       and vice versa.
    """
    cap_n = Vector((0.0, 0.0, 0.0))
    side_sum = 0.0
    n = len(cycle)
    for i in range(n):
        va = cycle[i]
        vb = cycle[(i + 1) % n]
        e = bm.edges.get((va, vb))
        if e is None or len(e.link_faces) != 1:
            continue
        af = e.link_faces[0]
        elen = (vb.co - va.co).length
        if elen < 1e-7:
            continue
        if orig_layer is not None and af[orig_layer] < 0.5 and abs(af.normal.dot(n_unit)) > 0.90:
            cap_n += af.normal * elen
        else:
            side_sum += (af.calc_center_median() - p0).dot(n_unit) * elen

    if cap_n.length > 1e-5:
        return cap_n.normalized()
    if side_sum < -1e-6:
        return n_unit.copy()
    if side_sum > 1e-6:
        return -n_unit
    return fallback_n.normalized()


def _tag_new_cap_face(
    f: bmesh.types.BMFace,
    dn: Vector,
    cut_mat_slot: int,
    orig_layer,
    nrm_layer,
    uv_layers: Sequence,
    matrix_world: Matrix,
    uv_scale: float,
) -> None:
    """Assign Cut Material, Box UV, fdt_orig=0.0, and outward flat normal to a newly created cap face."""
    f.normal_update()
    if f.normal.dot(dn) < 0.0:
        f.normal_flip()
        f.normal_update()
    f.material_index = cut_mat_slot
    f.smooth = True
    if orig_layer is not None:
        f[orig_layer] = 0.0
    _apply_world_box_uv_to_face(f, uv_layers, matrix_world, uv_scale)
    if nrm_layer is not None:
        fn = f.normal
        for lp in f.loops:
            lp[nrm_layer] = (fn.x, fn.y, fn.z)


def _triangulate_and_create_cap_faces(
    bm: bmesh.types.BMesh,
    cycle: Sequence[bmesh.types.BMVert],
    desired_normal: Vector,
    u_axis: Vector,
    v_axis: Vector,
    origin: Vector,
    cut_mat_slot: int,
    orig_layer,
    nrm_layer,
    uv_layers: Sequence,
    matrix_world: Matrix,
    uv_scale: float,
) -> int:
    """Create and BEAUTY-triangulate a planar closed cap loop with zero collinear boundary cracks."""
    if len(cycle) < 3:
        return 0
    dn = desired_normal.normalized()

    # Orient cycle winding to match desired_normal before creating face
    pts2 = [Vector(((v.co - origin).dot(u_axis), (v.co - origin).dot(v_axis), 0.0)) for v in cycle]
    area = _loop_signed_area_2d(pts2)
    basis_dot = u_axis.cross(v_axis).dot(dn)
    ordered = list(cycle)
    if (area < 0.0) != (basis_dot < 0.0):
        ordered.reverse()

    # Primary path: exact N-gon + Blender BEAUTY triangulation.
    # Preserves 100% of collinear boundary vertices (T-junctions) without dropping boundary edges!
    try:
        ngon = bm.faces.new(ordered)
    except ValueError:
        ngon = None

    if ngon is not None:
        ngon.normal_update()
        if ngon.normal.dot(dn) < 0.0:
            ngon.normal_flip()
            ngon.normal_update()
        if len(ngon.verts) > 3:
            tri_res = bmesh.ops.triangulate(
                bm, faces=[ngon], quad_method="BEAUTY", ngon_method="BEAUTY"
            )
            new_faces = [f for f in tri_res.get("faces", []) if f.is_valid]
            if not new_faces and ngon.is_valid:
                new_faces = [ngon]
        else:
            new_faces = [ngon]
        for f in new_faces:
            _tag_new_cap_face(f, dn, cut_mat_slot, orig_layer, nrm_layer, uv_layers, matrix_world, uv_scale)
        return len(new_faces)

    # Fallback path: 2D ear-clipping via tessellate_polygon (for non-simple / pinch loops)
    pts2_ord = [Vector(((v.co - origin).dot(u_axis), (v.co - origin).dot(v_axis), 0.0)) for v in ordered]
    tris_idx = tessellate_polygon([pts2_ord])
    if not tris_idx:
        return 0

    created = 0
    for i0, i1, i2 in tris_idx:
        v0, v1, v2 = ordered[i0], ordered[i1], ordered[i2]
        if v0 == v1 or v1 == v2 or v2 == v0:
            continue
        tri_n = (v1.co - v0.co).cross(v2.co - v0.co)
        if tri_n.length < 1e-9:
            continue
        verts = (v0, v2, v1) if tri_n.dot(dn) < 0.0 else (v0, v1, v2)
        try:
            f = bm.faces.new(verts)
        except ValueError:
            continue
        _tag_new_cap_face(f, dn, cut_mat_slot, orig_layer, nrm_layer, uv_layers, matrix_world, uv_scale)
        created += 1

    return created


def fill_unclosed_cut_planes_in_bmesh(
    bm: bmesh.types.BMesh,
    matrix_world: Matrix,
    cut_mat_slot: int,
    uv_scale: float,
    planes: Sequence[tuple[Vector, Vector]],
    is_cut_main: bool = False,
) -> tuple[int, int]:
    """Fill unfilled solid cavities strictly on `planes` inside an already-open BMesh.

    Returns `(num_loops_filled, num_tris_created)`.
    """
    if len(bm.verts) == 0 or not planes:
        return 0, 0

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Compute mesh diagonal for scale-adaptive plane tolerance (works on cm & meter assets!)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    diag = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))).length
    plane_tol = max(0.008, diag * 4e-4)

    orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
    if orig_layer is None:
        orig_layer = bm.faces.layers.float.new(ORIG_ATTR_NAME)
    nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)
    uv_layers = list(bm.loops.layers.uv.values())

    open_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    if not open_edges:
        return 0, 0

    bm.verts.index_update()
    edge_lookup = {
        (min(e.verts[0].index, e.verts[1].index), max(e.verts[0].index, e.verts[1].index)): e
        for e in open_edges
    }

    loops_filled = 0
    tris_created = 0

    for p0, pn in planes:
        u_ax, v_ax, n_unit = _plane_basis(pn)
        plane_edges = [
            e
            for e in open_edges
            if e.is_valid
            and len(e.link_faces) == 1
            and abs((e.verts[0].co - p0).dot(n_unit)) <= plane_tol
            and abs((e.verts[1].co - p0).dot(n_unit)) <= plane_tol
        ]
        if len(plane_edges) < 3:
            continue

        cycles = _trace_closed_loops_from_edges(plane_edges)
        if not cycles:
            continue

        # Expected fallback outward normal (flipped for `_Cut` main mesh cavity, outward for `_FDST_ID-*` piece)
        fallback_n = -n_unit if is_cut_main else n_unit
        # If existing cap faces exist on this plane, prefer their orientation as fallback
        for f in bm.faces:
            if orig_layer is not None and f[orig_layer] < 0.5 and abs(f.normal.dot(n_unit)) > 0.95:
                if abs((f.calc_center_median() - p0).dot(n_unit)) <= plane_tol * 2.0:
                    fallback_n = f.normal.normalized()
                    break

        for cycle in cycles:
            is_solid, ccw_cycle = _vote_loop_is_solid_cavity(
                cycle, u_ax, v_ax, p0, orig_layer, edge_lookup
            )
            if not is_solid:
                continue
            desired_n = _orient_desired_normal_for_cycle(
                bm, ccw_cycle, p0, n_unit, orig_layer, fallback_n
            )
            n_new = _triangulate_and_create_cap_faces(
                bm=bm,
                cycle=ccw_cycle,
                desired_normal=desired_n,
                u_axis=u_ax,
                v_axis=v_ax,
                origin=p0,
                cut_mat_slot=cut_mat_slot,
                orig_layer=orig_layer,
                nrm_layer=nrm_layer,
                uv_layers=uv_layers,
                matrix_world=matrix_world,
                uv_scale=uv_scale,
            )
            if n_new > 0:
                loops_filled += 1
                tris_created += n_new

    return loops_filled, tris_created


def extract_cutter_bboxes_local(
    cutter_objs: Sequence[bpy.types.Object],
    target_matrix_world: Matrix,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> list[tuple[Vector, Vector]]:
    """Return local-space `(bb_min, bb_max)` AABBs of each cutter in `target_matrix_world` space."""
    if not cutter_objs:
        return []
    try:
        inv_target = target_matrix_world.inverted()
    except ValueError:
        return []

    bboxes: list[tuple[Vector, Vector]] = []
    for c_obj in cutter_objs:
        if c_obj is None or c_obj.type != "MESH" or c_obj.data is None:
            continue
        eval_c = c_obj.evaluated_get(depsgraph) if depsgraph is not None else c_obj
        me = eval_c.to_mesh() if depsgraph is not None else c_obj.data
        try:
            if me is None or len(me.vertices) == 0:
                continue
            rel_m = inv_target @ eval_c.matrix_world
            pts = [rel_m @ v.co for v in me.vertices]
            bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
            bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
            bboxes.append((bmin, bmax))
        finally:
            if depsgraph is not None and me is not None:
                eval_c.to_mesh_clear()
    return bboxes


def _create_welded_liner_faces(
    bm: bmesh.types.BMesh,
    target_faces: Sequence[bmesh.types.BMFace],
    diag: float,
    cut_mat_slot: int,
    orig_layer,
    nrm_layer,
    uv_layers: Sequence,
    matrix_world: Matrix,
    uv_scale: float,
    clamp_planes: Sequence[tuple[Vector, Vector]] = (),
) -> int:
    """Create welded, Z-fighting-free inward-offset liner faces for single-sided exposed backfaces.

    Uses even-thickness miter vertex offset so adjacent walls/ceilings stay tightly welded
    at corners with zero cracks, assigns Cut Material + World-Space Box UV + flat inward normal,
    and clamps vertices so they never poke past cut planes.
    """
    if not target_faces:
        return 0

    offset = max(0.002, diag * 4e-4)  # ~2.8 mm on 7m asset, ~2 mm on smaller assets
    eps = max(0.05, diag * 2e-4)

    vert_nrms: dict[bmesh.types.BMVert, list[Vector]] = {}
    for f in target_faces:
        if not f.is_valid:
            continue
        fn = f.normal.copy()
        if fn.length < 1e-6:
            continue
        for v in f.verts:
            vert_nrms.setdefault(v, []).append(fn)

    new_vmap: dict[bmesh.types.BMVert, bmesh.types.BMVert] = {}
    for v, nlist in vert_nrms.items():
        avg = Vector((0.0, 0.0, 0.0))
        for n in nlist:
            avg += n
        if avg.length < 1e-6:
            avg = nlist[0].copy()
        else:
            avg.normalize()
        # Even-thickness miter scale (clamped so sharp corners never spike)
        min_dot = min(max(0.55, avg.dot(n)) for n in nlist)
        new_co = v.co - avg * (offset / min_dot)
        # Never push a perpendicular wall liner vertex outward past a cut plane
        for p0, out_n in clamp_planes:
            if abs(avg.dot(out_n)) > 0.5:
                continue
            d_old = (v.co - p0).dot(out_n)
            d_new = (new_co - p0).dot(out_n)
            if d_old <= eps * 2.0 and d_new > d_old:
                new_co -= out_n * (d_new - d_old)
        new_vmap[v] = bm.verts.new(new_co)

    created = 0
    for f in target_faces:
        if not f.is_valid:
            continue
        nverts = [new_vmap[v] for v in reversed(f.verts) if v in new_vmap]
        if len(nverts) < 3:
            continue
        desired_n = -f.normal.normalized()
        try:
            nf = bm.faces.new(nverts)
        except ValueError:
            continue
        _tag_new_cap_face(
            nf,
            desired_n,
            cut_mat_slot,
            orig_layer,
            nrm_layer,
            uv_layers,
            matrix_world,
            uv_scale,
        )
        created += 1

    return created


def line_exposed_backfaces_in_bmesh(
    bm: bmesh.types.BMesh,
    matrix_world: Matrix,
    cut_mat_slot: int,
    uv_scale: float,
    planes: Sequence[tuple[Vector, Vector]],
    is_cut_main: bool = False,
    planes_are_cap_outward: bool = False,
    cutter_bboxes_local: Sequence[tuple[Vector, Vector]] | None = None,
) -> int:
    """Detect single-sided original faces whose BACKSIDE is exposed through a cut opening
    (such as sealed attics, roof undersides, and partition walls in game buildings with
    Backface Culling) and create welded, Box-UV'd inner liner faces so nothing is see-through.

    - Never touches intact double-sided walls (rays hit the front of the inner wall).
    - Never closes open windows, doors, or hollow rooms (rays pass to air or hit front faces).
    - 100% idempotent (re-running hits the newly created front-facing liner -> 0 new faces).
    """
    if len(bm.verts) < 3 or len(bm.faces) == 0:
        return 0

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    diag = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))).length
    if diag < 1e-5:
        return 0

    orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
    if orig_layer is None:
        orig_layer = bm.faces.layers.float.new(ORIG_ATTR_NAME)
        for f in bm.faces:
            f[orig_layer] = 1.0
    nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)
    uv_layers = list(bm.loops.layers.uv.values())

    active_planes = list(planes)
    if not active_planes:
        active_planes = detect_existing_cap_planes(bm, orig_layer, diag)
        planes_are_cap_outward = True
    if not active_planes:
        return 0

    bvh = BVHTree.FromBMesh(bm)
    eps = max(0.05, diag * 2e-4)
    plane_tol = max(1.0, diag * 0.005)
    offset = max(0.002, diag * 4e-4)
    double_side_chk_eps = max(1e-4, offset * 0.02)
    double_side_max_dist = max(1.5, diag * 0.004)

    # Bound rays strictly to the cut region on large `_Cut` meshes:
    # - Prefer exact cutter AABBs during Apply;
    # - In Post-Apply Object Mode (no cutter object), infer per-plane cut bounds strictly from
    #   coplanar cap faces lying ON that cut plane (so inner liner faces never expand the box!).
    effective_bboxes = list(cutter_bboxes_local) if cutter_bboxes_local else []
    bbox_pad = max(2.0, diag * 0.01)

    def _in_any_cutter_bbox(pt: Vector) -> bool:
        if not effective_bboxes:
            return True
        for bmin, bmax in effective_bboxes:
            if (
                bmin.x - bbox_pad <= pt.x <= bmax.x + bbox_pad
                and bmin.y - bbox_pad <= pt.y <= bmax.y + bbox_pad
                and bmin.z - bbox_pad <= pt.z <= bmax.z + bbox_pad
            ):
                return True
        return False

    oriented_out_planes: list[tuple[Vector, Vector]] = []
    hits_map: dict[int, int] = {}

    for p0, pn in active_planes:
        n_unit = pn.normalized()
        if planes_are_cap_outward:
            out_n = n_unit
            inward = -n_unit
        else:
            # Cutter plane normal points OUT of cutter:
            # - For Main (_Cut = building - cutter): inward into remaining building is +n_unit
            # - For Piece (_FDST_ID-* = building & cutter): inward into piece is -n_unit
            inward = n_unit if is_cut_main else -n_unit
            out_n = -inward

        oriented_out_planes.append((p0, out_n))

        plane_cap_bbox: tuple[Vector, Vector] | None = None
        plane_cap_pad = bbox_pad
        if not effective_bboxes and is_cut_main:
            cap_pts = [
                v.co
                for f in bm.faces
                if f[orig_layer] < 0.5
                and abs(f.normal.dot(out_n)) > 0.90
                and abs((f.calc_center_median() - p0).dot(out_n)) <= plane_tol
                for v in f.verts
            ]
            if cap_pts:
                cmin = Vector((min(p.x for p in cap_pts), min(p.y for p in cap_pts), min(p.z for p in cap_pts)))
                cmax = Vector((max(p.x for p in cap_pts), max(p.y for p in cap_pts), max(p.z for p in cap_pts)))
                plane_cap_pad = max(10.0, (cmax - cmin).length * 0.15)
                plane_cap_bbox = (cmin, cmax)

        def _in_plane_region(pt: Vector) -> bool:
            if effective_bboxes:
                return _in_any_cutter_bbox(pt)
            if plane_cap_bbox is not None:
                bmin, bmax = plane_cap_bbox
                return (
                    bmin.x - plane_cap_pad <= pt.x <= bmax.x + plane_cap_pad
                    and bmin.y - plane_cap_pad <= pt.y <= bmax.y + plane_cap_pad
                    and bmin.z - plane_cap_pad <= pt.z <= bmax.z + plane_cap_pad
                )
            return True

        near_v = [
            vt
            for vt in bm.verts
            if abs((vt.co - p0).dot(n_unit)) <= plane_tol and _in_plane_region(vt.co)
        ]
        if len(near_v) < 3:
            continue

        u_ax, v_ax, _ = _plane_basis(inward)
        us = [(vt.co - p0).dot(u_ax) for vt in near_v]
        vs = [(vt.co - p0).dot(v_ax) for vt in near_v]
        u0, u1 = min(us), max(us)
        v0, v1 = min(vs), max(vs)
        span_u = u1 - u0
        span_v = v1 - v0
        if span_u < eps or span_v < eps:
            continue

        nu = max(8, min(44, int(span_u / max(2.0, span_u / 36.0))))
        nv = max(8, min(44, int(span_v / max(2.0, span_v / 36.0))))

        # 5-ray fan per grid sample catches both perpendicular back walls AND angled side walls/ceilings
        dirs = (
            inward,
            (inward + 0.4 * u_ax).normalized(),
            (inward - 0.4 * u_ax).normalized(),
            (inward + 0.4 * v_ax).normalized(),
            (inward - 0.4 * v_ax).normalized(),
        )
        max_dist = diag * 0.9

        for iu in range(nu):
            uu = u0 + span_u * (iu + 0.5) / nu
            for iv in range(nv):
                vv = v0 + span_v * (iv + 0.5) / nv
                origin = p0 + u_ax * uu + v_ax * vv - inward * eps
                for d in dirs:
                    _loc, nrm, idx, dist = bvh.ray_cast(origin, d, max_dist)
                    if idx is None or dist < eps * 2.0:
                        continue
                    # Backface hit: ray enters from behind the face (same hemisphere as face normal)
                    if d.dot(nrm) > 0.08:
                        f = bm.faces[idx]
                        if f[orig_layer] >= 0.5 and f.calc_area() > 1e-4:
                            hits_map[idx] = hits_map.get(idx, 0) + 1

    if not hits_map:
        return 0

    # Confirm truly single-sided faces (no close opposing inner face already backing it)
    def _has_close_inner_face(face: bmesh.types.BMFace) -> bool:
        c = face.calc_center_median()
        _loc2, nrm2, idx2, _dist2 = bvh.ray_cast(
            c - face.normal * double_side_chk_eps,
            -face.normal,
            double_side_max_dist,
        )
        return idx2 is not None and idx2 != face.index and nrm2.dot(face.normal) < -0.7

    confirmed_ids: set[int] = set()
    for fi, hit_cnt in hits_map.items():
        if hit_cnt < 2:
            continue
        f = bm.faces[fi]
        if _has_close_inner_face(f):
            continue
        confirmed_ids.add(fi)

    if not confirmed_ids:
        return 0

    # Catch adjacent coplanar sliver triangles on the same wall that got >= 1 hit
    for fi, hit_cnt in hits_map.items():
        if fi in confirmed_ids or hit_cnt < 1:
            continue
        f = bm.faces[fi]
        if _has_close_inner_face(f):
            continue
        for e in f.edges:
            if any(
                nf.index in confirmed_ids and nf.normal.dot(f.normal) > 0.96
                for nf in e.link_faces
            ):
                confirmed_ids.add(fi)
                break

    target_faces = [bm.faces[fi] for fi in sorted(confirmed_ids)]
    return _create_welded_liner_faces(
        bm=bm,
        target_faces=target_faces,
        diag=diag,
        cut_mat_slot=cut_mat_slot,
        orig_layer=orig_layer,
        nrm_layer=nrm_layer,
        uv_layers=uv_layers,
        matrix_world=matrix_world,
        uv_scale=uv_scale,
        clamp_planes=oriented_out_planes,
    )


def _store_existing_corner_normals_attr(mesh: bpy.types.Mesh) -> None:
    """Snapshot `mesh.corner_normals` into temp `fdt_nrm` attribute before BMesh editing."""
    nl = len(mesh.loops)
    if nl == 0:
        return
    nrm_flat = np.empty(nl * 3, dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", nrm_flat)
    attr = mesh.attributes.get(rust_bridge.PRESERVED_NRM_ATTR)
    if attr is not None and (attr.data_type != "FLOAT_VECTOR" or attr.domain != "CORNER"):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(name=rust_bridge.PRESERVED_NRM_ATTR, type="FLOAT_VECTOR", domain="CORNER")
    attr.data.foreach_set("vector", nrm_flat)


def _restore_normals_and_sharp_edges_after_fill(mesh: bpy.types.Mesh) -> None:
    """Restore preserved Custom Split Normals + Sharp Edges + Cut Rim Freestyle Edges after filling."""
    if len(mesh.polygons) == 0:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
    nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)

    border_edges: set[int] = set()
    sharp_edges: set[int] = set()

    for idx, edge in enumerate(bm.edges):
        if len(edge.link_faces) != 2:
            continue
        f0, f1 = edge.link_faces[0], edge.link_faces[1]
        is_border = False
        if orig_layer is not None:
            a = f0[orig_layer] >= 0.5
            b = f1[orig_layer] >= 0.5
            if a != b:
                is_border = True
                border_edges.add(idx)
                sharp_edges.add(idx)
        if not is_border and nrm_layer is not None:
            for sv in (edge.verts[0], edge.verts[1]):
                lp0 = next((lp for lp in f0.loops if lp.vert == sv), None)
                lp1 = next((lp for lp in f1.loops if lp.vert == sv), None)
                if lp0 is not None and lp1 is not None:
                    n0, n1 = lp0[nrm_layer], lp1[nrm_layer]
                    if n0.length > 1e-5 and n1.length > 1e-5 and n0.normalized().dot(n1.normalized()) < 0.995:
                        sharp_edges.add(idx)
                        break

    bm.free()

    ne = len(mesh.edges)
    if border_edges and ne > 0:
        fs_attr = mesh.attributes.get("freestyle_edge")
        if fs_attr is None:
            fs_attr = mesh.attributes.new(name="freestyle_edge", type="BOOLEAN", domain="EDGE")
        flags = np.zeros(ne, dtype=bool)
        flags[list(border_edges)] = True
        fs_attr.data.foreach_set("value", flags)

    if sharp_edges and ne > 0:
        sh_attr = mesh.attributes.get("sharp_edge")
        if sh_attr is not None and (sh_attr.data_type != "BOOLEAN" or sh_attr.domain != "EDGE"):
            mesh.attributes.remove(sh_attr)
            sh_attr = None
        if sh_attr is None:
            sh_attr = mesh.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
        sflags = np.zeros(ne, dtype=bool)
        sflags[list(sharp_edges)] = True
        sh_attr.data.foreach_set("value", sflags)

    nrm_attr = mesh.attributes.get(rust_bridge.PRESERVED_NRM_ATTR)
    if nrm_attr is not None and nrm_attr.domain == "CORNER" and len(mesh.loops) > 0:
        try:
            nl = len(mesh.loops)
            nrm_flat = np.empty(nl * 3, dtype=np.float32)
            nrm_attr.data.foreach_get("vector", nrm_flat)
            nrm_3 = nrm_flat.reshape(-1, 3)
            lens = np.linalg.norm(nrm_3, axis=1)
            good = lens > 1e-5
            nrm_3[good] /= lens[good, None]
            if not np.all(good):
                for poly in mesh.polygons:
                    pn = poly.normal
                    for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        if not good[li]:
                            nrm_3[li] = (pn.x, pn.y, pn.z)
            mesh.attributes.remove(nrm_attr)
            mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
            mesh.normals_split_custom_set(nrm_3)
        except Exception:
            pass


def _best_fit_plane_normal(verts: Sequence[bmesh.types.BMVert]) -> tuple[Vector, Vector]:
    """Compute Newell plane normal & centroid for an ordered vertex cycle."""
    n = len(verts)
    center = Vector((0.0, 0.0, 0.0))
    nx = ny = nz = 0.0
    for i in range(n):
        p0 = verts[i].co
        p1 = verts[(i + 1) % n].co
        center += p0
        nx += (p0.y - p1.y) * (p0.z + p1.z)
        ny += (p0.z - p1.z) * (p0.x + p1.x)
        nz += (p0.x - p1.x) * (p0.y + p1.y)
    center /= max(1, n)
    norm = Vector((nx, ny, nz))
    if norm.length < 1e-8:
        norm = Vector((0.0, 0.0, 1.0))
    else:
        norm.normalize()
    return center, norm


def manual_fill_cut_caps(
    context: bpy.types.Context,
    cut_material: bpy.types.Material | None,
    uv_scale: float,
) -> tuple[bool, str, int]:
    """1-click Manual Fill Cut Cap supporting BOTH Edit Mode (selected edges/verts) and Object Mode (auto cut-plane scan)."""
    active = context.active_object
    was_edit = active is not None and active.type == "MESH" and active.mode == "EDIT"

    if was_edit:
        # Step 1: Temporarily switch to Object Mode briefly so we can snapshot exact corner_normals
        # cleanly, then run targeted fill on selected boundary edges, and return to Edit Mode!
        obj = active
        me = obj.data
        sel_edge_indices: list[int] = []

        bm_ed = bmesh.from_edit_mesh(me)
        bm_ed.edges.ensure_lookup_table()
        bm_ed.faces.ensure_lookup_table()
        sel_edge_indices = [
            e.index for e in bm_ed.edges if e.select and len(e.link_faces) == 1
        ]
        if not sel_edge_indices:
            # Fallback: open edges whose both endpoints are selected vertices
            sel_edge_indices = [
                e.index
                for e in bm_ed.edges
                if e.verts[0].select and e.verts[1].select and len(e.link_faces) == 1
            ]
        sel_face_indices = (
            [f.index for f in bm_ed.faces if f.select] if not sel_edge_indices else []
        )
        if not sel_edge_indices and not sel_face_indices:
            return (
                False,
                "กรุณาเลือกขอบรูที่เปิดอยู่ (Open Boundary Edges) หรือเลือกผิวที่มองทะลุใน Edit Mode",
                0,
            )

        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            _store_existing_corner_normals_attr(me)
            cut_slot = _ensure_cut_material_slot(me, cut_material)

            bm = bmesh.new()
            bm.from_mesh(me)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
            if orig_layer is None:
                orig_layer = bm.faces.layers.float.new(ORIG_ATTR_NAME)
                for f in bm.faces:
                    f[orig_layer] = 1.0
            nrm_layer = bm.loops.layers.float_vector.get(rust_bridge.PRESERVED_NRM_ATTR)
            uv_layers = list(bm.loops.layers.uv.values())

            total_tris = 0
            filled_loops = 0
            lined_faces = 0

            if sel_edge_indices:
                sel_edges = [bm.edges[i] for i in sel_edge_indices if 0 <= i < len(bm.edges)]
                bm.verts.index_update()
                cycles = _trace_closed_loops_from_edges(sel_edges)

                if cycles:
                    for cyc in cycles:
                        p0, pn = _best_fit_plane_normal(cyc)
                        u_ax, v_ax, n_unit = _plane_basis(pn)
                        desired_n = _orient_desired_normal_for_cycle(
                            bm, cyc, p0, n_unit, orig_layer, n_unit
                        )
                        n_new = _triangulate_and_create_cap_faces(
                            bm=bm,
                            cycle=cyc,
                            desired_normal=desired_n,
                            u_axis=u_ax,
                            v_axis=v_ax,
                            origin=p0,
                            cut_mat_slot=cut_slot,
                            orig_layer=orig_layer,
                            nrm_layer=nrm_layer,
                            uv_layers=uv_layers,
                            matrix_world=obj.matrix_world,
                            uv_scale=uv_scale,
                        )
                        if n_new > 0:
                            filled_loops += 1
                            total_tris += n_new
                else:
                    # Open partial chain selected: use contextual_create + triangulate only new faces
                    existing_faces = set(bm.faces)
                    res = bmesh.ops.contextual_create(bm, geom=sel_edges)
                    new_faces = [f for f in res.get("faces", []) if f not in existing_faces]
                    if new_faces:
                        tri_res = bmesh.ops.triangulate(bm, faces=new_faces)
                        final_new = [f for f in tri_res.get("faces", []) if f.is_valid]
                        if not final_new:
                            final_new = [f for f in new_faces if f.is_valid]
                        for f in final_new:
                            f.material_index = cut_slot
                            f.smooth = True
                            f.normal_update()
                            f[orig_layer] = 0.0
                            _apply_world_box_uv_to_face(f, uv_layers, obj.matrix_world, uv_scale)
                            if nrm_layer is not None:
                                fn = f.normal
                                for lp in f.loops:
                                    lp[nrm_layer] = (fn.x, fn.y, fn.z)
                            total_tris += 1
                        filled_loops = 1
            elif sel_face_indices:
                xs = [v.co.x for v in bm.verts]
                ys = [v.co.y for v in bm.verts]
                zs = [v.co.z for v in bm.verts]
                diag = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))).length
                target_faces = [bm.faces[i] for i in sel_face_indices if 0 <= i < len(bm.faces)]
                lined_faces = _create_welded_liner_faces(
                    bm=bm,
                    target_faces=target_faces,
                    diag=diag,
                    cut_mat_slot=cut_slot,
                    orig_layer=orig_layer,
                    nrm_layer=nrm_layer,
                    uv_layers=uv_layers,
                    matrix_world=obj.matrix_world,
                    uv_scale=uv_scale,
                )
                total_tris = lined_faces

            bm.to_mesh(me)
            bm.free()
            me.update()
            _restore_normals_and_sharp_edges_after_fill(me)
        finally:
            bpy.ops.object.mode_set(mode="EDIT")

        if total_tris == 0:
            return False, "ไม่สามารถปิดช่องจากส่วนที่เลือกได้ (ลองเลือกขอบรอบรูให้ครบวง หรือเลือกผิวที่ทะลุ)", 0
        if lined_faces > 0:
            return (
                True,
                f"สร้างผิวปิดด้านใน {lined_faces} หน้า พร้อม Cut Material + Box UV แล้ว",
                lined_faces,
            )
        return (
            True,
            f"ปิดหน้าตัดเพิ่ม {total_tris} หน้า ({filled_loops} ช่อง) พร้อม Cut Material + Box UV + Sharp Edge แล้ว",
            total_tris,
        )

    # Object Mode: auto-scan selected mesh objects for BOTH:
    # 1) Unfilled solid cavities strictly on cut planes, AND
    # 2) Exposed single-sided backfaces visible through cut openings (attics/walls)
    targets = [o for o in context.selected_objects if o.type == "MESH" and o.data is not None]
    if not targets and active is not None and active.type == "MESH" and active.data is not None:
        targets = [active]
    if not targets:
        return False, "กรุณาเลือกโมเดลที่ตัดแล้วก่อนกด Fill Cut Cap", 0

    total_loops = 0
    total_cap_tris = 0
    total_lined_tris = 0
    for obj in targets:
        me = obj.data
        if len(me.polygons) == 0:
            continue
        _store_existing_corner_normals_attr(me)
        cut_slot = _ensure_cut_material_slot(me, cut_material)

        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        orig_layer = bm.faces.layers.float.get(ORIG_ATTR_NAME)
        xs = [v.co.x for v in bm.verts]
        ys = [v.co.y for v in bm.verts]
        zs = [v.co.z for v in bm.verts]
        diag = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))).length
        planes = detect_existing_cap_planes(bm, orig_layer, diag)

        lf, tc = fill_unclosed_cut_planes_in_bmesh(
            bm=bm,
            matrix_world=obj.matrix_world,
            cut_mat_slot=cut_slot,
            uv_scale=uv_scale,
            planes=planes,
            is_cut_main=obj.name.endswith("_Cut"),
        )
        bm.normal_update()
        lc = line_exposed_backfaces_in_bmesh(
            bm=bm,
            matrix_world=obj.matrix_world,
            cut_mat_slot=cut_slot,
            uv_scale=uv_scale,
            planes=planes,
            is_cut_main=obj.name.endswith("_Cut"),
            planes_are_cap_outward=True,
        )

        if (tc + lc) > 0:
            bm.to_mesh(me)
            bm.free()
            me.update()
            _restore_normals_and_sharp_edges_after_fill(me)
            total_loops += lf
            total_cap_tris += tc
            total_lined_tris += lc
        else:
            bm.free()
            # Clean up snapshot temp attr if nothing changed
            tmp_a = me.attributes.get(rust_bridge.PRESERVED_NRM_ATTR)
            if tmp_a is not None:
                me.attributes.remove(tmp_a)

    grand_total = total_cap_tris + total_lined_tris
    if grand_total == 0:
        return (
            True,
            "ไม่พบช่องโหว่หรือผิวทะลุบนแนวตัดแล้ว (ถ้าจะปิดจุดเฉพาะ ให้เข้า Edit Mode แล้วเลือกขอบหรือผิวแล้วกด Fill Cut Cap)",
            0,
        )
    parts = []
    if total_cap_tris > 0:
        parts.append(f"ปิดหน้าตัด {total_cap_tris} หน้า ({total_loops} ช่อง)")
    if total_lined_tris > 0:
        parts.append(f"ปิดผิวด้านในที่ทะลุ {total_lined_tris} หน้า")
    return True, f"{' + '.join(parts)} พร้อม Cut Material + Box UV + Sharp Edge แล้ว", grand_total
