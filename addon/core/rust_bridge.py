"""High-speed ctypes + NumPy bridge to `bin/fdt_bool_engine.dll`.

Designed as a reusable core module inside FullDestruc_Tool for both Apple Bite
and future Fracture workflows:
- Loads `fdt_bool_engine.dll` lazily with automatic graceful fallback if missing.
- Transfers Blender Mesh geometry via `foreach_get` / `foreach_set` + NumPy
  (zero Python per-vertex/per-loop iteration).
- Maintains stateful Rust session handles with per-target prefix caching so
  real-time cutter dragging only recomputes from the moving cutter onward.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Sequence

import bpy
import bmesh
import numpy as np
from mathutils import Matrix

from . import profiler
from .geonodes import ORIG_ATTR_NAME

PRESERVED_NRM_ATTR = "fdt_nrm"

# ---------------------------------------------------------------------------
# C-ABI Struct Mirrors
# ---------------------------------------------------------------------------


class FdtMeshInput(ctypes.Structure):
    _fields_ = [
        ("num_verts", ctypes.c_uint32),
        ("verts_xyz", ctypes.POINTER(ctypes.c_float)),
        ("num_loops", ctypes.c_uint32),
        ("loop_verts", ctypes.POINTER(ctypes.c_uint32)),
        ("num_polys", ctypes.c_uint32),
        ("poly_starts", ctypes.POINTER(ctypes.c_uint32)),
        ("mat_ids", ctypes.POINTER(ctypes.c_int32)),
        ("num_uv_layers", ctypes.c_uint32),
        ("uv_data", ctypes.POINTER(ctypes.c_float)),
        ("world_matrix", ctypes.POINTER(ctypes.c_float)),
        ("loop_normals", ctypes.POINTER(ctypes.c_float)),
    ]


class FdtMeshOutput(ctypes.Structure):
    _fields_ = [
        ("num_verts", ctypes.c_uint32),
        ("verts_xyz", ctypes.POINTER(ctypes.c_float)),
        ("num_loops", ctypes.c_uint32),
        ("loop_verts", ctypes.POINTER(ctypes.c_uint32)),
        ("num_polys", ctypes.c_uint32),
        ("poly_starts", ctypes.POINTER(ctypes.c_uint32)),
        ("mat_ids", ctypes.POINTER(ctypes.c_int32)),
        ("is_cut_face", ctypes.POINTER(ctypes.c_uint8)),
        ("num_uv_layers", ctypes.c_uint32),
        ("uv_data", ctypes.POINTER(ctypes.c_float)),
        ("num_edges", ctypes.c_uint32),
        ("edge_verts", ctypes.POINTER(ctypes.c_uint32)),
        ("loop_edges", ctypes.POINTER(ctypes.c_uint32)),
        ("loop_normals", ctypes.POINTER(ctypes.c_float)),
    ]


class FdtEvalOutput(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("dirty_from_idx", ctypes.c_uint32),
        ("remaining", FdtMeshOutput),
        ("num_pieces", ctypes.c_uint32),
        ("pieces", ctypes.POINTER(FdtMeshOutput)),
    ]


class FdtGpuSubMeshV130(ctypes.Structure):
    _fields_ = [
        ("mat_id", ctypes.c_int32),
        ("num_tris", ctypes.c_uint32),
        ("pos_xyz", ctypes.POINTER(ctypes.c_float)),
        ("nor_xyz", ctypes.POINTER(ctypes.c_float)),
        ("uv_xy", ctypes.POINTER(ctypes.c_float)),
    ]


class FdtGpuMeshDataV130(ctypes.Structure):
    _fields_ = [
        ("num_submeshes", ctypes.c_uint32),
        ("submeshes", ctypes.POINTER(FdtGpuSubMeshV130)),
    ]


class FdtGpuPreviewOutputV130(ctypes.Structure):
    _fields_ = [
        ("job_id", ctypes.c_uint64),
        ("compute_ms", ctypes.c_float),
        ("dirty_from_idx", ctypes.c_uint32),
        ("remaining", FdtGpuMeshDataV130),
        ("num_pieces", ctypes.c_uint32),
        ("pieces", ctypes.POINTER(FdtGpuMeshDataV130)),
    ]


class FdtGpuSubMesh(ctypes.Structure):
    _fields_ = [
        ("mat_id", ctypes.c_int32),
        ("num_tris", ctypes.c_uint32),
        ("pos_xyz", ctypes.POINTER(ctypes.c_float)),
        ("nor_xyz", ctypes.POINTER(ctypes.c_float)),
        ("uv_xy", ctypes.POINTER(ctypes.c_float)),
        ("uv1_xy", ctypes.POINTER(ctypes.c_float)),
    ]


class FdtGpuMeshData(ctypes.Structure):
    _fields_ = [
        ("num_submeshes", ctypes.c_uint32),
        ("submeshes", ctypes.POINTER(FdtGpuSubMesh)),
    ]


class FdtGpuPreviewOutput(ctypes.Structure):
    _fields_ = [
        ("job_id", ctypes.c_uint64),
        ("compute_ms", ctypes.c_float),
        ("dirty_from_idx", ctypes.c_uint32),
        ("remaining", FdtGpuMeshData),
        ("num_pieces", ctypes.c_uint32),
        ("pieces", ctypes.POINTER(FdtGpuMeshData)),
    ]


class FdtGpuChunkDelta(ctypes.Structure):
    _fields_ = [
        ("chunk_id", ctypes.c_uint32),
        ("mesh", FdtGpuMeshData),
    ]


class FdtGpuPieceDelta(ctypes.Structure):
    _fields_ = [
        ("cutter_id", ctypes.c_uint32),
        ("mesh", FdtGpuMeshData),
    ]


class FdtGpuChunkOwner(ctypes.Structure):
    _fields_ = [
        ("chunk_id", ctypes.c_uint32),
        ("owner_cutter_id", ctypes.c_uint32),
    ]


class FdtGpuDeltaOutputV200(ctypes.Structure):
    _fields_ = [
        ("job_id", ctypes.c_uint64),
        ("compute_ms", ctypes.c_float),
        ("recomputed_islands", ctypes.c_uint32),
        ("num_updated_chunks", ctypes.c_uint32),
        ("updated_chunks", ctypes.POINTER(FdtGpuChunkDelta)),
        ("num_removed_chunks", ctypes.c_uint32),
        ("removed_chunk_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("num_updated_pieces", ctypes.c_uint32),
        ("updated_pieces", ctypes.POINTER(FdtGpuPieceDelta)),
        ("num_removed_pieces", ctypes.c_uint32),
        ("removed_piece_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("num_ordered_pieces", ctypes.c_uint32),
        ("ordered_piece_ids", ctypes.POINTER(ctypes.c_uint32)),
    ]


class FdtGpuDeltaOutput(ctypes.Structure):
    _fields_ = [
        ("job_id", ctypes.c_uint64),
        ("compute_ms", ctypes.c_float),
        ("recomputed_islands", ctypes.c_uint32),
        ("num_updated_chunks", ctypes.c_uint32),
        ("updated_chunks", ctypes.POINTER(FdtGpuChunkDelta)),
        ("num_removed_chunks", ctypes.c_uint32),
        ("removed_chunk_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("num_updated_pieces", ctypes.c_uint32),
        ("updated_pieces", ctypes.POINTER(FdtGpuPieceDelta)),
        ("num_removed_pieces", ctypes.c_uint32),
        ("removed_piece_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("num_ordered_pieces", ctypes.c_uint32),
        ("ordered_piece_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("num_chunk_owners", ctypes.c_uint32),
        ("chunk_owners", ctypes.POINTER(FdtGpuChunkOwner)),
    ]


# ---------------------------------------------------------------------------
# Cross-Platform Engine Library Loader (.dll / .so / .dylib)
# ---------------------------------------------------------------------------

_DLL: ctypes.CDLL | None = None
_DLL_VERSION: int = 0
_DLL_LOAD_ATTEMPTED: bool = False
_DLL_ERROR: str = ""


def _dll_path() -> str:
    import sys

    addon_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys.platform == "win32":
        lib_name = "fdt_bool_engine.dll"
    elif sys.platform == "darwin":
        lib_name = "libfdt_bool_engine.dylib"
    else:
        lib_name = "libfdt_bool_engine.so"

    candidates = (
        os.path.join(addon_root, "bin", lib_name),
        os.path.join(addon_root, "rust_engine", "target", "release", lib_name),
    )
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0]


def get_dll() -> ctypes.CDLL | None:
    global _DLL, _DLL_VERSION, _DLL_LOAD_ATTEMPTED, _DLL_ERROR
    if _DLL_LOAD_ATTEMPTED:
        return _DLL
    _DLL_LOAD_ATTEMPTED = True
    path = _dll_path()
    if not os.path.isfile(path):
        _DLL_ERROR = f"Engine binary not found at {path}"
        return None
    try:
        dll = ctypes.CDLL(path)
        dll.fdt_engine_version.argtypes = []
        dll.fdt_engine_version.restype = ctypes.c_uint32

        ver = int(dll.fdt_engine_version())
        if ver < 140:
            _DLL_ERROR = f"Incompatible DLL version: {ver} (need >= 140 — restart Blender to load new DLL)"
            return None

        gpu_out_type = FdtGpuPreviewOutput if ver >= 131 else FdtGpuPreviewOutputV130

        dll.fdt_session_create.argtypes = []
        dll.fdt_session_create.restype = ctypes.c_uint64

        dll.fdt_session_destroy.argtypes = [ctypes.c_uint64]
        dll.fdt_session_destroy.restype = ctypes.c_int32

        dll.fdt_session_set_target.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(FdtMeshInput),
        ]
        dll.fdt_session_set_target.restype = ctypes.c_int32

        dll.fdt_session_eval_cutters.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(FdtMeshInput),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_uint8,
        ]
        dll.fdt_session_eval_cutters.restype = ctypes.POINTER(FdtEvalOutput)

        dll.fdt_eval_free.argtypes = [ctypes.POINTER(FdtEvalOutput)]
        dll.fdt_eval_free.restype = None

        dll.fdt_session_submit_async.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(FdtMeshInput),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        dll.fdt_session_submit_async.restype = ctypes.c_uint64

        dll.fdt_session_poll_gpu_preview.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        dll.fdt_session_poll_gpu_preview.restype = ctypes.POINTER(gpu_out_type)

        dll.fdt_gpu_preview_free.argtypes = [ctypes.POINTER(gpu_out_type)]
        dll.fdt_gpu_preview_free.restype = None

        dll.fdt_session_fetch_cached_eval.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint8,
        ]
        dll.fdt_session_fetch_cached_eval.restype = ctypes.POINTER(FdtEvalOutput)

        if ver >= 200:
            delta_out_type = FdtGpuDeltaOutput if ver >= 210 else FdtGpuDeltaOutputV200
            dll.fdt_session_submit_async_delta.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(FdtMeshInput),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_int32,
                ctypes.c_float,
            ]
            dll.fdt_session_submit_async_delta.restype = ctypes.c_uint64

            dll.fdt_session_poll_gpu_delta.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.c_uint64,
            ]
            dll.fdt_session_poll_gpu_delta.restype = ctypes.POINTER(delta_out_type)

            dll.fdt_gpu_delta_free.argtypes = [ctypes.POINTER(delta_out_type)]
            dll.fdt_gpu_delta_free.restype = None

        if ver >= 210:
            dll.fdt_session_set_target_async.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(FdtMeshInput),
            ]
            dll.fdt_session_set_target_async.restype = ctypes.c_int32

        _DLL = dll
        _DLL_VERSION = ver
        _DLL_ERROR = ""
        return _DLL
    except Exception as exc:
        _DLL_ERROR = str(exc)
        _DLL = None
        _DLL_VERSION = 0
        return None


def get_dll_version() -> int:
    get_dll()
    return _DLL_VERSION


def supports_delta_gpu() -> bool:
    return get_dll_version() >= 200


def supports_async_target() -> bool:
    return get_dll_version() >= 210


def is_available() -> bool:
    return get_dll() is not None


def get_last_error() -> str:
    return _DLL_ERROR


# ---------------------------------------------------------------------------
# Fast Mesh <-> NumPy / C-ABI Conversion
# ---------------------------------------------------------------------------


def _matrix_to_col_major_f32(mat: Matrix) -> np.ndarray:
    # Blender Matrix rows -> transposed -> flat column-major 16 float32 for glam::DMat4
    return np.ascontiguousarray(mat.transposed(), dtype=np.float32).ravel()


def _ordered_uv_layers(mesh: bpy.types.Mesh) -> list:
    """Return UV layers with the active render UV layer placed first (index 0) for GPU preview."""
    layers = list(mesh.uv_layers)
    if len(layers) > 1:
        active_uv = next((l for l in layers if getattr(l, "active_render", False)), None) or mesh.uv_layers.active
        if active_uv is not None and active_uv in layers:
            layers.remove(active_uv)
            layers.insert(0, active_uv)
    return layers


def bmesh_to_input(
    bm: bmesh.types.BMesh,
    matrix_world: Matrix,
) -> tuple[FdtMeshInput, list[np.ndarray]]:
    """Extract live Edit Mode BMesh geometry straight into FdtMeshInput (zero datablock allocs)."""
    bm.verts.index_update()
    nv = len(bm.verts)
    np_count = len(bm.faces)

    keepalive: list[np.ndarray] = []

    if nv > 0:
        verts = np.fromiter(
            (c for v in bm.verts for c in (v.co.x, v.co.y, v.co.z)),
            dtype=np.float32,
            count=nv * 3,
        )
    else:
        verts = np.empty(0, dtype=np.float32)
    keepalive.append(verts)

    loop_list: list[int] = []
    starts_list: list[int] = [0]
    mat_list: list[int] = []
    for f in bm.faces:
        if not f.is_valid:
            continue
        for lp in f.loops:
            loop_list.append(lp.vert.index)
        starts_list.append(len(loop_list))
        mat_list.append(f.material_index)

    nl = len(loop_list)
    real_np = len(mat_list)
    loop_verts = np.asarray(loop_list, dtype=np.uint32)
    poly_starts = np.asarray(starts_list, dtype=np.uint32)
    mat_ids = np.asarray(mat_list, dtype=np.int32)
    keepalive.extend([loop_verts, poly_starts, mat_ids])

    xform = _matrix_to_col_major_f32(matrix_world)
    keepalive.append(xform)

    inp = FdtMeshInput(
        num_verts=nv,
        verts_xyz=verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if nv > 0 else None,
        num_loops=nl,
        loop_verts=loop_verts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if nl > 0 else None,
        num_polys=real_np,
        poly_starts=poly_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        mat_ids=mat_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if real_np > 0 else None,
        num_uv_layers=0,
        uv_data=None,
        world_matrix=xform.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        loop_normals=None,
    )
    return inp, keepalive


def _compute_true_matrix_world(obj: bpy.types.Object) -> Matrix:
    """Compute world matrix from matrix_basis + parent chain (never stale even inside hidden/excluded Vaults)."""
    mat = obj.matrix_basis.copy()
    cur = obj
    while cur.parent is not None:
        mat = cur.parent.matrix_basis @ cur.matrix_parent_inverse @ mat
        cur = cur.parent
    return mat


def _cutter_obj_to_input(
    c_obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph | None,
    temp_meshes: list[bpy.types.Mesh],
) -> tuple[FdtMeshInput, list[np.ndarray], int]:
    """Extract live cutter geometry (handles BOTH Object Mode and live Edit Mode BMesh + modifiers)."""
    mw = _compute_true_matrix_world(c_obj)
    if c_obj.mode == "EDIT" and c_obj.data is not None and len(c_obj.modifiers) == 0:
        bm = bmesh.from_edit_mesh(c_obj.data)
        c_inp, ka = bmesh_to_input(bm, mw)
        return c_inp, ka, int(c_inp.num_polys)

    mesh_for_input = c_obj.data
    if depsgraph is not None and (len(c_obj.modifiers) > 0 or c_obj.mode == "EDIT"):
        try:
            eval_obj = c_obj.evaluated_get(depsgraph)
            eval_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
            if len(eval_mesh.polygons) > 0:
                temp_meshes.append(eval_mesh)
                mesh_for_input = eval_mesh
            else:
                bpy.data.meshes.remove(eval_mesh)
        except RuntimeError:
            mesh_for_input = c_obj.data

    c_inp, ka = mesh_to_input(mesh_for_input, mw, include_uvs=False)
    return c_inp, ka, len(mesh_for_input.polygons)


def mesh_to_input(
    mesh: bpy.types.Mesh,
    matrix_world: Matrix,
    include_uvs: bool = True,
) -> tuple[FdtMeshInput, list[np.ndarray]]:
    """Extract a Blender Mesh into an FdtMeshInput struct using fast foreach_get."""
    nv = len(mesh.vertices)
    nl = len(mesh.loops)
    np_count = len(mesh.polygons)

    keepalive: list[np.ndarray] = []

    verts = np.empty(nv * 3, dtype=np.float32)
    if nv > 0:
        mesh.vertices.foreach_get("co", verts)
    keepalive.append(verts)

    loop_verts = np.empty(nl, dtype=np.uint32)
    if nl > 0:
        mesh.loops.foreach_get("vertex_index", loop_verts)
    keepalive.append(loop_verts)

    poly_starts = np.empty(np_count + 1, dtype=np.uint32)
    if np_count > 0:
        starts = np.empty(np_count, dtype=np.uint32)
        mesh.polygons.foreach_get("loop_start", starts)
        poly_starts[:-1] = starts
        poly_starts[-1] = nl
    else:
        poly_starts[0] = 0
    keepalive.append(poly_starts)

    mat_ids = np.empty(np_count, dtype=np.int32)
    if np_count > 0:
        mesh.polygons.foreach_get("material_index", mat_ids)
    keepalive.append(mat_ids)

    uv_layers = _ordered_uv_layers(mesh) if include_uvs else []
    nuv = len(uv_layers)
    if nuv > 0 and nl > 0:
        uv_data = np.empty(nuv * nl * 2, dtype=np.float32)
        for idx, layer in enumerate(uv_layers):
            buf = np.empty(nl * 2, dtype=np.float32)
            layer.data.foreach_get("uv", buf)
            uv_data[idx * nl * 2 : (idx + 1) * nl * 2] = buf
    else:
        uv_data = np.empty(0, dtype=np.float32)
    keepalive.append(uv_data)

    mat_arr = _matrix_to_col_major_f32(matrix_world)
    keepalive.append(mat_arr)

    # Read exact per-corner split normals (preserves custom normals, sharp edges & smooth curves)
    nrm_ptr = None
    if include_uvs and nl > 0:
        try:
            nrm_arr = np.empty(nl * 3, dtype=np.float32)
            mesh.corner_normals.foreach_get("vector", nrm_arr)
            keepalive.append(nrm_arr)
            nrm_ptr = nrm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        except Exception:
            nrm_ptr = None

    inp = FdtMeshInput(
        num_verts=nv,
        verts_xyz=verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if nv > 0 else None,
        num_loops=nl,
        loop_verts=loop_verts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if nl > 0 else None,
        num_polys=np_count,
        poly_starts=poly_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        mat_ids=mat_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if np_count > 0 else None,
        num_uv_layers=nuv,
        uv_data=uv_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if uv_data.size > 0 else None,
        world_matrix=mat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        loop_normals=nrm_ptr,
    )
    return inp, keepalive


def ensure_materials_with_cut(
    dst_mesh: bpy.types.Mesh,
    src_materials: Sequence[bpy.types.Material | None],
    cut_material: bpy.types.Material | None,
) -> int:
    """Sync material slots on `dst_mesh` from `src_materials` + `cut_material`.

    Returns the material slot index assigned to `cut_material` (or 0 if none).
    """
    desired: list[bpy.types.Material | None] = list(src_materials)
    cut_idx = 0
    if cut_material is not None:
        if cut_material in desired:
            cut_idx = desired.index(cut_material)
        else:
            cut_idx = len(desired)
            desired.append(cut_material)

    current = [dst_mesh.materials[i] for i in range(len(dst_mesh.materials))]
    if current != desired:
        dst_mesh.materials.clear()
        for mat in desired:
            dst_mesh.materials.append(mat)
    return cut_idx


def write_output_to_mesh(
    out: FdtMeshOutput,
    dst_mesh: bpy.types.Mesh,
    uv_layer_names: Sequence[str],
    src_materials: Sequence[bpy.types.Material | None],
    cut_material: bpy.types.Material | None,
    write_orig_attr: bool = True,
    apply_custom_normals: bool = True,
    write_nrm_attr: bool = True,
) -> bool:
    """Write a Rust `FdtMeshOutput` directly into `dst_mesh` via foreach_set.

    When `out.num_edges > 0` (live drag fast path), pre-computed Rust edges and
    loop edge indices are written directly so Blender skips `calc_edges=True`.
    Returns True if the resulting mesh has at least one polygon.
    """
    nv = int(out.num_verts)
    nl = int(out.num_loops)
    np_count = int(out.num_polys)
    nuv = int(out.num_uv_layers)
    ne = int(out.num_edges)

    dst_mesh.clear_geometry()
    ensure_materials_with_cut(dst_mesh, src_materials, cut_material)

    if nv == 0 or nl == 0 or np_count == 0:
        dst_mesh.update()
        return False

    verts_arr = np.ctypeslib.as_array(out.verts_xyz, shape=(nv * 3,))
    loops_arr = np.ctypeslib.as_array(out.loop_verts, shape=(nl,))
    starts_arr = np.ctypeslib.as_array(out.poly_starts, shape=(np_count + 1,))
    mat_arr = np.ctypeslib.as_array(out.mat_ids, shape=(np_count,))

    poly_starts = starts_arr[:-1]
    poly_totals = starts_arr[1:] - starts_arr[:-1]

    dst_mesh.vertices.add(nv)
    dst_mesh.vertices.foreach_set("co", verts_arr)

    has_precomputed_edges = ne > 0 and bool(out.edge_verts) and bool(out.loop_edges)
    if has_precomputed_edges:
        edge_verts_arr = np.ctypeslib.as_array(out.edge_verts, shape=(ne * 2,))
        loop_edges_arr = np.ctypeslib.as_array(out.loop_edges, shape=(nl,))
        dst_mesh.edges.add(ne)
        dst_mesh.edges.foreach_set("vertices", edge_verts_arr)

    dst_mesh.loops.add(nl)
    dst_mesh.loops.foreach_set("vertex_index", loops_arr)
    if has_precomputed_edges:
        dst_mesh.loops.foreach_set("edge_index", loop_edges_arr)

    dst_mesh.polygons.add(np_count)
    dst_mesh.polygons.foreach_set("loop_start", poly_starts)
    dst_mesh.polygons.foreach_set("loop_total", poly_totals)
    dst_mesh.polygons.foreach_set("material_index", mat_arr)

    # Restore all UV layers with original names (untouched faces keep exact original UVs;
    # cut faces receive Rust-computed world-space Box UVs)
    if nuv > 0 and bool(out.uv_data):
        all_uv = np.ctypeslib.as_array(out.uv_data, shape=(nuv, nl * 2))
        for idx in range(nuv):
            layer_name = uv_layer_names[idx] if idx < len(uv_layer_names) else f"UVMap_{idx}"
            uv_layer = dst_mesh.uv_layers.get(layer_name) or dst_mesh.uv_layers.new(name=layer_name)
            uv_layer.data.foreach_set("uv", all_uv[idx])

    # Write `fdt_orig` FACE float attribute when needed (Apply / rebuild)
    if write_orig_attr:
        cut_arr = np.ctypeslib.as_array(out.is_cut_face, shape=(np_count,))
        orig_flags = np.where(cut_arr == 0, 1.0, 0.0).astype(np.float32)
        orig_attr = dst_mesh.attributes.get(ORIG_ATTR_NAME)
        if orig_attr is not None and (orig_attr.data_type != "FLOAT" or orig_attr.domain != "FACE"):
            dst_mesh.attributes.remove(orig_attr)
            orig_attr = None
        if orig_attr is None:
            orig_attr = dst_mesh.attributes.new(name=ORIG_ATTR_NAME, type="FLOAT", domain="FACE")
        orig_attr.data.foreach_set("value", orig_flags)

    # Store preserved per-loop split normals in `fdt_nrm` CORNER attribute when needed
    # (skipped for high-poly meshes without custom normals -> saves ~100MB & ~9s on 4M tris!)
    nrm_3 = None
    if write_nrm_attr and nl > 0 and bool(out.loop_normals):
        nrm_flat = np.ctypeslib.as_array(out.loop_normals, shape=(nl * 3,)).copy()
        nrm_3 = nrm_flat.reshape(-1, 3)
        if write_orig_attr:
            nrm_attr = dst_mesh.attributes.get(PRESERVED_NRM_ATTR)
            if nrm_attr is not None and (nrm_attr.data_type != "FLOAT_VECTOR" or nrm_attr.domain != "CORNER"):
                dst_mesh.attributes.remove(nrm_attr)
                nrm_attr = None
            if nrm_attr is None:
                nrm_attr = dst_mesh.attributes.new(name=PRESERVED_NRM_ATTR, type="FLOAT_VECTOR", domain="CORNER")
            nrm_attr.data.foreach_set("vector", nrm_flat)

    dst_mesh.update(calc_edges=not has_precomputed_edges)

    # Apply exact custom split normals on commit/rebuild (skipped on Apply because
    # `_postprocess_baked_mesh` applies them ONCE on the final baked copy!)
    if apply_custom_normals and write_orig_attr and nrm_3 is not None and len(dst_mesh.loops) == nl:
        try:
            # Sanitize any degenerate zero-length normals before setting custom normals
            lens = np.linalg.norm(nrm_3, axis=1)
            bad = lens < 1e-5
            if np.any(bad):
                nrm_3[bad] = (0.0, 0.0, 1.0)
            dst_mesh.polygons.foreach_set("use_smooth", np.ones(np_count, dtype=bool))
            dst_mesh.normals_split_custom_set(nrm_3)
        except Exception:
            pass

    return True


# ---------------------------------------------------------------------------
# Stateful Session Cache Manager
# ---------------------------------------------------------------------------


class _RustSessionEntry:
    def __init__(self, handle: int):
        self.handle = handle
        self.target_ids: dict[str, int] = {}
        self.target_sigs: dict[str, tuple] = {}
        self.next_target_id: int = 1
        # Stable monotonic cutter IDs for v200 Delta API
        self.cutter_ids: dict[str, int] = {}
        self.cutter_id_to_name: dict[int, str] = {}
        self.next_cutter_id: int = 1
        # target_id -> set of cutter_ids already uploaded to Rust
        self.uploaded_cutters_by_target: dict[int, set[int]] = {}

    def get_or_assign_id(self, target_name: str) -> int:
        tid = self.target_ids.get(target_name)
        if tid is None:
            tid = self.next_target_id
            self.next_target_id += 1
            self.target_ids[target_name] = tid
        return tid

    def get_or_assign_cutter_id(self, cutter_name: str) -> int:
        cid = self.cutter_ids.get(cutter_name)
        if cid is None:
            cid = self.next_cutter_id
            self.next_cutter_id += 1
            self.cutter_ids[cutter_name] = cid
            self.cutter_id_to_name[cid] = cutter_name
        return cid


_SESSION_CACHE: dict[str, _RustSessionEntry] = {}


def get_cutter_name_for_id(session_id: str, cutter_id: int) -> str:
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return ""
    return entry.cutter_id_to_name.get(int(cutter_id), "")


def _target_signature(obj: bpy.types.Object) -> tuple:
    mesh = obj.data
    m = _compute_true_matrix_world(obj)
    mat_tuple = tuple(round(m[r][c], 5) for r in range(4) for c in range(4))
    return (
        mesh.as_pointer(),
        len(mesh.vertices),
        len(mesh.polygons),
        len(mesh.uv_layers),
        mat_tuple,
    )


def has_session_target(session_id: str, target_obj: bpy.types.Object) -> bool:
    """Return True if `target_obj` is already uploaded in Rust for `session_id` with matching signature."""
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return False
    return entry.target_sigs.get(target_obj.name) == _target_signature(target_obj)


def ensure_session_target(
    session_id: str,
    target_obj: bpy.types.Object,
    force_upload: bool = False,
    allow_async: bool = False,
) -> tuple[int, int, bool] | None:
    """Ensure `target_obj` is uploaded and cached in Rust. Returns `(handle, target_id, uploaded)`."""
    dll = get_dll()
    if dll is None:
        return None

    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        handle = int(dll.fdt_session_create())
        if handle == 0:
            return None
        entry = _RustSessionEntry(handle)
        _SESSION_CACHE[session_id] = entry

    tid = entry.get_or_assign_id(target_obj.name)
    sig = _target_signature(target_obj)
    uploaded = False
    if force_upload or entry.target_sigs.get(target_obj.name) != sig:
        inp, keepalive = mesh_to_input(
            target_obj.data, _compute_true_matrix_world(target_obj), include_uvs=True
        )
        if allow_async and supports_async_target():
            rc = int(dll.fdt_session_set_target_async(entry.handle, tid, ctypes.byref(inp)))
        else:
            rc = int(dll.fdt_session_set_target(entry.handle, tid, ctypes.byref(inp)))
        del keepalive
        if rc != 0:
            return None
        entry.target_sigs[target_obj.name] = sig
        entry.uploaded_cutters_by_target.pop(tid, None)
        uploaded = True

    return entry.handle, tid, uploaded


def preupload_target_async(session_id: str, target_obj: bpy.types.Object) -> bool:
    """Kick off background target BVH + SpatialChunks + Base GPU build immediately on Enter Cut Mode."""
    if not supports_async_target():
        return False
    return ensure_session_target(session_id, target_obj, allow_async=True) is not None


def evaluate_target_with_cutters(
    session_id: str,
    target_obj: bpy.types.Object,
    cutter_objs: Sequence[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph | None,
    cut_material: bpy.types.Material | None,
    uv_scale: float,
    res_obj: bpy.types.Object | None,
    piece_objs: Sequence[bpy.types.Object | None],
    is_live: bool = False,
    force_write: bool = False,
    include_edges: bool | None = None,
    apply_custom_normals: bool = True,
) -> tuple[bool, list[bool]]:
    """Run Rust sequential split for `target_obj` against `cutter_objs` and write outputs directly.

    Returns `(res_has_geo, [piece_i_has_geo, ...])`.
    """
    t_start = time.perf_counter()
    dll = get_dll()
    if dll is None:
        return False, []

    info = ensure_session_target(session_id, target_obj)
    t_upload_done = time.perf_counter()
    if info is None:
        return False, []
    handle, tid, target_uploaded = info

    src_mesh = target_obj.data
    src_materials = [src_mesh.materials[i] for i in range(len(src_mesh.materials))]
    uv_names = [uv.name for uv in _ordered_uv_layers(src_mesh)]

    # Determine cut material slot index consistent across remaining & pieces
    cut_mat_idx = 0
    if cut_material is not None:
        if cut_material in src_materials:
            cut_mat_idx = src_materials.index(cut_material)
        else:
            cut_mat_idx = len(src_materials)

    # Extract evaluated cutter meshes (supports Rough Edge GeoNodes modifier on cutters!)
    nc = len(cutter_objs)
    c_array = (FdtMeshInput * nc)()
    all_keepalive: list[list[np.ndarray]] = []
    temp_meshes: list[bpy.types.Mesh] = []
    cutter_total_polys = 0

    try:
        for i, c_obj in enumerate(cutter_objs):
            c_inp, ka, n_cpolys = _cutter_obj_to_input(c_obj, depsgraph, temp_meshes)
            cutter_total_polys += n_cpolys
            c_array[i] = c_inp
            all_keepalive.append(ka)

        t_extract_done = time.perf_counter()

        gen_edges = bool(is_live) if include_edges is None else bool(include_edges)
        eval_ptr = dll.fdt_session_eval_cutters(
            handle,
            tid,
            c_array if nc > 0 else None,
            nc,
            cut_mat_idx,
            ctypes.c_float(float(uv_scale)),
            ctypes.c_uint8(1 if gen_edges else 0),
        )
        t_rust_done = time.perf_counter()
        if not eval_ptr:
            return False, [False] * nc

        try:
            eval_out = eval_ptr.contents
            if eval_out.status != 0:
                return False, [False] * nc

            dirty_from = int(eval_out.dirty_from_idx)
            write_orig_attr = not is_live
            write_nrm_attr = (len(src_mesh.polygons) < 50_000) or bool(
                getattr(src_mesh, "has_custom_normals", False)
            )

            res_ok = False
            out_rem_verts = int(eval_out.remaining.num_verts)
            out_rem_polys = int(eval_out.remaining.num_polys)
            rem_is_stub = (out_rem_verts == 0 and out_rem_polys > 0)
            if res_obj is not None and res_obj.type == "MESH":
                # Skip rewriting Remaining when all cutters hit cache (eliminates duplicate slider writebacks!)
                if (
                    not force_write
                    and (rem_is_stub or (nc > 0 and dirty_from >= nc))
                    and len(res_obj.data.polygons) == out_rem_polys
                ):
                    res_ok = out_rem_polys > 0
                elif not rem_is_stub:
                    res_ok = write_output_to_mesh(
                        eval_out.remaining,
                        res_obj.data,
                        uv_names,
                        src_materials,
                        cut_material,
                        write_orig_attr=write_orig_attr,
                        apply_custom_normals=apply_custom_normals,
                        write_nrm_attr=write_nrm_attr,
                    )

            pieces_ok: list[bool] = []
            out_pieces_verts = 0
            out_pieces_polys = 0
            num_out_pieces = int(eval_out.num_pieces)
            for idx in range(min(num_out_pieces, len(piece_objs))):
                p_out = eval_out.pieces[idx]
                p_nv = int(p_out.num_verts)
                p_np = int(p_out.num_polys)
                p_is_stub = (p_nv == 0 and p_np > 0)
                out_pieces_verts += p_nv
                out_pieces_polys += p_np
                p_obj = piece_objs[idx]
                if p_obj is not None and p_obj.type == "MESH":
                    # Dirty-Piece Mask: skip rewriting unchanged leading pieces!
                    if not force_write and (p_is_stub or idx < dirty_from) and len(p_obj.data.polygons) == p_np:
                        pieces_ok.append(p_np > 0)
                    elif not p_is_stub:
                        ok = write_output_to_mesh(
                            p_out,
                            p_obj.data,
                            uv_names,
                            src_materials,
                            cut_material,
                            write_orig_attr=write_orig_attr,
                            apply_custom_normals=apply_custom_normals,
                            write_nrm_attr=write_nrm_attr,
                        )
                        pieces_ok.append(ok)
                    else:
                        pieces_ok.append(len(p_obj.data.polygons) > 0)
                else:
                    pieces_ok.append(False)

            t_end = time.perf_counter()
            profiler.record_eval(
                profiler.EvalSample(
                    timestamp=t_end,
                    session_id=session_id,
                    target_name=target_obj.name,
                    solver="RUST",
                    target_verts=len(src_mesh.vertices),
                    target_polys=len(src_mesh.polygons),
                    num_cutters=nc,
                    cutter_total_polys=cutter_total_polys,
                    out_rem_verts=out_rem_verts,
                    out_rem_polys=out_rem_polys,
                    out_pieces_verts=out_pieces_verts,
                    out_pieces_polys=out_pieces_polys,
                    target_uploaded=target_uploaded,
                    upload_ms=(t_upload_done - t_start) * 1000.0,
                    extract_ms=(t_extract_done - t_upload_done) * 1000.0,
                    rust_ms=(t_rust_done - t_extract_done) * 1000.0,
                    writeback_ms=(t_end - t_rust_done) * 1000.0,
                    total_ms=(t_end - t_start) * 1000.0,
                )
            )

            return res_ok, pieces_ok
        finally:
            dll.fdt_eval_free(eval_ptr)
    finally:
        del all_keepalive
        for tm in temp_meshes:
            try:
                bpy.data.meshes.remove(tm)
            except Exception:
                pass


def _copy_gpu_mesh_arrays(
    g: FdtGpuMeshData | FdtGpuMeshDataV130,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Copy material-grouped `(mat_id, pos, nor, uv0, uv1)` NumPy arrays from Rust GPU output."""
    ns = int(g.num_submeshes)
    if ns == 0 or not bool(g.submeshes):
        return []
    has_uv1 = _DLL_VERSION >= 131
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(ns):
        s = g.submeshes[i]
        nt = int(s.num_tris)
        if nt == 0 or not bool(s.pos_xyz) or not bool(s.nor_xyz) or not bool(s.uv_xy):
            continue
        nv = nt * 3
        pos = np.ctypeslib.as_array(s.pos_xyz, shape=(nv, 3)).copy()
        nor = np.ctypeslib.as_array(s.nor_xyz, shape=(nv, 3)).copy()
        uv0 = np.ctypeslib.as_array(s.uv_xy, shape=(nv, 2)).copy()
        if has_uv1 and bool(s.uv1_xy):
            uv1 = np.ctypeslib.as_array(s.uv1_xy, shape=(nv, 2)).copy()
        else:
            uv1 = uv0
        out.append((int(s.mat_id), pos, nor, uv0, uv1))
    return out


def submit_target_async(
    session_id: str,
    target_obj: bpy.types.Object,
    cutter_objs: Sequence[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph | None,
    cut_material: bpy.types.Material | None,
    uv_scale: float,
) -> int:
    """Submit a non-blocking boolean job to Rust's background worker (< 0.3 ms on main thread)."""
    dll = get_dll()
    if dll is None:
        return 0

    info = ensure_session_target(session_id, target_obj)
    if info is None:
        return 0
    handle, tid, _ = info

    src_mesh = target_obj.data
    src_materials = [src_mesh.materials[i] for i in range(len(src_mesh.materials))]
    cut_mat_idx = 0
    if cut_material is not None:
        if cut_material in src_materials:
            cut_mat_idx = src_materials.index(cut_material)
        else:
            cut_mat_idx = len(src_materials)

    nc = len(cutter_objs)
    c_array = (FdtMeshInput * nc)()
    all_keepalive: list[list[np.ndarray]] = []
    temp_meshes: list[bpy.types.Mesh] = []

    try:
        for i, c_obj in enumerate(cutter_objs):
            c_inp, ka, _n_cpolys = _cutter_obj_to_input(c_obj, depsgraph, temp_meshes)
            c_array[i] = c_inp
            all_keepalive.append(ka)

        return int(
            dll.fdt_session_submit_async(
                handle,
                tid,
                c_array if nc > 0 else None,
                nc,
                cut_mat_idx,
                ctypes.c_float(float(uv_scale)),
            )
        )
    finally:
        del all_keepalive
        for tm in temp_meshes:
            try:
                bpy.data.meshes.remove(tm)
            except Exception:
                pass


def poll_target_gpu_preview(
    session_id: str,
    target_obj: bpy.types.Object,
    last_seen_job_id: int,
) -> dict | None:
    """Non-blocking poll for the latest finished world-space GPU preview arrays (legacy full frame)."""
    dll = get_dll()
    if dll is None:
        return None
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return None
    tid = entry.target_ids.get(target_obj.name)
    if tid is None:
        return None

    ptr = dll.fdt_session_poll_gpu_preview(entry.handle, tid, ctypes.c_uint64(last_seen_job_id))
    if not ptr:
        return None

    try:
        out = ptr.contents
        rem_arr = _copy_gpu_mesh_arrays(out.remaining)
        num_p = int(out.num_pieces)
        pieces_arr = [_copy_gpu_mesh_arrays(out.pieces[i]) for i in range(num_p)]
        return {
            "job_id": int(out.job_id),
            "compute_ms": float(out.compute_ms),
            "dirty_from_idx": int(out.dirty_from_idx),
            "remaining": rem_arr,
            "pieces": pieces_arr,
        }
    finally:
        dll.fdt_gpu_preview_free(ptr)


def submit_target_async_delta(
    session_id: str,
    target_obj: bpy.types.Object,
    cutter_objs: Sequence[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph | None,
    cut_material: bpy.types.Material | None,
    uv_scale: float,
    dirty_cutter_names: set[str] | None = None,
) -> int:
    """v200 Delta async submit: extracts & transfers ONLY dirty/new cutters (< 0.1 ms even with 500 boxes!)."""
    dll = get_dll()
    if dll is None:
        return 0
    if not supports_delta_gpu():
        return submit_target_async(session_id, target_obj, cutter_objs, depsgraph, cut_material, uv_scale)

    info = ensure_session_target(session_id, target_obj, allow_async=True)
    if info is None:
        return 0
    handle, tid, _ = info
    entry = _SESSION_CACHE[session_id]
    uploaded_set = entry.uploaded_cutters_by_target.setdefault(tid, set())

    src_mesh = target_obj.data
    src_materials = [src_mesh.materials[i] for i in range(len(src_mesh.materials))]
    cut_mat_idx = 0
    if cut_material is not None:
        if cut_material in src_materials:
            cut_mat_idx = src_materials.index(cut_material)
        else:
            cut_mat_idx = len(src_materials)

    ordered_ids: list[int] = []
    to_upload: list[tuple[int, bpy.types.Object]] = []
    for c_obj in cutter_objs:
        cid = entry.get_or_assign_cutter_id(c_obj.name)
        ordered_ids.append(cid)
        if (
            dirty_cutter_names is None
            or c_obj.name in dirty_cutter_names
            or cid not in uploaded_set
        ):
            to_upload.append((cid, c_obj))

    nu = len(to_upload)
    na = len(ordered_ids)
    upd_ids_arr = (ctypes.c_uint32 * nu)(*(cid for cid, _ in to_upload)) if nu > 0 else None
    upd_inps_arr = (FdtMeshInput * nu)() if nu > 0 else None
    act_ids_arr = (ctypes.c_uint32 * na)(*ordered_ids) if na > 0 else None

    all_keepalive: list[list[np.ndarray]] = []
    temp_meshes: list[bpy.types.Mesh] = []

    try:
        for i, (cid, c_obj) in enumerate(to_upload):
            c_inp, ka, _n_cpolys = _cutter_obj_to_input(c_obj, depsgraph, temp_meshes)
            upd_inps_arr[i] = c_inp
            all_keepalive.append(ka)

        jid = int(
            dll.fdt_session_submit_async_delta(
                handle,
                tid,
                upd_ids_arr,
                upd_inps_arr,
                nu,
                act_ids_arr,
                na,
                cut_mat_idx,
                ctypes.c_float(float(uv_scale)),
            )
        )
        if jid > 0:
            for cid, _ in to_upload:
                uploaded_set.add(cid)
        return jid
    finally:
        del all_keepalive
        for tm in temp_meshes:
            try:
                bpy.data.meshes.remove(tm)
            except Exception:
                pass


def poll_target_gpu_delta(
    session_id: str,
    target_obj: bpy.types.Object,
    last_seen_job_id: int,
) -> dict | None:
    """v210 Non-blocking Delta GPU poll: returns ONLY changed chunks, owner flips, & boundary pieces."""
    dll = get_dll()
    if dll is None or not supports_delta_gpu():
        return None
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return None
    tid = entry.target_ids.get(target_obj.name)
    if tid is None:
        return None

    ptr = dll.fdt_session_poll_gpu_delta(entry.handle, tid, ctypes.c_uint64(last_seen_job_id))
    if not ptr:
        return None

    try:
        out = ptr.contents
        updated_chunks: list[tuple[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]] = []
        nuc = int(out.num_updated_chunks)
        if nuc > 0 and bool(out.updated_chunks):
            for i in range(nuc):
                cd = out.updated_chunks[i]
                updated_chunks.append((int(cd.chunk_id), _copy_gpu_mesh_arrays(cd.mesh)))

        removed_chunk_ids: list[int] = []
        nrc = int(out.num_removed_chunks)
        if nrc > 0 and bool(out.removed_chunk_ids):
            removed_chunk_ids = [int(out.removed_chunk_ids[i]) for i in range(nrc)]

        updated_pieces: list[tuple[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]] = []
        nup = int(out.num_updated_pieces)
        if nup > 0 and bool(out.updated_pieces):
            for i in range(nup):
                pd = out.updated_pieces[i]
                updated_pieces.append((int(pd.cutter_id), _copy_gpu_mesh_arrays(pd.mesh)))

        removed_piece_ids: list[int] = []
        nrp = int(out.num_removed_pieces)
        if nrp > 0 and bool(out.removed_piece_ids):
            removed_piece_ids = [int(out.removed_piece_ids[i]) for i in range(nrp)]

        ordered_piece_ids: list[int] = []
        nop = int(out.num_ordered_pieces)
        if nop > 0 and bool(out.ordered_piece_ids):
            ordered_piece_ids = [int(out.ordered_piece_ids[i]) for i in range(nop)]

        chunk_owners: list[tuple[int, int]] = []
        if _DLL_VERSION >= 210:
            nco = int(out.num_chunk_owners)
            if nco > 0 and bool(out.chunk_owners):
                chunk_owners = [
                    (int(out.chunk_owners[i].chunk_id), int(out.chunk_owners[i].owner_cutter_id))
                    for i in range(nco)
                ]

        return {
            "job_id": int(out.job_id),
            "compute_ms": float(out.compute_ms),
            "recomputed_islands": int(out.recomputed_islands),
            "updated_chunks": updated_chunks,
            "removed_chunk_ids": removed_chunk_ids,
            "chunk_owners": chunk_owners,
            "updated_pieces": updated_pieces,
            "removed_piece_ids": removed_piece_ids,
            "ordered_piece_ids": ordered_piece_ids,
        }
    finally:
        dll.fdt_gpu_delta_free(ptr)


def commit_target_cached_eval(
    session_id: str,
    target_obj: bpy.types.Object,
    cut_material: bpy.types.Material | None,
    res_obj: bpy.types.Object | None,
    piece_objs: Sequence[bpy.types.Object | None],
    write_orig_attr: bool = False,
    include_edges: bool = True,
    apply_custom_normals: bool = True,
) -> tuple[bool, list[bool]]:
    """Fetch the already-computed Rust cached result (0 ms recompute) and write into real Blender meshes."""
    t_start = time.perf_counter()
    dll = get_dll()
    if dll is None:
        return False, []
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return False, []
    tid = entry.target_ids.get(target_obj.name)
    if tid is None:
        return False, []

    src_mesh = target_obj.data
    src_materials = [src_mesh.materials[i] for i in range(len(src_mesh.materials))]
    uv_names = [uv.name for uv in _ordered_uv_layers(src_mesh)]

    eval_ptr = dll.fdt_session_fetch_cached_eval(
        entry.handle, tid, ctypes.c_uint8(1 if include_edges else 0)
    )
    t_fetch_done = time.perf_counter()
    if not eval_ptr:
        return False, []

    try:
        eval_out = eval_ptr.contents
        if eval_out.status != 0:
            return False, []

        write_nrm_attr = (len(src_mesh.polygons) < 50_000) or bool(
            getattr(src_mesh, "has_custom_normals", False)
        )

        res_ok = False
        out_rem_verts = int(eval_out.remaining.num_verts)
        out_rem_polys = int(eval_out.remaining.num_polys)
        if res_obj is not None and res_obj.type == "MESH":
            res_ok = write_output_to_mesh(
                eval_out.remaining,
                res_obj.data,
                uv_names,
                src_materials,
                cut_material,
                write_orig_attr=write_orig_attr,
                apply_custom_normals=apply_custom_normals,
                write_nrm_attr=write_nrm_attr,
            )

        pieces_ok: list[bool] = []
        out_pieces_verts = 0
        out_pieces_polys = 0
        num_out_pieces = int(eval_out.num_pieces)
        for idx in range(min(num_out_pieces, len(piece_objs))):
            p_out = eval_out.pieces[idx]
            out_pieces_verts += int(p_out.num_verts)
            out_pieces_polys += int(p_out.num_polys)
            p_obj = piece_objs[idx]
            if p_obj is not None and p_obj.type == "MESH":
                ok = write_output_to_mesh(
                    p_out,
                    p_obj.data,
                    uv_names,
                    src_materials,
                    cut_material,
                    write_orig_attr=write_orig_attr,
                    apply_custom_normals=apply_custom_normals,
                    write_nrm_attr=write_nrm_attr,
                )
                pieces_ok.append(ok)
            else:
                pieces_ok.append(False)

        t_end = time.perf_counter()
        profiler.record_eval(
            profiler.EvalSample(
                timestamp=t_end,
                session_id=session_id,
                target_name=target_obj.name,
                solver="RUST",
                target_verts=len(src_mesh.vertices),
                target_polys=len(src_mesh.polygons),
                num_cutters=num_out_pieces,
                cutter_total_polys=0,
                out_rem_verts=out_rem_verts,
                out_rem_polys=out_rem_polys,
                out_pieces_verts=out_pieces_verts,
                out_pieces_polys=out_pieces_polys,
                target_uploaded=False,
                upload_ms=0.0,
                extract_ms=0.0,
                rust_ms=(t_fetch_done - t_start) * 1000.0,
                writeback_ms=(t_end - t_fetch_done) * 1000.0,
                total_ms=(t_end - t_start) * 1000.0,
            )
        )
        return res_ok, pieces_ok
    finally:
        dll.fdt_eval_free(eval_ptr)


def destroy_session(session_id: str) -> None:
    entry = _SESSION_CACHE.pop(session_id, None)
    if entry is None:
        return
    dll = get_dll()
    if dll is not None:
        try:
            dll.fdt_session_destroy(entry.handle)
        except Exception:
            pass


def destroy_all_sessions() -> None:
    for sid in list(_SESSION_CACHE.keys()):
        destroy_session(sid)
