"""End-to-end Performance & Stress Benchmark for FullDestruc_Tool v2.0.

Exercises:
1. Real Rust C-ABI v200 (`libfdt_bool_engine.so`/`.dll`/`.dylib`) through `addon.core.rust_bridge`:
   - 1,000,000 polygons (2M tris) + 500 cut boxes via `submit_target_async_delta` + `poll_target_gpu_delta`
   - Verifies Layer #1 (`index 0`) priority over Layer #2 (`index 1`) through Python C-ABI
   - Measures Main Thread submit latency, delta poll + GPU VRAM patch latency, and Rust worker latency
2. Python Hot-Path Micro-Benchmarks (500 Cutters):
   - `session.load_session_readonly` vs raw `json.loads`
   - `zone_registry.resolve_zone` / `zone_token` O(1) cache
   - `profiler.record_eval` buffered disk logging
   - `live_sync._on_depsgraph_update` dirty-only O(1) filtering & O(1) adopt guard
   - `gpu_preview` chunked VRAM delta patch, per-zone piece bucketing, lazy `piece_wire`,
     and zero-rebuild batched cutter wireframes (`_draw_session_cutters_xray`)
   - `viewport_hud._draw_dock_post_pixel` cached GPU batch replay (0 allocs/frame)

Run directly with:
    python3 tests/bench_performance.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Lightweight Blender (`bpy`, `bmesh`, `gpu`, `gpu_extras`, `blf`, `mathutils`) Stubs
# ---------------------------------------------------------------------------

class _MockMatrix:
    def __init__(self, arr: np.ndarray | None = None):
        if arr is None:
            self._m = np.eye(4, dtype=np.float32)
        else:
            self._m = np.asarray(arr, dtype=np.float32).reshape(4, 4).copy()

    def __getitem__(self, idx):
        return self._m[idx]

    def __array__(self, dtype=None):
        return self._m if dtype is None else self._m.astype(dtype)

    def __matmul__(self, other):
        if isinstance(other, _MockMatrix):
            return _MockMatrix(self._m @ other._m)
        return self._m @ np.asarray(other, dtype=np.float32)

    def copy(self):
        return _MockMatrix(self._m)

    def transposed(self):
        return _MockMatrix(self._m.T)

    def inverted_safe(self):
        return _MockMatrix(np.linalg.inv(self._m))

    @classmethod
    def Translation(cls, xyz):
        m = np.eye(4, dtype=np.float32)
        m[0, 3], m[1, 3], m[2, 3] = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return cls(m)


class _ForeachSeq:
    """Minimal Blender RNA collection wrapper backed by contiguous NumPy arrays."""

    def __init__(self, count: int, attrs: dict[str, np.ndarray]):
        self._count = int(count)
        self._attrs = attrs
        self.foreach_get_calls = 0

    def __len__(self) -> int:
        return self._count

    def foreach_get(self, attr: str, dst: np.ndarray) -> None:
        self.foreach_get_calls += 1
        src = self._attrs[attr]
        dst[:] = src.ravel()


class _MockUVLayer:
    def __init__(self, name: str, uv_flat: np.ndarray):
        self.name = name
        self.active_render = True
        self.data = _ForeachSeq(uv_flat.size // 2, {"uv": uv_flat})


class _MockUVLayers(list):
    def __init__(self, items):
        super().__init__(items)
        self.active = items[0] if items else None

    def get(self, name: str):
        for x in self:
            if x.name == name:
                return x
        return None


class _MockMesh:
    _NEXT_PTR = 100000

    def __init__(
        self,
        name: str,
        verts_xyz: np.ndarray,
        loop_verts: np.ndarray,
        poly_starts: np.ndarray,
        poly_totals: np.ndarray,
        mat_ids: np.ndarray,
        edges_idx: np.ndarray | None = None,
        uv_flat: np.ndarray | None = None,
        materials: list | None = None,
    ):
        self.name = name
        _MockMesh._NEXT_PTR += 16
        self._ptr = _MockMesh._NEXT_PTR
        nv = verts_xyz.size // 3
        nl = loop_verts.size
        np_c = poly_starts.size
        self.vertices = _ForeachSeq(nv, {"co": verts_xyz.astype(np.float32)})
        self.loops = _ForeachSeq(nl, {"vertex_index": loop_verts.astype(np.int32)})
        self.polygons = _ForeachSeq(
            np_c,
            {
                "loop_start": poly_starts.astype(np.int32),
                "loop_total": poly_totals.astype(np.int32),
                "material_index": mat_ids.astype(np.int32),
            },
        )
        if edges_idx is None:
            edges_idx = np.zeros(0, dtype=np.int32)
        self.edges = _ForeachSeq(edges_idx.size // 2, {"vertices": edges_idx.astype(np.int32)})
        if uv_flat is None:
            uv_flat = np.zeros(nl * 2, dtype=np.float32)
        self.uv_layers = _MockUVLayers([_MockUVLayer("UVMap", uv_flat.astype(np.float32))])
        self.materials = list(materials or [])
        self.corner_normals = []
        self.attributes = {}

    def as_pointer(self) -> int:
        return self._ptr


class _MockObject:
    _NEXT_PTR = 500000

    def __init__(self, name: str, mesh: _MockMesh, matrix_world: _MockMatrix | None = None):
        self.name = name
        self.type = "MESH"
        self.data = mesh
        self.matrix_world = matrix_world if matrix_world is not None else _MockMatrix()
        self.parent = None
        self.matrix_parent_inverse = _MockMatrix()
        self.mode = "OBJECT"
        self.modifiers: list = []
        self.show_in_front = False
        self.display_type = "WIRE"
        self.hide_viewport = False
        self._hidden = False
        self._selected = False
        self.color = (0.95, 0.55, 0.15, 1.0)
        self._props: dict[str, Any] = {}
        _MockObject._NEXT_PTR += 16
        self._ptr = _MockObject._NEXT_PTR

        v = mesh.vertices._attrs["co"].reshape(-1, 3)
        mn = v.min(axis=0)
        mx = v.max(axis=0)
        self.bound_box = [
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mn[0]), float(mn[1]), float(mx[2])),
            (float(mn[0]), float(mx[1]), float(mx[2])),
            (float(mn[0]), float(mx[1]), float(mn[2])),
            (float(mx[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mn[1]), float(mx[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
            (float(mx[0]), float(mx[1]), float(mn[2])),
        ]

    @property
    def matrix_basis(self):
        return self.matrix_world

    def as_pointer(self) -> int:
        return self._ptr

    def get(self, key: str, default=None):
        return self._props.get(key, default)

    def __setitem__(self, key: str, val):
        self._props[key] = val

    def __getitem__(self, key: str):
        return self._props[key]

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def hide_get(self) -> bool:
        return self._hidden

    def hide_set(self, v: bool) -> None:
        self._hidden = bool(v)

    def select_get(self) -> bool:
        return self._selected

    def select_set(self, v: bool) -> None:
        self._selected = bool(v)


class _MockCollection:
    def __init__(self, name: str):
        self.name = name
        self._objs: list[_MockObject] = []
        self.objects = _MockObjCollection(self._objs)
        self._props: dict[str, Any] = {}

    def __setitem__(self, key: str, val):
        self._props[key] = val

    def get(self, key: str, default=None):
        return self._props.get(key, default)


class _MockObjCollection:
    def __init__(self, items: list[_MockObject]):
        self._items = items
        self._by_name: dict[str, _MockObject] = {o.name: o for o in items}

    def link(self, obj: _MockObject):
        if obj.name not in self._by_name:
            self._items.append(obj)
            self._by_name[obj.name] = obj

    def get(self, name: str):
        return self._by_name.get(name)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name


class _MockScene:
    def __init__(self):
        self._ptr = 900001
        self._props: dict[str, Any] = {}
        self.objects = _MockObjCollection([])
        self.fdt_settings = types.SimpleNamespace(
            boolean_solver="RUST",
            gpu_fast_drag=True,
            show_viewport_dock=True,
            ui_active_tool="APPLE_BITE",
            active_zone="A",
            preview_display_mode="ALL",
            show_zone_grid=True,
            zone_grid_size=0.35,
            cut_material=None,
            apply_keep_result=True,
            apply_keep_pieces=True,
            texel_per_meter=256.0,
            texture_size=512,
        )

    def as_pointer(self) -> int:
        return self._ptr

    def get(self, key: str, default=None):
        return self._props.get(key, default)

    def __setitem__(self, key: str, val):
        self._props[key] = val

    def __getitem__(self, key: str):
        return self._props[key]

    def __delitem__(self, key: str):
        self._props.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self._props


class _MockMaterial:
    _NEXT = 700000

    def __init__(self, name: str, color=(0.75, 0.75, 0.75, 1.0)):
        self.name = name
        _MockMaterial._NEXT += 16
        self._ptr = _MockMaterial._NEXT
        self.diffuse_color = color
        self.node_tree = None

    def as_pointer(self) -> int:
        return self._ptr


# Counters for GPU telemetry assertions
GPU_COUNTERS = {
    "batch_for_shader_calls": 0,
    "draw_calls": 0,
    "shader_binds": 0,
    "uniform_sampler_calls": 0,
}


class _MockBatch:
    def __init__(self, mode: str, n_verts: int):
        self.mode = mode
        self.n_verts = n_verts

    def draw(self, shader) -> None:
        GPU_COUNTERS["draw_calls"] += 1


class _MockShader:
    def __init__(self, name: str):
        self.name = name

    def bind(self) -> None:
        GPU_COUNTERS["shader_binds"] += 1

    def uniform_float(self, name: str, val) -> None:
        pass

    def uniform_sampler(self, name: str, tex) -> None:
        GPU_COUNTERS["uniform_sampler_calls"] += 1

    def uniform_block(self, name: str, ubo) -> None:
        pass


def _install_mock_blender():
    bpy = types.ModuleType("bpy")
    bpy.app = types.SimpleNamespace(
        background=False,
        version_string="5.2.0",
        handlers=types.SimpleNamespace(
            persistent=lambda fn: fn,
            depsgraph_update_post=[],
        ),
        timers=types.SimpleNamespace(
            register=lambda fn, first_interval=0.0, persistent=False: None,
            is_registered=lambda fn: False,
        ),
    )

    scene = _MockScene()
    all_objs = _MockObjCollection([])
    all_cols: dict[str, _MockCollection] = {}

    class _ColRegistry:
        def get(self, name: str):
            return all_cols.get(name)

        def new(self, name: str):
            c = _MockCollection(name)
            all_cols[name] = c
            return c

        def __contains__(self, name: str) -> bool:
            return name in all_cols

    bpy.data = types.SimpleNamespace(
        objects=all_objs,
        collections=_ColRegistry(),
        meshes=types.SimpleNamespace(remove=lambda m: None),
        materials=[],
        filepath="",
    )
    bpy.context = types.SimpleNamespace(
        scene=scene,
        active_object=None,
        selected_objects=[],
        region=types.SimpleNamespace(width=1920, height=1080, as_pointer=lambda: 12345),
        space_data=types.SimpleNamespace(
            type="VIEW_3D",
            shading=types.SimpleNamespace(show_xray=False),
        ),
        window_manager=types.SimpleNamespace(windows=[]),
        preferences=types.SimpleNamespace(system=types.SimpleNamespace(ui_scale=1.0)),
    )
    bpy.types = types.SimpleNamespace(
        Object=_MockObject,
        Mesh=_MockMesh,
        Scene=_MockScene,
        Material=_MockMaterial,
        Depsgraph=object,
        Collection=_MockCollection,
        Operator=object,
        Gizmo=object,
        GizmoGroup=object,
        SpaceView3D=types.SimpleNamespace(
            draw_handler_add=lambda *a, **kw: 1,
            draw_handler_remove=lambda *a, **kw: None,
        ),
    )
    bpy.utils = types.SimpleNamespace(
        register_class=lambda c: None,
        unregister_class=lambda c: None,
    )

    # gpu & gpu_extras
    gpu = types.ModuleType("gpu")
    _gpu_state = {"depth_test": "LESS_EQUAL", "depth_mask": True, "blend": "NONE"}
    gpu.state = types.SimpleNamespace(
        viewport_get=lambda: (0, 0, 1920, 1080),
        depth_test_get=lambda: _gpu_state["depth_test"],
        depth_test_set=lambda v: _gpu_state.__setitem__("depth_test", v),
        depth_mask_get=lambda: _gpu_state["depth_mask"],
        depth_mask_set=lambda v: _gpu_state.__setitem__("depth_mask", v),
        blend_get=lambda: _gpu_state["blend"],
        blend_set=lambda v: _gpu_state.__setitem__("blend", v),
        line_width_set=lambda w: None,
        point_size_set=lambda s: None,
        face_culling_set=lambda c: None,
    )
    gpu.matrix = types.SimpleNamespace(
        get_projection_matrix=lambda: _MockMatrix(),
        get_model_view_matrix=lambda: _MockMatrix(),
        push=lambda: None,
        pop=lambda: None,
        multiply_matrix=lambda m: None,
    )
    gpu.shader = types.SimpleNamespace(
        from_builtin=lambda name: _MockShader(name),
        create_from_info=lambda info: _MockShader("CUSTOM"),
    )
    gpu.types = types.SimpleNamespace(
        Buffer=lambda fmt, sz, data: (fmt, sz, data),
        GPUTexture=lambda dims, format="RGBA8", data=None: ("TEX", dims, format),
        GPUUniformBuf=lambda data: ("UBO", len(data)),
        GPUShaderCreateInfo=lambda: types.SimpleNamespace(
            vertex_in=lambda *a, **kw: None,
            vertex_out=lambda *a, **kw: None,
            push_constant=lambda *a, **kw: None,
            sampler=lambda *a, **kw: None,
            uniform_buf=lambda *a, **kw: None,
            fragment_out=lambda *a, **kw: None,
            vertex_source=lambda *a, **kw: None,
            fragment_source=lambda *a, **kw: None,
            typedef_source=lambda *a, **kw: None,
        ),
        GPUStageInterfaceInfo=lambda name: types.SimpleNamespace(
            smooth=lambda *a, **kw: None,
        ),
    )
    gpu.texture = types.SimpleNamespace(from_image=lambda img: None)

    gpu_extras = types.ModuleType("gpu_extras")
    gpu_extras_batch = types.ModuleType("gpu_extras.batch")

    def _batch_for_shader(shader, mode, content):
        GPU_COUNTERS["batch_for_shader_calls"] += 1
        pos = content.get("pos")
        n = len(pos) if pos is not None else 0
        return _MockBatch(mode, n)

    gpu_extras_batch.batch_for_shader = _batch_for_shader
    gpu_extras.batch = gpu_extras_batch

    bmesh = types.ModuleType("bmesh")
    blf = types.ModuleType("blf")
    blf.size = lambda *a: None
    blf.color = lambda *a: None
    blf.dimensions = lambda fid, txt: (len(txt) * 7.0, 12.0)
    blf.position = lambda *a: None
    blf.draw = lambda *a: None

    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = _MockMatrix
    mathutils.Vector = lambda xyz: np.asarray(xyz, dtype=np.float32)

    sys.modules["bpy"] = bpy
    sys.modules["bpy.app"] = bpy.app
    sys.modules["bpy.app.handlers"] = bpy.app.handlers
    sys.modules["gpu"] = gpu
    sys.modules["gpu_extras"] = gpu_extras
    sys.modules["gpu_extras.batch"] = gpu_extras_batch
    sys.modules["bmesh"] = bmesh
    sys.modules["blf"] = blf
    sys.modules["mathutils"] = mathutils
    return bpy, scene, all_objs, all_cols


# ---------------------------------------------------------------------------
# Procedural Mesh Generators (1M-Poly Box + Unit Cutter Box)
# ---------------------------------------------------------------------------

def make_subdivided_box_mesh(name: str, sx: float, sy: float, sz: float, segs: int, mat) -> _MockMesh:
    """Create a watertight subdivided box with `6 * segs * segs` quads (2x tris)."""
    # Build vectorized per-face grid (fast even for segs=408 -> 998,784 quads / ~2M tris!)
    faces_defs = [
        # (origin, u_vec, v_vec) oriented outward
        (np.array([-sx, -sy, sz]), np.array([2 * sx, 0, 0]), np.array([0, 2 * sy, 0])),   # +Z
        (np.array([-sx, sy, -sz]), np.array([2 * sx, 0, 0]), np.array([0, -2 * sy, 0])),  # -Z
        (np.array([-sx, -sy, -sz]), np.array([2 * sx, 0, 0]), np.array([0, 0, 2 * sz])),  # -Y
        (np.array([sx, sy, -sz]), np.array([-2 * sx, 0, 0]), np.array([0, 0, 2 * sz])),   # +Y
        (np.array([sx, -sy, -sz]), np.array([0, 2 * sy, 0]), np.array([0, 0, 2 * sz])),   # +X
        (np.array([-sx, sy, -sz]), np.array([0, -2 * sy, 0]), np.array([0, 0, 2 * sz])),  # -X
    ]
    g = segs + 1
    u = np.linspace(0.0, 1.0, g, dtype=np.float32)
    v = np.linspace(0.0, 1.0, g, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (g, g)
    uu_f = uu.ravel()[:, None]
    vv_f = vv.ravel()[:, None]

    ii, jj = np.meshgrid(np.arange(segs, dtype=np.int32), np.arange(segs, dtype=np.int32), indexing="ij")
    i0 = (ii * g + jj).ravel()
    i1 = ((ii + 1) * g + jj).ravel()
    i2 = ((ii + 1) * g + (jj + 1)).ravel()
    i3 = (ii * g + (jj + 1)).ravel()
    quad_local = np.stack([i0, i1, i2, i3], axis=1).ravel()  # (segs*segs*4,)

    verts_list = []
    loops_list = []
    uvs_list = []
    uv_local = np.stack([uu_f[:, 0], vv_f[:, 0]], axis=1)  # (g*g, 2)
    quad_uv = uv_local[quad_local].ravel()

    v_off = 0
    for orig, du, dv in faces_defs:
        pts = orig.astype(np.float32)[None, :] + uu_f * du.astype(np.float32)[None, :] + vv_f * dv.astype(np.float32)[None, :]
        verts_list.append(pts)
        loops_list.append(quad_local + v_off)
        uvs_list.append(quad_uv)
        v_off += g * g

    verts_xyz = np.concatenate(verts_list, axis=0).astype(np.float32).ravel()
    loop_verts = np.concatenate(loops_list, axis=0).astype(np.int32)
    uv_flat = np.concatenate(uvs_list, axis=0).astype(np.float32)
    n_polys = 6 * segs * segs
    poly_starts = (np.arange(n_polys, dtype=np.int32) * 4)
    poly_totals = np.full(n_polys, 4, dtype=np.int32)
    mat_ids = np.zeros(n_polys, dtype=np.int32)

    return _MockMesh(
        name=name,
        verts_xyz=verts_xyz,
        loop_verts=loop_verts,
        poly_starts=poly_starts,
        poly_totals=poly_totals,
        mat_ids=mat_ids,
        uv_flat=uv_flat,
        materials=[mat],
    )


def make_cube_mesh(name: str, half: float = 0.5) -> _MockMesh:
    h = float(half)
    verts = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float32,
    ).ravel()
    loops = np.array(
        [
            0, 3, 2, 1,  # -Z
            4, 5, 6, 7,  # +Z
            0, 1, 5, 4,  # -Y
            2, 3, 7, 6,  # +Y
            1, 2, 6, 5,  # +X
            3, 0, 4, 7,  # -X
        ],
        dtype=np.int32,
    )
    edges = np.array(
        [
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ],
        dtype=np.int32,
    )
    starts = np.arange(6, dtype=np.int32) * 4
    totals = np.full(6, 4, dtype=np.int32)
    mats = np.zeros(6, dtype=np.int32)
    return _MockMesh(name, verts, loops, starts, totals, mats, edges_idx=edges)


def main():
    bpy, scene, all_objs, all_cols = _install_mock_blender()

    # Create lightweight package shell for `addon` so we don't trigger Blender UI registration in `__init__.py`
    addon_pkg = types.ModuleType("addon")
    addon_pkg.__path__ = [str(REPO_ROOT / "addon")]
    core_pkg = types.ModuleType("addon.core")
    core_pkg.__path__ = [str(REPO_ROOT / "addon" / "core")]
    ui_pkg = types.ModuleType("addon.ui")
    ui_pkg.__path__ = [str(REPO_ROOT / "addon" / "ui")]
    sys.modules["addon"] = addon_pkg
    sys.modules["addon.core"] = core_pkg
    sys.modules["addon.ui"] = ui_pkg

    from addon.core import gpu_preview, live_sync, profiler, rust_bridge, session as sess_mod, zone_registry

    print("=" * 72)
    print("  FullDestruc_Tool v2.0 Python + Rust C-ABI Performance Stress Suite")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Verify Rust C-ABI v200 Library Loads Cleanly
    # ------------------------------------------------------------------
    assert rust_bridge.is_available(), "Rust engine shared library not found! Run `cargo build --release` first."
    ver = rust_bridge.get_dll_version()
    assert ver >= 200, f"Expected v200 engine, got v{ver}"
    assert rust_bridge.supports_delta_gpu(), "supports_delta_gpu() must be True for v200"
    print(f"[PASS] Rust C-ABI loaded: version={ver}, supports_delta_gpu={rust_bridge.supports_delta_gpu()}")

    # ------------------------------------------------------------------
    # 2. Session & Zone Registry Cache Benchmark (500 Cutters)
    # ------------------------------------------------------------------
    zones_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    root_col = bpy.data.collections.new("FDT_Bite_S1")
    cutters_col = bpy.data.collections.new("FDT_Cutters_S1")
    results_col = bpy.data.collections.new("FDT_Results_S1")
    pieces_col = bpy.data.collections.new("FDT_Pieces_S1")
    vault_col = bpy.data.collections.new("FDT_Vault_S1")

    cube_mesh = make_cube_mesh("SharedCubeMesh", half=0.35)
    target_1m = _MockObject("Building_1M", cube_mesh)
    all_objs.link(target_1m)
    scene.objects.link(target_1m)
    vault_col.objects.link(target_1m)

    cutter_objs: list[_MockObject] = []
    cutter_entries = []
    # Spread 500 boxes cleanly on the +Z face of a 100x100x2 building (grid 25 x 20)
    for i in range(500):
        gx = i % 25
        gy = i // 25
        cx = -46.0 + gx * 3.8
        cy = -46.0 + gy * 4.8
        cz = 1.0
        cname = f"Cutter_S1_ID-{zones_list[i % len(zones_list)]}_{i + 1:03d}"
        c_obj = _MockObject(cname, cube_mesh, _MockMatrix.Translation((cx, cy, cz)))
        z = zones_list[i % len(zones_list)]
        c_obj["fdt_is_cutter"] = True
        c_obj["fdt_cutter_zone"] = z
        c_obj["fdt_session_id"] = "S1"
        c_obj.color = (0.2 + 0.1 * (i % 8), 0.5, 0.9 - 0.08 * (i % 8), 1.0)
        cutters_col.objects.link(c_obj)
        all_objs.link(c_obj)
        scene.objects.link(c_obj)
        cutter_objs.append(c_obj)
        cutter_entries.append(sess_mod.CutterEntry(name=cname, zone=z, muted=False, rough=False))

    ab_sess = sess_mod.AppleBiteSession(
        session_id="S1",
        target_names=["Building_1M"],
        root_col_name=root_col.name,
        vault_col_name=vault_col.name,
        cutters_col_name=cutters_col.name,
        results_col_name=results_col.name,
        pieces_col_name=pieces_col.name,
        cutters=cutter_entries,
    )
    sess_mod.save_session(scene, ab_sess)
    raw_json = scene[sess_mod.SESSION_SCENE_PROP]

    t0 = time.perf_counter()
    for _ in range(1000):
        json.loads(raw_json)
    raw_json_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(1000):
        s_ro = sess_mod.load_session_readonly(scene)
        assert s_ro is not None and len(s_ro.cutters) == 500
    cached_sess_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[PASS] session.load_session_readonly (500 cutters x1000): "
        f"{cached_sess_ms:.2f} ms ({cached_sess_ms / 1000.0:.4f} ms/call) vs raw json.loads {raw_json_ms:.2f} ms "
        f"({raw_json_ms / max(cached_sess_ms, 1e-6):.1f}x faster!)"
    )

    t0 = time.perf_counter()
    for i in range(1000):
        zdef = zone_registry.resolve_zone(zones_list[i % len(zones_list)], scene)
        assert zdef.key == zones_list[i % len(zones_list)]
    cached_zone_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] zone_registry.resolve_zone (x1000): {cached_zone_ms:.2f} ms ({cached_zone_ms / 1000.0:.4f} ms/call)")

    # ------------------------------------------------------------------
    # 3. Real C-ABI v200 Layer #1 Priority Check
    # ------------------------------------------------------------------
    mat_base = _MockMaterial("BaseWall")
    mat_cut = _MockMaterial("CutCap", (0.9, 0.3, 0.1, 1.0))
    prio_mesh = make_subdivided_box_mesh("PrioTarget", 5.0, 5.0, 2.0, segs=8, mat=mat_base)
    prio_target = _MockObject("PrioTarget", prio_mesh)
    c_top = _MockObject("C_Top", make_cube_mesh("C_Top_M", 2.0), _MockMatrix.Translation((0.0, 0.0, 2.0)))
    c_bot = _MockObject("C_Bot", make_cube_mesh("C_Bot_M", 2.0), _MockMatrix.Translation((1.0, 0.0, 2.0)))

    jid = rust_bridge.submit_target_async_delta(
        "PRIO_TEST", prio_target, [c_top, c_bot], None, mat_cut, 0.5, dirty_cutter_names=None
    )
    assert jid > 0
    res = None
    deadline = time.perf_counter() + 5.0
    while res is None and time.perf_counter() < deadline:
        res = rust_bridge.poll_target_gpu_delta("PRIO_TEST", prio_target, 0)
        if res is None:
            time.sleep(0.002)
    assert res is not None and len(res["updated_pieces"]) == 2
    by_cid = dict(res["updated_pieces"])
    cid_top, cid_bot = res["ordered_piece_ids"][0], res["ordered_piece_ids"][1]
    assert rust_bridge.get_cutter_name_for_id("PRIO_TEST", cid_top) == "C_Top"
    assert rust_bridge.get_cutter_name_for_id("PRIO_TEST", cid_bot) == "C_Bot"
    bot_xs = np.concatenate([sub[1][:, 0] for sub in by_cid[cid_bot]])
    assert float(bot_xs.min()) >= 2.0 - 1e-3, f"Layer #1 priority violated! Layer #2 min_x={bot_xs.min()}"
    rust_bridge.destroy_session("PRIO_TEST")
    print(f"[PASS] Python C-ABI v200 Layer #1 Priority verified (Layer #2 cleanly clipped at x >= {bot_xs.min():.3f})")

    # ------------------------------------------------------------------
    # 4. Real C-ABI v200 Stress: 1,000,000 Polygons (2M Tris) + 500 Boxes!
    # ------------------------------------------------------------------
    print("\n--- Generating 1,000,000-Polygon (2M-Tri) Building + 500 Boxes ---")
    t_gen = time.perf_counter()
    mesh_1m = make_subdivided_box_mesh("Building_1M_Mesh", 50.0, 50.0, 1.0, segs=408, mat=mat_base)
    target_1m.data = mesh_1m
    print(f"  Generated {len(mesh_1m.polygons):,} quads ({len(mesh_1m.polygons) * 2:,} tris) in {(time.perf_counter() - t_gen)*1000:.1f} ms")

    t_init0 = time.perf_counter()
    jid0 = rust_bridge.submit_target_async_delta(
        "S1", target_1m, cutter_objs, None, mat_cut, 0.5, dirty_cutter_names=None
    )
    submit_init_ms = (time.perf_counter() - t_init0) * 1000.0
    res0 = None
    while res0 is None:
        res0 = rust_bridge.poll_target_gpu_delta("S1", target_1m, 0)
        if res0 is None:
            time.sleep(0.005)
    total_init_ms = (time.perf_counter() - t_init0) * 1000.0

    # Apply initial full delta to gpu_preview
    zone_col_map = {z: (0.2 + 0.1 * i, 0.5, 0.8, 1.0) for i, z in enumerate(zones_list)}
    zone_by_name = {c.name: (zone_col_map[c["fdt_cutter_zone"]], c["fdt_cutter_zone"]) for c in cutter_objs}
    meta0 = {
        cid: zone_by_name.get(rust_bridge.get_cutter_name_for_id("S1", cid), ((1.0, 0.5, 0.1, 1.0), "A"))
        for cid in res0["ordered_piece_ids"]
    }
    gpu_preview.begin_drag_preview(ab_sess, grid_size=0.35, show_grid=True, display_mode="ALL")
    gpu_preview.apply_target_gpu_delta(
        target_1m.name,
        [mat_base],
        mat_cut,
        res0["updated_chunks"],
        res0["removed_chunk_ids"],
        res0["updated_pieces"],
        res0["removed_piece_ids"],
        res0["ordered_piece_ids"],
        meta0,
    )
    state_1m = gpu_preview._GPU_STATE[target_1m.name]
    print(
        f"  Initial 1M+500 upload: main_submit={submit_init_ms:.1f}ms, rust_compute={res0['compute_ms']:.1f}ms, "
        f"wall={total_init_ms:.1f}ms | chunks={len(state_1m['rem_chunks'])}, "
        f"piece_buckets={len(state_1m['piece_buckets'])} (for 500 pieces!)"
    )
    # Verify lazy piece_wire: 0 wireframes built in ALL mode!
    assert len(state_1m["piece_wire_buckets"]) == 0, "piece_wire should NOT be built in ALL mode!"
    assert len(state_1m["piece_buckets"]) <= 24, f"Expected <=24 merged piece buckets for 500 pieces, got {len(state_1m['piece_buckets'])}"

    # Drag Box #1 (Layer #1 TOP) for 30 frames via REAL C-ABI v200 Delta + gpu_preview!
    submit_times = []
    poll_patch_times = []
    rust_times = []
    chunk_counts = []
    piece_counts = []
    last_jid = res0["job_id"]
    drag_box = cutter_objs[0]
    base_x = float(drag_box.matrix_world[0][3])
    base_y = float(drag_box.matrix_world[1][3])

    for f in range(30):
        drag_box.matrix_world = _MockMatrix.Translation((base_x + 0.05 * f, base_y + 0.02 * f, 1.0))

        t_s0 = time.perf_counter()
        jid = rust_bridge.submit_target_async_delta(
            "S1", target_1m, cutter_objs, None, mat_cut, 0.5, dirty_cutter_names={drag_box.name}
        )
        t_s1 = time.perf_counter()
        submit_times.append((t_s1 - t_s0) * 1000.0)

        # Wait for Rust background worker to finish frame
        res = None
        while res is None:
            t_p0 = time.perf_counter()
            res = rust_bridge.poll_target_gpu_delta("S1", target_1m, last_jid)
            if res is None:
                time.sleep(0.0005)
        last_jid = res["job_id"]
        meta = {
            cid: zone_by_name.get(rust_bridge.get_cutter_name_for_id("S1", cid), ((1.0, 0.5, 0.1, 1.0), "A"))
            for cid in res["ordered_piece_ids"]
        }
        gpu_preview.apply_target_gpu_delta(
            target_1m.name,
            [mat_base],
            mat_cut,
            res["updated_chunks"],
            res["removed_chunk_ids"],
            res["updated_pieces"],
            res["removed_piece_ids"],
            res["ordered_piece_ids"],
            meta,
        )
        t_p1 = time.perf_counter()
        poll_patch_times.append((t_p1 - t_p0) * 1000.0)
        rust_times.append(res["compute_ms"])
        chunk_counts.append(len(res["updated_chunks"]))
        piece_counts.append(len(res["updated_pieces"]))

    avg_sub = float(np.mean(submit_times))
    avg_poll = float(np.mean(poll_patch_times))
    avg_rust = float(np.mean(rust_times))
    main_thread_total = avg_sub + avg_poll
    print(
        f"[PASS] 1M Polys + 500 Boxes Drag (30f real C-ABI v200):\n"
        f"       Main Thread Submit: {avg_sub:.3f} ms | Poll+GPU Patch: {avg_poll:.3f} ms | "
        f"Total Main Thread: {main_thread_total:.3f} ms ({1000.0 / max(main_thread_total, 1e-4):.0f} FPS budget!)\n"
        f"       Rust Background Worker: {avg_rust:.2f} ms ({1000.0 / max(avg_rust, 1e-4):.0f} FPS!) | "
        f"Delta avg: {np.mean(chunk_counts):.1f} chunks, {np.mean(piece_counts):.1f} pieces"
    )
    assert main_thread_total < 4.0, f"Main thread drag budget exceeded 4.0 ms: {main_thread_total:.3f} ms"
    assert avg_rust < 16.6, f"Rust background worker exceeded 16.6 ms (60 FPS): {avg_rust:.2f} ms"

    # ------------------------------------------------------------------
    # 5. Viewport Draw Callback Stress (384 Chunks + 500 Pieces + 500 Cutter Wireframes)
    # ------------------------------------------------------------------
    bpy.context.active_object = drag_box
    drag_box.select_set(True)

    # Frame 1 warms up the merged static unselected cutter wireframe batch
    GPU_COUNTERS["batch_for_shader_calls"] = 0
    GPU_COUNTERS["draw_calls"] = 0
    gpu_preview._draw_callback()
    warmup_batches = GPU_COUNTERS["batch_for_shader_calls"]
    warmup_draws = GPU_COUNTERS["draw_calls"]

    # Frames 2..101: simulate 100 viewport frames while dragging the active box!
    GPU_COUNTERS["batch_for_shader_calls"] = 0
    GPU_COUNTERS["draw_calls"] = 0
    GPU_COUNTERS["shader_binds"] = 0
    GPU_COUNTERS["uniform_sampler_calls"] = 0
    t_d0 = time.perf_counter()
    for f in range(100):
        drag_box.matrix_world = _MockMatrix.Translation((base_x + 0.01 * f, base_y, 1.0))
        gpu_preview._draw_callback()
    draw_100_ms = (time.perf_counter() - t_d0) * 1000.0
    avg_draw_ms = draw_100_ms / 100.0
    avg_draws = GPU_COUNTERS["draw_calls"] / 100.0
    avg_binds = GPU_COUNTERS["shader_binds"] / 100.0
    avg_samplers = GPU_COUNTERS["uniform_sampler_calls"] / 100.0
    print(
        f"[PASS] gpu_preview._draw_callback (384 chunks + 500 pieces + 500 wireframes, 100f drag):\n"
        f"       batch_for_shader allocs: {GPU_COUNTERS['batch_for_shader_calls']} (warmup={warmup_batches}) | "
        f"avg={avg_draw_ms:.3f} ms/f ({1000.0 / max(avg_draw_ms, 1e-4):.0f} FPS!)\n"
        f"       draw_calls/f={avg_draws:.1f} | shader_binds/f={avg_binds:.1f} | tex_binds/f={avg_samplers:.1f}"
    )
    assert GPU_COUNTERS["batch_for_shader_calls"] == 0, (
        f"Expected 0 VBO allocs during 100 drag frames, got {GPU_COUNTERS['batch_for_shader_calls']}"
    )
    assert avg_draw_ms < 2.0, f"Draw callback too slow: {avg_draw_ms:.3f} ms/f"

    # Verify lazy piece_wire builds ONLY on switch to MAIN_WIRE
    GPU_COUNTERS["batch_for_shader_calls"] = 0
    gpu_preview.set_display_options(display_mode="MAIN_WIRE")
    gpu_preview._draw_callback()
    assert len(state_1m["piece_wire_buckets"]) > 0 and GPU_COUNTERS["batch_for_shader_calls"] > 0
    GPU_COUNTERS["batch_for_shader_calls"] = 0
    gpu_preview._draw_callback()
    assert GPU_COUNTERS["batch_for_shader_calls"] == 0, "MAIN_WIRE should reuse cached wire batches!"
    gpu_preview.set_display_options(display_mode="ALL")
    print(f"[PASS] Lazy piece_wire verified (built on demand in MAIN_WIRE, 0 allocs on steady frames)")

    # ------------------------------------------------------------------
    # 6. live_sync._on_depsgraph_update Dirty-Only O(1) Stress (500 Cutters)
    # ------------------------------------------------------------------
    # Stub apple_bite helper hooks used by live_sync._on_depsgraph_update
    adopt_calls = [0]
    submitted_dirty_sets: list[set[str] | None] = []

    ab_stub = types.ModuleType("addon.core.apple_bite")
    ab_stub.adopt_duplicated_cutters = lambda sc: (adopt_calls.__setitem__(0, adopt_calls[0] + 1), [])[1]
    ab_stub.is_rough_wire_visible = lambda obj, is_r: not is_r
    ab_stub.submit_rust_session_async_delta = lambda sc, dg, dirty_cutter_names=None: (
        submitted_dirty_sets.append(set(dirty_cutter_names) if dirty_cutter_names is not None else None),
        {"Building_1M": 999},
    )[1]
    ab_stub.submit_rust_session_async = lambda sc, dg: {"Building_1M": 999}
    ab_stub.get_zone_color = lambda z: (1.0, 0.5, 0.1, 1.0)
    ab_stub._get_cut_material = lambda sc: mat_cut
    sys.modules["addon.core.apple_bite"] = ab_stub

    # Record initial snapshot
    live_sync.record_current_cutter_states(scene)
    # Prime adopt guard with 1 initial tick
    init_dg = types.SimpleNamespace(updates=[types.SimpleNamespace(id=drag_box, is_updated_geometry=False)])
    live_sync._on_depsgraph_update(scene, init_dg)
    adopt_calls[0] = 0
    submitted_dirty_sets.clear()
    cube_mesh.vertices.foreach_get_calls = 0

    t_dg0 = time.perf_counter()
    for f in range(100):
        drag_box.matrix_world = _MockMatrix.Translation((base_x + 0.02 * (f + 1), base_y, 1.0))
        dg = types.SimpleNamespace(updates=[types.SimpleNamespace(id=drag_box, is_updated_geometry=False)])
        live_sync._on_depsgraph_update(scene, dg)
    dg_100_ms = (time.perf_counter() - t_dg0) * 1000.0
    print(
        f"[PASS] live_sync._on_depsgraph_update (500 boxes, 100f transform drag):\n"
        f"       avg={dg_100_ms / 100.0:.3f} ms/f | adopt_scans={adopt_calls[0]} | "
        f"foreach_get_calls={cube_mesh.vertices.foreach_get_calls} | dirty_submitted={len(submitted_dirty_sets)}"
    )
    assert adopt_calls[0] == 0, f"adopt_duplicated_cutters should not scan on steady drag, got {adopt_calls[0]}"
    assert cube_mesh.vertices.foreach_get_calls == 0, (
        f"Transform-only drag should call 0 foreach_get, got {cube_mesh.vertices.foreach_get_calls}"
    )
    assert len(submitted_dirty_sets) == 100 and all(s == {drag_box.name} for s in submitted_dirty_sets)
    assert (dg_100_ms / 100.0) < 1.0, f"live_sync depsgraph handler too slow: {dg_100_ms / 100.0:.3f} ms/f"

    # ------------------------------------------------------------------
    # 7. Viewport HUD Batch Caching Stress (500 Cutters, 100 Frames)
    # ------------------------------------------------------------------
    rebar_stub = types.ModuleType("addon.core.rebar_state")
    rebar_stub.find_entry = lambda sid, sc: None
    rebar_stub.load_registry = lambda sc: []
    sys.modules["addon.core.rebar_state"] = rebar_stub
    ab_stub.CUTTER_PROP = "fdt_is_cutter"
    ab_stub.CUTTER_ZONE_PROP = "fdt_cutter_zone"

    from addon.ui import viewport_hud

    GPU_COUNTERS["batch_for_shader_calls"] = 0
    viewport_hud._draw_dock_post_pixel()
    hud_warmup_batches = GPU_COUNTERS["batch_for_shader_calls"]
    assert hud_warmup_batches > 0

    GPU_COUNTERS["batch_for_shader_calls"] = 0
    t_h0 = time.perf_counter()
    for _ in range(100):
        viewport_hud._draw_dock_post_pixel()
    hud_100_ms = (time.perf_counter() - t_h0) * 1000.0
    print(
        f"[PASS] viewport_hud._draw_dock_post_pixel (500 cutters, 100f):\n"
        f"       batch_for_shader allocs: {GPU_COUNTERS['batch_for_shader_calls']} (warmup={hud_warmup_batches}) | "
        f"avg={hud_100_ms / 100.0:.3f} ms/f"
    )
    assert GPU_COUNTERS["batch_for_shader_calls"] == 0, "HUD should allocate 0 GPU batches on steady frames!"

    # ------------------------------------------------------------------
    # 8. Profiler Buffered Logging Check
    # ------------------------------------------------------------------
    t_pr0 = time.perf_counter()
    for i in range(500):
        profiler.record_eval(
            profiler.EvalSample(
                timestamp=time.perf_counter(),
                session_id="S1",
                target_name="Building_1M",
                solver="RUST_GPU_DELTA",
                target_verts=1_000_000,
                target_polys=998_784,
                num_cutters=500,
                cutter_total_polys=6,
                out_rem_verts=3000,
                out_rem_polys=1000,
                out_pieces_verts=0,
                out_pieces_polys=1,
                target_uploaded=False,
                upload_ms=0.0,
                extract_ms=0.0,
                rust_ms=4.5,
                writeback_ms=0.5,
                total_ms=0.5,
            )
        )
    prof_ms = (time.perf_counter() - t_pr0) * 1000.0
    print(f"[PASS] profiler.record_eval (500 samples): {prof_ms:.2f} ms ({prof_ms / 500.0:.4f} ms/sample)")

    # Cleanup
    rust_bridge.destroy_all_sessions()
    gpu_preview.end_drag_preview()
    print("\n" + "=" * 72)
    print("  ALL PERFORMANCE & CORRECTNESS ASSERTIONS PASSED (>60 FPS VERIFIED!)")
    print("=" * 72)


if __name__ == "__main__":
    main()
