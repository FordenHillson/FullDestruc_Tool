"""On-Demand Single-View G-Buffer Imposter Drag Renderer for FullDestruc_Tool (v2.3).

Eliminates live-drag Boolean & VBO upload latency on ultra-high-poly meshes (1M-10M+ tris):
- Captures the current viewport angle ONCE on demand into a lightweight offscreen G-Buffer:
  1. `color_tex` (`RGBA8`, ~8.3 MB @ 1080p) : Exact shaded texture color + coverage alpha
  2. `pos_tex`   (`RGBA32F`, ~33.2 MB @ 1080p): Exact world_pos.xyz + packed (triplanar_axis, shade)
  3. `depth_tex` (`DEPTH_COMPONENT32F`, ~8.3 MB): Hardware depth written to `gl_FragDepth`
  Total transient VRAM: ~49.8 MB @ 1080p (ZERO pre-baked multi-angle image bank; freed on exit).
- During active cutter drag (`G`/`R`/`S`), renders 1 fullscreen quad (2 triangles, ~0.15 ms):
  - Box Cutters (1-512+): Analytical OBB test in exact Layer #1 Priority via tiny `RGBA32F` data texture.
  - Freehand Concave / Rough Cutters: Exact GPU Depth-Parity Mask (`RGBA16F` additive front/back count
    against captured `depth_tex`) supporting ANY arbitrary closed 3D mesh on both OpenGL & Vulkan.
  - Supports all 4 display modes (`ALL`, `MAIN`, `MAIN_WIRE`, `PIECES`) + World-Space Triplanar Zone Grid
    + pixel-perfect 3D depth occlusion for Blender's X-Ray cutter wireframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader

    HAS_GPU = True
except Exception:
    bpy = None  # type: ignore[assignment]
    gpu = None  # type: ignore[assignment]
    batch_for_shader = None  # type: ignore[assignment]
    HAS_GPU = False


MAX_OBB_CUTTERS = 512

# ---------------------------------------------------------------------------
# Pure-Python / NumPy Cutter Classification & OBB Packing (Unit-Testable!)
# ---------------------------------------------------------------------------


@dataclass
class ImposterCutterDesc:
    name: str
    layer_idx: int
    zone_letter: str
    zone_rgba: tuple[float, float, float, float]
    is_obb: bool
    # 4x4 float32 matrix mapping world_pos -> unit cube [-1, 1]^3 (valid when is_obb=True)
    inv_obb_mat: np.ndarray | None
    selected_in_pieces: bool = True
    # Reference to bpy.types.Object for special-shape (Rough / Freehand) GPU parity pass
    obj_ref: object | None = None
    is_rough: bool = False


def is_local_box_geometry(verts_xyz: np.ndarray, num_polys: int, tol: float = 1e-3) -> tuple[bool, np.ndarray, np.ndarray]:
    """Return `(is_box, center_xyz, half_extents_xyz)` for local vertex coordinates."""
    if verts_xyz.ndim != 2 or verts_xyz.shape[1] != 3 or verts_xyz.shape[0] != 8 or num_polys != 6:
        return False, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)

    mn = verts_xyz.min(axis=0).astype(np.float64)
    mx = verts_xyz.max(axis=0).astype(np.float64)
    half = 0.5 * (mx - mn)
    if np.any(half <= 1e-6):
        return False, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    center = 0.5 * (mn + mx)

    # Every vertex must lie on an AABB corner (|local_norm| == 1 on all 3 axes)
    norm_v = np.abs((verts_xyz.astype(np.float64) - center) / half)
    if not bool(np.all(np.abs(norm_v - 1.0) <= tol)):
        return False, center, half
    return True, center, half


def compute_inv_obb_matrix(
    matrix_world_4x4: np.ndarray,
    local_center: np.ndarray,
    local_half: np.ndarray,
) -> np.ndarray | None:
    """Compute 4x4 float32 matrix mapping World Pos -> unit cube `[-1, 1]^3`."""
    mw = np.asarray(matrix_world_4x4, dtype=np.float64).reshape(4, 4)
    # Build local-to-box transform: p_local = center + diag(half) * u
    local_to_world = np.eye(4, dtype=np.float64)
    local_to_world[0, 0] = float(local_half[0])
    local_to_world[1, 1] = float(local_half[1])
    local_to_world[2, 2] = float(local_half[2])
    local_to_world[0:3, 3] = local_center.astype(np.float64)
    box_to_world = mw @ local_to_world
    try:
        inv_m = np.linalg.inv(box_to_world)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inv_m)):
        return None
    return inv_m.astype(np.float32)


def eval_obb_cpu_point(inv_obb_mat: np.ndarray, world_xyz: Sequence[float]) -> bool:
    """Exact CPU mirror of the GLSL OBB inside test (used in automated tests)."""
    p4 = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float32)
    lp = inv_obb_mat[:3, :] @ p4
    return bool(abs(float(lp[0])) <= 1.0 and abs(float(lp[1])) <= 1.0 and abs(float(lp[2])) <= 1.0)


def pack_obb_cutters_texture_data(
    cutters: Sequence[ImposterCutterDesc],
    max_rows: int = MAX_OBB_CUTTERS,
) -> tuple[np.ndarray, int]:
    """Pack OBB cutters into `(rows, 5, 4)` float32 array for `GPUTexture((5, rows), 'RGBA32F')`.

    Row layout per OBB cutter `i`:
      col 0: inv_obb_mat row 0 (`vec4`)
      col 1: inv_obb_mat row 1 (`vec4`)
      col 2: inv_obb_mat row 2 (`vec4`)
      col 3: `(layer_idx, selected_flag, 0.0, 0.0)`
      col 4: `(zone_r, zone_g, zone_b, zone_a)`
    """
    obbs = [c for c in cutters if c.is_obb and c.inv_obb_mat is not None][:max_rows]
    n = len(obbs)
    rows = max(n, 1)
    arr = np.zeros((rows, 5, 4), dtype=np.float32)
    for i, c in enumerate(obbs):
        inv_m = c.inv_obb_mat
        arr[i, 0, :] = inv_m[0, :]
        arr[i, 1, :] = inv_m[1, :]
        arr[i, 2, :] = inv_m[2, :]
        arr[i, 3, 0] = float(c.layer_idx)
        arr[i, 3, 1] = 1.0 if c.selected_in_pieces else 0.0
        arr[i, 4, 0] = float(c.zone_rgba[0])
        arr[i, 4, 1] = float(c.zone_rgba[1])
        arr[i, 4, 2] = float(c.zone_rgba[2])
        arr[i, 4, 3] = float(c.zone_rgba[3])
    return np.ascontiguousarray(arr, dtype=np.float32), n


def should_activate_imposter(
    mode: str,
    poly_threshold: int,
    total_target_polys: int,
    num_cutters: int,
    cutter_threshold: int = 50,
) -> bool:
    """Decide whether Imposter Drag should activate based on user settings & scene complexity."""
    m = str(mode or "AUTO").upper()
    if m == "OFF":
        return False
    if m == "ON":
        return True
    return (total_target_polys >= max(int(poly_threshold), 0)) or (num_cutters >= cutter_threshold)


# ---------------------------------------------------------------------------
# GPU Shaders & Offscreen G-Buffer State (Blender Runtime)
# ---------------------------------------------------------------------------

_ACTIVE: bool = False
_INIT_FAILED: bool = False
_GBUF_VALID: bool = False
_GBUF_SIZE: tuple[int, int] = (0, 0)
_GBUF_MVP_KEY: tuple[float, ...] | None = None

_COLOR_TEX = None
_POS_TEX = None
_DEPTH_TEX = None
_GBUF_FBO = None

# Lazily allocated only when a Rough/Freehand special cutter is active
_PARITY_TEX = None
_PARITY_FBO = None
_SPECIAL_MASK_TEX = None
_SPECIAL_MASK_FBO = None

_CAPTURE_BASIC_SHADER = None
_CAPTURE_MULTI_SHADER = None
_PARITY_SHADER = None
_PARITY_STAMP_SHADER = None
_COMPOSITE_SHADER = None
_FS_QUAD_BATCH = None
_FS_STAMP_BATCH = None
_DUMMY_SPECIAL_TEX = None

_CUTTER_DESCS: list[ImposterCutterDesc] = []
_OBB_TEX = None
_OBB_COUNT: int = 0
_OBB_TEX_DIRTY: bool = True
_HAS_SPECIAL_CUTTERS: bool = False

# (mesh_ptr, nv, np) -> (is_box, center, half)
_BOX_CLASS_CACHE: dict[tuple[int, int, int], tuple[bool, np.ndarray, np.ndarray]] = {}
# (obj_name, mesh_ptr, nv, np, n_mods, is_rough) -> GPUBatch ("TRIS") in local space
_SPECIAL_MESH_BATCH_CACHE: dict[tuple, object] = {}


def is_active() -> bool:
    return _ACTIVE and not _INIT_FAILED


def invalidate_gbuffer() -> None:
    global _GBUF_VALID, _GBUF_MVP_KEY
    _GBUF_VALID = False
    _GBUF_MVP_KEY = None


def invalidate_cutter_cache(mesh_ptr: int | None = None) -> None:
    if mesh_ptr is None:
        _BOX_CLASS_CACHE.clear()
        _SPECIAL_MESH_BATCH_CACHE.clear()
        return
    for k in list(_BOX_CLASS_CACHE.keys()):
        if k[0] == mesh_ptr:
            _BOX_CLASS_CACHE.pop(k, None)
    for k in list(_SPECIAL_MESH_BATCH_CACHE.keys()):
        if len(k) >= 2 and k[1] == mesh_ptr:
            _SPECIAL_MESH_BATCH_CACHE.pop(k, None)


def _classify_blender_cutter(obj, is_rough: bool) -> tuple[bool, np.ndarray, np.ndarray]:
    if is_rough or obj.mode == "EDIT" or len(obj.modifiers) > 0:
        return False, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    me = getattr(obj, "data", None)
    if me is None or obj.type != "MESH":
        return False, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    nv = len(me.vertices)
    np_cnt = len(me.polygons)
    if nv != 8 or np_cnt != 6:
        return False, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    try:
        mptr = int(me.as_pointer())
    except Exception:
        mptr = id(me)
    ckey = (mptr, nv, np_cnt)
    cached = _BOX_CLASS_CACHE.get(ckey)
    if cached is not None:
        return cached
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    res = is_local_box_geometry(co.reshape(-1, 3), np_cnt)
    _BOX_CLASS_CACHE[ckey] = res
    return res


def update_imposter_cutters(
    active_cutters: Sequence[tuple[object, str]],
    zone_color_fn: Callable[[str], tuple[float, float, float, float]],
    rough_names: set[str] | None = None,
    selected_zones: set[str] | None = None,
) -> None:
    """Refresh OBB matrices & special-shape descriptors for all active cutters (< 0.1 ms)."""
    global _CUTTER_DESCS, _OBB_TEX_DIRTY, _HAS_SPECIAL_CUTTERS
    r_set = rough_names or set()
    sel_z = selected_zones or set()

    descs: list[ImposterCutterDesc] = []
    has_spec = False
    for idx, (c_obj, z_raw) in enumerate(active_cutters):
        z_let = str(z_raw or "A").strip() or "A"
        z_col = zone_color_fn(z_let)
        is_r = (c_obj.name in r_set) or any(m.name.startswith("FDT_Rough") for m in getattr(c_obj, "modifiers", ()))
        is_box, ctr, hlf = _classify_blender_cutter(c_obj, is_r)
        inv_m = None
        if is_box:
            mw = np.asarray(c_obj.matrix_world, dtype=np.float64)
            inv_m = compute_inv_obb_matrix(mw, ctr, hlf)
            if inv_m is None:
                is_box = False
        if not is_box:
            has_spec = True
        descs.append(
            ImposterCutterDesc(
                name=c_obj.name,
                layer_idx=idx,
                zone_letter=z_let,
                zone_rgba=(float(z_col[0]), float(z_col[1]), float(z_col[2]), 1.0),
                is_obb=is_box,
                inv_obb_mat=inv_m,
                selected_in_pieces=(not sel_z) or (z_let in sel_z),
                obj_ref=c_obj,
                is_rough=is_r,
            )
        )

    _CUTTER_DESCS = descs
    _HAS_SPECIAL_CUTTERS = has_spec
    _OBB_TEX_DIRTY = True


def activate_imposter_drag(
    active_cutters: Sequence[tuple[object, str]],
    zone_color_fn: Callable[[str], tuple[float, float, float, float]],
    rough_names: set[str] | None = None,
    selected_zones: set[str] | None = None,
) -> bool:
    """Activate Imposter Drag mode and update active cutter OBB/special descriptors."""
    global _ACTIVE
    if not HAS_GPU or _INIT_FAILED or (bpy is not None and bpy.app.background):
        return False
    update_imposter_cutters(active_cutters, zone_color_fn, rough_names=rough_names, selected_zones=selected_zones)
    _ACTIVE = True
    return True


def deactivate_imposter_drag() -> None:
    """Exit Imposter Drag cleanly (keeps offscreen buffers warm until end_drag_preview)."""
    global _ACTIVE
    _ACTIVE = False


# ---------------------------------------------------------------------------
# Shader Compilation (OpenGL & Vulkan Portable, <= 128B Push Constants)
# ---------------------------------------------------------------------------


def _get_capture_basic_shader():
    global _CAPTURE_BASIC_SHADER, _INIT_FAILED
    if _CAPTURE_BASIC_SHADER is not None:
        return _CAPTURE_BASIC_SHADER
    if _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_imp_cap_basic_iface")
        vert_out.smooth("VEC3", "v_world_pos")
        vert_out.smooth("VEC3", "v_world_nor")
        vert_out.smooth("VEC2", "v_uv")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "nor")
        info.vertex_in(2, "VEC2", "uv")
        info.vertex_out(vert_out)

        info.sampler(0, "FLOAT_2D", "base_tex")
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "base_color")
        info.push_constant("VEC4", "uv_transform")
        info.push_constant("VEC4", "params")
        info.fragment_out(0, "VEC4", "fragColor")
        info.fragment_out(1, "VEC4", "fragWorldPos")

        info.vertex_source(
            """
            void main()
            {
                v_world_pos = pos;
                v_world_nor = normalize(nor);
                v_uv = uv;
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            }
            """
        )

        info.fragment_source(
            """
            void main()
            {
                vec3 n = normalize(v_world_nor);
                if (!gl_FrontFacing) {
                    n = -n;
                }

                vec3 l1 = normalize(vec3(0.45, -0.55, 0.70));
                vec3 l2 = normalize(vec3(-0.60, 0.45, 0.35));
                float ndl1 = max(dot(n, l1), 0.0);
                float ndl2 = max(dot(n, l2), 0.0);
                float hemi = 0.5 + 0.5 * n.z;
                float shade = 0.36 + 0.16 * hemi + 0.38 * ndl1 + 0.14 * ndl2;

                float use_tex = params.x;
                float decode_srgb = params.y;

                vec3 col_lin;
                if (use_tex > 0.5) {
                    vec2 uv = v_uv * uv_transform.xy + uv_transform.zw;
                    vec3 tex_rgb = texture(base_tex, uv).rgb;
                    if (decode_srgb > 0.5) {
                        tex_rgb = pow(max(tex_rgb, vec3(0.0)), vec3(2.2));
                    }
                    col_lin = tex_rgb * base_color.rgb;
                } else {
                    col_lin = base_color.rgb;
                }
                vec3 rgb = col_lin * shade;

                vec3 an = abs(n);
                float axis = (an.z >= an.x && an.z >= an.y) ? 0.0 : ((an.x >= an.y) ? 1.0 : 2.0);
                float packed_meta = axis * 4.0 + clamp(shade, 0.0, 2.0);

                fragColor = vec4(rgb, 1.0);
                fragWorldPos = vec4(v_world_pos, packed_meta);
            }
            """
        )
        _CAPTURE_BASIC_SHADER = gpu.shader.create_from_info(info)
        return _CAPTURE_BASIC_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Imposter capture basic shader failed: {exc}")
        _INIT_FAILED = True
        return None


def _get_capture_multi_shader():
    global _CAPTURE_MULTI_SHADER
    if _CAPTURE_MULTI_SHADER is not None:
        return _CAPTURE_MULTI_SHADER
    if _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_imp_cap_multi_iface")
        vert_out.smooth("VEC3", "v_world_pos")
        vert_out.smooth("VEC3", "v_world_nor")
        vert_out.smooth("VEC2", "v_uv0")
        vert_out.smooth("VEC2", "v_uv1")

        info = gpu.types.GPUShaderCreateInfo()
        info.typedef_source(
            """
            struct MultiMatParams {
                vec4 col1;
                vec4 col2;
                vec4 col3;
                vec4 col4;
                vec4 global_col;
                vec4 uv1_tr;
                vec4 uv2_tr;
                vec4 uv3_tr;
                vec4 uv4_tr;
                vec4 uv_rot;
                vec4 glob_uv_tr;
                vec4 mask_scale;
                vec4 mask_sharp;
                vec4 mask_off;
                vec4 mcr_params;
                vec4 extra_params;
                vec4 srgb_flags;
            };
            """
        )
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "nor")
        info.vertex_in(2, "VEC2", "uv0")
        info.vertex_in(3, "VEC2", "uv1")
        info.vertex_out(vert_out)

        info.sampler(0, "FLOAT_2D", "bcr1_tex")
        info.sampler(1, "FLOAT_2D", "bcr2_tex")
        info.sampler(2, "FLOAT_2D", "bcr3_tex")
        info.sampler(3, "FLOAT_2D", "bcr4_tex")
        info.sampler(4, "FLOAT_2D", "mask_tex")
        info.sampler(5, "FLOAT_2D", "mcr_tex")
        info.uniform_buf(0, "MultiMatParams", "u_mm")

        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.fragment_out(0, "VEC4", "fragColor")
        info.fragment_out(1, "VEC4", "fragWorldPos")

        info.vertex_source(
            """
            void main()
            {
                v_world_pos = pos;
                v_world_nor = normalize(nor);
                v_uv0 = uv0;
                v_uv1 = uv1;
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            }
            """
        )

        info.fragment_source(
            """
            vec2 eval_enf_uv(vec2 uv, vec4 tr, float rot_rad)
            {
                vec2 p = vec2(uv.x, uv.y - 1.0);
                p = vec2(p.x * tr.x + tr.z, p.y * tr.y - tr.w);
                if (abs(rot_rad) > 0.0001) {
                    float c = cos(rot_rad);
                    float s = sin(rot_rad);
                    p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
                }
                return p;
            }

            vec3 sample_bcr(sampler2D tex, vec2 uv, float dec_srgb)
            {
                vec3 c = texture(tex, uv).rgb;
                if (dec_srgb > 0.5) {
                    c = pow(max(c, vec3(0.0)), vec3(2.2));
                }
                return c;
            }

            float modify_mask(float gmask, float sharpness, float offset)
            {
                if (gmask <= 0.001) {
                    return 0.0;
                }
                float t = clamp(gmask + offset, 0.0, 1.0);
                float base = mix(-sharpness, 1.0, t);
                return clamp(0.5 * sharpness + base, 0.0, 1.0);
            }

            vec3 blend_overlay(vec3 a, vec3 b)
            {
                vec3 lo = 2.0 * a * b;
                vec3 hi = vec3(1.0) - 2.0 * (vec3(1.0) - a) * (vec3(1.0) - b);
                return mix(lo, hi, step(vec3(0.5), a));
            }

            vec3 apply_layer(vec3 colA, vec3 colB, float m, float combine_mode)
            {
                if (combine_mode > 0.5) {
                    return mix(colA, blend_overlay(colA, colB), m);
                }
                return mix(colA, colB, m);
            }

            void main()
            {
                vec3 n = normalize(v_world_nor);
                if (!gl_FrontFacing) {
                    n = -n;
                }

                vec2 uv_1 = eval_enf_uv(v_uv0, u_mm.uv1_tr, u_mm.uv_rot.x);
                vec2 uv_2 = eval_enf_uv(v_uv0, u_mm.uv2_tr, u_mm.uv_rot.y);
                vec2 uv_3 = eval_enf_uv(v_uv0, u_mm.uv3_tr, u_mm.uv_rot.z);
                vec2 uv_4 = eval_enf_uv(v_uv0, u_mm.uv4_tr, u_mm.uv_rot.w);
                vec2 uv_g = eval_enf_uv(v_uv1, u_mm.glob_uv_tr, u_mm.mask_sharp.w);

                vec3 c1 = sample_bcr(bcr1_tex, uv_1, u_mm.srgb_flags.x) * u_mm.col1.rgb;
                vec3 c2 = sample_bcr(bcr2_tex, uv_2, u_mm.srgb_flags.y) * u_mm.col2.rgb;
                vec3 c3 = sample_bcr(bcr3_tex, uv_3, u_mm.srgb_flags.z) * u_mm.col3.rgb;
                vec3 c4 = sample_bcr(bcr4_tex, uv_4, u_mm.srgb_flags.w) * u_mm.col4.rgb;

                vec3 gmask = (u_mm.mask_scale.w > 0.5) ? texture(mask_tex, uv_g).rgb : vec3(0.0);

                float m2 = modify_mask(gmask.r * u_mm.mask_scale.x, u_mm.mask_sharp.x, u_mm.mask_off.x) * u_mm.col2.a;
                float m3 = modify_mask(gmask.g * u_mm.mask_scale.y, u_mm.mask_sharp.y, u_mm.mask_off.y) * u_mm.col3.a;
                float m4 = modify_mask(gmask.b * u_mm.mask_scale.z, u_mm.mask_sharp.z, u_mm.mask_off.z) * u_mm.col4.a;

                vec3 col_lin = c1;
                col_lin = apply_layer(col_lin, c2, m2, u_mm.extra_params.y);
                col_lin = apply_layer(col_lin, c3, m3, u_mm.extra_params.z);
                col_lin = apply_layer(col_lin, c4, m4, u_mm.extra_params.w);

                if (u_mm.global_col.a > 0.5) {
                    vec3 mcr = texture(mcr_tex, uv_g).rgb;
                    if (u_mm.extra_params.x > 0.5) {
                        mcr = pow(max(mcr, vec3(0.0)), vec3(2.2));
                    }
                    float mcr_f = clamp(u_mm.mcr_params.x, 0.0, 1.0);
                    float mcr_p = max(u_mm.mcr_params.y, 0.1);
                    vec3 mcr_mod = pow(max(mcr, vec3(0.0001)), vec3(mcr_p));
                    if (u_mm.mcr_params.z > 0.5) {
                        col_lin = mix(col_lin, col_lin * mcr_mod, mcr_f);
                    } else {
                        col_lin = mix(col_lin, blend_overlay(col_lin, mcr_mod), mcr_f);
                    }
                }
                col_lin *= u_mm.global_col.rgb;

                vec3 l1 = normalize(vec3(0.45, -0.55, 0.70));
                vec3 l2 = normalize(vec3(-0.60, 0.45, 0.35));
                float ndl1 = max(dot(n, l1), 0.0);
                float ndl2 = max(dot(n, l2), 0.0);
                float hemi = 0.5 + 0.5 * n.z;
                float shade = 0.36 + 0.16 * hemi + 0.38 * ndl1 + 0.14 * ndl2;

                vec3 rgb = col_lin * shade;
                vec3 an = abs(n);
                float axis = (an.z >= an.x && an.z >= an.y) ? 0.0 : ((an.x >= an.y) ? 1.0 : 2.0);
                float packed_meta = axis * 4.0 + clamp(shade, 0.0, 2.0);

                fragColor = vec4(rgb, 1.0);
                fragWorldPos = vec4(v_world_pos, packed_meta);
            }
            """
        )
        _CAPTURE_MULTI_SHADER = gpu.shader.create_from_info(info)
        return _CAPTURE_MULTI_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Imposter capture multi shader failed: {exc}")
        return None


def _get_parity_shader():
    global _PARITY_SHADER
    if _PARITY_SHADER is not None:
        return _PARITY_SHADER
    if _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "pos")
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.fragment_out(0, "VEC4", "fragParity")
        info.vertex_source(
            """
            void main()
            {
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            }
            """
        )
        info.fragment_source(
            """
            void main()
            {
                fragParity = gl_FrontFacing ? vec4(1.0, 0.0, 0.0, 0.0) : vec4(0.0, 1.0, 0.0, 0.0);
            }
            """
        )
        _PARITY_SHADER = gpu.shader.create_from_info(info)
        return _PARITY_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Imposter parity shader failed: {exc}")
        return None


def _get_parity_stamp_shader():
    global _PARITY_STAMP_SHADER
    if _PARITY_STAMP_SHADER is not None:
        return _PARITY_STAMP_SHADER
    if _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_imp_stamp_iface")
        vert_out.smooth("VEC2", "v_uv")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC2", "pos")
        info.vertex_out(vert_out)
        info.sampler(0, "FLOAT_2D", "parity_tex")
        info.push_constant("VEC4", "stamp_meta")  # (signed_layer_plus1, z_r, z_g, z_b)
        info.fragment_out(0, "VEC4", "fragMask")

        info.vertex_source(
            """
            void main()
            {
                v_uv = pos * 0.5 + 0.5;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
            """
        )
        info.fragment_source(
            """
            void main()
            {
                vec2 p = texture(parity_tex, v_uv).rg;
                if ((p.g - p.r) <= 0.5) {
                    discard;
                }
                fragMask = stamp_meta;
            }
            """
        )
        _PARITY_STAMP_SHADER = gpu.shader.create_from_info(info)
        return _PARITY_STAMP_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Imposter parity stamp shader failed: {exc}")
        return None


def _get_composite_shader():
    global _COMPOSITE_SHADER, _INIT_FAILED
    if _COMPOSITE_SHADER is not None:
        return _COMPOSITE_SHADER
    if _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_imp_comp_iface")
        vert_out.smooth("VEC2", "v_uv")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC2", "pos")
        info.vertex_out(vert_out)
        try:
            info.depth_write("ANY")
        except Exception:
            pass

        info.sampler(0, "FLOAT_2D", "color_tex")
        info.sampler(1, "FLOAT_2D", "pos_tex")
        info.sampler(2, "FLOAT_2D", "depth_tex")
        info.sampler(3, "FLOAT_2D", "obb_tex")
        info.sampler(4, "FLOAT_2D", "special_mask_tex")

        # 32 bytes total push constants (well within 128B Vulkan limit)
        # u_params0 = (obb_count, has_special_mask, display_mode_id, show_grid)
        # u_params1 = (grid_size, grid_strength, 0.0, 0.0)
        info.push_constant("VEC4", "u_params0")
        info.push_constant("VEC4", "u_params1")
        info.fragment_out(0, "VEC4", "fragColor")

        info.vertex_source(
            """
            void main()
            {
                v_uv = pos * 0.5 + 0.5;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
            """
        )

        info.fragment_source(
            """
            void main()
            {
                vec4 base_c = texture(color_tex, v_uv);
                if (base_c.a < 0.5) {
                    discard;
                }

                float d = texture(depth_tex, v_uv).r;
                vec4 wp_meta = texture(pos_tex, v_uv);
                vec3 wp = wp_meta.xyz;
                float axis = floor(wp_meta.w / 4.0 + 0.001);
                float shade = wp_meta.w - axis * 4.0;

                int obb_count = int(u_params0.x + 0.5);
                bool has_spec = u_params0.y > 0.5;
                int disp_mode = int(u_params0.z + 0.5); // 0=ALL, 1=MAIN/MAIN_WIRE, 2=PIECES
                bool show_grid = u_params0.w > 0.5;

                int best_layer = 999999;
                vec3 hit_zcol = vec3(1.0, 0.5, 0.12);
                bool hit_sel = true;

                if (has_spec) {
                    vec4 sm = texture(special_mask_tex, v_uv);
                    float raw_l = abs(sm.x);
                    if (raw_l > 0.5) {
                        best_layer = int(raw_l + 0.5) - 1;
                        hit_zcol = sm.yzw;
                        hit_sel = (sm.x > 0.0);
                    }
                }

                vec4 wp4 = vec4(wp, 1.0);
                for (int i = 0; i < obb_count; ++i) {
                    vec4 meta = texelFetch(obb_tex, ivec2(3, i), 0);
                    int lidx = int(meta.x + 0.5);
                    if (lidx >= best_layer) {
                        break;
                    }
                    vec4 r0 = texelFetch(obb_tex, ivec2(0, i), 0);
                    float lx = dot(r0, wp4);
                    if (abs(lx) > 1.0) continue;
                    vec4 r1 = texelFetch(obb_tex, ivec2(1, i), 0);
                    float ly = dot(r1, wp4);
                    if (abs(ly) > 1.0) continue;
                    vec4 r2 = texelFetch(obb_tex, ivec2(2, i), 0);
                    float lz = dot(r2, wp4);
                    if (abs(lz) > 1.0) continue;

                    best_layer = lidx;
                    hit_sel = (meta.y > 0.5);
                    hit_zcol = texelFetch(obb_tex, ivec2(4, i), 0).rgb;
                    break;
                }

                bool is_hit = (best_layer < 999999);
                if (disp_mode == 1) {
                    // MAIN / MAIN_WIRE: hide pixels inside any active cutter
                    if (is_hit) {
                        discard;
                    }
                    gl_FragDepth = d;
                    fragColor = vec4(base_c.rgb, 1.0);
                    return;
                }
                if (disp_mode == 2) {
                    // PIECES: show only pixels inside selected zone cutter(s)
                    if (!is_hit || !hit_sel) {
                        discard;
                    }
                }

                vec3 rgb = base_c.rgb;
                if (is_hit && show_grid) {
                    rgb = mix(rgb, hit_zcol * shade * 1.1, 0.22);
                    float gs = max(u_params1.x, 0.02);
                    float gstr = u_params1.y;
                    vec2 guv;
                    if (axis < 0.5) {
                        guv = wp.xy / gs;
                    } else if (axis < 1.5) {
                        guv = wp.yz / gs;
                    } else {
                        guv = wp.xz / gs;
                    }
                    vec2 fw = max(fwidth(guv), vec2(0.0005));
                    vec2 grid = abs(fract(guv - 0.5) - 0.5) / fw;
                    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
                    vec3 line_col = mix(vec3(1.0, 0.96, 0.85), hit_zcol + vec3(0.35), 0.5);
                    rgb = mix(rgb, line_col, line * gstr);
                }

                gl_FragDepth = d;
                fragColor = vec4(rgb, 1.0);
            }
            """
        )

        _COMPOSITE_SHADER = gpu.shader.create_from_info(info)
        return _COMPOSITE_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Imposter composite shader failed: {exc}")
        _INIT_FAILED = True
        return None


def _ensure_gbuf_resources(w: int, h: int, need_special: bool) -> bool:
    global _GBUF_SIZE, _GBUF_VALID, _COLOR_TEX, _POS_TEX, _DEPTH_TEX, _GBUF_FBO
    global _PARITY_TEX, _PARITY_FBO, _SPECIAL_MASK_TEX, _SPECIAL_MASK_FBO
    global _FS_QUAD_BATCH, _FS_STAMP_BATCH, _DUMMY_SPECIAL_TEX, _INIT_FAILED

    w = max(int(w), 1)
    h = max(int(h), 1)

    if _GBUF_SIZE != (w, h) or _GBUF_FBO is None:
        _COLOR_TEX = None
        _POS_TEX = None
        _DEPTH_TEX = None
        _GBUF_FBO = None
        _PARITY_TEX = None
        _PARITY_FBO = None
        _SPECIAL_MASK_TEX = None
        _SPECIAL_MASK_FBO = None
        _GBUF_VALID = False
        try:
            _COLOR_TEX = gpu.types.GPUTexture((w, h), format="RGBA8")
            _POS_TEX = gpu.types.GPUTexture((w, h), format="RGBA32F")
            _DEPTH_TEX = gpu.types.GPUTexture((w, h), format="DEPTH_COMPONENT32F")
            _GBUF_FBO = gpu.types.GPUFrameBuffer(
                depth_slot=_DEPTH_TEX,
                color_slots=(_COLOR_TEX, _POS_TEX),
            )
            _GBUF_SIZE = (w, h)
        except Exception as exc:
            print(f"[FullDestruc_Tool] Imposter G-Buffer alloc failed ({w}x{h}): {exc}")
            _INIT_FAILED = True
            return False

    if need_special and (_PARITY_FBO is None or _SPECIAL_MASK_FBO is None):
        try:
            _PARITY_TEX = gpu.types.GPUTexture((w, h), format="RGBA16F")
            _PARITY_FBO = gpu.types.GPUFrameBuffer(
                depth_slot=_DEPTH_TEX,
                color_slots=(_PARITY_TEX,),
            )
            _SPECIAL_MASK_TEX = gpu.types.GPUTexture((w, h), format="RGBA16F")
            _SPECIAL_MASK_FBO = gpu.types.GPUFrameBuffer(color_slots=(_SPECIAL_MASK_TEX,))
        except Exception as exc:
            print(f"[FullDestruc_Tool] Imposter special parity FBO alloc failed: {exc}")
            return False

    if _DUMMY_SPECIAL_TEX is None:
        try:
            buf = gpu.types.Buffer("FLOAT", 4, [0.0, 0.0, 0.0, 0.0])
            _DUMMY_SPECIAL_TEX = gpu.types.GPUTexture((1, 1), format="RGBA16F", data=buf)
        except Exception:
            pass

    if _FS_QUAD_BATCH is None:
        comp_sh = _get_composite_shader()
        if comp_sh is None:
            return False
        quad = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float32,
        )
        _FS_QUAD_BATCH = batch_for_shader(comp_sh, "TRIS", {"pos": quad})

    if need_special and _FS_STAMP_BATCH is None:
        stamp_sh = _get_parity_stamp_shader()
        if stamp_sh is not None:
            quad = np.array(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                dtype=np.float32,
            )
            _FS_STAMP_BATCH = batch_for_shader(stamp_sh, "TRIS", {"pos": quad})

    return True


def _ensure_obb_texture() -> None:
    global _OBB_TEX, _OBB_COUNT, _OBB_TEX_DIRTY
    if not _OBB_TEX_DIRTY and _OBB_TEX is not None:
        return
    arr, n = pack_obb_cutters_texture_data(_CUTTER_DESCS, MAX_OBB_CUTTERS)
    rows = arr.shape[0]
    buf = gpu.types.Buffer("FLOAT", arr.size, arr.ravel())
    _OBB_TEX = gpu.types.GPUTexture((5, rows), format="RGBA32F", data=buf)
    _OBB_COUNT = n
    _OBB_TEX_DIRTY = False


def _get_special_cutter_batch(obj, is_rough: bool, parity_shader):
    """Return cached local-space triangulated `GPUBatch` for a Freehand/Rough cutter."""
    me = getattr(obj, "data", None)
    if me is None or obj.type != "MESH":
        return None
    try:
        mptr = int(me.as_pointer())
    except Exception:
        mptr = id(me)
    n_mods = len(getattr(obj, "modifiers", ()))
    ckey = (obj.name, mptr, len(me.vertices), len(me.polygons), n_mods, bool(is_rough), obj.mode)
    if obj.mode != "EDIT":
        cached = _SPECIAL_MESH_BATCH_CACHE.get(ckey)
        if cached is not None:
            return cached

    eval_me = None
    eval_obj = None
    try:
        if n_mods > 0 or obj.mode == "EDIT":
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            eval_me = eval_obj.to_mesh()
            src_me = eval_me
        else:
            src_me = me
        src_me.calc_loop_triangles()
        n_lt = len(src_me.loop_triangles)
        if n_lt == 0:
            return None
        n_v = len(src_me.vertices)
        v_co = np.empty(n_v * 3, dtype=np.float32)
        src_me.vertices.foreach_get("co", v_co)
        v3 = v_co.reshape(-1, 3)
        tri_idx = np.empty(n_lt * 3, dtype=np.int32)
        src_me.loop_triangles.foreach_get("vertices", tri_idx)
        tri_pos = np.ascontiguousarray(v3[tri_idx], dtype=np.float32)
        batch = batch_for_shader(parity_shader, "TRIS", {"pos": tri_pos})
        if obj.mode != "EDIT":
            _SPECIAL_MESH_BATCH_CACHE[ckey] = batch
        return batch
    except Exception:
        return None
    finally:
        if eval_obj is not None and eval_me is not None:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass


def _render_special_cutter_masks(mvp) -> bool:
    """Render GPU Depth-Parity Mask (`RGBA16F`) for Rough/Freehand cutters in reverse Layer order."""
    if not _HAS_SPECIAL_CUTTERS or _PARITY_FBO is None or _SPECIAL_MASK_FBO is None:
        return False
    parity_sh = _get_parity_shader()
    stamp_sh = _get_parity_stamp_shader()
    if parity_sh is None or stamp_sh is None or _FS_STAMP_BATCH is None:
        return False

    specials = [c for c in _CUTTER_DESCS if not c.is_obb and c.obj_ref is not None]
    if not specials:
        return False

    with _SPECIAL_MASK_FBO.bind():
        _SPECIAL_MASK_FBO.clear(color=(0.0, 0.0, 0.0, 0.0))

    prev_cull = gpu.state.face_culling_get()
    try:
        # Stamp in reverse Layer order so Layer #0 overwrites Layer #1, etc.
        for c in reversed(specials):
            batch = _get_special_cutter_batch(c.obj_ref, c.is_rough, parity_sh)
            if batch is None:
                continue
            # Pass 1: Count front (+R) and back (+G) faces behind captured surface depth
            with _PARITY_FBO.bind():
                _PARITY_FBO.clear(color=(0.0, 0.0, 0.0, 0.0))
                gpu.state.face_culling_set("NONE")
                gpu.state.depth_test_set("GREATER")
                gpu.state.depth_mask_set(False)
                gpu.state.blend_set("ADDITIVE")
                parity_sh.bind()
                parity_sh.uniform_float("ModelViewProjectionMatrix", mvp @ c.obj_ref.matrix_world)
                batch.draw(parity_sh)

            # Pass 2: Stamp inside pixels `(G - R) > 0.5` into special_mask_tex
            with _SPECIAL_MASK_FBO.bind():
                gpu.state.depth_test_set("NONE")
                gpu.state.depth_mask_set(False)
                gpu.state.blend_set("NONE")
                stamp_sh.bind()
                stamp_sh.uniform_sampler("parity_tex", _PARITY_TEX)
                signed_layer = float(c.layer_idx + 1) * (1.0 if c.selected_in_pieces else -1.0)
                stamp_sh.uniform_float(
                    "stamp_meta",
                    (signed_layer, c.zone_rgba[0], c.zone_rgba[1], c.zone_rgba[2]),
                )
                _FS_STAMP_BATCH.draw(stamp_sh)
    finally:
        gpu.state.face_culling_set(prev_cull)

    return True


def _mvp_cache_key(mvp, w: int, h: int) -> tuple:
    try:
        vals = [round(float(mvp[r][c]), 5) for r in range(4) for c in range(4)]
    except Exception:
        return (w, h, 0)
    vals.append(w)
    vals.append(h)
    return tuple(vals)


def draw_imposter_frame(
    mvp,
    display_mode: str,
    show_grid: bool,
    grid_size: float,
    capture_scene_fn: Callable[[object, object], None],
) -> bool:
    """Draw the Imposter frame if active and camera is stationary.

    Returns `True` when the Imposter rendered the mesh (caller skips heavy VBO draw and
    proceeds straight to 3D X-Ray cutter wireframes). Returns `False` when orbiting or
    unavailable so caller draws real VBOs cleanly with zero blur.
    """
    global _GBUF_VALID, _GBUF_MVP_KEY

    if not _ACTIVE or _INIT_FAILED or not HAS_GPU or bpy.app.background:
        return False

    try:
        vp = gpu.state.viewport_get()
        w, h = int(vp[2]), int(vp[3])
    except Exception:
        return False
    if w <= 1 or h <= 1:
        return False

    mvp_k = _mvp_cache_key(mvp, w, h)
    # Camera moved (Orbit/Pan/Zoom): draw real static VBOs smoothly this frame & re-capture once stationary!
    if _GBUF_VALID and _GBUF_MVP_KEY is not None and _GBUF_MVP_KEY != mvp_k:
        _GBUF_VALID = False
        _GBUF_MVP_KEY = mvp_k
        return False

    cap_basic = _get_capture_basic_shader()
    cap_multi = _get_capture_multi_shader()
    comp_sh = _get_composite_shader()
    if cap_basic is None or comp_sh is None:
        return False

    if not _ensure_gbuf_resources(w, h, need_special=_HAS_SPECIAL_CUTTERS):
        return False

    # Capture clean base G-Buffer once per stationary camera angle (~1-3 ms GPU)
    if not _GBUF_VALID:
        try:
            with _GBUF_FBO.bind():
                _GBUF_FBO.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
                gpu.state.depth_test_set("LESS_EQUAL")
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set("NONE")
                capture_scene_fn(cap_basic, cap_multi)
            _GBUF_VALID = True
            _GBUF_MVP_KEY = mvp_k
        except Exception as exc:
            print(f"[FullDestruc_Tool] Imposter G-Buffer capture failed: {exc}")
            _GBUF_VALID = False
            return False

    _ensure_obb_texture()
    has_spec_drawn = _render_special_cutter_masks(mvp) if _HAS_SPECIAL_CUTTERS else False

    mode_u = str(display_mode or "ALL").upper()
    mode_id = 0 if mode_u == "ALL" else (2 if mode_u == "PIECES" else 1)

    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set("NONE")

    comp_sh.bind()
    comp_sh.uniform_sampler("color_tex", _COLOR_TEX)
    comp_sh.uniform_sampler("pos_tex", _POS_TEX)
    comp_sh.uniform_sampler("depth_tex", _DEPTH_TEX)
    comp_sh.uniform_sampler("obb_tex", _OBB_TEX)
    comp_sh.uniform_sampler(
        "special_mask_tex",
        _SPECIAL_MASK_TEX if (has_spec_drawn and _SPECIAL_MASK_TEX is not None) else _DUMMY_SPECIAL_TEX,
    )
    comp_sh.uniform_float(
        "u_params0",
        (float(_OBB_COUNT), 1.0 if has_spec_drawn else 0.0, float(mode_id), 1.0 if show_grid else 0.0),
    )
    comp_sh.uniform_float("u_params1", (float(grid_size), 0.65 if show_grid else 0.0, 0.0, 0.0))
    _FS_QUAD_BATCH.draw(comp_sh)
    return True


def release() -> None:
    """Free all transient GPU G-Buffer textures, FBOs, and caches (0 MB VRAM leak)."""
    global _ACTIVE, _GBUF_VALID, _GBUF_SIZE, _GBUF_MVP_KEY
    global _COLOR_TEX, _POS_TEX, _DEPTH_TEX, _GBUF_FBO
    global _PARITY_TEX, _PARITY_FBO, _SPECIAL_MASK_TEX, _SPECIAL_MASK_FBO
    global _OBB_TEX, _OBB_COUNT, _OBB_TEX_DIRTY, _HAS_SPECIAL_CUTTERS
    global _FS_QUAD_BATCH, _FS_STAMP_BATCH, _DUMMY_SPECIAL_TEX

    _ACTIVE = False
    _GBUF_VALID = False
    _GBUF_SIZE = (0, 0)
    _GBUF_MVP_KEY = None
    _COLOR_TEX = None
    _POS_TEX = None
    _DEPTH_TEX = None
    _GBUF_FBO = None
    _PARITY_TEX = None
    _PARITY_FBO = None
    _SPECIAL_MASK_TEX = None
    _SPECIAL_MASK_FBO = None
    _OBB_TEX = None
    _OBB_COUNT = 0
    _OBB_TEX_DIRTY = True
    _HAS_SPECIAL_CUTTERS = False
    _CUTTER_DESCS.clear()
    _BOX_CLASS_CACHE.clear()
    _SPECIAL_MESH_BATCH_CACHE.clear()
    _FS_QUAD_BATCH = None
    _FS_STAMP_BATCH = None
    _DUMMY_SPECIAL_TEX = None


def shutdown_shaders() -> None:
    """Release compiled shaders on addon unregister."""
    global _CAPTURE_BASIC_SHADER, _CAPTURE_MULTI_SHADER, _PARITY_SHADER
    global _PARITY_STAMP_SHADER, _COMPOSITE_SHADER, _INIT_FAILED
    release()
    _CAPTURE_BASIC_SHADER = None
    _CAPTURE_MULTI_SHADER = None
    _PARITY_SHADER = None
    _PARITY_STAMP_SHADER = None
    _COMPOSITE_SHADER = None
    _INIT_FAILED = False
