"""Direct Multi-Material & Enfusion MatPBRMulti Textured GPU Viewport Renderer (`POST_VIEW`).

Renders the building's REAL textures directly on the GPU at 60-144+ FPS with zero switching:
- Enfusion `MatPBRMulti_shader` (4 BCR layers on UVMap0 with Color_1..4 + UVTransform_1..4,
  blended via MaskMap on UVMap1 + GlobalMCRMap BlendOverlay3Power/MCRMultiply on UVMap1)
- Enfusion `MatPBRBasic_shader` & standard Blender Principled BSDF / Cut Face materials
- Crisp World-Space Zone Grid + subtle Zone tint on cut pieces
"""

from __future__ import annotations

from collections import deque
import math
import re
from typing import Sequence

import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

from . import rust_bridge
from .session import AppleBiteSession

_BASIC_SHADER = None
_BASIC_SHADER_FAILED = False
_MULTI_SHADER = None
_MULTI_SHADER_FAILED = False
_LINE_SHADER = None
_FALLBACK_TEX = None
_DRAW_HANDLE = None

_ACTIVE: bool = False
_HIDDEN_OBJECTS: list[bpy.types.Object] = []
_CUTTERS_COL_NAME: str = ""

# target_name -> {
#   "materials": [mat_desc_dict, ...],
#   "rem_batches": [(mat_id, is_multi_bool, GPUBatch), ...],
#   "piece_batches": [([(mat_id, is_multi_bool, GPUBatch), ...], zone_rgba, zone_letter), ...],
#   "piece_wire": [(GPUBatch, zone_rgba), ...],
# }
_GPU_STATE: dict[str, dict] = {}
# (mat_ptr, mat_name, dll_ver) -> mat_desc_dict
_MAT_CACHE: dict[tuple[int, str, int], dict] = {}
_LAST_MAT_DEBUG: list[str] = []
_GRID_SIZE: float = 0.35
_SHOW_GRID: bool = True
_DISPLAY_MODE: str = "ALL"  # "ALL" | "PIECES" | "MAIN" | "MAIN_WIRE"

_NON_COLOR_KEYWORDS = (
    "_nmo",
    "_nm",
    "_normal",
    "normal",
    "bump",
    "rough",
    "metal",
    "_ao",
    "mask",
    "spec",
    "height",
    "_ml",
    "_mcr",
    "_s.",
    "_disp",
    "cavity",
)
_COLOR_KEYWORDS = (
    "_bcr",
    "bcr",
    "albedo",
    "diffuse",
    "basecolor",
    "base_color",
    "color",
    "_co",
    "_bc",
    "_d.",
    "_diff",
)
_SKIP_SOCKET_KEYWORDS = (
    "normal",
    "rough",
    "metal",
    "alpha",
    "height",
    "bump",
    "vector",
    "nmo",
    "ao",
    "fac",
    "factor",
    "mask",
    "opacity",
    "specular",
    "cavity",
    "disp",
    "emission",
    "coat",
    "sheen",
    "subsurface",
    "transmission",
    "ior",
    "aniso",
    "weight",
)

_DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "roof": ("roof", "tile", "rooftile", "slate", "shingle", "thatch", "eternit", "terracotta", "asbestos", "ridge", "clay", "ceramic", "corrugated"),
    "ext": ("plaster", "stucco", "brizolit", "wall", "brick", "facade", "concrete", "mortar"),
    "int": ("plaster", "bare", "wallpaper", "paint", "interior", "wall", "gypsum"),
    "wall": ("plaster", "stucco", "brizolit", "wall", "brick", "facade", "concrete"),
    "wood": ("wood", "plank", "timber", "splinter", "beam", "bark", "log"),
    "timber": ("wood", "plank", "timber", "beam", "log"),
    "plank": ("wood", "plank", "timber", "board"),
    "metal": ("metal", "iron", "steel", "rust", "zinc", "tin", "sheet", "copper"),
    "gutter": ("metal", "zinc", "copper", "rust", "gutter", "tin"),
    "floor": ("floor", "tile", "plank", "concrete", "cobble", "pave", "wood"),
    "stone": ("stone", "brick", "rock", "cobble", "mason"),
    "brick": ("brick", "stone", "mason"),
    "glass": ("glass", "window"),
    "window": ("window", "glass", "frame", "wood"),
    "door": ("door", "wood", "plank"),
    "trim": ("trim", "wood", "metal", "plaster"),
}
_GENERIC_MAT_TOKENS = {
    "pub",
    "village",
    "house",
    "building",
    "mat",
    "material",
    "bcr",
    "st",
    "dst",
    "lod",
    "prop",
    "asset",
    "tex",
    "map",
}

_TOKEN_RE = re.compile(r"[a-z]+")
_LAYER0_RE = re.compile(r"(?:^|[\s_\-.])(?:layer[\s_]*0|bcr(?:map)?[\s_]*0|albedo[\s_]*0|base[\s_]*(?:color|bcr|layer|map)?|primary|color[\s_]*0)(?:$|[\s_\-.])")
_OVERLAY_LAYER_RE = re.compile(r"(?:^|[\s_\-.])(?:layer[\s_]*[1-9]|bcr(?:map)?[\s_]*[1-9]|albedo[\s_]*[1-9]|color[\s_]*[1-9]|secondary|overlay|detail|dirt|moss|grime|damage|macro|super)(?:$|[\s_\-.])")


def get_material_debug_info() -> list[str]:
    return list(_LAST_MAT_DEBUG)


def invalidate_material_cache() -> None:
    _MAT_CACHE.clear()


def _get_fallback_texture():
    global _FALLBACK_TEX
    if _FALLBACK_TEX is not None:
        return _FALLBACK_TEX
    if bpy.app.background:
        return None
    try:
        buf = gpu.types.Buffer("FLOAT", 4, [1.0, 1.0, 1.0, 1.0])
        _FALLBACK_TEX = gpu.types.GPUTexture((1, 1), format="RGBA8", data=buf)
    except Exception:
        try:
            buf = gpu.types.Buffer("FLOAT", 4, [1.0, 1.0, 1.0, 1.0])
            _FALLBACK_TEX = gpu.types.GPUTexture((1, 1), format="RGBA16F", data=buf)
        except Exception:
            _FALLBACK_TEX = None
    return _FALLBACK_TEX


def _is_candidate_image(img) -> bool:
    if img is None:
        return False
    try:
        if img.size[0] > 0 and img.size[1] > 0:
            return True
        if getattr(img, "packed_file", None) is not None:
            return True
        if getattr(img, "filepath", ""):
            return True
    except Exception:
        return False
    return False


def _load_gpu_tex(img) -> tuple[object | None, bool, str]:
    """Return `(gpu_texture | None, needs_srgb_decode, fmt_str)` safely."""
    if img is None or bpy.app.background:
        return None, False, ""
    try:
        if img.size[0] == 0 and getattr(img, "filepath", ""):
            try:
                img.reload()
            except Exception:
                pass
        gtex = gpu.texture.from_image(img)
        if gtex is None:
            return None, False, ""
        cs_name = ""
        try:
            cs_name = str(img.colorspace_settings.name).lower()
        except Exception:
            pass
        fmt = ""
        try:
            fmt = str(gtex.format).upper()
        except Exception:
            pass
        is_float_fmt = ("16F" in fmt) or ("32F" in fmt) or ("FLOAT" in fmt)
        needs_srgb = ("SRGB" not in fmt) and (not is_float_fmt) and ("non-color" not in cs_name) and ("raw" not in cs_name)
        return gtex, needs_srgb, fmt
    except Exception:
        return None, False, ""


# ---------------------------------------------------------------------------
# Enfusion MatPBRMulti / MatPBRBasic Extraction
# ---------------------------------------------------------------------------


def _find_enfusion_inner_tree(mat: bpy.types.Material):
    """Unwrap Enfusion's 1-node outer Group wrapper (`Shader Nodetree -> Group`) if present."""
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return None, None
    # Direct Main node
    main = nt.nodes.get("Main")
    if main is not None and main.type == "GROUP" and getattr(main, "node_tree", None) is not None:
        return nt, main
    # 1-level wrapper group (standard EBT material structure)
    for n in nt.nodes:
        if n.type == "GROUP" and getattr(n, "node_tree", None) is not None:
            sub = n.node_tree
            sub_main = sub.nodes.get("Main")
            if sub_main is not None and sub_main.type == "GROUP" and getattr(sub_main, "node_tree", None) is not None:
                return sub, sub_main
    return None, None


def _read_float_input(node, name: str, default: float) -> float:
    if node is None:
        return float(default)
    inp = node.inputs.get(name)
    if inp is not None and hasattr(inp, "default_value"):
        try:
            return float(inp.default_value)
        except Exception:
            return float(default)
    return float(default)


def _read_rgba_input(node, name: str, default=(1.0, 1.0, 1.0, 1.0)) -> tuple[float, float, float, float]:
    if node is None:
        return default
    inp = node.inputs.get(name)
    if inp is not None and hasattr(inp, "default_value"):
        try:
            dv = inp.default_value
            return (float(dv[0]), float(dv[1]), float(dv[2]), float(dv[3]) if len(dv) > 3 else 1.0)
        except Exception:
            return default
    return default


def _read_uv_transform_group(nt, node_name: str) -> tuple[float, float, float, float, float]:
    """Read `(TilingU, TilingV, OffsetU, OffsetV, Rotation_rad)` from an Enfusion `MapUVTransform` node."""
    nd = nt.nodes.get(node_name)
    if nd is None:
        return (1.0, 1.0, 0.0, 0.0, 0.0)
    tu = _read_float_input(nd, "TilingU", 1.0)
    tv = _read_float_input(nd, "TilingV", 1.0)
    ou = _read_float_input(nd, "OffsetU", 0.0)
    ov = _read_float_input(nd, "OffsetV", 0.0)
    rot_deg = _read_float_input(nd, "Rotation", 0.0)
    return (tu, tv, ou, ov, math.radians(rot_deg))


def _get_tex_image_by_name(nt, node_name: str):
    nd = nt.nodes.get(node_name)
    if nd is not None and nd.type == "TEX_IMAGE":
        img = getattr(nd, "image", None)
        if _is_candidate_image(img):
            return img
    return None


def _try_resolve_enfusion_multimat(mat: bpy.types.Material, has_uv1: bool) -> dict | None:
    """Detect Enfusion `MatPBRMulti_shader` and build a cached 6-sampler GPU descriptor."""
    inner_nt, main = _find_enfusion_inner_tree(mat)
    if inner_nt is None or main is None:
        return None
    main_tree_name = str(getattr(main.node_tree, "name", ""))
    if not main_tree_name.startswith("MatPBRMulti"):
        return None

    bcr_imgs = [
        _get_tex_image_by_name(inner_nt, "BCR_1"),
        _get_tex_image_by_name(inner_nt, "BCR_2"),
        _get_tex_image_by_name(inner_nt, "BCR_3"),
        _get_tex_image_by_name(inner_nt, "BCR_4"),
    ]
    mask_img = _get_tex_image_by_name(inner_nt, "MaskMap")
    mcr_img = _get_tex_image_by_name(inner_nt, "GlobalMCRMap")

    # If DLL is still v130 (before Blender restart), we don't have UVMap1 yet -> safe Layer 1 fallback with Color_1 tint!
    if not has_uv1 or _get_multimat_shader() is None:
        img1 = bcr_imgs[0]
        c1 = _read_rgba_input(main, "Color_1", (1.0, 1.0, 1.0, 1.0))
        tu, tv, ou, ov, _ = _read_uv_transform_group(inner_nt, "UVTransform_1")
        gtex, dec, fmt = _load_gpu_tex(img1) if img1 is not None else (None, False, "")
        return {
            "kind": "BASIC",
            "gtex": gtex,
            "base_col": (c1[0], c1[1], c1[2], 1.0),
            "uv_xform": (tu, tv, ou, -ov),
            "use_tex": gtex is not None,
            "dec_srgb": dec,
            "note": f"{mat.name} -> ENF_L1:{img1.name if img1 else 'None'} (restart Blender for full 4-layer UVMap1 MultiMat)",
        }

    # Load all 6 GPU textures once (cached in _MAT_CACHE)
    bcr_texs = []
    bcr_decs = []
    bcr_names = []
    for img in bcr_imgs:
        if img is not None:
            gt, dec, _ = _load_gpu_tex(img)
            bcr_texs.append(gt)
            bcr_decs.append(1.0 if dec else 0.0)
            bcr_names.append(img.name)
        else:
            bcr_texs.append(None)
            bcr_decs.append(0.0)
            bcr_names.append("-")

    mask_tex, _, _ = _load_gpu_tex(mask_img) if mask_img is not None else (None, False, "")
    mcr_tex, _, _ = _load_gpu_tex(mcr_img) if mcr_img is not None else (None, False, "")

    c1 = _read_rgba_input(main, "Color_1", (1.0, 1.0, 1.0, 1.0))
    c2 = _read_rgba_input(main, "Color_2", (1.0, 1.0, 1.0, 1.0))
    c3 = _read_rgba_input(main, "Color_3", (1.0, 1.0, 1.0, 1.0))
    c4 = _read_rgba_input(main, "Color_4", (1.0, 1.0, 1.0, 1.0))
    gcol = _read_rgba_input(main, "Color", (1.0, 1.0, 1.0, 1.0))

    en2 = _read_float_input(main, "Enabled_2", 1.0) if (bcr_texs[1] is not None or c2[:3] != (1.0, 1.0, 1.0)) else 0.0
    en3 = _read_float_input(main, "Enabled_3", 1.0) if (bcr_texs[2] is not None) else 0.0
    en4 = _read_float_input(main, "Enabled_4", 1.0) if (bcr_texs[3] is not None) else 0.0

    uv1_tr = _read_uv_transform_group(inner_nt, "UVTransform_1")
    uv2_tr = _read_uv_transform_group(inner_nt, "UVTransform_2")
    uv3_tr = _read_uv_transform_group(inner_nt, "UVTransform_3")
    uv4_tr = _read_uv_transform_group(inner_nt, "UVTransform_4")
    glob_tr = _read_uv_transform_group(inner_nt, "GlobalMapsUVTransform")

    m_r_scale = _read_float_input(main, "MaskRedScale", 1.0)
    m_g_scale = _read_float_input(main, "MaskGreenScale", 1.0)
    m_b_scale = _read_float_input(main, "MaskBlueScale", 1.0)

    m_sh2 = _read_float_input(main, "MaskSharpness_2", 0.0)
    m_sh3 = _read_float_input(main, "MaskSharpness_3", 0.0)
    m_sh4 = _read_float_input(main, "MaskSharpness_4", 0.0)

    m_off2 = _read_float_input(main, "MaskOffset_2", 0.0)
    m_off3 = _read_float_input(main, "MaskOffset_3", 0.0)
    m_off4 = _read_float_input(main, "MaskOffset_4", 0.0)

    mcr_mode = _read_float_input(main, "MCRMode", 0.0)
    mcr_op = _read_float_input(main, "MCROpacityBlend", 1.0)
    ov_w = _read_float_input(main, "OverlayWhitePower", 1.0)
    ov_b = _read_float_input(main, "OverlayBlackPower", 1.0)
    mcr_br = _read_float_input(main, "MCRBrightness", 0.0)

    cm2 = _read_float_input(main, "CombineMode_2", 0.0)
    cm3 = _read_float_input(main, "CombineMode_3", 0.0)
    cm4 = _read_float_input(main, "CombineMode_4", 0.0)

    # Pack 17 std140 vec4 floats (272 bytes) into GPUUniformBuf
    ubo_arr = np.array(
        [
            c1[0], c1[1], c1[2], 1.0 if bcr_texs[0] is not None else 0.0,
            c2[0], c2[1], c2[2], float(en2),
            c3[0], c3[1], c3[2], float(en3),
            c4[0], c4[1], c4[2], float(en4),
            gcol[0], gcol[1], gcol[2], 1.0 if mcr_tex is not None else 0.0,
            uv1_tr[0], uv1_tr[1], uv1_tr[2], uv1_tr[3],
            uv2_tr[0], uv2_tr[1], uv2_tr[2], uv2_tr[3],
            uv3_tr[0], uv3_tr[1], uv3_tr[2], uv3_tr[3],
            uv4_tr[0], uv4_tr[1], uv4_tr[2], uv4_tr[3],
            uv1_tr[4], uv2_tr[4], uv3_tr[4], uv4_tr[4],
            glob_tr[0], glob_tr[1], glob_tr[2], glob_tr[3],
            m_r_scale, m_g_scale, m_b_scale, 1.0 if mask_tex is not None else 0.0,
            m_sh2, m_sh3, m_sh4, glob_tr[4],
            m_off2, m_off3, m_off4, 0.0,
            mcr_mode, mcr_op, ov_w, ov_b,
            mcr_br, cm2, cm3, cm4,
            bcr_decs[0], bcr_decs[1], bcr_decs[2], bcr_decs[3],
        ],
        dtype=np.float32,
    )
    ubo = None
    if not bpy.app.background:
        try:
            ubo = gpu.types.GPUUniformBuf(ubo_arr.tobytes())
        except Exception as exc:
            print(f"[FullDestruc_Tool] MultiMat UBO creation failed for {mat.name}: {exc}")
            return None

    return {
        "kind": "MULTIMAT",
        "bcr_texs": bcr_texs,
        "mask_tex": mask_tex,
        "mcr_tex": mcr_tex,
        "ubo": ubo,
        "note": (
            f"{mat.name} -> MULTIMAT(4L+UV1): "
            f"L1={bcr_names[0]}, L2={bcr_names[1]}, L3={bcr_names[2]}, L4={bcr_names[3]}, "
            f"Mask={mask_img.name if mask_img else '-'}, MCR={mcr_img.name if mcr_img else '-'}"
        ),
    }


# ---------------------------------------------------------------------------
# Generic / Standard Shader & MatPBRBasic Resolver
# ---------------------------------------------------------------------------


def _score_layer_hint(text: str) -> int:
    tl = text.strip().lower()
    if not tl:
        return 0
    if _OVERLAY_LAYER_RE.search(tl):
        return -14
    if _LAYER0_RE.search(tl) or tl in ("bcr", "bcrmap", "base color", "color", "albedo", "diffuse"):
        return 16
    return 0


def _score_material_match(mat_name: str, img_text_l: str) -> int:
    if not mat_name or not img_text_l:
        return 0
    tokens = [t for t in _TOKEN_RE.findall(mat_name.lower()) if len(t) >= 3 and t not in _GENERIC_MAT_TOKENS]
    if not tokens:
        return 0
    bonus = 0
    for tok in tokens:
        if tok in img_text_l:
            bonus = max(bonus, 24)
        syns = _DOMAIN_SYNONYMS.get(tok)
        if syns and any(s in img_text_l for s in syns):
            bonus = max(bonus, 32 if tok == "roof" else 20)
    return bonus


def _score_image_node(node, img, mat_name: str = "") -> int:
    node_meta = f"{getattr(node, 'name', '')} {getattr(node, 'label', '')}".lower()
    img_meta = f"{getattr(img, 'name', '')} {getattr(img, 'filepath', '')}".lower()
    name_l = f"{node_meta} {img_meta}"
    cs_name = ""
    try:
        cs_name = str(img.colorspace_settings.name).lower()
    except Exception:
        pass
    score = 0
    if any(k in name_l for k in _COLOR_KEYWORDS):
        score += 14
    if "srgb" in cs_name:
        score += 6
    if "non-color" in cs_name or "raw" in cs_name:
        score -= 10
    if any(k in name_l for k in _NON_COLOR_KEYWORDS):
        score -= 18
    score += _score_layer_hint(node_meta)
    score += _score_material_match(mat_name, img_meta)
    return score


def _extract_uv_transform(tex_node) -> tuple[float, float, float, float]:
    vec_inp = getattr(tex_node, "inputs", None)
    if vec_inp is None:
        return (1.0, 1.0, 0.0, 0.0)
    vsock = vec_inp.get("Vector")
    if vsock is None or not vsock.is_linked:
        return (1.0, 1.0, 0.0, 0.0)
    cur = vsock.links[0].from_node
    for _ in range(3):
        if cur is None:
            break
        if cur.type == "MAPPING":
            try:
                loc = cur.inputs.get("Location")
                sca = cur.inputs.get("Scale")
                ox = float(loc.default_value[0]) if (loc and not loc.is_linked) else 0.0
                oy = float(loc.default_value[1]) if (loc and not loc.is_linked) else 0.0
                sx = float(sca.default_value[0]) if (sca and not sca.is_linked) else 1.0
                sy = float(sca.default_value[1]) if (sca and not sca.is_linked) else 1.0
                if abs(sx) > 1e-4 and abs(sy) > 1e-4:
                    return (sx, sy, ox, oy)
            except Exception:
                return (1.0, 1.0, 0.0, 0.0)
        if cur.type == "REROUTE" and cur.inputs and cur.inputs[0].is_linked:
            cur = cur.inputs[0].links[0].from_node
            continue
        if cur.type == "VECT_MATH" and getattr(cur, "operation", "") == "SCALE":
            try:
                s_sock = cur.inputs.get("Scale")
                if s_sock is not None and not s_sock.is_linked:
                    sv = float(s_sock.default_value)
                    if abs(sv) > 1e-4:
                        return (sv, sv, 0.0, 0.0)
            except Exception:
                pass
        break
    return (1.0, 1.0, 0.0, 0.0)


def _mul_rgb(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        max(0.05, min(2.0, a[0] * b[0])),
        max(0.05, min(2.0, a[1] * b[1])),
        max(0.05, min(2.0, a[2] * b[2])),
    )


def _lerp_tint(rgb: tuple[float, float, float], fac: float) -> tuple[float, float, float]:
    f = max(0.0, min(1.0, float(fac)))
    return (1.0 - f + rgb[0] * f, 1.0 - f + rgb[1] * f, 1.0 - f + rgb[2] * f)


def _get_mix_color_inputs(node):
    inputs = getattr(node, "inputs", ())
    if node.type == "MIX":
        if getattr(node, "data_type", "RGBA") != "RGBA":
            return None, None, None
        return (
            inputs[0] if len(inputs) > 0 else None,
            inputs[6] if len(inputs) > 6 else None,
            inputs[7] if len(inputs) > 7 else None,
        )
    if node.type == "MIX_RGB":
        return (
            inputs.get("Fac") or (inputs[0] if len(inputs) > 0 else None),
            inputs.get("Color1") or (inputs[1] if len(inputs) > 1 else None),
            inputs.get("Color2") or (inputs[2] if len(inputs) > 2 else None),
        )
    return None, None, None


def _collect_all_image_nodes_recursive(
    node_tree,
    mat_name: str,
    base_bonus: int = 0,
    depth: int = 0,
    max_depth: int = 4,
    tint_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[tuple[int, object, tuple[float, float, float], tuple[float, float, float, float]]]:
    if node_tree is None or depth > max_depth:
        return []
    found = []
    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if _is_candidate_image(img):
                sc = _score_image_node(node, img, mat_name) + base_bonus
                found.append((sc, img, tint_rgb, _extract_uv_transform(node)))
        elif node.type == "GROUP":
            sub_tree = getattr(node, "node_tree", None)
            if sub_tree is not None:
                found.extend(
                    _collect_all_image_nodes_recursive(
                        sub_tree, mat_name, base_bonus - 2, depth + 1, max_depth, tint_rgb
                    )
                )
    return found


def _trace_candidates_upstream(
    start_sockets: Sequence,
    mat_name: str,
    initial_bonus: int = 0,
    initial_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_hops: int = 14,
    group_depth: int = 0,
) -> list[tuple[int, object, tuple[float, float, float], tuple[float, float, float, float]]]:
    queue = deque()
    for sock in start_sockets:
        if sock is None or not sock.is_linked:
            continue
        sb = _score_layer_hint(getattr(sock, "name", ""))
        for link in sock.links:
            queue.append((link.from_node, 0, initial_bonus + sb, initial_tint))

    best_bonus_at_node: dict[int, int] = {}
    candidates = []
    steps = 0

    while queue and steps < 256:
        node, depth, path_bonus, tint_rgb = queue.popleft()
        steps += 1
        if node is None or depth > max_hops:
            continue

        nid = id(node)
        prev_b = best_bonus_at_node.get(nid)
        if prev_b is not None and path_bonus <= prev_b:
            continue
        best_bonus_at_node[nid] = path_bonus

        if node.type == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if _is_candidate_image(img):
                sc = _score_image_node(node, img, mat_name) + path_bonus
                candidates.append((sc, img, tint_rgb, _extract_uv_transform(node)))
            continue

        if node.type in ("MIX", "MIX_RGB"):
            fac_sock, sa, sb = _get_mix_color_inputs(node)
            btype = str(getattr(node, "blend_type", "MIX")).upper()
            fac_val = 1.0
            if fac_sock is not None and not fac_sock.is_linked and hasattr(fac_sock, "default_value"):
                try:
                    fac_val = float(fac_sock.default_value)
                except Exception:
                    fac_val = 1.0

            if btype == "MULTIPLY":
                a_linked = sa is not None and sa.is_linked
                b_linked = sb is not None and sb.is_linked
                if a_linked and not b_linked and sb is not None and hasattr(sb, "default_value"):
                    dv = sb.default_value
                    new_tint = _mul_rgb(tint_rgb, _lerp_tint((float(dv[0]), float(dv[1]), float(dv[2])), fac_val)) if max(float(dv[0]), float(dv[1]), float(dv[2])) > 0.08 else tint_rgb
                    for link in sa.links:
                        queue.append((link.from_node, depth + 1, path_bonus + 8, new_tint))
                    continue
                if b_linked and not a_linked and sa is not None and hasattr(sa, "default_value"):
                    dv = sa.default_value
                    new_tint = _mul_rgb(tint_rgb, _lerp_tint((float(dv[0]), float(dv[1]), float(dv[2])), fac_val)) if max(float(dv[0]), float(dv[1]), float(dv[2])) > 0.08 else tint_rgb
                    for link in sb.links:
                        queue.append((link.from_node, depth + 1, path_bonus + 8, new_tint))
                    continue
                if a_linked:
                    for link in sa.links:
                        queue.append((link.from_node, depth + 1, path_bonus + 12, tint_rgb))
                if b_linked:
                    for link in sb.links:
                        queue.append((link.from_node, depth + 1, path_bonus - 6, tint_rgb))
                continue

            if sa is not None and sa.is_linked:
                for link in sa.links:
                    queue.append((link.from_node, depth + 1, path_bonus + 18, tint_rgb))
            if sb is not None and sb.is_linked:
                for link in sb.links:
                    queue.append((link.from_node, depth + 1, path_bonus - 10, tint_rgb))
            continue

        if node.type == "GROUP":
            group_tint = tint_rgb
            for inp in getattr(node, "inputs", ()):
                if inp.is_linked or inp.type != "RGBA":
                    continue
                iname = inp.name.lower()
                if any(k in iname for k in ("tint", "modulate", "multiply")) and not _OVERLAY_LAYER_RE.search(iname):
                    try:
                        dv = inp.default_value
                        if max(float(dv[0]), float(dv[1]), float(dv[2])) > 0.08:
                            group_tint = _mul_rgb(group_tint, (float(dv[0]), float(dv[1]), float(dv[2])))
                    except Exception:
                        pass

            for inp in getattr(node, "inputs", ()):
                if not inp.is_linked:
                    continue
                lname = inp.name.lower()
                if any(k in lname for k in _SKIP_SOCKET_KEYWORDS):
                    continue
                sb = _score_layer_hint(lname)
                for link in inp.links:
                    queue.append((link.from_node, depth + 1, path_bonus + sb + 4, group_tint))

            sub_tree = getattr(node, "node_tree", None)
            if sub_tree is not None and group_depth < 3:
                internal_starts = []
                for sn in sub_tree.nodes:
                    if sn.type in ("BSDF_PRINCIPLED", "BSDF_DIFFUSE", "EMISSION"):
                        for sname in ("Base Color", "Color"):
                            s = sn.inputs.get(sname)
                            if s is not None and s.is_linked:
                                internal_starts.append(s)
                    elif sn.type == "GROUP_OUTPUT":
                        for s in sn.inputs:
                            if s.is_linked and not any(k in s.name.lower() for k in _SKIP_SOCKET_KEYWORDS):
                                internal_starts.append(s)
                if internal_starts:
                    candidates.extend(
                        _trace_candidates_upstream(
                            internal_starts,
                            mat_name,
                            initial_bonus=path_bonus - 2,
                            initial_tint=group_tint,
                            max_hops=max_hops - 2,
                            group_depth=group_depth + 1,
                        )
                    )
                else:
                    candidates.extend(
                        _collect_all_image_nodes_recursive(
                            sub_tree, mat_name, path_bonus - 6, 0, 2, group_tint
                        )
                    )
            continue

        for inp in getattr(node, "inputs", ()):
            if not inp.is_linked:
                continue
            lname = inp.name.lower()
            if any(k in lname for k in _SKIP_SOCKET_KEYWORDS):
                continue
            sb = _score_layer_hint(lname)
            for link in inp.links:
                queue.append((link.from_node, depth + 1, path_bonus + sb, tint_rgb))

    return candidates


def _find_color_image_in_tree(node_tree, mat_name: str):
    if node_tree is None:
        return None, (1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 0.0), ""

    candidates = []
    bsdf_starts = []
    for node in node_tree.nodes:
        if node.type in ("BSDF_PRINCIPLED", "BSDF_DIFFUSE", "EMISSION"):
            for sock_name in ("Base Color", "Color"):
                sock = node.inputs.get(sock_name)
                if sock is not None and sock.is_linked:
                    bsdf_starts.append(sock)
    if bsdf_starts:
        candidates.extend(_trace_candidates_upstream(bsdf_starts, mat_name, initial_bonus=10))

    if not candidates:
        out_starts = []
        for node in node_tree.nodes:
            if node.type == "OUTPUT_MATERIAL":
                surf = node.inputs.get("Surface")
                if surf is not None and surf.is_linked:
                    out_starts.append(surf)
        if out_starts:
            candidates.extend(_trace_candidates_upstream(out_starts, mat_name, initial_bonus=6))

    if not candidates:
        candidates.extend(_collect_all_image_nodes_recursive(node_tree, mat_name, base_bonus=0, depth=0, max_depth=4))

    if not candidates:
        return None, (1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 0.0), ""

    by_img = {}
    for item in candidates:
        iid = id(item[1])
        prev = by_img.get(iid)
        if prev is None or item[0] > prev[0]:
            by_img[iid] = item

    ranked = sorted(by_img.values(), key=lambda x: x[0], reverse=True)
    _, best_img, best_tint, best_uv = ranked[0]

    cand_str = ""
    if len(ranked) > 1:
        top_names = ", ".join(f"{it[1].name}({it[0]})" for it in ranked[:4])
        cand_str = f" | cand=[{top_names}]"

    return best_img, best_tint, best_uv, cand_str


def _resolve_material_gpu_info(mat: bpy.types.Material | None) -> dict:
    """Resolve and cache a material descriptor (`MULTIMAT` or `BASIC`)."""
    if mat is None:
        return {
            "kind": "BASIC",
            "gtex": None,
            "base_col": (0.76, 0.77, 0.80, 1.0),
            "uv_xform": (1.0, 1.0, 0.0, 0.0),
            "use_tex": False,
            "dec_srgb": False,
            "note": "None -> default clay",
        }

    get_ver = getattr(rust_bridge, "get_dll_version", None)
    dll_ver = int(get_ver()) if callable(get_ver) else 130
    cache_key = (int(mat.as_pointer()), str(mat.name), dll_ver)
    cached = _MAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # 1. Enfusion MatPBRMulti_shader fast path (4 layers + MaskMap + GlobalMCRMap on UVMap1)
    mm_desc = _try_resolve_enfusion_multimat(mat, has_uv1=(dll_ver >= 131))
    if mm_desc is not None:
        _MAT_CACHE[cache_key] = mm_desc
        return mm_desc

    # 2. Enfusion MatPBRBasic_shader fast path
    inner_nt, main = _find_enfusion_inner_tree(mat)
    if inner_nt is not None and main is not None:
        main_tree_name = str(getattr(main.node_tree, "name", ""))
        if main_tree_name.startswith("MatPBRBasic"):
            bcr_img = _get_tex_image_by_name(inner_nt, "BCRMap")
            col = _read_rgba_input(main, "Color", (1.0, 1.0, 1.0, 1.0))
            tu, tv, ou, ov, _ = _read_uv_transform_group(inner_nt, "GlobalMapsUVTransform")
            gtex, dec, fmt = _load_gpu_tex(bcr_img) if bcr_img is not None else (None, False, "")
            res = {
                "kind": "BASIC",
                "gtex": gtex,
                "base_col": (col[0], col[1], col[2], 1.0),
                "uv_xform": (tu, tv, ou, -ov),
                "use_tex": gtex is not None,
                "dec_srgb": dec,
                "note": f"{mat.name} -> ENF_BASIC:{bcr_img.name if bcr_img else 'None'} ({bcr_img.size[0]}x{bcr_img.size[1]}, fmt={fmt}, srgb_dec={dec})" if bcr_img else f"{mat.name} -> ENF_BASIC_SOLID",
            }
            _MAT_CACHE[cache_key] = res
            return res

    # 3. General Blender Principled BSDF / Custom Node Tree fallback
    base_col = (
        float(mat.diffuse_color[0]),
        float(mat.diffuse_color[1]),
        float(mat.diffuse_color[2]),
        1.0,
    )
    img = None
    tint_rgb = (1.0, 1.0, 1.0)
    uv_xform = (1.0, 1.0, 0.0, 0.0)
    cand_str = ""

    nt = getattr(mat, "node_tree", None)
    if nt is not None:
        for node in nt.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bc = node.inputs.get("Base Color")
                if bc is not None and hasattr(bc, "default_value"):
                    dv = bc.default_value
                    base_col = (float(dv[0]), float(dv[1]), float(dv[2]), 1.0)
                break
        img, tint_rgb, uv_xform, cand_str = _find_color_image_in_tree(nt, mat.name)

    if img is not None and not bpy.app.background:
        gtex, dec, fmt = _load_gpu_tex(img)
        if gtex is not None:
            res = {
                "kind": "BASIC",
                "gtex": gtex,
                "base_col": (tint_rgb[0], tint_rgb[1], tint_rgb[2], 1.0),
                "uv_xform": uv_xform,
                "use_tex": True,
                "dec_srgb": dec,
                "note": f"{mat.name} -> TEX:{img.name} ({img.size[0]}x{img.size[1]}, fmt={fmt}, srgb_dec={dec}){cand_str}",
            }
            _MAT_CACHE[cache_key] = res
            return res

    res = {
        "kind": "BASIC",
        "gtex": None,
        "base_col": base_col,
        "uv_xform": (1.0, 1.0, 0.0, 0.0),
        "use_tex": False,
        "dec_srgb": False,
        "note": f"{mat.name} -> SOLID_COLOR {tuple(round(c, 2) for c in base_col[:3])}",
    }
    _MAT_CACHE[cache_key] = res
    return res


# ---------------------------------------------------------------------------
# Shaders (1: Basic Single-Texture, 2: Enfusion MatPBRMulti 4-Layer + UV1)
# ---------------------------------------------------------------------------


def _get_preview_shader():
    global _BASIC_SHADER, _BASIC_SHADER_FAILED
    if _BASIC_SHADER is not None:
        return _BASIC_SHADER
    if _BASIC_SHADER_FAILED or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_gpu_tex_preview_iface")
        vert_out.smooth("VEC3", "v_world_pos")
        vert_out.smooth("VEC3", "v_world_nor")
        vert_out.smooth("VEC2", "v_uv")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "nor")
        info.vertex_in(2, "VEC2", "uv")
        info.vertex_out(vert_out)

        info.sampler(0, "FLOAT_2D", "base_tex")
        info.push_constant("MAT4", "ModelViewProjectionMatrix")  # 64B
        info.push_constant("VEC4", "base_color")                 # 16B
        info.push_constant("VEC4", "zone_tint")                  # 16B
        info.push_constant("VEC4", "uv_transform")               # 16B
        info.push_constant("VEC4", "params")                     # 16B
        info.fragment_out(0, "VEC4", "fragColor")

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
                float grid_size = params.z;
                float grid_strength = params.w;

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

                if (zone_tint.a > 0.001) {
                    rgb = mix(rgb, zone_tint.rgb * shade * 1.1, zone_tint.a);
                }
                if (grid_strength > 0.001) {
                    vec3 an = abs(n);
                    float gs = max(grid_size, 0.02);
                    vec2 guv;
                    if (an.z >= an.x && an.z >= an.y) {
                        guv = v_world_pos.xy / gs;
                    } else if (an.x >= an.y) {
                        guv = v_world_pos.yz / gs;
                    } else {
                        guv = v_world_pos.xz / gs;
                    }
                    vec2 fw = max(fwidth(guv), vec2(0.0005));
                    vec2 grid = abs(fract(guv - 0.5) - 0.5) / fw;
                    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
                    vec3 line_col = mix(vec3(1.0, 0.96, 0.85), zone_tint.rgb + vec3(0.35), 0.5);
                    rgb = mix(rgb, line_col, line * grid_strength);
                }

                fragColor = vec4(rgb, 1.0);
            }
            """
        )

        _BASIC_SHADER = gpu.shader.create_from_info(info)
        return _BASIC_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] Basic GPU Preview shader compile failed: {exc}")
        _BASIC_SHADER_FAILED = True
        return None


def _get_multimat_shader():
    global _MULTI_SHADER, _MULTI_SHADER_FAILED
    if _MULTI_SHADER is not None:
        return _MULTI_SHADER
    if _MULTI_SHADER_FAILED or bpy.app.background:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_gpu_multimat_iface")
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

        # 96 bytes total push constants (<= 128B Vulkan limit)
        info.push_constant("MAT4", "ModelViewProjectionMatrix")  # 64B
        info.push_constant("VEC4", "zone_tint")                  # 16B
        info.push_constant("VEC4", "grid_params")                # 16B (grid_size, grid_strength, 0, 0)
        info.fragment_out(0, "VEC4", "fragColor")

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

                // ApplyGlobalLayer (GlobalMCRMap on UVMap1)
                if (u_mm.global_col.a > 0.5) {
                    vec3 mcr = texture(mcr_tex, uv_g).rgb;
                    if (u_mm.mcr_params.x < 0.5) {
                        vec3 diff = mcr - vec3(0.5);
                        vec3 pwr = mix(vec3(u_mm.mcr_params.z), vec3(u_mm.mcr_params.w), step(diff, vec3(0.0)));
                        vec3 adj = clamp(diff * pwr + vec3(0.5), vec3(0.0), vec3(1.0));
                        col_lin = blend_overlay(adj, col_lin);
                    } else {
                        vec3 mul_col = mcr * col_lin * pow(2.0, u_mm.extra_params.x);
                        col_lin = mix(col_lin, mul_col, u_mm.mcr_params.y);
                    }
                }
                col_lin *= u_mm.global_col.rgb;

                // Natural viewport lighting closely matching Blender Material Preview
                vec3 l1 = normalize(vec3(0.45, -0.55, 0.70));
                vec3 l2 = normalize(vec3(-0.60, 0.45, 0.35));
                float ndl1 = max(dot(n, l1), 0.0);
                float ndl2 = max(dot(n, l2), 0.0);
                float hemi = 0.5 + 0.5 * n.z;
                float shade = 0.40 + 0.16 * hemi + 0.36 * ndl1 + 0.14 * ndl2;

                vec3 rgb = col_lin * shade;

                if (zone_tint.a > 0.001) {
                    rgb = mix(rgb, zone_tint.rgb * shade * 1.1, zone_tint.a);
                }
                float grid_size = grid_params.x;
                float grid_strength = grid_params.y;
                if (grid_strength > 0.001) {
                    vec3 an = abs(n);
                    float gs = max(grid_size, 0.02);
                    vec2 guv;
                    if (an.z >= an.x && an.z >= an.y) {
                        guv = v_world_pos.xy / gs;
                    } else if (an.x >= an.y) {
                        guv = v_world_pos.yz / gs;
                    } else {
                        guv = v_world_pos.xz / gs;
                    }
                    vec2 fw = max(fwidth(guv), vec2(0.0005));
                    vec2 grid = abs(fract(guv - 0.5) - 0.5) / fw;
                    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
                    vec3 line_col = mix(vec3(1.0, 0.96, 0.85), zone_tint.rgb + vec3(0.35), 0.5);
                    rgb = mix(rgb, line_col, line * grid_strength);
                }

                fragColor = vec4(rgb, 1.0);
            }
            """
        )

        _MULTI_SHADER = gpu.shader.create_from_info(info)
        return _MULTI_SHADER
    except Exception as exc:
        print(f"[FullDestruc_Tool] MultiMat GPU Preview shader compile failed: {exc}")
        _MULTI_SHADER_FAILED = True
        return None


def _tag_all_view3d_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        if win.screen is None:
            continue
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _draw_submesh_list(
    basic_shader,
    multi_shader,
    mvp,
    fallback_tex,
    mat_infos: list[dict],
    sub_batches: list[tuple[int, bool, object]],
    zone_tint: tuple[float, float, float, float],
    grid_strength: float,
) -> None:
    bound = None
    for mat_id, is_multi, batch in sub_batches:
        desc = mat_infos[mat_id] if (0 <= mat_id < len(mat_infos)) else None

        if is_multi and multi_shader is not None and desc is not None and desc.get("kind") == "MULTIMAT":
            if bound != "MULTI":
                multi_shader.bind()
                multi_shader.uniform_float("ModelViewProjectionMatrix", mvp)
                multi_shader.uniform_float("zone_tint", zone_tint)
                multi_shader.uniform_float("grid_params", (_GRID_SIZE, grid_strength, 0.0, 0.0))
                bound = "MULTI"
            bcr_texs = desc["bcr_texs"]
            multi_shader.uniform_sampler("bcr1_tex", bcr_texs[0] or fallback_tex)
            multi_shader.uniform_sampler("bcr2_tex", bcr_texs[1] or fallback_tex)
            multi_shader.uniform_sampler("bcr3_tex", bcr_texs[2] or fallback_tex)
            multi_shader.uniform_sampler("bcr4_tex", bcr_texs[3] or fallback_tex)
            multi_shader.uniform_sampler("mask_tex", desc["mask_tex"] or fallback_tex)
            multi_shader.uniform_sampler("mcr_tex", desc["mcr_tex"] or fallback_tex)
            multi_shader.uniform_block("u_mm", desc["ubo"])
            batch.draw(multi_shader)
        elif basic_shader is not None:
            if bound != "BASIC":
                basic_shader.bind()
                basic_shader.uniform_float("ModelViewProjectionMatrix", mvp)
                basic_shader.uniform_float("zone_tint", zone_tint)
                bound = "BASIC"
            if desc is not None and desc.get("kind") == "BASIC":
                gtex = desc["gtex"]
                bcol = desc["base_col"]
                uv_xf = desc["uv_xform"]
                has_tex = bool(desc["use_tex"] and gtex is not None)
                dec_srgb = bool(desc["dec_srgb"])
            else:
                gtex = None
                bcol = (0.76, 0.77, 0.80, 1.0)
                uv_xf = (1.0, 1.0, 0.0, 0.0)
                has_tex = False
                dec_srgb = False

            tex_to_bind = gtex if has_tex else fallback_tex
            if tex_to_bind is not None:
                basic_shader.uniform_sampler("base_tex", tex_to_bind)
            basic_shader.uniform_float("base_color", bcol)
            basic_shader.uniform_float("uv_transform", uv_xf)
            basic_shader.uniform_float(
                "params",
                (1.0 if has_tex else 0.0, 1.0 if dec_srgb else 0.0, _GRID_SIZE, grid_strength),
            )
            batch.draw(basic_shader)


_BBOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _get_line_shader():
    global _LINE_SHADER
    if _LINE_SHADER is not None:
        return _LINE_SHADER
    if bpy.app.background:
        return None
    try:
        _LINE_SHADER = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    except Exception:
        try:
            _LINE_SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
        except Exception:
            _LINE_SHADER = None
    return _LINE_SHADER


def _extract_cutter_line_positions(obj: bpy.types.Object) -> np.ndarray | None:
    """Return `(N*2, 3)` float32 world-space line endpoints for a cutter wireframe."""
    me = getattr(obj, "data", None)
    if me is None:
        return None
    mat = np.asarray(obj.matrix_world, dtype=np.float32)
    rot_scale = mat[:3, :3]
    trans = mat[:3, 3]

    if obj.mode == "EDIT":
        try:
            bm = bmesh.from_edit_mesh(me)
            ne = len(bm.edges)
            if 0 < ne <= 512:
                pts_list = []
                for e in bm.edges:
                    if e.is_valid:
                        v0, v1 = e.verts[0].co, e.verts[1].co
                        pts_list.append((v0.x, v0.y, v0.z))
                        pts_list.append((v1.x, v1.y, v1.z))
                if pts_list:
                    pts_arr = np.asarray(pts_list, dtype=np.float32)
                    return np.ascontiguousarray((pts_arr @ rot_scale.T) + trans, dtype=np.float32)
        except Exception:
            pass

    num_edges = len(me.edges)
    if 0 < num_edges <= 256:
        nv = len(me.vertices)
        if nv == 0:
            return None
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        verts_w = (co.reshape(-1, 3) @ rot_scale.T) + trans
        e_idx = np.empty(num_edges * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", e_idx)
        return np.ascontiguousarray(verts_w[e_idx], dtype=np.float32)

    # Dense mesh (e.g. Rough Edge cutter): draw clean 12-edge bounding box
    bb = np.asarray([tuple(c) for c in obj.bound_box], dtype=np.float32)
    if bb.shape != (8, 3):
        return None
    bb_w = (bb @ rot_scale.T) + trans
    idx = np.asarray([i for pair in _BBOX_EDGES for i in pair], dtype=np.int32)
    return np.ascontiguousarray(bb_w[idx], dtype=np.float32)


def _draw_edit_mode_cutter_overlay(
    obj: bpy.types.Object,
    line_shader,
    is_poly: bool,
) -> None:
    """Draw always-visible Edit Mode overlay (selected face tint, live edges, face `*` dots, verts)
    on top of the GPU building preview so cutter faces are effortless to see and click."""
    me = getattr(obj, "data", None)
    if me is None:
        return
    try:
        bm = bmesh.from_edit_mesh(me)
    except Exception:
        return
    if not bm or len(bm.verts) == 0:
        return

    mw = obj.matrix_world
    ts = getattr(bpy.context, "tool_settings", None)
    sel_mode = tuple(ts.mesh_select_mode) if ts is not None else (True, False, False)
    show_verts, _show_edges, show_faces = bool(sel_mode[0]), bool(sel_mode[1]), bool(sel_mode[2])

    flat_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("ALPHA")

    # 1. Selected face translucent highlight (instant feedback of selected cutter face)
    sel_tri_pts = []
    for f in bm.faces:
        if not f.is_valid or not f.select or len(f.verts) < 3:
            continue
        wverts = [mw @ v.co for v in f.verts]
        p0 = wverts[0]
        for k in range(1, len(wverts) - 1):
            sel_tri_pts.append((p0.x, p0.y, p0.z))
            p1, p2 = wverts[k], wverts[k + 1]
            sel_tri_pts.append((p1.x, p1.y, p1.z))
            sel_tri_pts.append((p2.x, p2.y, p2.z))
    if sel_tri_pts:
        tri_arr = np.asarray(sel_tri_pts, dtype=np.float32)
        flat_shader.bind()
        flat_shader.uniform_float("color", (1.0, 0.56, 0.12, 0.22))
        batch_for_shader(flat_shader, "TRIS", {"pos": tri_arr}).draw(flat_shader)

    # 2. Live Edit Mode edges (unselected orange, selected bright yellow-white)
    unsel_e = []
    sel_e = []
    for e in bm.edges:
        if not e.is_valid:
            continue
        a = mw @ e.verts[0].co
        b = mw @ e.verts[1].co
        dst = sel_e if e.select else unsel_e
        dst.append((a.x, a.y, a.z))
        dst.append((b.x, b.y, b.z))

    line_shader.bind()
    if unsel_e:
        arr = np.asarray(unsel_e, dtype=np.float32)
        if is_poly:
            line_shader.uniform_float("lineWidth", 1.8)
        line_shader.uniform_float("color", (1.0, 0.62, 0.16, 0.88))
        batch_for_shader(line_shader, "LINES", {"pos": arr}).draw(line_shader)
    if sel_e:
        arr = np.asarray(sel_e, dtype=np.float32)
        if is_poly:
            line_shader.uniform_float("lineWidth", 2.8)
        line_shader.uniform_float("color", (1.0, 0.94, 0.36, 1.0))
        batch_for_shader(line_shader, "LINES", {"pos": arr}).draw(line_shader)

    # 3. High-contrast Face Center Dots (`*`) when in Face Select Mode (never hidden by building!)
    if show_faces:
        unsel_fc = []
        sel_fc = []
        for f in bm.faces:
            if not f.is_valid:
                continue
            c = mw @ f.calc_center_median()
            dst = sel_fc if f.select else unsel_fc
            dst.append((c.x, c.y, c.z))

        flat_shader.bind()
        if unsel_fc:
            arr = np.asarray(unsel_fc, dtype=np.float32)
            pb = batch_for_shader(flat_shader, "POINTS", {"pos": arr})
            gpu.state.point_size_set(11.0)
            flat_shader.uniform_float("color", (0.05, 0.05, 0.05, 0.85))
            pb.draw(flat_shader)
            gpu.state.point_size_set(6.5)
            flat_shader.uniform_float("color", (1.0, 0.66, 0.18, 1.0))
            pb.draw(flat_shader)
        if sel_fc:
            arr = np.asarray(sel_fc, dtype=np.float32)
            pb = batch_for_shader(flat_shader, "POINTS", {"pos": arr})
            gpu.state.point_size_set(14.0)
            flat_shader.uniform_float("color", (0.05, 0.05, 0.05, 0.95))
            pb.draw(flat_shader)
            gpu.state.point_size_set(8.5)
            flat_shader.uniform_float("color", (1.0, 0.96, 0.45, 1.0))
            pb.draw(flat_shader)

    # 4. High-contrast Vertex Dots when in Vertex Select Mode
    if show_verts and len(bm.verts) <= 1024:
        unsel_v = []
        sel_v = []
        for v in bm.verts:
            if not v.is_valid:
                continue
            p = mw @ v.co
            dst = sel_v if v.select else unsel_v
            dst.append((p.x, p.y, p.z))

        flat_shader.bind()
        if unsel_v:
            arr = np.asarray(unsel_v, dtype=np.float32)
            pb = batch_for_shader(flat_shader, "POINTS", {"pos": arr})
            gpu.state.point_size_set(8.5)
            flat_shader.uniform_float("color", (0.05, 0.05, 0.05, 0.82))
            pb.draw(flat_shader)
            gpu.state.point_size_set(4.8)
            flat_shader.uniform_float("color", (1.0, 0.64, 0.18, 1.0))
            pb.draw(flat_shader)
        if sel_v:
            arr = np.asarray(sel_v, dtype=np.float32)
            pb = batch_for_shader(flat_shader, "POINTS", {"pos": arr})
            gpu.state.point_size_set(11.5)
            flat_shader.uniform_float("color", (0.05, 0.05, 0.05, 0.95))
            pb.draw(flat_shader)
            gpu.state.point_size_set(6.8)
            flat_shader.uniform_float("color", (1.0, 0.96, 0.45, 1.0))
            pb.draw(flat_shader)

    gpu.state.point_size_set(1.0)
    # Re-bind line shader for any remaining cutters in loop
    line_shader.bind()


def get_selected_cutter_zones() -> set[str]:
    """Return uppercase Zone letters (e.g. {'B'}) for all currently selected/active cutter boxes."""
    zones: set[str] = set()
    ctx = bpy.context
    sel = list(getattr(ctx, "selected_objects", ()) or ())
    act = getattr(ctx, "active_object", None)
    if act is not None and act not in sel:
        sel.append(act)
    for o in sel:
        if o is not None and bool(o.get("fdt_is_cutter", False)):
            z = str(o.get("fdt_cutter_zone", "A") or "A").strip() or "A"
            if z:
                zones.add(z)
    return zones


def _draw_session_cutters_xray() -> None:
    """Draw dual-pass X-ray cutter wireframes so newly added boxes inside buildings are always visible."""
    if not _CUTTERS_COL_NAME:
        return
    col = bpy.data.collections.get(_CUTTERS_COL_NAME)
    if col is None or len(col.objects) == 0:
        return
    line_shader = _get_line_shader()
    if line_shader is None:
        return

    is_poly = True
    try:
        vp = gpu.state.viewport_get()
        vp_size = (float(max(1, vp[2])), float(max(1, vp[3])))
    except Exception:
        vp_size = (1920.0, 1080.0)

    space = getattr(bpy.context, "space_data", None)
    use_xray = bool(
        space is not None
        and getattr(space, "type", "") == "VIEW_3D"
        and getattr(getattr(space, "shading", None), "show_xray", False)
    )

    active_obj = getattr(bpy.context, "active_object", None)
    sel_zones = get_selected_cutter_zones() if _DISPLAY_MODE == "PIECES" else set()
    line_shader.bind()
    try:
        line_shader.uniform_float("viewportSize", vp_size)
    except Exception:
        is_poly = False

    gpu.state.depth_mask_set(False)
    gpu.state.blend_set("ALPHA")

    for obj in col.objects:
        if obj.type != "MESH" or obj.hide_get() or obj.hide_viewport:
            continue
        if obj.mode == "EDIT":
            _draw_edit_mode_cutter_overlay(obj, line_shader, is_poly)
            continue
        pts = _extract_cutter_line_positions(obj)
        if pts is None or pts.size == 0:
            continue
        batch = batch_for_shader(line_shader, "LINES", {"pos": pts})

        is_active = (obj == active_obj)
        is_sel = obj.select_get()
        c_zone = str(obj.get("fdt_cutter_zone", "A") or "A").strip() or "A"
        is_other_zone_dimmed = bool(sel_zones) and (c_zone not in sel_zones)
        if is_active:
            rgb = (1.0, 0.64, 0.16)
            w_front, w_xray = 2.2, 1.4
            a_front, a_xray = 1.0, 0.45
        elif is_sel:
            rgb = (1.0, 0.45, 0.08)
            w_front, w_xray = 1.9, 1.2
            a_front, a_xray = 0.95, 0.38
        elif is_other_zone_dimmed:
            oc = obj.color
            rgb = (float(oc[0]), float(oc[1]), float(oc[2]))
            w_front, w_xray = 1.0, 0.8
            a_front, a_xray = 0.22, 0.08
        else:
            oc = obj.color
            rgb = (float(oc[0]), float(oc[1]), float(oc[2]))
            w_front, w_xray = 1.4, 1.0
            a_front, a_xray = 0.78, 0.25

        # Pass A: Occluded lines only when Blender X-Ray (Alt+Z) or obj.show_in_front is enabled
        if use_xray or obj.show_in_front:
            gpu.state.depth_test_set("GREATER")
            if is_poly:
                line_shader.uniform_float("lineWidth", w_xray)
            line_shader.uniform_float("color", (rgb[0], rgb[1], rgb[2], a_xray))
            batch.draw(line_shader)

        # Pass B: Depth-tested visible outline (occluded naturally behind walls/roof)
        gpu.state.depth_test_set("LESS_EQUAL")
        if is_poly:
            line_shader.uniform_float("lineWidth", w_front)
        line_shader.uniform_float("color", (rgb[0], rgb[1], rgb[2], a_front))
        batch.draw(line_shader)


def _build_piece_wire_batches(
    p_subs: Sequence[tuple],
    z_col: tuple[float, float, float, float],
) -> list[tuple[object, tuple[float, float, float, float]]]:
    """Turn the cut-piece triangle batches into zone-colored line batches for Hole + Wire."""
    line_shader = _get_line_shader()
    if line_shader is None:
        return []
    result: list[tuple[object, tuple[float, float, float, float]]] = []
    for item in p_subs:
        if len(item) < 2:
            continue
        try:
            pos = np.asarray(item[1], dtype=np.float32).reshape(-1, 3)
        except Exception:
            continue
        tri_count = pos.shape[0] // 3
        if tri_count < 1:
            continue
        tris = pos[: tri_count * 3].reshape(-1, 3, 3)
        edges = np.concatenate(
            (tris[:, [0, 1], :], tris[:, [1, 2], :], tris[:, [2, 0], :]),
            axis=1,
        )
        pts = np.ascontiguousarray(edges.reshape(-1, 3), dtype=np.float32)
        if pts.shape[0] == 0:
            continue
        try:
            batch = batch_for_shader(line_shader, "LINES", {"pos": pts})
        except Exception:
            continue
        result.append((batch, z_col))
    return result


def _draw_piece_wireframes() -> None:
    """Overlay the removed/cut meshes as wireframe in their cutter zone color."""
    line_shader = _get_line_shader()
    if line_shader is None:
        return
    try:
        vp = gpu.state.viewport_get()
        vp_size = (float(max(1, vp[2])), float(max(1, vp[3])))
    except Exception:
        vp_size = (1920.0, 1080.0)

    gpu.state.depth_mask_set(False)
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    line_shader.bind()
    try:
        line_shader.uniform_float("viewportSize", vp_size)
    except Exception:
        pass
    try:
        line_shader.uniform_float("lineWidth", 1.0)
    except Exception:
        pass
    for state in _GPU_STATE.values():
        for batch, z_col in state.get("piece_wire", ()):
            line_shader.uniform_float("color", (z_col[0], z_col[1], z_col[2], 0.85))
            batch.draw(line_shader)


def _draw_callback() -> None:
    if not _ACTIVE or not _GPU_STATE:
        return
    basic_shader = _get_preview_shader()
    multi_shader = _get_multimat_shader()
    if basic_shader is None and multi_shader is None:
        return
    fallback_tex = _get_fallback_texture()

    mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()

    prev_depth_test = gpu.state.depth_test_get()
    prev_depth_mask = gpu.state.depth_mask_get()
    prev_blend = gpu.state.blend_get()

    try:
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

        draw_rem = _DISPLAY_MODE != "PIECES"
        draw_pieces = _DISPLAY_MODE not in ("MAIN", "MAIN_WIRE")
        sel_zones = get_selected_cutter_zones() if _DISPLAY_MODE == "PIECES" else set()

        for state in _GPU_STATE.values():
            mat_infos = state["materials"]

            if draw_rem:
                _draw_submesh_list(
                    basic_shader,
                    multi_shader,
                    mvp,
                    fallback_tex,
                    mat_infos,
                    state["rem_batches"],
                    zone_tint=(0.0, 0.0, 0.0, 0.0),
                    grid_strength=0.0,
                )

            if draw_pieces:
                for p_entry in state["piece_batches"]:
                    if len(p_entry) >= 3:
                        p_subs, z_col, p_zone = p_entry[0], p_entry[1], p_entry[2]
                    else:
                        p_subs, z_col, p_zone = p_entry[0], p_entry[1], "A"
                    if sel_zones and p_zone not in sel_zones:
                        continue
                    if _SHOW_GRID:
                        zt = (z_col[0], z_col[1], z_col[2], 0.22)
                        gs = 0.65
                    else:
                        zt = (0.0, 0.0, 0.0, 0.0)
                        gs = 0.0
                    _draw_submesh_list(
                        basic_shader,
                        multi_shader,
                        mvp,
                        fallback_tex,
                        mat_infos,
                        p_subs,
                        zone_tint=zt,
                        grid_strength=gs,
                    )

        if _DISPLAY_MODE == "MAIN_WIRE":
            _draw_piece_wireframes()

        # Pass 3: Dual-pass X-Ray Cutter Wireframes (visible immediately even inside building!)
        _draw_session_cutters_xray()
    except Exception as exc:
        print(f"[FullDestruc_Tool] Textured GPU Preview draw error: {exc}")
    finally:
        gpu.state.depth_test_set(prev_depth_test)
        gpu.state.depth_mask_set(prev_depth_mask)
        gpu.state.blend_set(prev_blend)


def is_active() -> bool:
    return _ACTIVE


def set_display_options(
    display_mode: str | None = None,
    show_grid: bool | None = None,
    grid_size: float | None = None,
) -> None:
    """Zero-cost instant viewport toggle (no Rust recompute, no mesh rebuild)."""
    global _DISPLAY_MODE, _SHOW_GRID, _GRID_SIZE
    if display_mode is not None:
        m = str(display_mode).upper()
        if m in ("ALL", "PIECES", "MAIN", "MAIN_WIRE"):
            _DISPLAY_MODE = m
    if show_grid is not None:
        _SHOW_GRID = bool(show_grid)
    if grid_size is not None:
        _GRID_SIZE = float(grid_size)
    if _ACTIVE:
        _tag_all_view3d_redraw()


def begin_drag_preview(
    session: AppleBiteSession,
    grid_size: float = 0.35,
    show_grid: bool = True,
    display_mode: str = "ALL",
) -> None:
    """Activate Zero-Switch Textured GPU preview and hide working Blender meshes."""
    global _ACTIVE, _GRID_SIZE, _SHOW_GRID, _DISPLAY_MODE, _CUTTERS_COL_NAME
    _GRID_SIZE = float(grid_size)
    _SHOW_GRID = bool(show_grid)
    m = str(display_mode).upper()
    _DISPLAY_MODE = m if m in ("ALL", "PIECES", "MAIN", "MAIN_WIRE") else "ALL"
    _ACTIVE = True

    cols = session.get_collections()
    cut_col = cols.get("cutters")
    _CUTTERS_COL_NAME = cut_col.name if cut_col is not None else ""
    for key in ("results", "pieces"):
        col = cols.get(key)
        if col is None:
            continue
        for obj in col.objects:
            if obj.type == "MESH" and not obj.hide_get():
                try:
                    obj.hide_set(True)
                    if obj not in _HIDDEN_OBJECTS:
                        _HIDDEN_OBJECTS.append(obj)
                except Exception:
                    pass


def _build_submesh_batches(
    submeshes: Sequence[tuple],
    mat_infos: list[dict],
    basic_shader,
    multi_shader,
) -> list[tuple[int, bool, object]]:
    batches: list[tuple[int, bool, object]] = []
    for item in submeshes:
        if len(item) >= 5:
            mat_id, pos, nor, uv0, uv1 = item[:5]
        else:
            mat_id, pos, nor, uv0 = item[:4]
            uv1 = uv0
        if pos.size == 0:
            continue
        desc = mat_infos[mat_id] if (0 <= mat_id < len(mat_infos)) else None
        if (
            desc is not None
            and desc.get("kind") == "MULTIMAT"
            and desc.get("ubo") is not None
            and multi_shader is not None
        ):
            b = batch_for_shader(
                multi_shader, "TRIS", {"pos": pos, "nor": nor, "uv0": uv0, "uv1": uv1}
            )
            batches.append((mat_id, True, b))
        elif basic_shader is not None:
            b = batch_for_shader(basic_shader, "TRIS", {"pos": pos, "nor": nor, "uv": uv0})
            batches.append((mat_id, False, b))
    return batches


def update_target_gpu_batches(
    target_name: str,
    src_materials: Sequence[bpy.types.Material | None],
    cut_material: bpy.types.Material | None,
    rem_submeshes: Sequence[tuple],
    pieces_submeshes: Sequence[tuple[Sequence[tuple], tuple[float, float, float, float]]],
) -> None:
    """Upload material-grouped world-space NumPy arrays from Rust straight into GPU VRAM batches."""
    global _LAST_MAT_DEBUG
    basic_shader = _get_preview_shader()
    multi_shader = _get_multimat_shader()
    if basic_shader is None and multi_shader is None:
        return

    mat_slots: list[bpy.types.Material | None] = list(src_materials)
    if cut_material is not None and cut_material not in mat_slots:
        mat_slots.append(cut_material)

    mat_infos = []
    dbg_notes = []
    for idx, m in enumerate(mat_slots):
        desc = _resolve_material_gpu_info(m)
        mat_infos.append(desc)
        dbg_notes.append(f"[{idx}] {desc['note']}")
    _LAST_MAT_DEBUG = dbg_notes

    rem_batches = _build_submesh_batches(rem_submeshes, mat_infos, basic_shader, multi_shader)

    piece_batches: list[tuple[list[tuple[int, bool, object]], tuple[float, float, float, float], str]] = []
    piece_wire: list[tuple[object, tuple[float, float, float, float]]] = []
    for p_item in pieces_submeshes:
        if len(p_item) >= 3:
            sub_list, z_col, z_let = p_item[0], p_item[1], str(p_item[2] or "A").strip() or "A"
        else:
            sub_list, z_col, z_let = p_item[0], p_item[1], "A"
        p_subs = _build_submesh_batches(sub_list, mat_infos, basic_shader, multi_shader)
        if p_subs:
            piece_batches.append((p_subs, z_col, z_let))
            piece_wire.extend(_build_piece_wire_batches(sub_list, z_col))

    _GPU_STATE[target_name] = {
        "materials": mat_infos,
        "rem_batches": rem_batches,
        "piece_batches": piece_batches,
        "piece_wire": piece_wire,
    }
    _tag_all_view3d_redraw()


def end_drag_preview() -> None:
    """Deactivate GPU preview and restore visibility of the committed real Blender meshes."""
    global _ACTIVE, _CUTTERS_COL_NAME
    was_active = _ACTIVE
    _ACTIVE = False
    _CUTTERS_COL_NAME = ""
    _GPU_STATE.clear()
    _MAT_CACHE.clear()

    for obj in _HIDDEN_OBJECTS:
        try:
            if obj.name in bpy.data.objects:
                obj.hide_set(False)
        except Exception:
            pass
    _HIDDEN_OBJECTS.clear()

    if was_active:
        _tag_all_view3d_redraw()


def register() -> None:
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        try:
            _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _draw_callback, (), "WINDOW", "POST_VIEW"
            )
        except Exception:
            _DRAW_HANDLE = None


def unregister() -> None:
    global _DRAW_HANDLE, _BASIC_SHADER, _BASIC_SHADER_FAILED, _MULTI_SHADER, _MULTI_SHADER_FAILED, _LINE_SHADER, _FALLBACK_TEX
    end_drag_preview()
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        except Exception:
            pass
        _DRAW_HANDLE = None
    _BASIC_SHADER = None
    _BASIC_SHADER_FAILED = False
    _MULTI_SHADER = None
    _MULTI_SHADER_FAILED = False
    _LINE_SHADER = None
    _FALLBACK_TEX = None
