"""Blender 5.2 Geometry Nodes builders for Apple Bite v2.

Node groups:
- FDT_GN_MarkOrig: stamps face float attribute `fdt_orig = 1.0` before booleans.
- FDT_GN_RoughCutter: optional rough-edge volume + 4D noise displacement on cutters.
- FDT_GN_CutFaceUV: assigns cut material and tri-planar box UVs (`UVMap0`) to cut faces (`fdt_orig < 0.5`).
"""

from __future__ import annotations

import bpy

GN_MARK_ORIG = "FDT_GN_MarkOrig"
GN_ROUGH_CUTTER = "FDT_GN_RoughCutter"
GN_CUT_FACE_UV = "FDT_GN_CutFaceUV"

ROUGH_CUTTER_VERSION = 8

ORIG_ATTR_NAME = "fdt_orig"
UV_ATTR_NAME = "UVMap0"


def _get_socket(sockets, *names: str, index: int = 0):
    """Look up an enabled node socket by candidate names with positional fallback (Blender 5.2+ safe)."""
    for name in names:
        for sock in sockets:
            if sock.name == name and getattr(sock, "enabled", True):
                return sock
    for name in names:
        sock = sockets.get(name)
        if sock is not None:
            return sock
    return sockets[index]


def _set_node_enum(node, prop_name: str, socket_name: str, value: str) -> None:
    """Set a Geometry Node enum safely in Blender 5.2+ (menu input socket or node property)."""
    if hasattr(node, prop_name):
        setattr(node, prop_name, value)
        return
    sock = node.inputs.get(socket_name)
    if sock is not None and hasattr(sock, "default_value"):
        try:
            sock.default_value = value
        except Exception:
            pass


def _new_group(name: str) -> bpy.types.NodeTree:
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        return existing
    ng = bpy.data.node_groups.new(name=name, type="GeometryNodeTree")
    return ng


def ensure_mark_orig_group() -> bpy.types.NodeTree:
    """Create or return FDT_GN_MarkOrig node group."""
    existing = bpy.data.node_groups.get(GN_MARK_ORIG)
    if existing is not None:
        return existing

    ng = _new_group(GN_MARK_ORIG)
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-300, 0)

    n_store = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store.location = (0, 0)
    n_store.data_type = "FLOAT"
    n_store.domain = "FACE"
    n_store.inputs["Name"].default_value = ORIG_ATTR_NAME
    n_store.inputs["Value"].default_value = 1.0

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (300, 0)

    links.new(n_in.outputs["Geometry"], n_store.inputs["Geometry"])
    links.new(n_store.outputs["Geometry"], n_out.inputs["Geometry"])
    return ng


def ensure_rough_cutter_group() -> bpy.types.NodeTree:
    """Create or return FDT_GN_RoughCutter (v8) for bold, blocky, spike-free concrete fracture.

    1. Subdivides cleanly (Level 1-3) and runs 1-pass topological `Blur Attribute` on raw 4D Noise
       BEFORE contrast stretching (`Map Range 0.30..0.70 -> 0..1`) so single-vertex needles are
       eliminated while full bold 0..1 jagged concrete chunks are preserved.
    2. Displaces strictly inward (`-Normal`, up to 42% of min cutter dimension) and applies light
       position blunting (max weight 0.20) to trim razor tips without softening the big chunks.
    3. Uniformly scales the jagged shape inward toward its bounding-box center by `Inward Depth`
       (`scale = 1.0 - Inward Depth * 0.85`), preserving the exact silhouette.
    4. Finishes with shortest-diagonal Triangulate + Merge by Distance + Flat Shading.
    """
    existing = bpy.data.node_groups.get(GN_ROUGH_CUTTER)
    if existing is not None:
        if int(existing.get("fdt_gn_version", 1)) >= ROUGH_CUTTER_VERSION:
            return existing
        # Upgrade legacy v1-v7 node group in-place so existing modifier links stay intact
        ng = existing
        ng.links.clear()
        ng.nodes.clear()
        for item in list(ng.interface.items_tree):
            ng.interface.remove(item)
    else:
        ng = _new_group(GN_ROUGH_CUTTER)

    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")

    s_inward = ng.interface.new_socket(name="Inward Depth", in_out="INPUT", socket_type="NodeSocketFloat")
    s_inward.default_value = 0.0
    if hasattr(s_inward, "min_value"):
        s_inward.min_value = 0.0
        s_inward.max_value = 1.0
    if hasattr(s_inward, "subtype"):
        s_inward.subtype = "FACTOR"

    s_jagged = ng.interface.new_socket(name="Jaggedness", in_out="INPUT", socket_type="NodeSocketFloat")
    s_jagged.default_value = 0.45
    if hasattr(s_jagged, "min_value"):
        s_jagged.min_value = 0.0
        s_jagged.max_value = 1.0
    if hasattr(s_jagged, "subtype"):
        s_jagged.subtype = "FACTOR"

    s_scale = ng.interface.new_socket(name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s_scale.default_value = 1.5
    if hasattr(s_scale, "min_value"):
        s_scale.min_value = 0.2
        s_scale.max_value = 15.0

    s_detail = ng.interface.new_socket(name="Detail", in_out="INPUT", socket_type="NodeSocketInt")
    s_detail.default_value = 2
    if hasattr(s_detail, "min_value"):
        s_detail.min_value = 1
        s_detail.max_value = 3

    s_seed = ng.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketFloat")
    s_seed.default_value = 0.0

    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-1400, 0)

    # 1. Bounding Box -> center (for inward scaling) & bold max jagged amplitude (0.42 * min_dim)
    n_bbox = nodes.new("GeometryNodeBoundBox")
    n_bbox.location = (-1140, 360)
    links.new(n_in.outputs["Geometry"], n_bbox.inputs["Geometry"])

    n_center_sum = nodes.new("ShaderNodeVectorMath")
    n_center_sum.location = (-920, 480)
    n_center_sum.operation = "ADD"
    links.new(n_bbox.outputs["Min"], n_center_sum.inputs[0])
    links.new(n_bbox.outputs["Max"], n_center_sum.inputs[1])

    n_center = nodes.new("ShaderNodeVectorMath")
    n_center.location = (-720, 480)
    n_center.operation = "SCALE"
    n_center.inputs["Scale"].default_value = 0.5
    links.new(n_center_sum.outputs["Vector"], n_center.inputs[0])

    n_size = nodes.new("ShaderNodeVectorMath")
    n_size.location = (-920, 300)
    n_size.operation = "SUBTRACT"
    links.new(n_bbox.outputs["Max"], n_size.inputs[0])
    links.new(n_bbox.outputs["Min"], n_size.inputs[1])

    n_sep_sz = nodes.new("ShaderNodeSeparateXYZ")
    n_sep_sz.location = (-720, 300)
    links.new(n_size.outputs["Vector"], n_sep_sz.inputs["Vector"])

    n_min_xy = nodes.new("ShaderNodeMath")
    n_min_xy.location = (-520, 340)
    n_min_xy.operation = "MINIMUM"
    links.new(n_sep_sz.outputs["X"], n_min_xy.inputs[0])
    links.new(n_sep_sz.outputs["Y"], n_min_xy.inputs[1])

    n_min_xyz = nodes.new("ShaderNodeMath")
    n_min_xyz.location = (-340, 300)
    n_min_xyz.operation = "MINIMUM"
    links.new(n_min_xy.outputs["Value"], n_min_xyz.inputs[0])
    links.new(n_sep_sz.outputs["Z"], n_min_xyz.inputs[1])

    n_max_amp = nodes.new("ShaderNodeMath")
    n_max_amp.location = (-140, 300)
    n_max_amp.operation = "MULTIPLY"
    n_max_amp.inputs[1].default_value = 0.42
    links.new(n_min_xyz.outputs["Value"], n_max_amp.inputs[0])

    # jagged_amp = 0.42 * min_dim * Jaggedness
    n_jagged_amp = nodes.new("ShaderNodeMath")
    n_jagged_amp.location = (60, 260)
    n_jagged_amp.operation = "MULTIPLY"
    links.new(n_max_amp.outputs["Value"], n_jagged_amp.inputs[0])
    links.new(n_in.outputs["Jaggedness"], n_jagged_amp.inputs[1])

    # 2. Low-poly clean subdivision (Level 1 = 48 tris, Level 2 = 192 tris, Level 3 = 768 tris on a box)
    n_subdiv = nodes.new("GeometryNodeSubdivideMesh")
    n_subdiv.location = (-520, 80)
    links.new(n_in.outputs["Geometry"], n_subdiv.inputs["Mesh"])
    links.new(n_in.outputs["Detail"], n_subdiv.inputs["Level"])

    # 3. Bold 4D Noise -> 1-pass topological blur (kills single-vertex needles) -> FULL contrast stretch!
    n_noise = nodes.new("ShaderNodeTexNoise")
    n_noise.location = (-720, -220)
    n_noise.noise_dimensions = "4D"
    n_noise.inputs["Detail"].default_value = 2.0
    n_noise.inputs["Roughness"].default_value = 0.55
    n_noise.inputs["Distortion"].default_value = 0.0
    links.new(n_in.outputs["Noise Scale"], n_noise.inputs["Scale"])
    links.new(n_in.outputs["Seed"], n_noise.inputs["W"])

    n_blur_val = nodes.new("GeometryNodeBlurAttribute")
    n_blur_val.location = (-460, -220)
    n_blur_val.data_type = "FLOAT"
    n_blur_val.inputs["Iterations"].default_value = 1
    n_blur_val.inputs["Weight"].default_value = 0.55
    links.new(n_noise.outputs["Fac"], _get_socket(n_blur_val.inputs, "Value", index=0))

    # Stretch contrast AFTER blur so full 0..1 jagged depth is preserved without single-vertex needles
    n_contrast = nodes.new("ShaderNodeMapRange")
    n_contrast.location = (-200, -220)
    n_contrast.data_type = "FLOAT"
    n_contrast.interpolation_type = "LINEAR"
    n_contrast.clamp = True
    n_contrast.inputs["From Min"].default_value = 0.30
    n_contrast.inputs["From Max"].default_value = 0.70
    n_contrast.inputs["To Min"].default_value = 0.0
    n_contrast.inputs["To Max"].default_value = 1.0
    links.new(_get_socket(n_blur_val.outputs, "Value", index=0), n_contrast.inputs["Value"])

    # 4. Inward jagged fracture displacement along -Normal
    n_disp = nodes.new("ShaderNodeMath")
    n_disp.location = (100, -160)
    n_disp.operation = "MULTIPLY"
    links.new(n_contrast.outputs["Result"], n_disp.inputs[0])
    links.new(n_jagged_amp.outputs["Value"], n_disp.inputs[1])

    n_neg = nodes.new("ShaderNodeMath")
    n_neg.location = (280, -160)
    n_neg.operation = "MULTIPLY"
    n_neg.inputs[1].default_value = -1.0
    links.new(n_disp.outputs["Value"], n_neg.inputs[0])

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (280, -340)

    n_vec_scale = nodes.new("ShaderNodeVectorMath")
    n_vec_scale.location = (460, -220)
    n_vec_scale.operation = "SCALE"
    links.new(n_norm.outputs["Normal"], n_vec_scale.inputs[0])
    links.new(n_neg.outputs["Value"], n_vec_scale.inputs["Scale"])

    n_set_jagged = nodes.new("GeometryNodeSetPosition")
    n_set_jagged.location = (660, 80)
    links.new(n_subdiv.outputs["Mesh"], n_set_jagged.inputs["Geometry"])
    links.new(n_vec_scale.outputs["Vector"], n_set_jagged.inputs["Offset"])

    # 5. Light Anti-Spike Position Blunting: trims razor tips (max weight 0.20) while keeping chunks bold
    n_blunt_mul = nodes.new("ShaderNodeMath")
    n_blunt_mul.location = (460, -40)
    n_blunt_mul.operation = "MULTIPLY"
    n_blunt_mul.inputs[1].default_value = 0.25
    links.new(n_in.outputs["Jaggedness"], n_blunt_mul.inputs[0])

    n_blunt_w = nodes.new("ShaderNodeMath")
    n_blunt_w.location = (640, -40)
    n_blunt_w.operation = "MINIMUM"
    n_blunt_w.inputs[1].default_value = 0.20
    links.new(n_blunt_mul.outputs["Value"], n_blunt_w.inputs[0])

    n_pos_blunt_in = nodes.new("GeometryNodeInputPosition")
    n_pos_blunt_in.location = (640, -220)

    n_blur_pos = nodes.new("GeometryNodeBlurAttribute")
    n_blur_pos.location = (840, -160)
    n_blur_pos.data_type = "FLOAT_VECTOR"
    n_blur_pos.inputs["Iterations"].default_value = 1
    links.new(n_pos_blunt_in.outputs["Position"], _get_socket(n_blur_pos.inputs, "Value", index=0))
    links.new(n_blunt_w.outputs["Value"], n_blur_pos.inputs["Weight"])

    n_set_blunt = nodes.new("GeometryNodeSetPosition")
    n_set_blunt.location = (1040, 80)
    links.new(n_set_jagged.outputs["Geometry"], n_set_blunt.inputs["Geometry"])
    links.new(_get_socket(n_blur_pos.outputs, "Value", index=0), n_set_blunt.inputs["Position"])

    # 6. Inward Depth: uniform scale around bbox center (preserves exact blunted jagged silhouette!)
    # scale_factor = 1.0 - Inward Depth * 0.85
    n_inward_scale = nodes.new("ShaderNodeMath")
    n_inward_scale.location = (840, 360)
    n_inward_scale.operation = "MULTIPLY_ADD"
    n_inward_scale.inputs[1].default_value = -0.85
    n_inward_scale.inputs[2].default_value = 1.0
    links.new(n_in.outputs["Inward Depth"], n_inward_scale.inputs[0])

    n_pos2 = nodes.new("GeometryNodeInputPosition")
    n_pos2.location = (840, 520)

    n_rel = nodes.new("ShaderNodeVectorMath")
    n_rel.location = (1040, 480)
    n_rel.operation = "SUBTRACT"
    links.new(n_pos2.outputs["Position"], n_rel.inputs[0])
    links.new(n_center.outputs["Vector"], n_rel.inputs[1])

    n_rel_scaled = nodes.new("ShaderNodeVectorMath")
    n_rel_scaled.location = (1220, 440)
    n_rel_scaled.operation = "SCALE"
    links.new(n_rel.outputs["Vector"], n_rel_scaled.inputs[0])
    links.new(n_inward_scale.outputs["Value"], n_rel_scaled.inputs["Scale"])

    n_pos_final = nodes.new("ShaderNodeVectorMath")
    n_pos_final.location = (1400, 440)
    n_pos_final.operation = "ADD"
    links.new(n_rel_scaled.outputs["Vector"], n_pos_final.inputs[0])
    links.new(n_center.outputs["Vector"], n_pos_final.inputs[1])

    n_set_scale = nodes.new("GeometryNodeSetPosition")
    n_set_scale.location = (1400, 80)
    links.new(n_set_blunt.outputs["Geometry"], n_set_scale.inputs["Geometry"])
    links.new(n_pos_final.outputs["Vector"], n_set_scale.inputs["Position"])

    # 7. Clean planar low-poly triangulation + weld degenerate slivers + crisp flat normals
    n_tri = nodes.new("GeometryNodeTriangulate")
    n_tri.location = (1620, 80)
    _set_node_enum(n_tri, "quad_method", "Quad Method", "SHORTEST_DIAGONAL")
    links.new(n_set_scale.outputs["Geometry"], n_tri.inputs["Mesh"])

    n_merge = nodes.new("GeometryNodeMergeByDistance")
    n_merge.location = (1840, 80)
    n_merge.inputs["Distance"].default_value = 0.015
    links.new(n_tri.outputs["Mesh"], n_merge.inputs["Geometry"])

    n_flat = nodes.new("GeometryNodeSetShadeSmooth")
    n_flat.location = (2060, 80)
    n_flat.inputs["Shade Smooth"].default_value = False
    links.new(n_merge.outputs["Geometry"], n_flat.inputs["Geometry"])

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (2280, 80)
    links.new(n_flat.outputs["Geometry"], n_out.inputs["Geometry"])

    ng["fdt_gn_version"] = ROUGH_CUTTER_VERSION
    return ng


def ensure_cut_face_uv_group() -> bpy.types.NodeTree:
    """Create or return FDT_GN_CutFaceUV node group.

    Selects newly cut faces (`fdt_orig < 0.5`), assigns Cut Material,
    and writes tri-planar box-projected UVs scaled by TexelPerMeter / TextureSize to `UVMap0`.
    """
    existing = bpy.data.node_groups.get(GN_CUT_FACE_UV)
    if existing is not None:
        return existing

    ng = _new_group(GN_CUT_FACE_UV)
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Cut Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    s_tpm = ng.interface.new_socket(name="Texel Per Meter", in_out="INPUT", socket_type="NodeSocketFloat")
    s_tpm.default_value = 512.0
    s_tex = ng.interface.new_socket(name="Texture Size", in_out="INPUT", socket_type="NodeSocketFloat")
    s_tex.default_value = 1024.0
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-900, 0)

    # Read fdt_orig attribute -> is_cut = (fdt_orig < 0.5)
    n_attr = nodes.new("GeometryNodeInputNamedAttribute")
    n_attr.location = (-650, 220)
    n_attr.data_type = "FLOAT"
    n_attr.inputs["Name"].default_value = ORIG_ATTR_NAME

    n_cmp = nodes.new("FunctionNodeCompare")
    n_cmp.location = (-420, 220)
    n_cmp.data_type = "FLOAT"
    n_cmp.operation = "LESS_THAN"
    n_cmp.inputs["B"].default_value = 0.5

    links.new(n_attr.outputs["Attribute"], n_cmp.inputs["A"])

    # Set Material on cut faces
    n_set_mat = nodes.new("GeometryNodeSetMaterial")
    n_set_mat.location = (-120, 120)
    links.new(n_in.outputs["Geometry"], n_set_mat.inputs["Geometry"])
    links.new(n_cmp.outputs["Result"], n_set_mat.inputs["Selection"])
    links.new(n_in.outputs["Cut Material"], n_set_mat.inputs["Material"])

    # Compute UV scale = TexelPerMeter / max(TextureSize, 1.0)
    n_max = nodes.new("ShaderNodeMath")
    n_max.location = (-650, -300)
    n_max.operation = "MAXIMUM"
    n_max.inputs[1].default_value = 1.0
    links.new(n_in.outputs["Texture Size"], n_max.inputs[0])

    n_div = nodes.new("ShaderNodeMath")
    n_div.location = (-450, -260)
    n_div.operation = "DIVIDE"
    links.new(n_in.outputs["Texel Per Meter"], n_div.inputs[0])
    links.new(n_max.outputs["Value"], n_div.inputs[1])

    # Tri-planar box UV from Position & Normal
    n_pos = nodes.new("GeometryNodeInputPosition")
    n_pos.location = (-650, -80)

    n_pos_scaled = nodes.new("ShaderNodeVectorMath")
    n_pos_scaled.location = (-260, -120)
    n_pos_scaled.operation = "SCALE"
    links.new(n_pos.outputs["Position"], n_pos_scaled.inputs[0])
    links.new(n_div.outputs["Value"], n_pos_scaled.inputs["Scale"])

    n_sep_pos = nodes.new("ShaderNodeSeparateXYZ")
    n_sep_pos.location = (-80, -120)
    links.new(n_pos_scaled.outputs["Vector"], n_sep_pos.inputs["Vector"])

    # Abs Normal to pick dominant projection plane (XY, XZ, or YZ)
    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (-650, -480)

    n_abs_norm = nodes.new("ShaderNodeVectorMath")
    n_abs_norm.location = (-460, -480)
    n_abs_norm.operation = "ABSOLUTE"
    links.new(n_norm.outputs["Normal"], n_abs_norm.inputs[0])

    n_sep_norm = nodes.new("ShaderNodeSeparateXYZ")
    n_sep_norm.location = (-260, -480)
    links.new(n_abs_norm.outputs["Vector"], n_sep_norm.inputs["Vector"])

    # Candidate UVs as 3D vectors (Z=0):
    # X-dominant -> (Y, Z), Y-dominant -> (X, Z), Z-dominant -> (X, Y)
    n_uv_x = nodes.new("ShaderNodeCombineXYZ")
    n_uv_x.location = (120, -40)
    links.new(n_sep_pos.outputs["Y"], n_uv_x.inputs["X"])
    links.new(n_sep_pos.outputs["Z"], n_uv_x.inputs["Y"])

    n_uv_y = nodes.new("ShaderNodeCombineXYZ")
    n_uv_y.location = (120, -180)
    links.new(n_sep_pos.outputs["X"], n_uv_y.inputs["X"])
    links.new(n_sep_pos.outputs["Z"], n_uv_y.inputs["Y"])

    n_uv_z = nodes.new("ShaderNodeCombineXYZ")
    n_uv_z.location = (120, -320)
    links.new(n_sep_pos.outputs["X"], n_uv_z.inputs["X"])
    links.new(n_sep_pos.outputs["Y"], n_uv_z.inputs["Y"])

    # Compare |nx| > |ny|
    n_x_gt_y = nodes.new("FunctionNodeCompare")
    n_x_gt_y.location = (-40, -440)
    n_x_gt_y.data_type = "FLOAT"
    n_x_gt_y.operation = "GREATER_THAN"
    links.new(n_sep_norm.outputs["X"], n_x_gt_y.inputs["A"])
    links.new(n_sep_norm.outputs["Y"], n_x_gt_y.inputs["B"])

    n_max_xy = nodes.new("ShaderNodeMath")
    n_max_xy.location = (-40, -580)
    n_max_xy.operation = "MAXIMUM"
    links.new(n_sep_norm.outputs["X"], n_max_xy.inputs[0])
    links.new(n_sep_norm.outputs["Y"], n_max_xy.inputs[1])

    # Compare |nz| >= max(|nx|, |ny|)
    n_z_dom = nodes.new("FunctionNodeCompare")
    n_z_dom.location = (140, -540)
    n_z_dom.data_type = "FLOAT"
    n_z_dom.operation = "GREATER_EQUAL"
    links.new(n_sep_norm.outputs["Z"], n_z_dom.inputs["A"])
    links.new(n_max_xy.outputs["Value"], n_z_dom.inputs["B"])

    n_sw_xy = nodes.new("GeometryNodeSwitch")
    n_sw_xy.location = (340, -140)
    n_sw_xy.input_type = "VECTOR"
    links.new(n_x_gt_y.outputs["Result"], n_sw_xy.inputs["Switch"])
    links.new(n_uv_y.outputs["Vector"], n_sw_xy.inputs["False"])
    links.new(n_uv_x.outputs["Vector"], n_sw_xy.inputs["True"])

    n_sw_final = nodes.new("GeometryNodeSwitch")
    n_sw_final.location = (540, -180)
    n_sw_final.input_type = "VECTOR"
    links.new(n_z_dom.outputs["Result"], n_sw_final.inputs["Switch"])
    links.new(n_sw_xy.outputs["Output"], n_sw_final.inputs["False"])
    links.new(n_uv_z.outputs["Vector"], n_sw_final.inputs["True"])

    # Store UVMap0 on face corners of cut faces
    n_store_uv = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store_uv.location = (760, 120)
    n_store_uv.data_type = "FLOAT2"
    n_store_uv.domain = "CORNER"
    n_store_uv.inputs["Name"].default_value = UV_ATTR_NAME

    links.new(n_set_mat.outputs["Geometry"], n_store_uv.inputs["Geometry"])
    links.new(n_cmp.outputs["Result"], n_store_uv.inputs["Selection"])
    links.new(n_sw_final.outputs["Output"], n_store_uv.inputs["Value"])

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (1000, 120)
    links.new(n_store_uv.outputs["Geometry"], n_out.inputs["Geometry"])
    return ng


def set_modifier_socket(mod: bpy.types.NodesModifier, socket_name: str, value) -> None:
    """Safely assign a Geometry Nodes modifier input by its UI socket name (Blender 5.2+ compatible)."""
    if mod.node_group is None:
        return
    for item in mod.node_group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name == socket_name:
            ident = item.identifier
            # Blender 5.2+ uses mod.properties.inputs[ident]["value"]
            props = getattr(mod, "properties", None)
            inputs = getattr(props, "inputs", None) if props is not None else None
            if inputs is not None and ident in inputs:
                inputs[ident]["value"] = value
                return
            # Fallback for Blender 4.x
            mod[ident] = value
            return
