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

ROUGH_CUTTER_VERSION = 2

ORIG_ATTR_NAME = "fdt_orig"
UV_ATTR_NAME = "UVMap0"


def _get_socket(sockets, *names: str, index: int = 0):
    """Look up a node socket by candidate names with positional fallback (Blender 4.x/5.2+ safe)."""
    for name in names:
        sock = sockets.get(name)
        if sock is not None:
            return sock
    return sockets[index]


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
    """Create or return FDT_GN_RoughCutter (v2) node group for edge-band inward rough cuts.

    Extracts sharp outer edges before subdivision, computes normalized vertex proximity to those
    edges, and carves strictly inward (-Normal) within the configurable Inward Ratio band so the
    cutter stays 100% solid, watertight, manifold, and never bulges past its drawn boundary.
    """
    existing = bpy.data.node_groups.get(GN_ROUGH_CUTTER)
    if existing is not None:
        if int(existing.get("fdt_gn_version", 1)) >= ROUGH_CUTTER_VERSION:
            return existing
        # Upgrade legacy v1 node group in-place so existing modifier links stay intact
        ng = existing
        ng.links.clear()
        ng.nodes.clear()
        for item in list(ng.interface.items_tree):
            ng.interface.remove(item)
    else:
        ng = _new_group(GN_ROUGH_CUTTER)

    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")

    s_inward = ng.interface.new_socket(name="Inward Ratio", in_out="INPUT", socket_type="NodeSocketFloat")
    s_inward.default_value = 0.35
    if hasattr(s_inward, "min_value"):
        s_inward.min_value = 0.0
        s_inward.max_value = 1.0
    if hasattr(s_inward, "subtype"):
        s_inward.subtype = "FACTOR"

    s_strength = ng.interface.new_socket(name="Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    s_strength.default_value = 0.08
    if hasattr(s_strength, "min_value"):
        s_strength.min_value = 0.0
        s_strength.max_value = 2.0

    s_scale = ng.interface.new_socket(name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s_scale.default_value = 3.0
    if hasattr(s_scale, "min_value"):
        s_scale.min_value = 0.1
        s_scale.max_value = 50.0

    s_detail = ng.interface.new_socket(name="Noise Detail", in_out="INPUT", socket_type="NodeSocketFloat")
    s_detail.default_value = 0.5
    if hasattr(s_detail, "min_value"):
        s_detail.min_value = 0.0
        s_detail.max_value = 1.0
    if hasattr(s_detail, "subtype"):
        s_detail.subtype = "FACTOR"

    s_subdiv = ng.interface.new_socket(name="Subdiv Level", in_out="INPUT", socket_type="NodeSocketInt")
    s_subdiv.default_value = 3
    if hasattr(s_subdiv, "min_value"):
        s_subdiv.min_value = 1
        s_subdiv.max_value = 5

    s_seed = ng.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketFloat")
    s_seed.default_value = 0.0

    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-1200, 0)

    # 1. Extract sharp outer edges before subdivision (angle > ~20 deg = 0.35 rad)
    n_edge_angle = nodes.new("GeometryNodeInputMeshEdgeAngle")
    n_edge_angle.location = (-1000, 320)

    n_is_sharp = nodes.new("FunctionNodeCompare")
    n_is_sharp.location = (-800, 320)
    n_is_sharp.data_type = "FLOAT"
    n_is_sharp.operation = "GREATER_THAN"
    n_is_sharp.inputs["B"].default_value = 0.35

    n_sep_edges = nodes.new("GeometryNodeSeparateGeometry")
    n_sep_edges.location = (-600, 320)
    n_sep_edges.domain = "EDGE"

    links.new(n_edge_angle.outputs["Unsigned Angle"], n_is_sharp.inputs["A"])
    links.new(n_in.outputs["Geometry"], n_sep_edges.inputs["Geometry"])
    links.new(n_is_sharp.outputs["Result"], n_sep_edges.inputs["Selection"])

    # 2. Watertight uniform subdivision (keeps cutter 100% manifold for Rust Boolean Engine)
    n_subdiv = nodes.new("GeometryNodeSubdivideMesh")
    n_subdiv.location = (-600, 80)
    links.new(n_in.outputs["Geometry"], n_subdiv.inputs["Mesh"])
    links.new(n_in.outputs["Subdiv Level"], n_subdiv.inputs["Level"])

    # 3. Normalized edge-distance mask (1.0 at sharp edge -> 0.0 at band_width inward)
    n_prox = nodes.new("GeometryNodeProximity")
    n_prox.location = (-360, 300)
    n_prox.target_element = "EDGES"
    links.new(
        n_sep_edges.outputs["Selection"],
        _get_socket(n_prox.inputs, "Geometry", "Target", index=0),
    )

    n_stat = nodes.new("GeometryNodeAttributeStatistic")
    n_stat.location = (-140, 360)
    n_stat.data_type = "FLOAT"
    n_stat.domain = "POINT"
    links.new(n_subdiv.outputs["Mesh"], n_stat.inputs["Geometry"])
    links.new(n_prox.outputs["Distance"], n_stat.inputs["Attribute"])

    n_max_d = nodes.new("ShaderNodeMath")
    n_max_d.location = (60, 360)
    n_max_d.operation = "MAXIMUM"
    n_max_d.inputs[1].default_value = 1e-4
    links.new(n_stat.outputs["Max"], n_max_d.inputs[0])

    n_band = nodes.new("ShaderNodeMath")
    n_band.location = (240, 320)
    n_band.operation = "MULTIPLY"
    links.new(n_max_d.outputs["Value"], n_band.inputs[0])
    links.new(n_in.outputs["Inward Ratio"], n_band.inputs[1])

    n_band_safe = nodes.new("ShaderNodeMath")
    n_band_safe.location = (420, 320)
    n_band_safe.operation = "MAXIMUM"
    n_band_safe.inputs[1].default_value = 1e-5
    links.new(n_band.outputs["Value"], n_band_safe.inputs[0])

    n_map = nodes.new("ShaderNodeMapRange")
    n_map.location = (600, 260)
    n_map.data_type = "FLOAT"
    n_map.interpolation_type = "SMOOTHSTEP"
    n_map.clamp = True
    n_map.inputs["From Min"].default_value = 0.0
    n_map.inputs["To Min"].default_value = 1.0
    n_map.inputs["To Max"].default_value = 0.0
    links.new(n_prox.outputs["Distance"], n_map.inputs["Value"])
    links.new(n_band_safe.outputs["Value"], n_map.inputs["From Max"])

    # Clean zero guard when Inward Ratio == 0.0
    n_ratio_on = nodes.new("FunctionNodeCompare")
    n_ratio_on.location = (600, 40)
    n_ratio_on.data_type = "FLOAT"
    n_ratio_on.operation = "GREATER_THAN"
    n_ratio_on.inputs["B"].default_value = 0.001
    links.new(n_in.outputs["Inward Ratio"], n_ratio_on.inputs["A"])

    n_mask = nodes.new("ShaderNodeMath")
    n_mask.location = (800, 180)
    n_mask.operation = "MULTIPLY"
    links.new(n_map.outputs["Result"], n_mask.inputs[0])
    links.new(n_ratio_on.outputs["Result"], n_mask.inputs[1])

    # 4. 4D Noise with customizable Detail & Roughness
    n_det_map = nodes.new("ShaderNodeMath")
    n_det_map.location = (-800, -240)
    n_det_map.operation = "MULTIPLY_ADD"
    n_det_map.inputs[1].default_value = 8.0
    n_det_map.inputs[2].default_value = 2.0
    links.new(n_in.outputs["Noise Detail"], n_det_map.inputs[0])

    n_rgh_map = nodes.new("ShaderNodeMath")
    n_rgh_map.location = (-800, -400)
    n_rgh_map.operation = "MULTIPLY_ADD"
    n_rgh_map.inputs[1].default_value = 0.6
    n_rgh_map.inputs[2].default_value = 0.3
    links.new(n_in.outputs["Noise Detail"], n_rgh_map.inputs[0])

    n_noise = nodes.new("ShaderNodeTexNoise")
    n_noise.location = (-560, -260)
    n_noise.noise_dimensions = "4D"
    links.new(n_in.outputs["Noise Scale"], n_noise.inputs["Scale"])
    links.new(n_in.outputs["Seed"], n_noise.inputs["W"])
    links.new(n_det_map.outputs["Value"], n_noise.inputs["Detail"])
    links.new(n_rgh_map.outputs["Value"], n_noise.inputs["Roughness"])

    # 5. Strictly inward displacement: Offset = Normal * (-Fac * edge_mask * Strength)
    n_masked_noise = nodes.new("ShaderNodeMath")
    n_masked_noise.location = (980, -40)
    n_masked_noise.operation = "MULTIPLY"
    links.new(n_noise.outputs["Fac"], n_masked_noise.inputs[0])
    links.new(n_mask.outputs["Value"], n_masked_noise.inputs[1])

    n_depth = nodes.new("ShaderNodeMath")
    n_depth.location = (1160, -40)
    n_depth.operation = "MULTIPLY"
    links.new(n_masked_noise.outputs["Value"], n_depth.inputs[0])
    links.new(n_in.outputs["Strength"], n_depth.inputs[1])

    n_neg = nodes.new("ShaderNodeMath")
    n_neg.location = (1340, -40)
    n_neg.operation = "MULTIPLY"
    n_neg.inputs[1].default_value = -1.0
    links.new(n_depth.outputs["Value"], n_neg.inputs[0])

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (1340, -220)

    n_vec_scale = nodes.new("ShaderNodeVectorMath")
    n_vec_scale.location = (1540, -120)
    n_vec_scale.operation = "SCALE"
    links.new(n_norm.outputs["Normal"], n_vec_scale.inputs[0])
    links.new(n_neg.outputs["Value"], n_vec_scale.inputs["Scale"])

    n_set_pos = nodes.new("GeometryNodeSetPosition")
    n_set_pos.location = (1760, 80)
    links.new(n_subdiv.outputs["Mesh"], n_set_pos.inputs["Geometry"])
    links.new(n_vec_scale.outputs["Vector"], n_set_pos.inputs["Offset"])

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (1980, 80)
    links.new(n_set_pos.outputs["Geometry"], n_out.inputs["Geometry"])

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
