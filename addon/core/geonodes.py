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

ORIG_ATTR_NAME = "fdt_orig"
UV_ATTR_NAME = "UVMap0"


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
    """Create or return FDT_GN_RoughCutter node group for rough/jagged cut edges.

    Uses Subdivide Mesh + 4D Noise displacement along normals so the cutter stays
    100% solid, manifold, and fast (avoiding hollow-shell volume artifacts).
    """
    existing = bpy.data.node_groups.get(GN_ROUGH_CUTTER)
    if existing is not None:
        return existing

    ng = _new_group(GN_ROUGH_CUTTER)
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_scale = ng.interface.new_socket(name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s_scale.default_value = 3.0
    s_strength = ng.interface.new_socket(name="Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    s_strength.default_value = 0.08
    s_voxel = ng.interface.new_socket(name="Voxel Size", in_out="INPUT", socket_type="NodeSocketFloat")
    s_voxel.default_value = 0.06
    s_seed = ng.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketFloat")
    s_seed.default_value = 0.0
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-800, 0)

    n_subdiv = nodes.new("GeometryNodeSubdivideMesh")
    n_subdiv.location = (-420, 100)
    n_subdiv.inputs["Level"].default_value = 3

    n_noise = nodes.new("ShaderNodeTexNoise")
    n_noise.location = (-550, -200)
    n_noise.noise_dimensions = "4D"

    n_sub = nodes.new("ShaderNodeMath")
    n_sub.location = (-320, -200)
    n_sub.operation = "SUBTRACT"
    n_sub.inputs[1].default_value = 0.5

    n_mul = nodes.new("ShaderNodeMath")
    n_mul.location = (-120, -200)
    n_mul.operation = "MULTIPLY"

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (-120, -360)

    n_vec_scale = nodes.new("ShaderNodeVectorMath")
    n_vec_scale.location = (80, -240)
    n_vec_scale.operation = "SCALE"

    n_set_pos = nodes.new("GeometryNodeSetPosition")
    n_set_pos.location = (300, 80)

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (550, 80)

    links.new(n_in.outputs["Geometry"], n_subdiv.inputs["Mesh"])

    links.new(n_in.outputs["Noise Scale"], n_noise.inputs["Scale"])
    links.new(n_in.outputs["Seed"], n_noise.inputs["W"])
    links.new(n_noise.outputs["Fac"], n_sub.inputs[0])
    links.new(n_sub.outputs["Value"], n_mul.inputs[0])
    links.new(n_in.outputs["Strength"], n_mul.inputs[1])

    links.new(n_norm.outputs["Normal"], n_vec_scale.inputs[0])
    links.new(n_mul.outputs["Value"], n_vec_scale.inputs["Scale"])

    links.new(n_subdiv.outputs["Mesh"], n_set_pos.inputs["Geometry"])
    links.new(n_vec_scale.outputs["Vector"], n_set_pos.inputs["Offset"])
    links.new(n_set_pos.outputs["Geometry"], n_out.inputs["Geometry"])
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
