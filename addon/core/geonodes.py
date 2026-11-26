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

ROUGH_CUTTER_VERSION = 4

ORIG_ATTR_NAME = "fdt_orig"
UV_ATTR_NAME = "UVMap0"


def _get_socket(sockets, *names: str, index: int = 0):
    """Look up a node socket by candidate names with positional fallback (Blender 5.2+ safe)."""
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
    """Create or return FDT_GN_RoughCutter (v4) node group for natural concrete fracture cuts.

    Computes the cutter's Signed Distance Field (SDF) inside its padded bounding box and erodes
    the outer boundary strictly inward using a blended 4D Fractal Noise + 4D Voronoi (F1) rock
    fracture field: `density = signed_dist - carve_factor * carve_depth`.
    Converts directly via Volume-to-Mesh (Threshold = 0.0) with zero internal Boolean overhead,
    eliminating straight box edges and Swiss-cheese holes while staying 100% watertight & manifold.
    """
    existing = bpy.data.node_groups.get(GN_ROUGH_CUTTER)
    if existing is not None:
        if int(existing.get("fdt_gn_version", 1)) >= ROUGH_CUTTER_VERSION:
            return existing
        # Upgrade legacy v1/v2/v3 node group in-place so existing modifier links stay intact
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

    s_carve = ng.interface.new_socket(name="Carve Amount", in_out="INPUT", socket_type="NodeSocketFloat")
    s_carve.default_value = 0.50
    if hasattr(s_carve, "min_value"):
        s_carve.min_value = 0.0
        s_carve.max_value = 1.0
    if hasattr(s_carve, "subtype"):
        s_carve.subtype = "FACTOR"

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

    s_res = ng.interface.new_socket(name="Resolution", in_out="INPUT", socket_type="NodeSocketInt")
    s_res.default_value = 3
    if hasattr(s_res, "min_value"):
        s_res.min_value = 1
        s_res.max_value = 5

    s_seed = ng.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketFloat")
    s_seed.default_value = 0.0

    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-1680, 0)

    # 1. Bounding Box -> padded Volume Cube bounds & max inward depth
    n_bbox = nodes.new("GeometryNodeBoundBox")
    n_bbox.location = (-1380, 360)
    links.new(n_in.outputs["Geometry"], n_bbox.inputs["Geometry"])

    n_pad_min = nodes.new("ShaderNodeVectorMath")
    n_pad_min.location = (-1140, 480)
    n_pad_min.operation = "SUBTRACT"
    n_pad_min.inputs[1].default_value = (0.05, 0.05, 0.05)
    links.new(n_bbox.outputs["Min"], n_pad_min.inputs[0])

    n_pad_max = nodes.new("ShaderNodeVectorMath")
    n_pad_max.location = (-1140, 320)
    n_pad_max.operation = "ADD"
    n_pad_max.inputs[1].default_value = (0.05, 0.05, 0.05)
    links.new(n_bbox.outputs["Max"], n_pad_max.inputs[0])

    n_size = nodes.new("ShaderNodeVectorMath")
    n_size.location = (-1140, 140)
    n_size.operation = "SUBTRACT"
    links.new(n_bbox.outputs["Max"], n_size.inputs[0])
    links.new(n_bbox.outputs["Min"], n_size.inputs[1])

    n_sep_sz = nodes.new("ShaderNodeSeparateXYZ")
    n_sep_sz.location = (-940, 140)
    links.new(n_size.outputs["Vector"], n_sep_sz.inputs["Vector"])

    n_min_xy = nodes.new("ShaderNodeMath")
    n_min_xy.location = (-740, 180)
    n_min_xy.operation = "MINIMUM"
    links.new(n_sep_sz.outputs["X"], n_min_xy.inputs[0])
    links.new(n_sep_sz.outputs["Y"], n_min_xy.inputs[1])

    n_min_xyz = nodes.new("ShaderNodeMath")
    n_min_xyz.location = (-560, 140)
    n_min_xyz.operation = "MINIMUM"
    links.new(n_min_xy.outputs["Value"], n_min_xyz.inputs[0])
    links.new(n_sep_sz.outputs["Z"], n_min_xyz.inputs[1])

    n_half_dim = nodes.new("ShaderNodeMath")
    n_half_dim.location = (-380, 140)
    n_half_dim.operation = "MULTIPLY"
    n_half_dim.inputs[1].default_value = 0.5
    links.new(n_min_xyz.outputs["Value"], n_half_dim.inputs[0])

    n_max_depth = nodes.new("ShaderNodeMath")
    n_max_depth.location = (-200, 140)
    n_max_depth.operation = "MAXIMUM"
    n_max_depth.inputs[1].default_value = 0.05
    links.new(n_half_dim.outputs["Value"], n_max_depth.inputs[0])

    # carve_depth = max_depth * Inward Ratio
    n_carve_depth = nodes.new("ShaderNodeMath")
    n_carve_depth.location = (0, 140)
    n_carve_depth.operation = "MULTIPLY"
    links.new(n_max_depth.outputs["Value"], n_carve_depth.inputs[0])
    links.new(n_in.outputs["Inward Ratio"], n_carve_depth.inputs[1])

    # 2. Exact Signed Distance Field (SDF): +Distance inside cutter, -Distance outside cutter
    n_pos = nodes.new("GeometryNodeInputPosition")
    n_pos.location = (-940, -100)

    n_prox = nodes.new("GeometryNodeProximity")
    n_prox.location = (-740, -40)
    _set_node_enum(n_prox, "target_element", "Target Element", "FACES")
    links.new(
        n_in.outputs["Geometry"],
        _get_socket(n_prox.inputs, "Geometry", "Target", index=0),
    )

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (-940, -240)

    n_sample_n = nodes.new("GeometryNodeSampleNearestSurface")
    n_sample_n.location = (-740, -220)
    n_sample_n.data_type = "FLOAT_VECTOR"
    links.new(n_in.outputs["Geometry"], n_sample_n.inputs["Mesh"])
    links.new(n_norm.outputs["Normal"], n_sample_n.inputs["Value"])

    # to_surface = closest_pos - P; dot(to_surface, N_closest) > 0 => INSIDE
    n_to_surf = nodes.new("ShaderNodeVectorMath")
    n_to_surf.location = (-500, -80)
    n_to_surf.operation = "SUBTRACT"
    links.new(n_prox.outputs["Position"], n_to_surf.inputs[0])
    links.new(n_pos.outputs["Position"], n_to_surf.inputs[1])

    n_dot = nodes.new("ShaderNodeVectorMath")
    n_dot.location = (-300, -120)
    n_dot.operation = "DOT_PRODUCT"
    links.new(n_to_surf.outputs["Vector"], n_dot.inputs[0])
    links.new(n_sample_n.outputs["Value"], n_dot.inputs[1])

    n_is_inside = nodes.new("FunctionNodeCompare")
    n_is_inside.location = (-100, -120)
    n_is_inside.data_type = "FLOAT"
    n_is_inside.operation = "GREATER_THAN"
    n_is_inside.inputs["B"].default_value = 0.0
    links.new(n_dot.outputs["Value"], n_is_inside.inputs["A"])

    n_neg_dist = nodes.new("ShaderNodeMath")
    n_neg_dist.location = (-100, 10)
    n_neg_dist.operation = "MULTIPLY"
    n_neg_dist.inputs[1].default_value = -1.0
    links.new(n_prox.outputs["Distance"], n_neg_dist.inputs[0])

    n_sdf = nodes.new("GeometryNodeSwitch")
    n_sdf.location = (120, -40)
    n_sdf.input_type = "FLOAT"
    links.new(n_is_inside.outputs["Result"], n_sdf.inputs["Switch"])
    links.new(n_neg_dist.outputs["Value"], n_sdf.inputs["False"])
    links.new(n_prox.outputs["Distance"], n_sdf.inputs["True"])

    # 3. Natural Concrete Fracture Field: 65% 4D Fractal Noise + 35% 4D Voronoi (F1) angular facets
    n_det_map = nodes.new("ShaderNodeMath")
    n_det_map.location = (-940, -440)
    n_det_map.operation = "MULTIPLY_ADD"
    n_det_map.inputs[1].default_value = 8.0
    n_det_map.inputs[2].default_value = 2.0
    links.new(n_in.outputs["Noise Detail"], n_det_map.inputs[0])

    n_rgh_map = nodes.new("ShaderNodeMath")
    n_rgh_map.location = (-940, -600)
    n_rgh_map.operation = "MULTIPLY_ADD"
    n_rgh_map.inputs[1].default_value = 0.5
    n_rgh_map.inputs[2].default_value = 0.4
    links.new(n_in.outputs["Noise Detail"], n_rgh_map.inputs[0])

    n_noise = nodes.new("ShaderNodeTexNoise")
    n_noise.location = (-660, -460)
    n_noise.noise_dimensions = "4D"
    n_noise.inputs["Distortion"].default_value = 0.15
    links.new(n_in.outputs["Noise Scale"], n_noise.inputs["Scale"])
    links.new(n_in.outputs["Seed"], n_noise.inputs["W"])
    links.new(n_det_map.outputs["Value"], n_noise.inputs["Detail"])
    links.new(n_rgh_map.outputs["Value"], n_noise.inputs["Roughness"])

    n_vor_scale = nodes.new("ShaderNodeMath")
    n_vor_scale.location = (-940, -760)
    n_vor_scale.operation = "MULTIPLY"
    n_vor_scale.inputs[1].default_value = 1.4
    links.new(n_in.outputs["Noise Scale"], n_vor_scale.inputs[0])

    n_vor = nodes.new("ShaderNodeTexVoronoi")
    n_vor.location = (-660, -740)
    n_vor.voronoi_dimensions = "4D"
    _set_node_enum(n_vor, "feature", "Feature", "F1")
    links.new(n_vor_scale.outputs["Value"], n_vor.inputs["Scale"])
    links.new(n_in.outputs["Seed"], n_vor.inputs["W"])

    # blend = 0.65 * noise + 0.35 * voronoi
    n_n_w = nodes.new("ShaderNodeMath")
    n_n_w.location = (-400, -480)
    n_n_w.operation = "MULTIPLY"
    n_n_w.inputs[1].default_value = 0.65
    links.new(n_noise.outputs["Fac"], n_n_w.inputs[0])

    n_blend = nodes.new("ShaderNodeMath")
    n_blend.location = (-200, -560)
    n_blend.operation = "MULTIPLY_ADD"
    n_blend.inputs[1].default_value = 0.35
    links.new(n_vor.outputs["Distance"], n_blend.inputs[0])
    links.new(n_n_w.outputs["Value"], n_blend.inputs[2])

    # White-noise carve shaping via Carve Amount:
    # low = (1.0 - Carve Amount) * 0.55; carve_factor = MapRange(blend, low..0.85 -> 0..1)
    n_low = nodes.new("ShaderNodeMath")
    n_low.location = (-200, -740)
    n_low.operation = "MULTIPLY_ADD"
    n_low.inputs[1].default_value = -0.55
    n_low.inputs[2].default_value = 0.55
    links.new(n_in.outputs["Carve Amount"], n_low.inputs[0])

    n_carve_fac = nodes.new("ShaderNodeMapRange")
    n_carve_fac.location = (40, -560)
    n_carve_fac.data_type = "FLOAT"
    n_carve_fac.interpolation_type = "LINEAR"
    n_carve_fac.clamp = True
    n_carve_fac.inputs["From Max"].default_value = 0.85
    n_carve_fac.inputs["To Min"].default_value = 0.0
    n_carve_fac.inputs["To Max"].default_value = 1.0
    links.new(n_blend.outputs["Value"], n_carve_fac.inputs["Value"])
    links.new(n_low.outputs["Value"], n_carve_fac.inputs["From Min"])

    # 4. Erode SDF strictly inward: density = signed_dist - carve_factor * carve_depth
    n_erode_amt = nodes.new("ShaderNodeMath")
    n_erode_amt.location = (300, -300)
    n_erode_amt.operation = "MULTIPLY"
    links.new(n_carve_fac.outputs["Result"], n_erode_amt.inputs[0])
    links.new(n_carve_depth.outputs["Value"], n_erode_amt.inputs[1])

    n_density = nodes.new("ShaderNodeMath")
    n_density.location = (520, -100)
    n_density.operation = "SUBTRACT"
    links.new(n_sdf.outputs["Output"], n_density.inputs[0])
    links.new(n_erode_amt.outputs["Value"], n_density.inputs[1])

    # 5. Volume Cube (grid 22..62 from Resolution 1..5) -> Volume to Mesh (Threshold = 0.0)
    n_grid_res = nodes.new("ShaderNodeMath")
    n_grid_res.location = (300, 320)
    n_grid_res.operation = "MULTIPLY_ADD"
    n_grid_res.inputs[1].default_value = 10.0
    n_grid_res.inputs[2].default_value = 12.0
    links.new(n_in.outputs["Resolution"], n_grid_res.inputs[0])

    n_vol = nodes.new("GeometryNodeVolumeCube")
    n_vol.location = (760, 140)
    n_vol.inputs["Background"].default_value = -1.0
    links.new(n_density.outputs["Value"], n_vol.inputs["Density"])
    links.new(n_pad_min.outputs["Vector"], n_vol.inputs["Min"])
    links.new(n_pad_max.outputs["Vector"], n_vol.inputs["Max"])
    links.new(n_grid_res.outputs["Value"], n_vol.inputs["Resolution X"])
    links.new(n_grid_res.outputs["Value"], n_vol.inputs["Resolution Y"])
    links.new(n_grid_res.outputs["Value"], n_vol.inputs["Resolution Z"])

    n_v2m = nodes.new("GeometryNodeVolumeToMesh")
    n_v2m.location = (1020, 140)
    _set_node_enum(n_v2m, "resolution_mode", "Resolution Mode", "GRID")
    n_v2m.inputs["Threshold"].default_value = 0.0
    n_v2m.inputs["Adaptivity"].default_value = 0.08
    links.new(n_vol.outputs["Volume"], n_v2m.inputs["Volume"])

    # 6. Clean bypass when Inward Ratio == 0 or Carve Amount == 0
    n_ratio_on = nodes.new("FunctionNodeCompare")
    n_ratio_on.location = (760, -240)
    n_ratio_on.data_type = "FLOAT"
    n_ratio_on.operation = "GREATER_THAN"
    n_ratio_on.inputs["B"].default_value = 0.001
    links.new(n_in.outputs["Inward Ratio"], n_ratio_on.inputs["A"])

    n_carve_on = nodes.new("FunctionNodeCompare")
    n_carve_on.location = (760, -400)
    n_carve_on.data_type = "FLOAT"
    n_carve_on.operation = "GREATER_THAN"
    n_carve_on.inputs["B"].default_value = 0.001
    links.new(n_in.outputs["Carve Amount"], n_carve_on.inputs["A"])

    n_active = nodes.new("FunctionNodeBooleanMath")
    n_active.location = (1020, -300)
    n_active.operation = "AND"
    links.new(n_ratio_on.outputs["Result"], n_active.inputs[0])
    links.new(n_carve_on.outputs["Result"], n_active.inputs[1])

    n_sw = nodes.new("GeometryNodeSwitch")
    n_sw.location = (1280, 140)
    n_sw.input_type = "GEOMETRY"
    links.new(n_active.outputs["Boolean"], n_sw.inputs["Switch"])
    links.new(n_in.outputs["Geometry"], n_sw.inputs["False"])
    links.new(n_v2m.outputs["Mesh"], n_sw.inputs["True"])

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (1520, 140)
    links.new(n_sw.outputs["Output"], n_out.inputs["Geometry"])

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
