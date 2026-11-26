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

ROUGH_CUTTER_VERSION = 10

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
    """Create or return FDT_GN_RoughCutter (v10) — Angular Concrete Cleavage Cutter.

    Chisels the cutter's outer edges, corners, and perimeter using randomly tilted planar cleavage
    slabs and bold obtuse concrete chunk notches in a single Exact Mesh Boolean (DIFFERENCE) pass:
    - Produces crisp, flat, low-poly broken concrete facets (like real fractured concrete slabs).
    - Strictly inward-only (mathematically cannot bulge outside the original cutter boundary).
    - Zero wavy gelatin blobs, zero Marching-Cubes slivers, and zero needle spikes.
    - Uniform `Inward Depth` scaling preserves the exact fractured silhouette while shrinking inward.
    """
    existing = bpy.data.node_groups.get(GN_ROUGH_CUTTER)
    if existing is not None:
        if int(existing.get("fdt_gn_version", 1)) >= ROUGH_CUTTER_VERSION:
            return existing
        # Upgrade legacy v1-v9 node group in-place so existing modifier links stay intact
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
    s_scale.default_value = 2.5
    if hasattr(s_scale, "min_value"):
        s_scale.min_value = 0.5
        s_scale.max_value = 10.0

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
    n_in.location = (-1800, 0)

    # 1. Bounding Box -> center, size, diagonal length
    n_bbox = nodes.new("GeometryNodeBoundBox")
    n_bbox.location = (-1520, 460)
    links.new(n_in.outputs["Geometry"], n_bbox.inputs["Geometry"])

    n_center_sum = nodes.new("ShaderNodeVectorMath")
    n_center_sum.location = (-1300, 620)
    n_center_sum.operation = "ADD"
    links.new(n_bbox.outputs["Min"], n_center_sum.inputs[0])
    links.new(n_bbox.outputs["Max"], n_center_sum.inputs[1])

    n_center = nodes.new("ShaderNodeVectorMath")
    n_center.location = (-1100, 620)
    n_center.operation = "SCALE"
    n_center.inputs["Scale"].default_value = 0.5
    links.new(n_center_sum.outputs["Vector"], n_center.inputs[0])

    n_size = nodes.new("ShaderNodeVectorMath")
    n_size.location = (-1300, 420)
    n_size.operation = "SUBTRACT"
    links.new(n_bbox.outputs["Max"], n_size.inputs[0])
    links.new(n_bbox.outputs["Min"], n_size.inputs[1])

    n_diag = nodes.new("ShaderNodeVectorMath")
    n_diag.location = (-1100, 380)
    n_diag.operation = "LENGTH"
    links.new(n_size.outputs["Vector"], n_diag.inputs[0])

    n_diag_safe = nodes.new("ShaderNodeMath")
    n_diag_safe.location = (-900, 380)
    n_diag_safe.operation = "MAXIMUM"
    n_diag_safe.inputs[1].default_value = 0.1
    links.new(n_diag.outputs["Value"], n_diag_safe.inputs[0])

    # 2. Subdivide cutter lightly (Level 1 for Detail=1, Level 2 for Detail=2..3) to provide anchor points
    n_sub_lvl = nodes.new("ShaderNodeMath")
    n_sub_lvl.location = (-1520, 140)
    n_sub_lvl.operation = "MINIMUM"
    n_sub_lvl.inputs[1].default_value = 2.0
    links.new(n_in.outputs["Detail"], n_sub_lvl.inputs[0])

    n_subdiv = nodes.new("GeometryNodeSubdivideMesh")
    n_subdiv.location = (-1300, 140)
    links.new(n_in.outputs["Geometry"], n_subdiv.inputs["Mesh"])
    links.new(n_sub_lvl.outputs["Value"], n_subdiv.inputs["Level"])

    # Compute exact selection probability = target_cuts / max(point_count, 1)
    # Detail 1 -> ~12 cuts, Detail 2 -> ~20 cuts, Detail 3 -> ~28 cuts (safe on any custom mesh!)
    n_dom = nodes.new("GeometryNodeAttributeDomainSize")
    n_dom.location = (-1100, 220)
    _set_node_enum(n_dom, "component", "Component", "MESH")
    links.new(n_subdiv.outputs["Mesh"], n_dom.inputs["Geometry"])

    n_pt_safe = nodes.new("ShaderNodeMath")
    n_pt_safe.location = (-900, 220)
    n_pt_safe.operation = "MAXIMUM"
    n_pt_safe.inputs[1].default_value = 1.0
    links.new(n_dom.outputs["Point Count"], n_pt_safe.inputs[0])

    n_target_cuts = nodes.new("ShaderNodeMath")
    n_target_cuts.location = (-1100, 40)
    n_target_cuts.operation = "MULTIPLY_ADD"
    n_target_cuts.inputs[1].default_value = 8.0
    n_target_cuts.inputs[2].default_value = 4.0
    links.new(n_in.outputs["Detail"], n_target_cuts.inputs[0])

    n_keep_prob = nodes.new("ShaderNodeMath")
    n_keep_prob.location = (-700, 140)
    n_keep_prob.operation = "DIVIDE"
    links.new(n_target_cuts.outputs["Value"], n_keep_prob.inputs[0])
    links.new(n_pt_safe.outputs["Value"], n_keep_prob.inputs[1])

    # 3. Deterministic per-vertex random values (using 4D White Noise seeded by Index + Seed)
    n_idx = nodes.new("GeometryNodeInputIndex")
    n_idx.location = (-1520, -260)

    n_seed_v1 = nodes.new("ShaderNodeCombineXYZ")
    n_seed_v1.location = (-1300, -220)
    n_seed_v1.inputs["Y"].default_value = 1.73
    n_seed_v1.inputs["Z"].default_value = 0.0
    links.new(n_idx.outputs["Index"], n_seed_v1.inputs["X"])

    n_wn1 = nodes.new("ShaderNodeTexWhiteNoise")
    n_wn1.location = (-1100, -220)
    n_wn1.noise_dimensions = "4D"
    links.new(n_seed_v1.outputs["Vector"], n_wn1.inputs["Vector"])
    links.new(n_in.outputs["Seed"], n_wn1.inputs["W"])

    n_seed_v2 = nodes.new("ShaderNodeCombineXYZ")
    n_seed_v2.location = (-1300, -420)
    n_seed_v2.inputs["Y"].default_value = 13.37
    n_seed_v2.inputs["Z"].default_value = 7.11
    links.new(n_idx.outputs["Index"], n_seed_v2.inputs["X"])

    n_wn2 = nodes.new("ShaderNodeTexWhiteNoise")
    n_wn2.location = (-1100, -420)
    n_wn2.noise_dimensions = "4D"
    links.new(n_seed_v2.outputs["Vector"], n_wn2.inputs["Vector"])
    links.new(n_in.outputs["Seed"], n_wn2.inputs["W"])

    n_sep_r2 = nodes.new("ShaderNodeSeparateXYZ")
    n_sep_r2.location = (-880, -440)
    links.new(n_wn2.outputs["Color"], n_sep_r2.inputs["Vector"])

    # Selection: keep_prob > r_select
    n_sel = nodes.new("FunctionNodeCompare")
    n_sel.location = (-480, 120)
    n_sel.data_type = "FLOAT"
    n_sel.operation = "LESS_THAN"
    links.new(n_wn1.outputs["Value"], n_sel.inputs["A"])
    links.new(n_keep_prob.outputs["Value"], n_sel.inputs["B"])

    # 4. Tilted cleavage plane normal: N_cut = normalize(Normal + (wn1.Color - 0.5) * 1.1)
    n_tilt_raw = nodes.new("ShaderNodeVectorMath")
    n_tilt_raw.location = (-880, -200)
    n_tilt_raw.operation = "SUBTRACT"
    n_tilt_raw.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(n_wn1.outputs["Color"], n_tilt_raw.inputs[0])

    n_tilt = nodes.new("ShaderNodeVectorMath")
    n_tilt.location = (-680, -200)
    n_tilt.operation = "SCALE"
    n_tilt.inputs["Scale"].default_value = 1.1
    links.new(n_tilt_raw.outputs["Vector"], n_tilt.inputs[0])

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (-880, -60)

    n_ncut_sum = nodes.new("ShaderNodeVectorMath")
    n_ncut_sum.location = (-480, -120)
    n_ncut_sum.operation = "ADD"
    links.new(n_norm.outputs["Normal"], n_ncut_sum.inputs[0])
    links.new(n_tilt.outputs["Vector"], n_ncut_sum.inputs[1])

    n_ncut = nodes.new("ShaderNodeVectorMath")
    n_ncut.location = (-280, -120)
    n_ncut.operation = "NORMALIZE"
    links.new(n_ncut_sum.outputs["Vector"], n_ncut.inputs[0])

    # 5. Anisotropic safe bite depth along N_cut:
    # extent_along_N = dot(abs(N_cut), size); bite_depth = 0.28 * extent_along_N * Jaggedness * (0.35 + 0.65 * r_depth)
    n_abs_n = nodes.new("ShaderNodeVectorMath")
    n_abs_n.location = (-80, -260)
    n_abs_n.operation = "ABSOLUTE"
    links.new(n_ncut.outputs["Vector"], n_abs_n.inputs[0])

    n_ext = nodes.new("ShaderNodeVectorMath")
    n_ext.location = (120, -260)
    n_ext.operation = "DOT_PRODUCT"
    links.new(n_abs_n.outputs["Vector"], n_ext.inputs[0])
    links.new(n_size.outputs["Vector"], n_ext.inputs[1])

    n_max_bite = nodes.new("ShaderNodeMath")
    n_max_bite.location = (320, -260)
    n_max_bite.operation = "MULTIPLY"
    n_max_bite.inputs[1].default_value = 0.28
    links.new(n_ext.outputs["Value"], n_max_bite.inputs[0])

    n_bite_jag = nodes.new("ShaderNodeMath")
    n_bite_jag.location = (500, -260)
    n_bite_jag.operation = "MULTIPLY"
    links.new(n_max_bite.outputs["Value"], n_bite_jag.inputs[0])
    links.new(n_in.outputs["Jaggedness"], n_bite_jag.inputs[1])

    n_rand_depth = nodes.new("ShaderNodeMath")
    n_rand_depth.location = (320, -440)
    n_rand_depth.operation = "MULTIPLY_ADD"
    n_rand_depth.inputs[1].default_value = 0.65
    n_rand_depth.inputs[2].default_value = 0.35
    links.new(n_wn2.outputs["Value"], n_rand_depth.inputs[0])

    n_bite_depth = nodes.new("ShaderNodeMath")
    n_bite_depth.location = (700, -320)
    n_bite_depth.operation = "MULTIPLY"
    links.new(n_bite_jag.outputs["Value"], n_bite_depth.inputs[0])
    links.new(n_rand_depth.outputs["Value"], n_bite_depth.inputs[1])

    n_neg_bite = nodes.new("ShaderNodeMath")
    n_neg_bite.location = (880, -320)
    n_neg_bite.operation = "MULTIPLY"
    n_neg_bite.inputs[1].default_value = -1.0
    links.new(n_bite_depth.outputs["Value"], n_neg_bite.inputs[0])

    n_cleave_off = nodes.new("ShaderNodeVectorMath")
    n_cleave_off.location = (1060, -240)
    n_cleave_off.operation = "SCALE"
    links.new(n_ncut.outputs["Vector"], n_cleave_off.inputs[0])
    links.new(n_neg_bite.outputs["Value"], n_cleave_off.inputs["Scale"])

    # Store per-point inward cleavage offset & random Z spin before instancing
    n_store_off = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store_off.location = (-280, 200)
    n_store_off.data_type = "FLOAT_VECTOR"
    n_store_off.domain = "POINT"
    n_store_off.inputs["Name"].default_value = "_fdt_cleave_off"
    links.new(n_subdiv.outputs["Mesh"], n_store_off.inputs["Geometry"])
    links.new(n_cleave_off.outputs["Vector"], _get_socket(n_store_off.inputs, "Value", index=3))

    n_spin_rad = nodes.new("ShaderNodeMath")
    n_spin_rad.location = (-680, -520)
    n_spin_rad.operation = "MULTIPLY"
    n_spin_rad.inputs[1].default_value = 6.283185
    links.new(n_sep_r2.outputs["Z"], n_spin_rad.inputs[0])

    n_spin_vec = nodes.new("ShaderNodeCombineXYZ")
    n_spin_vec.location = (-480, -520)
    links.new(n_spin_rad.outputs["Value"], n_spin_vec.inputs["Z"])

    n_store_spin = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store_spin.location = (-60, 200)
    n_store_spin.data_type = "FLOAT_VECTOR"
    n_store_spin.domain = "POINT"
    n_store_spin.inputs["Name"].default_value = "_fdt_cleave_spin"
    links.new(n_store_off.outputs["Geometry"], n_store_spin.inputs["Geometry"])
    links.new(n_spin_vec.outputs["Vector"], _get_socket(n_store_spin.inputs, "Value", index=3))

    # 6. Prototype Cleavage Block: unit cube shifted by +0.5 Z so bottom face sits at z=0 and block extends +Z
    n_cube = nodes.new("GeometryNodeMeshCube")
    n_cube.location = (-280, 560)
    n_cube.inputs["Size"].default_value = (1.0, 1.0, 1.0)

    n_cube_shift = nodes.new("GeometryNodeSetPosition")
    n_cube_shift.location = (-60, 560)
    n_cube_shift.inputs["Offset"].default_value = (0.0, 0.0, 0.5)
    links.new(n_cube.outputs["Mesh"], n_cube_shift.inputs["Geometry"])

    # Instance scale: chunk_base = diag / max(Noise Scale * 0.55, 0.5)
    # Mix wide planar slabs (sx,sy ~ 1.3x) and bold angular notches (sx,sy ~ 0.65x) naturally
    n_scl_den = nodes.new("ShaderNodeMath")
    n_scl_den.location = (-680, 380)
    n_scl_den.operation = "MULTIPLY"
    n_scl_den.inputs[1].default_value = 0.55
    links.new(n_in.outputs["Noise Scale"], n_scl_den.inputs[0])

    n_scl_den_safe = nodes.new("ShaderNodeMath")
    n_scl_den_safe.location = (-480, 380)
    n_scl_den_safe.operation = "MAXIMUM"
    n_scl_den_safe.inputs[1].default_value = 0.5
    links.new(n_scl_den.outputs["Value"], n_scl_den_safe.inputs[0])

    n_chunk_base = nodes.new("ShaderNodeMath")
    n_chunk_base.location = (-280, 380)
    n_chunk_base.operation = "DIVIDE"
    links.new(n_diag_safe.outputs["Value"], n_chunk_base.inputs[0])
    links.new(n_scl_den_safe.outputs["Value"], n_chunk_base.inputs[1])

    n_rx_fac = nodes.new("ShaderNodeMath")
    n_rx_fac.location = (-480, -340)
    n_rx_fac.operation = "MULTIPLY_ADD"
    n_rx_fac.inputs[1].default_value = 0.85
    n_rx_fac.inputs[2].default_value = 0.55
    links.new(n_sep_r2.outputs["X"], n_rx_fac.inputs[0])

    n_ry_fac = nodes.new("ShaderNodeMath")
    n_ry_fac.location = (-480, -430)
    n_ry_fac.operation = "MULTIPLY_ADD"
    n_ry_fac.inputs[1].default_value = 0.85
    n_ry_fac.inputs[2].default_value = 0.55
    links.new(n_sep_r2.outputs["Y"], n_ry_fac.inputs[0])

    n_sx = nodes.new("ShaderNodeMath")
    n_sx.location = (-60, 380)
    n_sx.operation = "MULTIPLY"
    links.new(n_chunk_base.outputs["Value"], n_sx.inputs[0])
    links.new(n_rx_fac.outputs["Value"], n_sx.inputs[1])

    n_sy = nodes.new("ShaderNodeMath")
    n_sy.location = (-60, 290)
    n_sy.operation = "MULTIPLY"
    links.new(n_chunk_base.outputs["Value"], n_sy.inputs[0])
    links.new(n_ry_fac.outputs["Value"], n_sy.inputs[1])

    n_sz = nodes.new("ShaderNodeMath")
    n_sz.location = (-60, 470)
    n_sz.operation = "MULTIPLY"
    n_sz.inputs[1].default_value = 0.8
    links.new(n_diag_safe.outputs["Value"], n_sz.inputs[0])

    n_inst_scale = nodes.new("ShaderNodeCombineXYZ")
    n_inst_scale.location = (160, 380)
    links.new(n_sx.outputs["Value"], n_inst_scale.inputs["X"])
    links.new(n_sy.outputs["Value"], n_inst_scale.inputs["Y"])
    links.new(n_sz.outputs["Value"], n_inst_scale.inputs["Z"])

    # Align local +Z of each cleavage cube to N_cut
    n_align = nodes.new("FunctionNodeAlignRotationToVector")
    n_align.location = (160, 80)
    _set_node_enum(n_align, "axis", "Axis", "Z")
    _set_node_enum(n_align, "pivot_axis", "Pivot Axis", "AUTO")
    links.new(n_ncut.outputs["Vector"], n_align.inputs["Vector"])

    # 7. Instance cleavage blocks -> Spin around local Z -> Translate inward by exact offset -> Realize
    n_inst = nodes.new("GeometryNodeInstanceOnPoints")
    n_inst.location = (420, 220)
    links.new(n_store_spin.outputs["Geometry"], n_inst.inputs["Points"])
    links.new(n_sel.outputs["Result"], n_inst.inputs["Selection"])
    links.new(n_cube_shift.outputs["Geometry"], n_inst.inputs["Instance"])
    links.new(n_align.outputs["Rotation"], n_inst.inputs["Rotation"])
    links.new(n_inst_scale.outputs["Vector"], n_inst.inputs["Scale"])

    n_read_spin = nodes.new("GeometryNodeInputNamedAttribute")
    n_read_spin.location = (420, -40)
    n_read_spin.data_type = "FLOAT_VECTOR"
    n_read_spin.inputs["Name"].default_value = "_fdt_cleave_spin"

    n_rot_inst = nodes.new("GeometryNodeRotateInstances")
    n_rot_inst.location = (660, 220)
    n_rot_inst.inputs["Local Space"].default_value = True
    links.new(n_inst.outputs["Instances"], n_rot_inst.inputs["Instances"])
    links.new(_get_socket(n_read_spin.outputs, "Attribute", index=0), n_rot_inst.inputs["Rotation"])

    n_read_off = nodes.new("GeometryNodeInputNamedAttribute")
    n_read_off.location = (660, -40)
    n_read_off.data_type = "FLOAT_VECTOR"
    n_read_off.inputs["Name"].default_value = "_fdt_cleave_off"

    n_trans_inst = nodes.new("GeometryNodeTranslateInstances")
    n_trans_inst.location = (900, 220)
    n_trans_inst.inputs["Local Space"].default_value = False
    links.new(n_rot_inst.outputs["Instances"], n_trans_inst.inputs["Instances"])
    links.new(_get_socket(n_read_off.outputs, "Attribute", index=0), n_trans_inst.inputs["Translation"])

    n_realize = nodes.new("GeometryNodeRealizeInstances")
    n_realize.location = (1140, 220)
    links.new(n_trans_inst.outputs["Instances"], n_realize.inputs["Geometry"])

    # 8. Single-pass Exact Mesh Boolean DIFFERENCE (Cutter - All Cleavage Slabs)
    n_bool = nodes.new("GeometryNodeMeshBoolean")
    n_bool.location = (1380, 260)
    _set_node_enum(n_bool, "operation", "Operation", "DIFFERENCE")
    _set_node_enum(n_bool, "solver", "Solver", "EXACT")
    s_self = n_bool.inputs.get("Self Intersection")
    if s_self is not None and hasattr(s_self, "default_value"):
        s_self.default_value = True
    links.new(n_in.outputs["Geometry"], _get_socket(n_bool.inputs, "Mesh 1", index=0))
    links.new(n_realize.outputs["Geometry"], _get_socket(n_bool.inputs, "Mesh 2", index=1))

    # Bypass Boolean cleanly when Jaggedness == 0.0
    n_jag_on = nodes.new("FunctionNodeCompare")
    n_jag_on.location = (1380, 0)
    n_jag_on.data_type = "FLOAT"
    n_jag_on.operation = "GREATER_THAN"
    n_jag_on.inputs["B"].default_value = 0.001
    links.new(n_in.outputs["Jaggedness"], n_jag_on.inputs["A"])

    n_sw_jag = nodes.new("GeometryNodeSwitch")
    n_sw_jag.location = (1620, 220)
    n_sw_jag.input_type = "GEOMETRY"
    links.new(n_jag_on.outputs["Result"], n_sw_jag.inputs["Switch"])
    links.new(n_in.outputs["Geometry"], n_sw_jag.inputs["False"])
    links.new(n_bool.outputs["Mesh"], n_sw_jag.inputs["True"])

    # 9. Inward Depth: uniform scale around bbox center (preserves exact angular cleavage silhouette!)
    n_inward_scale = nodes.new("ShaderNodeMath")
    n_inward_scale.location = (1380, 520)
    n_inward_scale.operation = "MULTIPLY_ADD"
    n_inward_scale.inputs[1].default_value = -0.85
    n_inward_scale.inputs[2].default_value = 1.0
    links.new(n_in.outputs["Inward Depth"], n_inward_scale.inputs[0])

    n_pos2 = nodes.new("GeometryNodeInputPosition")
    n_pos2.location = (1380, 680)

    n_rel = nodes.new("ShaderNodeVectorMath")
    n_rel.location = (1580, 620)
    n_rel.operation = "SUBTRACT"
    links.new(n_pos2.outputs["Position"], n_rel.inputs[0])
    links.new(n_center.outputs["Vector"], n_rel.inputs[1])

    n_rel_scaled = nodes.new("ShaderNodeVectorMath")
    n_rel_scaled.location = (1760, 580)
    n_rel_scaled.operation = "SCALE"
    links.new(n_rel.outputs["Vector"], n_rel_scaled.inputs[0])
    links.new(n_inward_scale.outputs["Value"], n_rel_scaled.inputs["Scale"])

    n_pos_final = nodes.new("ShaderNodeVectorMath")
    n_pos_final.location = (1940, 580)
    n_pos_final.operation = "ADD"
    links.new(n_rel_scaled.outputs["Vector"], n_pos_final.inputs[0])
    links.new(n_center.outputs["Vector"], n_pos_final.inputs[1])

    n_set_scale = nodes.new("GeometryNodeSetPosition")
    n_set_scale.location = (1940, 220)
    links.new(n_sw_jag.outputs["Output"], n_set_scale.inputs["Geometry"])
    links.new(n_pos_final.outputs["Vector"], n_set_scale.inputs["Position"])

    # 10. Crisp flat concrete facet shading
    n_flat = nodes.new("GeometryNodeSetShadeSmooth")
    n_flat.location = (2180, 220)
    n_flat.inputs["Shade Smooth"].default_value = False
    links.new(n_set_scale.outputs["Geometry"], n_flat.inputs["Geometry"])

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (2400, 220)
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
