"""3D Gear Icon Gizmo on selected Cutter -> Wheel Menu -> Interactive 3D Axis Picker."""

from __future__ import annotations

import blf
import bmesh
import bpy
import gpu
from bpy.props import EnumProperty, StringProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from ..core import apple_bite, zone_registry
from ..core.apple_bite import CUTTER_PROP
from ..core.session import has_active_session, load_session

_AXIS_DEFS: list[tuple[str, Vector, tuple[float, float, float, float]]] = [
    ("+X", Vector((1.0, 0.0, 0.0)), (0.95, 0.25, 0.25, 1.0)),
    ("-X", Vector((-1.0, 0.0, 0.0)), (0.75, 0.20, 0.20, 0.9)),
    ("+Y", Vector((0.0, 1.0, 0.0)), (0.25, 0.90, 0.35, 1.0)),
    ("-Y", Vector((0.0, -1.0, 0.0)), (0.20, 0.70, 0.28, 0.9)),
    ("+Z", Vector((0.0, 0.0, 1.0)), (0.25, 0.58, 1.00, 1.0)),
    ("-Z", Vector((0.0, 0.0, -1.0)), (0.20, 0.45, 0.80, 0.9)),
]

_active_axis_picker: "FDT_OT_applebite_pick_axis | None" = None
_active_layer_picker: "FDT_OT_applebite_pick_layer_target | None" = None


def _compute_axis_segments(cutter: bpy.types.Object) -> list[tuple[str, Vector, Vector, tuple[float, float, float, float]]]:
    """Return list of (dir_name, start_world, tip_world, rgba) for the 6 local axes of cutter."""
    center = cutter.matrix_world.translation.copy()
    rot_mat = cutter.matrix_world.to_3x3().normalized()
    dims = cutter.dimensions
    segments = []
    for idx, (dname, local_dir, color) in enumerate(_AXIS_DEFS):
        axis_i = idx // 2
        half_ext = max(float(dims[axis_i]) * 0.5, 0.25)
        arrow_len = max(half_ext * 0.65, 0.45)
        w_dir = rot_mat @ local_dir
        start = center + w_dir * (half_ext * 0.55)
        tip = center + w_dir * (half_ext + arrow_len)
        segments.append((dname, start, tip, color))
    return segments


class FDT_MT_cutter_wheel(bpy.types.Menu):
    bl_idname = "FDT_MT_cutter_wheel"
    bl_label = "Cutter Actions"

    def draw(self, context):
        pie = self.layout.menu_pie()
        obj = context.active_object
        cname = obj.name if (obj is not None and bool(obj.get(CUTTER_PROP, False))) else ""

        pie.operator("fdt.applebite_pick_axis", text="Next Box (3D Axis)", icon="ORIENTATION_LOCAL")
        op_z = pie.operator("fdt.applebite_cycle_zone", text="Change Zone", icon="COLOR")
        op_z.cutter_name = cname
        op_del = pie.operator("fdt.applebite_delete_cutter", text="Delete Box", icon="X")
        op_del.cutter_name = cname
        op_m = pie.operator("fdt.applebite_toggle_mute", text="Mute / Unmute", icon="HIDE_OFF")
        op_m.cutter_name = cname
        pie.operator("fdt.applebite_add_box", text="New Box", icon="MESH_CUBE")
        op_draw = pie.operator("fdt.applebite_draw_sculpt_cutter", text="Draw Cut", icon="GREASEPENCIL")
        op_draw.cutter_name = cname
        pie.operator("fdt.applebite_apply", text="Apply", icon="CHECKMARK")
        settings = getattr(context.scene, "fdt_settings", None)
        view_order = ("ALL", "PIECES", "MAIN", "MAIN_WIRE")
        view_labels = {
            "ALL": "All",
            "PIECES": "Piece Only",
            "MAIN": "Hole Only",
            "MAIN_WIRE": "Hole + Wire",
        }
        cur_view = str(getattr(settings, "preview_display_mode", "ALL") or "ALL") if settings is not None else "ALL"
        if cur_view not in view_order:
            cur_view = "ALL"
        next_view = view_order[(view_order.index(cur_view) + 1) % len(view_order)]
        op_view = pie.operator(
            "fdt.applebite_set_view_mode",
            text=f"View: {view_labels[next_view]}",
            icon="HIDE_OFF",
        )
        op_view.mode = next_view


class FDT_OT_applebite_pick_axis(bpy.types.Operator):
    bl_idname = "fdt.applebite_pick_axis"
    bl_label = "Pick 3D Axis for Next Box"
    bl_description = "Show 3D directional axes on the selected cutter; click any axis to spawn the next flush zone box"

    _handle_3d = None
    _handle_2d = None
    _cutter_name: str = ""
    _hovered_dir: str | None = None

    def _cleanup(self, context: bpy.types.Context | None = None) -> None:
        global _active_axis_picker
        if self._handle_3d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, "WINDOW")
            except Exception:
                pass
            self._handle_3d = None
        if self._handle_2d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, "WINDOW")
            except Exception:
                pass
            self._handle_2d = None
        if _active_axis_picker is self:
            _active_axis_picker = None
        if context is not None and context.area is not None:
            context.area.tag_redraw()

    def _draw_3d(self) -> None:
        cutter = bpy.data.objects.get(self._cutter_name)
        if cutter is None:
            return
        segments = _compute_axis_segments(cutter)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")

        for dname, start, tip, base_col in segments:
            is_hover = dname == self._hovered_dir
            col = (1.0, 0.88, 0.20, 1.0) if is_hover else base_col
            gpu.state.line_width_set(5.5 if is_hover else 3.0)

            # Main axis shaft
            batch = batch_for_shader(shader, "LINES", {"pos": [start, tip]})
            shader.bind()
            shader.uniform_float("color", col)
            batch.draw(shader)

            # Tip point handle
            gpu.state.point_size_set(14.0 if is_hover else 9.0)
            pt_batch = batch_for_shader(shader, "POINTS", {"pos": [tip]})
            shader.bind()
            shader.uniform_float("color", col)
            pt_batch.draw(shader)

        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("NONE")

    def _draw_2d(self) -> None:
        context = bpy.context
        cutter = bpy.data.objects.get(self._cutter_name)
        if cutter is None:
            return
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return

        font_id = 0
        # Axis labels at tips
        for dname, _start, tip, base_col in _compute_axis_segments(cutter):
            pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, tip)
            if pos_2d is None:
                continue
            is_hover = dname == self._hovered_dir
            blf.size(font_id, 17 if is_hover else 14)
            if is_hover:
                blf.color(font_id, 1.0, 0.90, 0.25, 1.0)
            else:
                blf.color(font_id, base_col[0], base_col[1], base_col[2], 1.0)
            blf.position(font_id, pos_2d.x + 10, pos_2d.y - 5, 0)
            blf.draw(font_id, dname)

        # Bottom guide banner
        blf.size(font_id, 15)
        blf.color(font_id, 1.0, 0.90, 0.30, 1.0)
        if self._hovered_dir:
            msg = f"[Next Box] Left-click to place next zone along {self._hovered_dir}  (RMB / ESC Cancel)"
        else:
            msg = "[Next Box] Hover and left-click a 3D axis (+X, -X, +Y, -Y, +Z, -Z)  (ESC Cancel)"
        blf.position(font_id, 24, 42, 0)
        blf.draw(font_id, msg)

    def _hit_test_axis(self, context: bpy.types.Context, mouse_x: float, mouse_y: float) -> str | None:
        cutter = bpy.data.objects.get(self._cutter_name)
        if cutter is None:
            return None
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None

        mouse = Vector((mouse_x, mouse_y))
        best_dir = None
        best_dist = 44.0  # pixel hit radius

        for dname, start, tip, _col in _compute_axis_segments(cutter):
            p_tip = view3d_utils.location_3d_to_region_2d(region, rv3d, tip)
            p_mid = view3d_utils.location_3d_to_region_2d(region, rv3d, (start + tip) * 0.5)
            for p in (p_tip, p_mid):
                if p is None:
                    continue
                d = (Vector((p.x, p.y)) - mouse).length
                if d < best_dist:
                    best_dist = d
                    best_dir = dname
        return best_dir

    def modal(self, context, event):
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}

        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            new_hover = self._hit_test_axis(context, event.mouse_region_x, event.mouse_region_y)
            if new_hover != self._hovered_dir:
                self._hovered_dir = new_hover
                if context.area:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            picked = self._hit_test_axis(context, event.mouse_region_x, event.mouse_region_y)
            if picked is not None:
                self._cleanup(context)
                ok, msg, _ = apple_bite.add_adjacent_box_cutter(
                    context,
                    direction=picked,
                    auto_next_zone=True,
                )
                settings = getattr(context.scene, "fdt_settings", None)
                if settings is not None:
                    settings.last_status = msg
                self.report({"INFO"} if ok else {"WARNING"}, msg)
                return {"FINISHED"} if ok else {"CANCELLED"}
            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        global _active_axis_picker
        if _active_axis_picker is not None:
            _active_axis_picker._cleanup(context)

        obj = context.active_object
        if obj is None or not bool(obj.get(CUTTER_PROP, False)):
            self.report({"WARNING"}, "Select a cutter box first.")
            return {"CANCELLED"}

        self._cutter_name = obj.name
        self._hovered_dir = None
        _active_axis_picker = self

        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_3d, (), "WINDOW", "POST_VIEW"
        )
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_2d, (), "WINDOW", "POST_PIXEL"
        )
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


def _cutter_world_geom(obj: bpy.types.Object) -> tuple[Vector, list[Vector], list[Vector]]:
    """Return `(center_world, edge_endpoints_world, tri_verts_world)` for a cutter (Object or Edit Mode)."""
    mw = obj.matrix_world
    me = obj.data
    edge_pts: list[Vector] = []
    tri_pts: list[Vector] = []

    if obj.mode == "EDIT" and me is not None:
        try:
            bm = bmesh.from_edit_mesh(me)
            if len(bm.verts) > 0:
                center = sum((mw @ v.co for v in bm.verts), Vector((0.0, 0.0, 0.0))) / len(bm.verts)
                for e in bm.edges:
                    if e.is_valid:
                        edge_pts.append(mw @ e.verts[0].co)
                        edge_pts.append(mw @ e.verts[1].co)
                for f in bm.faces:
                    if f.is_valid and len(f.verts) >= 3:
                        wv = [mw @ v.co for v in f.verts]
                        for k in range(1, len(wv) - 1):
                            tri_pts.extend([wv[0], wv[k], wv[k + 1]])
                return center, edge_pts, tri_pts
        except Exception:
            pass

    corners = [mw @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector((0.0, 0.0, 0.0))) / max(1, len(corners))
    if me is not None and 0 < len(me.edges) <= 256:
        wverts = [mw @ v.co for v in me.vertices]
        for e in me.edges:
            edge_pts.append(wverts[e.vertices[0]])
            edge_pts.append(wverts[e.vertices[1]])
        for p in me.polygons:
            vids = p.vertices
            if len(vids) >= 3:
                for k in range(1, len(vids) - 1):
                    tri_pts.extend([wverts[vids[0]], wverts[vids[k]], wverts[vids[k + 1]]])
    else:
        bb_edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for a, b in bb_edges:
            edge_pts.append(corners[a])
            edge_pts.append(corners[b])
        bb_quads = (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (1, 2, 6, 5), (3, 0, 4, 7),
        )
        for q0, q1, q2, q3 in bb_quads:
            tri_pts.extend([corners[q0], corners[q1], corners[q2], corners[q0], corners[q2], corners[q3]])

    return center, edge_pts, tri_pts


def _draw_2d_pill_badge(
    shader,
    font_id: int,
    text: str,
    x: float,
    y: float,
    font_size: int,
    bg_col: tuple[float, float, float, float],
    rim_col: tuple[float, float, float, float],
    txt_col: tuple[float, float, float, float],
) -> None:
    """Draw a crisp readable 2D rounded-box label centered at (x, y)."""
    blf.size(font_id, font_size)
    tw, th = blf.dimensions(font_id, text)
    pad_x = 10.0
    pad_y = 6.0
    x0, x1 = x - tw * 0.5 - pad_x, x + tw * 0.5 + pad_x
    y0, y1 = y - th * 0.5 - pad_y, y + th * 0.5 + pad_y

    gpu.state.blend_set("ALPHA")
    tris = [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
    b_bg = batch_for_shader(shader, "TRIS", {"pos": tris})
    shader.bind()
    shader.uniform_float("color", bg_col)
    b_bg.draw(shader)

    gpu.state.line_width_set(1.8)
    b_rim = batch_for_shader(shader, "LINE_LOOP", {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]})
    shader.bind()
    shader.uniform_float("color", rim_col)
    b_rim.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")

    blf.color(font_id, *txt_col)
    blf.position(font_id, x - tw * 0.5, y - th * 0.35, 0)
    blf.draw(font_id, text)


def _ray_intersect_local_aabb(
    origin: Vector,
    direction: Vector,
    bound_box,
) -> tuple[bool, Vector | None]:
    """Exact slab ray-AABB intersection in object-local space (fallback when obj.ray_cast fails)."""
    if not bound_box:
        return False, None
    xs = [c[0] for c in bound_box]
    ys = [c[1] for c in bound_box]
    zs = [c[2] for c in bound_box]
    bmin = (min(xs), min(ys), min(zs))
    bmax = (max(xs), max(ys), max(zs))
    tmin = 0.0
    tmax = float("inf")
    for i in range(3):
        o = origin[i]
        d = direction[i]
        if abs(d) < 1e-8:
            if o < bmin[i] or o > bmax[i]:
                return False, None
        else:
            t1 = (bmin[i] - o) / d
            t2 = (bmax[i] - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False, None
    return True, origin + direction * tmin


class FDT_OT_applebite_pick_layer_target(bpy.types.Operator):
    """Interactive 3D Cutter Picker: click any other cutter box to place the source cutter ABOVE or BELOW it."""

    bl_idname = "fdt.applebite_pick_layer_target"
    bl_label = "Pick Target Box for Layer Order"
    bl_description = "Hover and left-click another cutter box in the 3D View to place this box above or below it"

    source_cutter: StringProperty(name="Source Cutter", default="")
    position: EnumProperty(
        name="Position",
        items=[
            ("ABOVE", "Above", "Place source cutter directly above the target cutter"),
            ("BELOW", "Below", "Place source cutter directly below the target cutter"),
        ],
        default="ABOVE",
    )

    _handle_3d = None
    _handle_2d = None
    _hovered_target: str | None = None

    def _cleanup(self, context: bpy.types.Context | None = None) -> None:
        global _active_layer_picker
        if self._handle_3d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, "WINDOW")
            except Exception:
                pass
            self._handle_3d = None
        if self._handle_2d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, "WINDOW")
            except Exception:
                pass
            self._handle_2d = None
        if _active_layer_picker is self:
            _active_layer_picker = None
        if context is not None and context.area is not None:
            context.area.tag_redraw()

    def _iter_session_cutters(self, context: bpy.types.Context):
        session = load_session(context.scene) if context.scene is not None else None
        if session is None:
            return []
        result = []
        for idx, entry in enumerate(session.cutters):
            obj = bpy.data.objects.get(entry.name)
            if obj is not None and obj.type == "MESH" and not obj.hide_viewport and not obj.hide_get():
                result.append((idx, entry, obj))
        return result

    def _predict_new_layer_number(self, entries_list, target_name: str) -> int:
        names = [e.name for _, e, _ in entries_list if e.name != self.source_cutter]
        if target_name not in names:
            return 1
        t_idx = names.index(target_name)
        return (t_idx + 1) if self.position == "ABOVE" else (t_idx + 2)

    def _hit_test_target(self, context: bpy.types.Context, mouse_x: float, mouse_y: float) -> str | None:
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None

        items = self._iter_session_cutters(context)
        candidates = [(idx, entry, obj) for idx, entry, obj in items if entry.name != self.source_cutter]
        if not candidates:
            return None

        mouse = Vector((mouse_x, mouse_y))

        # 1. Priority hit-test: clicking directly on a cutter's 2D floating badge (48 px radius)
        best_badge = None
        best_badge_dist = 48.0
        centers: dict[str, Vector] = {}
        for _idx, entry, obj in candidates:
            c_w, _edges, _tris = _cutter_world_geom(obj)
            centers[entry.name] = c_w
            p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, c_w)
            if p2 is not None:
                d = (Vector((p2.x, p2.y)) - mouse).length
                if d < best_badge_dist:
                    best_badge_dist = d
                    best_badge = entry.name
        if best_badge is not None:
            return best_badge

        # 2. 3D raycast against each candidate cutter mesh / volume (closest hit wins!)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse).normalized()
        depsgraph = context.evaluated_depsgraph_get()
        best_ray = None
        best_ray_dist = float("inf")
        for _idx, entry, obj in candidates:
            try:
                inv_m = obj.matrix_world.inverted()
            except ValueError:
                continue
            lo = inv_m @ ray_origin
            ld_raw = inv_m.to_3x3() @ ray_dir
            if ld_raw.length < 1e-8:
                continue
            ld = ld_raw.normalized()
            hit = False
            loc = None
            try:
                hit, loc, _nrm, _fidx = obj.ray_cast(lo, ld, depsgraph=depsgraph)
            except RuntimeError:
                # Unevaluated / BOUNDS / empty-eval cutter: fall back to exact local AABB slab test
                hit, loc = _ray_intersect_local_aabb(lo, ld, obj.bound_box)
            if hit and loc is not None:
                world_hit = obj.matrix_world @ loc
                dist = (world_hit - ray_origin).length
                if dist < best_ray_dist:
                    best_ray_dist = dist
                    best_ray = entry.name
        if best_ray is not None:
            return best_ray

        # 3. Fallback: nearest bounding-box corner in 2D screen space within 32 px
        best_fb = None
        best_fb_dist = 32.0
        for _idx, entry, obj in candidates:
            mw = obj.matrix_world
            for corner in obj.bound_box:
                p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ Vector(corner))
                if p2 is not None:
                    d = (Vector((p2.x, p2.y)) - mouse).length
                    if d < best_fb_dist:
                        best_fb_dist = d
                        best_fb = entry.name
        return best_fb

    def _draw_3d(self) -> None:
        context = bpy.context
        items = self._iter_session_cutters(context)
        if not items:
            return

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("NONE")

        accent_fill = (0.18, 0.85, 0.60, 0.22) if self.position == "ABOVE" else (0.98, 0.62, 0.16, 0.22)

        for _idx, entry, obj in items:
            is_src = entry.name == self.source_cutter
            is_hover = entry.name == self._hovered_target
            _center, edge_pts, tri_pts = _cutter_world_geom(obj)

            if is_hover and tri_pts:
                b_tri = batch_for_shader(shader, "TRIS", {"pos": tri_pts})
                shader.bind()
                shader.uniform_float("color", accent_fill)
                b_tri.draw(shader)

            if edge_pts:
                if is_hover:
                    col = (1.0, 0.94, 0.35, 1.0)
                    lw = 3.6
                elif is_src:
                    col = (0.78, 0.82, 0.90, 0.45)
                    lw = 1.6
                else:
                    r, g, b, _ = apple_bite.get_zone_color(entry.zone)
                    col = (r, g, b, 0.82)
                    lw = 2.0
                gpu.state.line_width_set(lw)
                b_ed = batch_for_shader(shader, "LINES", {"pos": edge_pts})
                shader.bind()
                shader.uniform_float("color", col)
                b_ed.draw(shader)

        gpu.state.line_width_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("NONE")

    def _draw_2d(self) -> None:
        context = bpy.context
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return

        items = self._iter_session_cutters(context)
        if not items:
            return

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        font_id = 0

        src_entry = next((e for _, e, _ in items if e.name == self.source_cutter), None)
        src_zone = zone_registry.zone_token(src_entry.zone, context.scene) if src_entry is not None else "?"
        dir_word = "ABOVE" if self.position == "ABOVE" else "BELOW"
        accent_col = (0.22, 0.86, 0.62, 0.96) if self.position == "ABOVE" else (0.98, 0.62, 0.16, 0.96)

        # Floating layer badges on every cutter box
        for idx, entry, obj in items:
            c_w, _edges, _tris = _cutter_world_geom(obj)
            p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, c_w)
            if p2 is None:
                continue

            r, g, b, _ = apple_bite.get_zone_color(entry.zone)
            is_src = entry.name == self.source_cutter
            is_hover = entry.name == self._hovered_target

            if is_src:
                _draw_2d_pill_badge(
                    shader,
                    font_id,
                    f"#{idx + 1} [Zone {zone_registry.zone_token(entry.zone, context.scene)}] (Moving)",
                    p2.x,
                    p2.y,
                    12,
                    (0.10, 0.12, 0.16, 0.78),
                    (0.65, 0.70, 0.80, 0.65),
                    (0.82, 0.86, 0.92, 0.90),
                )
            elif is_hover:
                pred_num = self._predict_new_layer_number(items, entry.name)
                _draw_2d_pill_badge(
                    shader,
                    font_id,
                    f"#{idx + 1} [Zone {zone_registry.zone_token(entry.zone, context.scene)}]  -> Place [{src_zone}] {dir_word} (#{pred_num})",
                    p2.x,
                    p2.y,
                    14,
                    (accent_col[0], accent_col[1], accent_col[2], 0.96),
                    (1.0, 1.0, 1.0, 1.0),
                    (0.06, 0.06, 0.08, 1.0),
                )
            else:
                _draw_2d_pill_badge(
                    shader,
                    font_id,
                    f"#{idx + 1} [Zone {zone_registry.zone_token(entry.zone, context.scene)}]",
                    p2.x,
                    p2.y,
                    13,
                    (0.10, 0.12, 0.16, 0.88),
                    (r, g, b, 0.92),
                    (r * 0.4 + 0.6, g * 0.4 + 0.6, b * 0.4 + 0.6, 1.0),
                )

        # Bottom guide banner
        if self._hovered_target:
            tgt_entry = next((e for _, e, _ in items if e.name == self._hovered_target), None)
            tz = zone_registry.zone_token(tgt_entry.zone, context.scene) if tgt_entry is not None else "?"
            pred = self._predict_new_layer_number(items, self._hovered_target)
            hud = f"[Layer {dir_word}] Left-click to place [Zone {src_zone}] {dir_word} [Zone {tz}] (-> Layer #{pred})   |   RMB / ESC Cancel"
        else:
            hud = f"[Layer {dir_word}] Hover & left-click target cutter box to place [Zone {src_zone}] {dir_word} it   |   MMB Orbit  |  RMB / ESC Cancel"

        _draw_2d_pill_badge(
            shader,
            font_id,
            hud,
            region.width * 0.5,
            44.0,
            14,
            (0.08, 0.10, 0.14, 0.90),
            accent_col,
            (1.0, 0.95, 0.78, 1.0),
        )

    def modal(self, context, event):
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}

        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            new_hover = self._hit_test_target(context, event.mouse_region_x, event.mouse_region_y)
            if new_hover != self._hovered_target:
                self._hovered_target = new_hover
                if context.area:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            picked = self._hit_test_target(context, event.mouse_region_x, event.mouse_region_y)
            if picked is not None:
                self._cleanup(context)
                ok, msg = apple_bite.move_cutter_relative_to(
                    context,
                    source_name=self.source_cutter,
                    target_name=picked,
                    position=self.position,
                )
                settings = getattr(context.scene, "fdt_settings", None)
                if settings is not None:
                    settings.last_status = msg
                self.report({"INFO"} if ok else {"WARNING"}, msg)
                return {"FINISHED"} if ok else {"CANCELLED"}
            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        global _active_layer_picker
        if _active_layer_picker is not None:
            _active_layer_picker._cleanup(context)

        if not self.source_cutter:
            obj = context.active_object
            if obj is not None and bool(obj.get(CUTTER_PROP, False)):
                self.source_cutter = obj.name

        items = self._iter_session_cutters(context)
        others = [e for _, e, _ in items if e.name != self.source_cutter]
        if not self.source_cutter or not others:
            self.report({"WARNING"}, "ต้องมีกล่องตัดอย่างน้อย 2 กล่องเพื่อสลับลำดับ Layer")
            return {"CANCELLED"}

        self._hovered_target = self._hit_test_target(context, event.mouse_region_x, event.mouse_region_y)
        _active_layer_picker = self

        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_3d, (), "WINDOW", "POST_VIEW"
        )
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_2d, (), "WINDOW", "POST_PIXEL"
        )
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


class FDT_GGT_cutter_gear(bpy.types.GizmoGroup):
    """Floating Gear Icon on the selected Cutter — click to open the Wheel Menu at mouse cursor."""

    bl_idname = "FDT_GGT_cutter_gear"
    bl_label = "FullDestruc Cutter Gear"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        if context.scene is None or not has_active_session(context.scene):
            return False
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and bool(obj.get(CUTTER_PROP, False))

    def setup(self, context):
        gz = self.gizmos.new("GIZMO_GT_button_2d")
        gz.icon = "PREFERENCES"
        gz.draw_options = {"BACKDROP", "OUTLINE"}
        gz.scale_basis = 0.22
        gz.color = (0.18, 0.20, 0.24)
        gz.alpha = 0.88
        gz.color_highlight = (0.95, 0.55, 0.15)
        gz.alpha_highlight = 1.0

        gz.target_set_operator("fdt.cutter_wheel_menu")
        self.gear_gizmo = gz

    def draw_prepare(self, context):
        obj = context.active_object
        if obj is None:
            return
        # Float the gear neatly slightly above the cutter's top face
        top_z = max(float(obj.dimensions.z) * 0.5, 0.3) + 0.18
        pos = obj.matrix_world.translation + Vector((0.0, 0.0, top_z))
        self.gear_gizmo.matrix_basis = Matrix.Translation(pos)


CLASSES = (
    FDT_MT_cutter_wheel,
    FDT_OT_applebite_pick_axis,
    FDT_OT_applebite_pick_layer_target,
    FDT_GGT_cutter_gear,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    global _active_axis_picker, _active_layer_picker
    if _active_axis_picker is not None:
        _active_axis_picker._cleanup(None)
    if _active_layer_picker is not None:
        _active_layer_picker._cleanup(None)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
