"""Blender Operators for Apple Bite v2.1 (Guided UX + Cutter List + Zone Color Overlay)."""

from __future__ import annotations

import json

import blf
import bmesh
import bpy
import gpu
import numpy as np
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from ..core import apple_bite, zone_registry
from ..core.session import has_active_session, load_session

# Track viewport draw handles cleanly so unregister never leaks
_hud_draw_handle = None
_wire_draw_handle = None
_interactive_active = False
_zone_grid_shader = None
_zone_grid_shader_failed = False
_zone_grid_cache_sig = None
_zone_grid_cache_entries: list[tuple[object, str, str]] = []  # (batch, zone, cutter_name)
_zone_grid_deps_rev: int = 0


def invalidate_zone_grid_cache(tag_redraw: bool = True) -> None:
    """Force Zone Grid overlay to rebuild from the latest Rust-updated Piece meshes."""
    global _zone_grid_deps_rev, _zone_grid_cache_sig
    _zone_grid_deps_rev += 1
    _zone_grid_cache_sig = None
    if tag_redraw:
        wm = getattr(bpy.context, "window_manager", None)
        if wm is not None:
            for win in wm.windows:
                if win.screen is None:
                    continue
                for area in win.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph) -> None:
    """Bump Zone Grid cache revision whenever geometry or transforms change during an active session."""
    global _zone_grid_deps_rev
    if scene is None or not has_active_session(scene):
        return
    if depsgraph is None:
        return
    for upd in depsgraph.updates:
        if upd.is_updated_geometry or upd.is_updated_transform:
            _zone_grid_deps_rev += 1
            return


def _remove_hud_handler() -> None:
    global _hud_draw_handle, _interactive_active
    if _hud_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_hud_draw_handle, "WINDOW")
        except Exception:
            pass
        _hud_draw_handle = None
    _interactive_active = False
    try:
        from ..ui import draw_overlay

        draw_overlay.close_draw_overlay()
    except Exception:
        pass


def _ensure_wire_overlay() -> None:
    global _wire_draw_handle
    if _wire_draw_handle is None:
        try:
            _wire_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _draw_zone_wireframes, (), "WINDOW", "POST_VIEW"
            )
        except Exception:
            _wire_draw_handle = None


def _remove_wire_overlay() -> None:
    global _wire_draw_handle
    if _wire_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_wire_draw_handle, "WINDOW")
        except Exception:
            pass
        _wire_draw_handle = None


def _get_zone_grid_shader():
    """Compile and cache the Blender 5.2 tri-planar world-space Zone Grid GPU shader."""
    global _zone_grid_shader, _zone_grid_shader_failed
    if _zone_grid_shader is not None:
        return _zone_grid_shader
    if _zone_grid_shader_failed:
        return None
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fdt_zone_grid_iface")
        vert_out.smooth("VEC3", "v_world_pos")
        vert_out.smooth("VEC3", "v_world_nor")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "nor")
        info.vertex_out(vert_out)
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "zone_color")
        info.push_constant("FLOAT", "grid_size")
        info.push_constant("FLOAT", "fill_alpha")
        info.push_constant("FLOAT", "line_alpha")
        info.fragment_out(0, "VEC4", "fragColor")

        info.vertex_source(
            """
            void main()
            {
                vec3 n = normalize(nor);
                v_world_pos = pos;
                v_world_nor = n;
                gl_Position = ModelViewProjectionMatrix * vec4(pos + n * 0.0025, 1.0);
            }
            """
        )
        info.fragment_source(
            """
            void main()
            {
                vec3 an = abs(normalize(v_world_nor));
                float gs = max(grid_size, 0.02);
                vec2 uv;
                if (an.z >= an.x && an.z >= an.y) {
                    uv = v_world_pos.xy / gs;
                } else if (an.x >= an.y) {
                    uv = v_world_pos.yz / gs;
                } else {
                    uv = v_world_pos.xz / gs;
                }

                vec2 fw = max(fwidth(uv), vec2(0.0005));
                vec2 grid = abs(fract(uv - 0.5) - 0.5) / fw;
                float line = 1.0 - min(min(grid.x, grid.y), 1.0);
                // Soften line thickness slightly so grid lines look crisp and readable
                line = clamp(line * 1.15, 0.0, 1.0);

                float alpha = mix(fill_alpha, line_alpha, line);
                vec3 bright = min(zone_color.rgb * 1.25 + vec3(0.14), vec3(1.0));
                vec3 rgb = mix(zone_color.rgb, bright, line);
                fragColor = vec4(rgb, alpha);
            }
            """
        )
        _zone_grid_shader = gpu.shader.create_from_info(info)
        return _zone_grid_shader
    except Exception as exc:
        print(f"[FullDestruc_Tool] Zone Grid shader fallback: {exc}")
        _zone_grid_shader_failed = True
        return None


def _build_zone_grid_cache(context, session, shader) -> list[tuple[object, str, str]]:
    """Build triangulated world-space GPU batches for all active Piece meshes in FDT_Pieces_*."""
    global _zone_grid_cache_sig, _zone_grid_cache_entries

    cols = session.get_collections()
    pieces_col = cols.get("pieces")
    if pieces_col is None or len(pieces_col.objects) == 0:
        _zone_grid_cache_sig = None
        _zone_grid_cache_entries = []
        return []

    # Lightweight signature: depsgraph rev + cutter order/live Edit Mode state + rough settings + piece names
    from ..core import live_sync

    cutter_sig = []
    for entry in session.cutters:
        c_obj = bpy.data.objects.get(entry.name)
        if c_obj is None:
            continue
        tok = live_sync._cutter_state_token(c_obj)
        cr = getattr(c_obj, "fdt_rough", None)
        cr_sig = (
            (
                round(float(cr.inward_depth), 4),
                round(float(cr.jaggedness), 4),
                round(float(cr.noise_scale), 4),
                int(cr.detail),
                round(float(cr.seed), 4),
            )
            if cr is not None
            else ()
        )
        cutter_sig.append((entry.name, entry.zone, entry.muted, entry.rough, cr_sig, tok))
    settings = getattr(context.scene, "fdt_settings", None)
    rough_sig = (bool(settings.enable_rough),) if settings is not None else ()
    piece_names = tuple(o.name for o in pieces_col.objects)
    sig = (session.session_id, _zone_grid_deps_rev, piece_names, rough_sig, tuple(cutter_sig))

    if sig == _zone_grid_cache_sig:
        return _zone_grid_cache_entries

    use_rust = settings is not None and getattr(settings, "boolean_solver", "") == "RUST"
    depsgraph = None if use_rust else context.evaluated_depsgraph_get()
    entries: list[tuple[object, str, str]] = []

    for piece_obj in pieces_col.objects:
        if piece_obj.type != "MESH" or piece_obj.hide_viewport or piece_obj.hide_get():
            continue
        eval_obj = None
        if use_rust and len(piece_obj.modifiers) == 0:
            mesh = piece_obj.data
        else:
            if depsgraph is None:
                depsgraph = context.evaluated_depsgraph_get()
            eval_obj = piece_obj.evaluated_get(depsgraph)
            try:
                mesh = eval_obj.to_mesh()
            except Exception:
                continue
        if mesh is None:
            continue
        try:
            n_verts = len(mesh.vertices)
            if n_verts == 0 or len(mesh.polygons) == 0:
                continue
            mesh.calc_loop_triangles()
            n_tris = len(mesh.loop_triangles)
            if n_tris == 0:
                continue

            v_co = np.empty(n_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", v_co)
            v_co = v_co.reshape((n_verts, 3))

            tri_idx = np.empty(n_tris * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tri_idx)
            tri_idx = tri_idx.reshape((n_tris, 3))

            # Transform to world space
            mw = np.array(piece_obj.matrix_world, dtype=np.float32)
            rot = mw[:3, :3]
            trans = mw[:3, 3]
            world_v = v_co @ rot.T + trans

            tri_pos = world_v[tri_idx]  # (T, 3, 3)
            e1 = tri_pos[:, 1, :] - tri_pos[:, 0, :]
            e2 = tri_pos[:, 2, :] - tri_pos[:, 0, :]
            fn = np.cross(e1, e2)
            fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
            fn = fn / np.where(fn_len > 1e-8, fn_len, 1.0)
            tri_nor = np.repeat(fn[:, np.newaxis, :], 3, axis=1)  # (T, 3, 3)

            pos_flat = np.ascontiguousarray(tri_pos.reshape((-1, 3)), dtype=np.float32)
            nor_flat = np.ascontiguousarray(tri_nor.reshape((-1, 3)), dtype=np.float32)

            if shader is not None:
                batch = batch_for_shader(shader, "TRIS", {"pos": pos_flat, "nor": nor_flat})
            else:
                fallback = gpu.shader.from_builtin("UNIFORM_COLOR")
                batch = batch_for_shader(fallback, "TRIS", {"pos": pos_flat})

            zone = str(piece_obj.get(apple_bite.PIECE_ZONE_PROP, "A") or "A").strip() or "A"
            cname = str(piece_obj.get("fdt_piece_cutter", ""))
            entries.append((batch, zone, cname))
        finally:
            if eval_obj is not None:
                eval_obj.to_mesh_clear()

    _zone_grid_cache_sig = sig
    _zone_grid_cache_entries = entries
    return entries


def _get_selected_cutter_names(context) -> set[str]:
    """Return names of all currently selected Apple Bite cutter objects in the viewport."""
    names: set[str] = set()
    for obj in getattr(context, "selected_objects", ()):
        if obj is not None and bool(obj.get(apple_bite.CUTTER_PROP, False)):
            names.add(obj.name)
    return names


def _draw_zone_grid_overlay(context, session, selected_cutters: set[str]) -> None:
    """Render the tri-planar colored Zone Grid onto mesh areas overlapped by active cutters."""
    from ..core import gpu_preview

    if gpu_preview.is_active():
        return
    settings = getattr(context.scene, "fdt_settings", None)
    if settings is not None and not bool(settings.show_zone_grid):
        return

    shader = _get_zone_grid_shader()
    entries = _build_zone_grid_cache(context, session, shader)
    if not entries:
        return

    has_sel = bool(selected_cutters)
    grid_size = float(settings.zone_grid_size) if settings is not None else 0.35

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)

    if shader is not None:
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        shader.bind()
        shader.uniform_float("ModelViewProjectionMatrix", mvp)
        shader.uniform_float("grid_size", grid_size)
        for batch, zone, cname in entries:
            r, g, b, _ = apple_bite.get_zone_color(zone)
            is_sel = cname in selected_cutters
            if is_sel:
                fill_a, line_a = 0.25, 0.92
            elif has_sel:
                fill_a, line_a = 0.05, 0.20
            else:
                fill_a, line_a = 0.09, 0.40
            shader.uniform_float("zone_color", (r, g, b, 1.0))
            shader.uniform_float("fill_alpha", fill_a)
            shader.uniform_float("line_alpha", line_a)
            batch.draw(shader)
    else:
        fallback = gpu.shader.from_builtin("UNIFORM_COLOR")
        fallback.bind()
        for batch, zone, cname in entries:
            r, g, b, _ = apple_bite.get_zone_color(zone)
            is_sel = cname in selected_cutters
            if is_sel:
                alpha = 0.28
            elif has_sel:
                alpha = 0.07
            else:
                alpha = 0.14
            fallback.uniform_float("color", (r, g, b, alpha))
            batch.draw(fallback)

    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _draw_zone_wireframes():
    """Draw tri-planar Zone Grid overlay + focus-aware zone-colored wireframes for cutters."""
    from ..core import gpu_preview

    context = bpy.context
    if context.scene is None:
        return
    session = load_session(context.scene)
    if session is None or not session.cutters:
        return

    # When Zero-Switch GPU Preview is active, it already renders the Zone Grid + depth-tested
    # cutters + in-front Edit Mode overlays directly — avoid drawing stale duplicate wireframes!
    if gpu_preview.is_active():
        return

    selected_cutters = _get_selected_cutter_names(context)
    has_sel = bool(selected_cutters)

    # 1. Projected Zone Grid on overlapped mesh surfaces (respects Layer order + focus highlight)
    _draw_zone_grid_overlay(context, session, selected_cutters)

    # 2. Focus-aware zone-colored cutter wireframes (depth-tested in Object Mode, in-front in Edit Mode/X-Ray)
    space = getattr(context, "space_data", None)
    use_xray = bool(
        space is not None
        and getattr(space, "type", "") == "VIEW_3D"
        and getattr(getattr(space, "shading", None), "show_xray", False)
    )
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_mask_set(False)

    unselected_entries = [e for e in session.cutters if e.name not in selected_cutters]
    selected_entries = [e for e in session.cutters if e.name in selected_cutters]

    for entry in unselected_entries + selected_entries:
        obj = bpy.data.objects.get(entry.name)
        if obj is None or obj.type != "MESH" or obj.hide_viewport or obj.hide_get():
            continue
        mesh = obj.data
        mw = obj.matrix_world
        coords = []
        if obj.mode == "EDIT":
            try:
                bm = bmesh.from_edit_mesh(mesh)
                for e in bm.edges:
                    if e.is_valid:
                        coords.append(mw @ e.verts[0].co)
                        coords.append(mw @ e.verts[1].co)
            except Exception:
                coords = []
        else:
            if len(mesh.edges) == 0:
                continue
            verts = mesh.vertices
            for edge in mesh.edges:
                coords.append(mw @ verts[edge.vertices[0]].co)
                coords.append(mw @ verts[edge.vertices[1]].co)
        if not coords:
            continue

        is_sel = entry.name in selected_cutters
        r, g, b, _ = apple_bite.get_zone_color(entry.zone)
        if entry.muted:
            line_w = 1.6 if is_sel else 1.0
            alpha = 0.55 if is_sel else (0.14 if has_sel else 0.25)
            color = (0.55, 0.55, 0.55, alpha)
        elif is_sel:
            line_w = 2.5
            color = (r, g, b, 0.96)
        elif has_sel:
            line_w = 1.0
            color = (r, g, b, 0.18)
        else:
            line_w = 1.4
            color = (r, g, b, 0.42)

        gpu.state.depth_test_set("NONE" if (obj.mode == "EDIT" or obj.show_in_front or use_xray) else "LESS_EQUAL")
        gpu.state.line_width_set(line_w)
        batch = batch_for_shader(shader, "LINES", {"pos": coords})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _draw_hud_callback():
    context = bpy.context
    session = load_session(context.scene)
    if session is None or not _interactive_active:
        return

    font_id = 0
    blf.size(font_id, 15)
    blf.color(font_id, 1.0, 0.85, 0.25, 1.0)
    settings = getattr(context.scene, "fdt_settings", None)
    zone = settings.active_zone if settings else "A"
    zone_label = zone_registry.zone_token(zone, context.scene)
    text = (
        f"[Apple Bite] Zone: {zone_label}  |  "
        f"Ctrl+RMB: Place Box on Surface  |  Ctrl+Shift+RMB: Pick Mesh Cutter  |  ESC: Stop"
    )
    blf.position(font_id, 24, 42, 0)
    blf.draw(font_id, text)

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return
    blf.size(font_id, 14)
    for entry in session.cutters:
        cutter_obj = bpy.data.objects.get(entry.name)
        if cutter_obj is None:
            continue
        pos_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cutter_obj.matrix_world.translation)
        if pos_2d is not None:
            r, g, b, _ = apple_bite.get_zone_color(entry.zone)
            zone_label = zone_registry.zone_token(entry.zone, context.scene)
            if entry.muted:
                blf.color(font_id, 0.6, 0.6, 0.6, 0.8)
                label = f"[{zone_label}] (พัก)"
            else:
                blf.color(font_id, r, g, b, 1.0)
                label = f"[{zone_label}]"
            blf.position(font_id, pos_2d.x + 8, pos_2d.y + 8, 0)
            blf.draw(font_id, label)


def _set_status(context: bpy.types.Context, msg: str) -> None:
    settings = getattr(context.scene, "fdt_settings", None)
    if settings is not None:
        settings.last_status = msg


class FDT_OT_applebite_smart_start(bpy.types.Operator):
    bl_idname = "fdt.applebite_smart_start"
    bl_label = "Start or Cut Further"
    bl_description = "Start a new Apple Bite cut on selected meshes (including cutting already-applied pieces further)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        info = apple_bite.detect_step1_action(context)
        if info["mode"] in ("START", "BOTH"):
            ok, msg, _ = apple_bite.enter_session(context, target_objects=info["targets"])
            if ok:
                _ensure_wire_overlay()
            _set_status(context, msg)
            self.report({"INFO"} if ok else {"WARNING"}, msg)
            return {"FINISHED"} if ok else {"CANCELLED"}
        if info["mode"] == "REEDIT" and info["session_id"]:
            ok, msg, _ = apple_bite.resume_session(context, info["session_id"])
            if ok:
                _ensure_wire_overlay()
            _set_status(context, msg)
            self.report({"INFO"} if ok else {"WARNING"}, msg)
            return {"FINISHED"} if ok else {"CANCELLED"}

        _set_status(context, info["guide_text"])
        self.report({"WARNING"}, info["guide_text"])
        return {"CANCELLED"}


class FDT_OT_applebite_reedit(bpy.types.Operator):
    bl_idname = "fdt.applebite_reedit"
    bl_label = "Re-edit Previous Cut"
    bl_description = "Re-open the previous Apple Bite session for this object to adjust its old cutters"
    bl_options = {"REGISTER", "UNDO"}

    session_id: StringProperty(name="Session ID", default="", options={"SKIP_SAVE"})

    def execute(self, context):
        sid = self.session_id
        if not sid:
            info = apple_bite.detect_step1_action(context)
            sid = info.get("session_id") or ""
        if not sid:
            self.report({"WARNING"}, "ไม่พบงานตัดเดิมของชิ้นที่เลือก")
            return {"CANCELLED"}
        ok, msg, _ = apple_bite.resume_session(context, sid)
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_enter(bpy.types.Operator):
    bl_idname = "fdt.applebite_enter"
    bl_label = "Start Apple Bite"
    bl_description = "Start cutting 1 or more selected mesh objects (originals stored safely in Protected Vault)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ok, msg, _session = apple_bite.enter_session(context)
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_exit(bpy.types.Operator):
    bl_idname = "fdt.applebite_exit"
    bl_label = "Pause / Exit"
    bl_description = "Pause Apple Bite mode cleanly (click any piece later to resume editing)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _remove_hud_handler()
        ok, msg = apple_bite.exit_session(context)
        _set_status(context, msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class FDT_OT_applebite_cancel(bpy.types.Operator):
    bl_idname = "fdt.applebite_cancel"
    bl_label = "Cancel & Restore Originals"
    bl_description = "Cancel Apple Bite, restore original meshes from Protected Vault, and remove temporary cuts"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _remove_hud_handler()
        ok, msg = apple_bite.cancel_session(context)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_reset_original(bpy.types.Operator):
    bl_idname = "fdt.applebite_reset_original"
    bl_label = "Reset to Original"
    bl_description = "Delete all related cuts (including Cut Further & Final pieces) and restore only the original mesh"
    bl_options = {"REGISTER", "UNDO"}

    session_id: StringProperty(name="Session ID", default="", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        if not self.session_id:
            info = apple_bite.detect_step1_action(context)
            self.session_id = info.get("session_id") or ""
        if not self.session_id:
            self.report({"WARNING"}, "ไม่พบงานตัดของชิ้นที่เลือก")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        layout = self.layout
        preview = apple_bite.get_reset_preview_info(context.scene, self.session_id)
        orig_names = preview["original_names"]
        fam_count = len(preview["family_ids"])

        if orig_names:
            display_names = ", ".join(orig_names[:3])
            if len(orig_names) > 3:
                display_names += f" +{len(orig_names) - 3}"
        else:
            display_names = self.session_id

        layout.label(text="Delete all cuts and restore original?", icon="ERROR")
        box = layout.box()
        box.label(text=f"Original: {display_names}", icon="MESH_DATA")
        box.label(text=f"Cut sessions to delete: {fam_count}", icon="TRASH")
        if fam_count > 1:
            box.label(text="(Includes all Cut Further & Final pieces)", icon="INFO")

    def execute(self, context):
        _remove_hud_handler()
        sid = self.session_id
        if not sid:
            info = apple_bite.detect_step1_action(context)
            sid = info.get("session_id") or ""
        if not sid:
            self.report({"WARNING"}, "ไม่พบงานตัดของชิ้นที่เลือก")
            return {"CANCELLED"}
        ok, msg, _restored = apple_bite.reset_to_original(context, sid)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_add_box(bpy.types.Operator):
    bl_idname = "fdt.applebite_add_box"
    bl_label = "+ วางกล่องตัด"
    bl_description = "Spawn a new Box Cutter at the center of the selected building, ready to move/scale (G/R/S)"
    bl_options = {"REGISTER", "UNDO"}

    zone: StringProperty(name="Zone", default="", options={"SKIP_SAVE"})

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        size = settings.cutter_size if settings else 2.0
        zone = self.zone or (settings.active_zone if settings else "A")
        ok, msg, _cutter = apple_bite.add_box_cutter(context, size=size, zone=zone)
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_add_adjacent_box(bpy.types.Operator):
    bl_idname = "fdt.applebite_add_adjacent_box"
    bl_label = "ต่อกล่องชิดขอบ"
    bl_description = "Clone the active/last cutter flush-adjacent with a small overlap so no thin slivers are left"
    bl_options = {"REGISTER", "UNDO"}

    direction: StringProperty(name="Direction", default="+X", options={"SKIP_SAVE"})
    auto_next_zone: BoolProperty(name="Auto Next Zone", default=True, options={"SKIP_SAVE"})

    def execute(self, context):
        ok, msg, _cutter = apple_bite.add_adjacent_box_cutter(
            context,
            direction=self.direction,
            auto_next_zone=self.auto_next_zone,
        )
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_add_mesh(bpy.types.Operator):
    bl_idname = "fdt.applebite_add_mesh"
    bl_label = "+ ใช้โมเดลที่เลือกเป็นตัวตัด"
    bl_description = "Convert currently selected custom mesh object(s) into Apple Bite cutters"
    bl_options = {"REGISTER", "UNDO"}

    zone: StringProperty(name="Zone", default="", options={"SKIP_SAVE"})

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        zone = self.zone or (settings.active_zone if settings else "A")
        ok, msg, _cutters = apple_bite.add_custom_mesh_cutters(context, list(context.selected_objects), zone=zone)
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_draw_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_draw_cutter"
    bl_label = "Draw Cutter (2D -> Extrude)"
    bl_description = "Draw a polygon or freehand shape on a standalone PySide6 overlay and extrude it into a cutter"
    bl_options = {"REGISTER"}

    zone: StringProperty(name="Zone", default="", options={"SKIP_SAVE"})

    def execute(self, context):
        from ..ui import draw_overlay

        ok, msg = draw_overlay.open_draw_cutter_overlay(context, zone=self.zone or None)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


_pending_sculpt_payload: dict | None = None


def stage_sculpt_payload(
    cutter_name: str,
    vertices,
    faces,
    operation: str,
    mask_face_indices: list[int] | None = None,
    mask_depth: float = 1.5,
) -> None:
    """Stage mesh data from PySide6 Draw Overlay before invoking the UNDO-tracked apply operator."""
    global _pending_sculpt_payload
    _pending_sculpt_payload = {
        "cutter_name": str(cutter_name),
        "vertices": vertices,
        "faces": faces,
        "operation": str(operation),
        "mask_face_indices": list(mask_face_indices) if mask_face_indices else [],
        "mask_depth": float(mask_depth),
    }


class FDT_OT_applebite_draw_sculpt_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_draw_sculpt_cutter"
    bl_label = "Draw Cut on Cutter"
    bl_description = "Draw a shape from any camera angle to Carve Out, Keep Inside, or Add onto the selected cutter"
    bl_options = {"REGISTER"}

    cutter_name: StringProperty(name="Cutter Name", default="", options={"SKIP_SAVE"})

    def execute(self, context):
        from ..ui import draw_overlay

        cname = self.cutter_name
        if not cname and context.active_object is not None:
            if bool(context.active_object.get(apple_bite.CUTTER_PROP, False)):
                cname = context.active_object.name
        if not cname:
            msg = "กรุณาเลือกตัวตัดที่ต้องการวาดแต่งก่อนครับ"
            _set_status(context, msg)
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        ok, msg = draw_overlay.open_draw_sculpt_overlay(context, cutter_name=cname)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_apply_sculpt_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_apply_sculpt_cutter"
    bl_label = "Apply Draw Sculpt to Cutter"
    bl_description = "Apply the staged 2D-to-3D sculpt boolean onto an existing cutter (with Undo support)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        global _pending_sculpt_payload
        payload = _pending_sculpt_payload
        _pending_sculpt_payload = None
        if not payload:
            return {"CANCELLED"}

        ok, msg, _cutter = apple_bite.sculpt_existing_cutter_with_mesh_data(
            context,
            cutter_name=payload["cutter_name"],
            vertices=payload["vertices"],
            faces=payload["faces"],
            operation=payload["operation"],
            mask_face_indices=payload.get("mask_face_indices") or None,
            mask_depth=float(payload.get("mask_depth", 1.5)),
        )
        if ok:
            _ensure_wire_overlay()
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_set_zone(bpy.types.Operator):
    bl_idname = "fdt.applebite_set_zone"
    bl_label = "Set Zone"
    bl_description = "Set active Zone ID (and update any selected cutters to this Zone)"
    bl_options = {"REGISTER", "UNDO"}

    zone: StringProperty(name="Zone", default="A")

    def execute(self, context):
        zone = zone_registry.resolve_zone(self.zone, context.scene).key
        settings = getattr(context.scene, "fdt_settings", None)
        if settings is not None:
            settings.active_zone = zone
        updated = apple_bite.set_cutters_zone(context, list(context.selected_objects), zone)
        msg = f"โซนที่เลือก: {zone_registry.zone_token(zone, context.scene)}" + (f" (อัปเดตตัวตัด {updated} อัน)" if updated else "")
        _set_status(context, msg)
        return {"FINISHED"}


class FDT_OT_applebite_select_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_select_cutter"
    bl_label = "Select Cutter"
    bl_description = "Select this cutter in the 3D Viewport"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")

    def execute(self, context):
        ok = apple_bite.select_cutter(context, self.cutter_name)
        if ok:
            _set_status(context, f"เลือกตัวตัด '{self.cutter_name}' แล้ว — กด G/R/S เพื่อปรับตำแหน่ง")
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_cycle_zone(bpy.types.Operator):
    bl_idname = "fdt.applebite_cycle_zone"
    bl_label = "Change Cutter Zone"
    bl_description = "Cycle this cutter to the next Zone (A -> B -> C ...)"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")

    def execute(self, context):
        result = apple_bite.cycle_cutter_zone(context, self.cutter_name)
        if result:
            new_key, new_name = result
            label = zone_registry.zone_token(new_key, context.scene)
            _set_status(context, f"เปลี่ยน '{new_name}' เป็นโซน {label} แล้ว")
            return {"FINISHED"}
        return {"CANCELLED"}


class FDT_OT_applebite_toggle_mute(bpy.types.Operator):
    bl_idname = "fdt.applebite_toggle_mute"
    bl_label = "Pause/Resume Cutter"
    bl_description = "Temporarily pause or resume this cutter without deleting it"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")

    def execute(self, context):
        ok, is_muted = apple_bite.toggle_cutter_mute(context, self.cutter_name)
        if ok:
            state = "Paused" if is_muted else "Active"
            _set_status(context, f"{self.cutter_name}: {state}")
            return {"FINISHED"}
        return {"CANCELLED"}


class FDT_OT_applebite_toggle_rough(bpy.types.Operator):
    bl_idname = "fdt.applebite_toggle_rough"
    bl_label = "Toggle Rough Cut"
    bl_description = "Toggle rough noise cut edge independently on this cutter"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")

    def execute(self, context):
        ok, is_rough = apple_bite.toggle_cutter_rough(context, self.cutter_name)
        if ok:
            state = "Rough ON" if is_rough else "Smooth (Flat)"
            _set_status(context, f"{self.cutter_name}: {state}")
            return {"FINISHED"}
        return {"CANCELLED"}


class FDT_OT_applebite_delete_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_delete_cutter"
    bl_label = "Delete Cutter"
    bl_description = "Remove this cutter from the session cleanly"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")

    def execute(self, context):
        ok, msg = apple_bite.delete_cutter(context, self.cutter_name)
        _set_status(context, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_move_cutter(bpy.types.Operator):
    bl_idname = "fdt.applebite_move_cutter"
    bl_label = "Move Cutter Layer"
    bl_description = "Move this cutter up or down in layer priority (top layer wins overlapping cuts)"
    bl_options = {"REGISTER", "UNDO"}

    cutter_name: StringProperty(name="Cutter Name", default="")
    direction: StringProperty(name="Direction", default="UP")

    def execute(self, context):
        ok, msg = apple_bite.move_cutter_layer(context, self.cutter_name, self.direction)
        _set_status(context, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_reorder_cutters(bpy.types.Operator):
    bl_idname = "fdt.applebite_reorder_cutters"
    bl_label = "Reorder Cutter Layers"
    bl_description = "Apply a new cutter layer order (from drag-and-drop) and rebuild cuts"
    bl_options = {"REGISTER", "UNDO"}

    order_json: StringProperty(name="Order JSON", default="[]")

    def execute(self, context):
        try:
            names = json.loads(self.order_json)
            if not isinstance(names, list):
                return {"CANCELLED"}
        except Exception:
            return {"CANCELLED"}
        ok, msg = apple_bite.reorder_cutter_layers(context, [str(n) for n in names])
        _set_status(context, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_apply(bpy.types.Operator):
    bl_idname = "fdt.applebite_apply"
    bl_label = "APPLY APPLE BITE"
    bl_description = "Bake cut main mesh and/or zone-tagged pieces cleanly (overwrites old Final on re-edit)"
    bl_options = {"REGISTER", "UNDO"}

    keep_result: BoolProperty(name="Keep Cut Main Mesh", default=True, options={"SKIP_SAVE"})
    keep_pieces: BoolProperty(name="Keep Cut Pieces", default=True, options={"SKIP_SAVE"})
    use_scene_options: BoolProperty(name="Use Scene Options", default=True, options={"SKIP_SAVE"})

    def execute(self, context):
        _remove_hud_handler()
        settings = getattr(context.scene, "fdt_settings", None)
        if self.use_scene_options and settings is not None:
            keep_res = bool(settings.apply_keep_result)
            keep_pcs = bool(settings.apply_keep_pieces)
        else:
            keep_res = bool(self.keep_result)
            keep_pcs = bool(self.keep_pieces)

        ok, msg, _baked = apple_bite.apply_session(context, keep_result=keep_res, keep_pieces=keep_pcs)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_interactive(bpy.types.Operator):
    bl_idname = "fdt.applebite_interactive"
    bl_label = "จิ้มวางบนผิว (Ctrl+คลิกขวา)"
    bl_description = "Ctrl+RMB on any wall surface to place a Box Cutter aligned to the surface; ESC to stop"

    def modal(self, context, event):
        global _interactive_active
        if not _interactive_active or not has_active_session(context.scene):
            _remove_hud_handler()
            if context.area:
                context.area.tag_redraw()
            return {"FINISHED"}

        if event.type == "ESC" and event.value == "PRESS":
            _remove_hud_handler()
            _set_status(context, "หยุดโหมดจิ้มวางแล้ว (ยังอยู่ในงานตัด ปรับกล่องต่อหรือกด Apply ได้)")
            if context.area:
                context.area.tag_redraw()
            return {"FINISHED"}

        if event.type == "RIGHTMOUSE" and event.value == "PRESS" and event.ctrl:
            region = context.region
            rv3d = context.region_data
            if region is not None and rv3d is not None:
                coord = (event.mouse_region_x, event.mouse_region_y)
                ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
                depsgraph = context.evaluated_depsgraph_get()
                hit, loc, norm, _idx, hit_obj, _mat = context.scene.ray_cast(depsgraph, ray_origin, view_vec)
                if hit:
                    settings = getattr(context.scene, "fdt_settings", None)
                    zone = settings.active_zone if settings else "A"
                    if event.shift and hit_obj is not None:
                        ok, msg, _ = apple_bite.add_custom_mesh_cutters(context, [hit_obj], zone=zone)
                    else:
                        size = settings.cutter_size if settings else 2.0
                        ok, msg, _ = apple_bite.add_box_cutter(context, location=loc, normal=norm, size=size, zone=zone)
                    _set_status(context, msg)
                    if context.area:
                        context.area.tag_redraw()
                    return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        global _hud_draw_handle, _interactive_active
        if not has_active_session(context.scene):
            ok, msg, _ = apple_bite.enter_session(context)
            _set_status(context, msg)
            if not ok:
                self.report({"WARNING"}, msg)
                return {"CANCELLED"}

        if _interactive_active:
            _remove_hud_handler()
            _set_status(context, "ปิดโหมดจิ้มวางแล้ว")
            return {"FINISHED"}

        _ensure_wire_overlay()
        _interactive_active = True
        _hud_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud_callback, (), "WINDOW", "POST_PIXEL"
        )
        context.window_manager.modal_handler_add(self)
        _set_status(context, "เปิดโหมดจิ้มวางแล้ว: กด Ctrl+คลิกขวา บนผิวตึกเพื่อวางกล่อง (ESC เพื่อหยุด)")
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


class FDT_OT_applebite_assign_cut_material(bpy.types.Operator):
    bl_idname = "fdt.applebite_assign_cut_material"
    bl_label = "Apply Cut Material to Selected"
    bl_description = "Assign the selected Cut Material to cut faces (fdt_orig < 0.5) of already-applied meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        mat = settings.cut_material if settings is not None else None
        targets = list(context.selected_objects)
        if not targets and context.active_object is not None:
            targets = [context.active_object]
        ok, msg, _count = apple_bite.assign_cut_material_to_objects(context, targets, mat)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_applebite_fill_cut_caps(bpy.types.Operator):
    bl_idname = "fdt.applebite_fill_cut_caps"
    bl_label = "Fill Cut Cap"
    bl_description = (
        "Fill unclosed cut-plane gaps with Cut Material, Box UV, and preserved normals. "
        "In Edit Mode fills selected open edges/loops; in Object Mode auto-scans cut planes"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ok, msg, _count = apple_bite.fill_cut_caps(context)
        _set_status(context, msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


_VIEW_MODE_LABELS = {
    "ALL": "All — ตึกหลัก + ชิ้นที่ตัด + กล่องตัด",
    "PIECES": "Piece Only — เฉพาะชิ้นที่ตัดตาม Zone ของกล่องที่เลือก",
    "MAIN": "Hole Only — ตึกหลักที่มีรู + กล่องตัด",
    "MAIN_WIRE": "Hole + Wire — ตึกหลัก solid + ชิ้นที่ตัดเป็น wireframe สีโซน",
}


class FDT_OT_applebite_set_view_mode(bpy.types.Operator):
    bl_idname = "fdt.applebite_set_view_mode"
    bl_label = "Set Preview View Mode"
    bl_description = "Instantly switch the 3D viewport view mode (0 ms, no Rust recompute)"
    bl_options = {"REGISTER"}

    mode: EnumProperty(
        name="View Mode",
        description="Which geometry to show while editing cutters",
        items=[
            ("ALL", "All", "Main building + cut pieces + cutters"),
            ("PIECES", "Piece Only", "Only the cut pieces of the selected cutter zone(s) + cutters"),
            ("MAIN", "Hole Only", "Only the main building with the hole + cutters"),
            ("MAIN_WIRE", "Hole + Wire", "Main building solid + removed pieces as zone-colored wireframe"),
        ],
        default="ALL",
    )

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        if settings is None:
            return {"CANCELLED"}
        mode = str(self.mode or "ALL").upper()
        if mode not in _VIEW_MODE_LABELS:
            return {"CANCELLED"}
        settings.preview_display_mode = mode
        _set_status(context, _VIEW_MODE_LABELS[mode])
        return {"FINISHED"}


CLASSES = (
    FDT_OT_applebite_smart_start,
    FDT_OT_applebite_reedit,
    FDT_OT_applebite_enter,
    FDT_OT_applebite_exit,
    FDT_OT_applebite_cancel,
    FDT_OT_applebite_reset_original,
    FDT_OT_applebite_add_box,
    FDT_OT_applebite_add_adjacent_box,
    FDT_OT_applebite_add_mesh,
    FDT_OT_applebite_draw_cutter,
    FDT_OT_applebite_draw_sculpt_cutter,
    FDT_OT_applebite_apply_sculpt_cutter,
    FDT_OT_applebite_set_zone,
    FDT_OT_applebite_select_cutter,
    FDT_OT_applebite_cycle_zone,
    FDT_OT_applebite_toggle_mute,
    FDT_OT_applebite_toggle_rough,
    FDT_OT_applebite_delete_cutter,
    FDT_OT_applebite_move_cutter,
    FDT_OT_applebite_reorder_cutters,
    FDT_OT_applebite_apply,
    FDT_OT_applebite_assign_cut_material,
    FDT_OT_applebite_fill_cut_caps,
    FDT_OT_applebite_set_view_mode,
    FDT_OT_applebite_interactive,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _ensure_wire_overlay()
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    global _zone_grid_shader, _zone_grid_shader_failed, _zone_grid_cache_sig, _zone_grid_cache_entries, _zone_grid_deps_rev
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _remove_hud_handler()
    _remove_wire_overlay()
    _zone_grid_shader = None
    _zone_grid_shader_failed = False
    _zone_grid_cache_sig = None
    _zone_grid_cache_entries = []
    _zone_grid_deps_rev = 0
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
