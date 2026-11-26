"""Step-by-Step Guided N-Panel for FullDestruc_Tool (English UI, Blender 5.2+)."""

from __future__ import annotations

import bpy

from ..core import apple_bite, rename_tool, rust_bridge, zone_registry
from ..core.session import load_session
from ..ops import apple_bite_ops
from ..ops.material_picker import draw_cut_material_row
from .panel_rebar import draw as draw_rebar_section


# Zen UV-style tool rail.  The left column is the tool selector; the right
# column renders only the selected tool's detail.  Icons are checked against
# Blender 5.2 in tests/test_rebar_ops.py.
_TOOL_TABS = (
    ("APPLE_BITE", "MOD_BOOLEAN"),
    ("REBAR", "MOD_CURVE"),
    ("CLEANUP", "BRUSH_DATA"),
    ("BATCH", "MOD_ARRAY"),
)


def _draw_rename_box(layout: bpy.types.UILayout, context: bpy.types.Context, settings) -> None:
    """User-driven Object Rename box — always available, never tied to Zone Registry."""
    targets = rename_tool.get_target_objects(context)
    count = len(targets)

    box = layout.box()
    hdr = box.row(align=True)
    is_open = bool(getattr(settings, "ui_show_rename", True))
    hdr.prop(
        settings,
        "ui_show_rename",
        text=f"Rename Objects ({count})" if count else "Rename Objects",
        icon="TRIA_DOWN" if is_open else "TRIA_RIGHT",
        emboss=False,
    )
    if not is_open:
        return

    mode = str(getattr(settings, "rename_mode", "FULL") or "FULL")
    row_mode = box.row(align=True)
    row_mode.prop(settings, "rename_mode", expand=True)

    row_input = box.row(align=True)
    row_input.prop(settings, "rename_text", text="ID" if mode == "TAG" else "Name")
    row_input.operator("fdt.rename_pick_active", text="", icon="EYEDROPPER")

    row_opts = box.row(align=True)
    if mode == "FULL":
        row_opts.prop(settings, "rename_auto_number", text="Auto #")
        if bool(getattr(settings, "rename_auto_number", True)):
            row_opts.prop(settings, "rename_start_index", text="Start")
    row_opts.prop(settings, "rename_sync_mesh", text="Sync Mesh")

    raw_text = str(getattr(settings, "rename_text", "") or "").strip()
    preview_plan = (
        rename_tool.build_target_names(
            targets,
            mode=mode,
            text=raw_text,
            auto_number=bool(getattr(settings, "rename_auto_number", True)),
            start_index=int(getattr(settings, "rename_start_index", 1)),
        )
        if targets and raw_text
        else []
    )
    if not targets:
        box.label(text="Select object(s) in Viewport or Outliner", icon="INFO")
    elif preview_plan:
        first_obj, first_new = preview_plan[0]
        extra = f" (+{len(preview_plan) - 1})" if len(preview_plan) > 1 else ""
        box.label(text=f"-> {first_new}{extra}", icon="GREASEPENCIL")
    else:
        hint = "Type new ID (e.g. A, B, A#B)" if mode == "TAG" else "Type new object name"
        box.label(text=hint, icon="INFO")

    row_btn = box.row()
    row_btn.scale_y = 1.2
    row_btn.enabled = bool(preview_plan)
    btn_label = f"Rename ({count})" if count > 1 else "Rename"
    row_btn.operator("fdt.rename_objects", text=btn_label, icon="CHECKMARK")



_CUTTER_LIST_SYNC_PENDING = False


def _deferred_sync_cutter_list() -> float | None:
    global _CUTTER_LIST_SYNC_PENDING
    _CUTTER_LIST_SYNC_PENDING = False
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is not None:
        from ..props import sync_cutter_layer_items

        sync_cutter_layer_items(scene)
    wm = getattr(getattr(bpy, "context", None), "window_manager", None)
    if wm is not None:
        for win in wm.windows:
            screen = getattr(win, "screen", None)
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    return None


def _schedule_cutter_list_sync() -> None:
    """Defer syncing CollectionProperty outside Panel.draw() (e.g. legacy .blend files)."""
    global _CUTTER_LIST_SYNC_PENDING
    if _CUTTER_LIST_SYNC_PENDING:
        return
    _CUTTER_LIST_SYNC_PENDING = True
    try:
        bpy.app.timers.register(_deferred_sync_cutter_list, first_interval=0.0)
    except Exception:
        _CUTTER_LIST_SYNC_PENDING = False


@bpy.app.handlers.persistent
def _on_cutter_list_handler(_dummy=None) -> None:
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is not None:
        from ..props import sync_cutter_layer_items

        sync_cutter_layer_items(scene)


class FDT_UL_cutter_layers(bpy.types.UIList):
    """Compact scrollable Cutter Layers list with inline Zone, Rough & Mute controls."""

    bl_idname = "FDT_UL_cutter_layers"

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_propname,
        index=0,
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)

            z_col = row.row(align=True)
            z_col.ui_units_x = 2.6
            z_label = zone_registry.zone_token(item.zone, context.scene)
            op_z = z_col.operator("fdt.applebite_cycle_zone", text=f"[{z_label[:10]}]")
            op_z.cutter_name = item.name

            lbl_col = row.row(align=True)
            lbl_col.active = not bool(item.muted)
            lbl_col.label(text=item.name)

            act = row.row(align=True)
            op_rough = act.operator(
                "fdt.applebite_toggle_rough",
                text="",
                icon="MOD_DISPLACE",
                depress=bool(item.rough),
                emboss=bool(item.rough),
            )
            op_rough.cutter_name = item.name

            op_mute = act.operator(
                "fdt.applebite_toggle_mute",
                text="",
                icon="HIDE_ON" if item.muted else "HIDE_OFF",
                depress=bool(item.muted),
                emboss=bool(item.muted),
            )
            op_mute.cutter_name = item.name
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


class FDT_OT_set_active_tool(bpy.types.Operator):
    bl_idname = "fdt.set_active_tool"
    bl_label = "Select Tool"
    bl_description = "เลือกเครื่องมือที่แสดงรายละเอียดด้านขวา — เช่น Apple Bite สำหรับตัดผิว หรือ Rebar สำหรับใส่เหล็ก"
    bl_options = {"REGISTER"}

    tool: bpy.props.EnumProperty(
        name="Tool",
        items=[
            ("APPLE_BITE", "Apple Bite", "Cut & Zones"),
            ("REBAR", "Rebar", "Concrete Rebars"),
            ("CLEANUP", "Cleanup", "Cleanup tools"),
            ("BATCH", "Batch", "Reserved"),
        ],
        default="APPLE_BITE",
    )

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        if settings is not None:
            settings.ui_active_tool = self.tool
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {"FINISHED"}


class FDT_PT_main_panel(bpy.types.Panel):
    bl_label = "FullDestruc Tool"
    bl_idname = "FDT_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FullDestruc"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.fdt_settings
        session = load_session(scene)

        # Quick UI launcher bar
        top_row = layout.row(align=True)
        top_row.operator("fdt.open_qt_window", text="Python Window", icon="WINDOW")
        top_row.operator("fdt.quick_popup", text="Popup", icon="PREFERENCES")
        top_row.operator("fdt.open_debug_window", text="Debug", icon="SYSTEM")

        # Zen UV-style body: tool icons on the left, that tool's detail right.
        body = layout.split(factor=0.16)
        rail = body.column(align=True)
        detail = body.column()

        for tool_id, icon in _TOOL_TABS:
            op = rail.operator(
                "fdt.set_active_tool",
                text="",
                icon=icon,
                depress=settings.ui_active_tool == tool_id,
            )
            op.tool = tool_id

        active = settings.ui_active_tool
        if active == "APPLE_BITE":
            # The boolean engine is Apple Bite's own setting, so it lives in
            # this detail area instead of floating above every tool tab.
            eng_box = detail.box()
            eng_row = eng_box.row(align=True)
            eng_row.prop(settings, "boolean_solver", text="Engine")
            if settings.boolean_solver == "RUST":
                if rust_bridge.is_available():
                    eng_box.prop(settings, "gpu_fast_drag", text="GPU Fast Drag (60+ FPS)", icon="SHADING_RENDERED")
                    eng_box.label(text="Rust Engine v1.5: CDT + GPU Ready", icon="CHECKMARK")
                else:
                    eng_box.label(text="Rust DLL missing — using Blender Exact", icon="ERROR")

            box_dst = detail.box()
            if session is None:
                self._draw_step1_idle(box_dst, context)
            else:
                self._draw_step2_and_step3_active(box_dst, context, settings, session)
        elif active == "REBAR":
            # The tab itself is the top-level switch — show the detail directly.
            draw_rebar_section(detail, context, force_open=True)
        elif active == "CLEANUP":
            box_clean = detail.box()
            box_clean.label(text="Find Texture / Del by Material / Replace / Fix Slots", icon="INFO")
        elif active == "BATCH":
            box_batch = detail.box()
            box_batch.label(text="Reserved slot for future Batch Pipeline", icon="TIME")

        _draw_rename_box(detail, context, settings)

    def _draw_step1_idle(self, layout: bpy.types.UILayout, context: bpy.types.Context) -> None:
        info = apple_bite.detect_step1_action(context)

        # Guide banner
        guide = layout.box()
        icon = "INFO" if info["mode"] == "NONE" else ("LOOP_BACK" if info["mode"] in ("REEDIT", "BOTH") else "CHECKMARK")
        guide.label(text=info["guide_text"], icon=icon)

        # Primary button
        row = layout.row()
        row.scale_y = 1.4
        row.enabled = info["mode"] != "NONE"
        btn_icon = "GREASEPENCIL" if info["mode"] == "REEDIT" else "MOD_BOOLEAN"
        row.operator("fdt.applebite_smart_start", text=info["button_label"], icon=btn_icon)

        # Secondary button when clicking an already-applied piece: Re-edit old cut
        if info["mode"] == "BOTH" and info["session_id"]:
            row2 = layout.row()
            op = row2.operator("fdt.applebite_reedit", text=info["reedit_label"], icon="LOOP_BACK")
            op.session_id = info["session_id"]

        # Reset button: remove all related cuts and restore the root original mesh
        show_applied_tools = info["mode"] in ("BOTH", "REEDIT") and bool(info["session_id"])
        if show_applied_tools:
            row3 = layout.row()
            row3.alert = True
            op_reset = row3.operator(
                "fdt.applebite_reset_original",
                text=info["reset_label"],
                icon="TRASH",
            )
            op_reset.session_id = info["session_id"]

        active_obj = context.active_object
        has_mesh = (active_obj is not None and active_obj.type == "MESH") or any(
            o.type == "MESH" for o in context.selected_objects
        )
        if show_applied_tools or has_mesh:
            settings = context.scene.fdt_settings
            mat_box = layout.box()
            mat_box.label(
                text="Cut Face Material & Cap Fill:" if show_applied_tools else "Cut Cap Fill:",
                icon="MATERIAL",
            )
            draw_cut_material_row(mat_box, settings, text="Mat")
            if show_applied_tools:
                row_am = mat_box.row()
                row_am.enabled = settings.cut_material is not None
                row_am.operator("fdt.applebite_assign_cut_material", text="Apply Cut Material", icon="BRUSH_DATA")
            if has_mesh:
                row_fill = mat_box.row()
                fill_lbl = (
                    "Fill Selected Cut Cap"
                    if context.mode == "EDIT_MESH"
                    else "Fill Cut Cap (Auto)"
                )
                row_fill.operator("fdt.applebite_fill_cut_caps", text=fill_lbl, icon="FACESEL")

    def _draw_step2_and_step3_active(
        self,
        layout: bpy.types.UILayout,
        context: bpy.types.Context,
        settings,
        session,
    ) -> None:
        # Dynamic Guide Banner
        guide = layout.box()
        if not session.cutters:
            guide.label(text="Step 2: Click '+ Add Box' or 'Click Place (Ctrl+RMB)'", icon="INFO")
        else:
            guide.label(text="Step 2: Move/Scale box (G/R/S) or click Gear icon on box", icon="PLAY")
        guide.label(text=f"Session: {session.session_id}  |  Targets: {len(session.target_names)}", icon="HOME")

        # --- Step 2: Place & Manage Cutters ---
        step2 = layout.box()
        step2.label(text="2. Place & Manage Cutters", icon="MESH_CUBE")

        # Zone quick selector + searchable dropdown backed by the Scene zone registry.
        grid = step2.grid_flow(row_major=True, columns=4, even_columns=True, align=True)
        for zdef in zone_registry.load_zones(context.scene)[:8]:
            op = grid.operator(
                "fdt.applebite_set_zone",
                text=zdef.token,
                depress=(settings.active_zone == zdef.key),
            )
            op.zone = zdef.key
        step2.prop(settings, "active_zone", text="Zone (type to search)")
        zone_manage = step2.row(align=True)
        op_new = zone_manage.operator("fdt.zone_manage", text="New Zone", icon="ADD")
        op_new.action = "NEW"
        op_new.zone_key = settings.active_zone
        zone_manage.operator("fdt.rename_popup", text="Rename", icon="GREASEPENCIL")
        op_del_z = zone_manage.operator("fdt.zone_manage", text="Delete", icon="TRASH")
        op_del_z.action = "DELETE"
        op_del_z.zone_key = settings.active_zone

        row_add = step2.row(align=True)
        row_add.scale_y = 1.2
        row_add.operator("fdt.applebite_add_box", text="+ Add Box", icon="MESH_CUBE")
        row_add.operator("fdt.applebite_add_mesh", text="+ Use Mesh", icon="MOD_SOLIDIFY")

        row_draw = step2.row(align=True)
        row_draw.scale_y = 1.25
        row_draw.operator("fdt.applebite_draw_cutter", text="Draw Cutter (2D -> Extrude)", icon="GREASEPENCIL")

        is_placing = bool(apple_bite_ops._interactive_active)
        step2.operator(
            "fdt.applebite_interactive",
            text="Stop Click Place (ESC)" if is_placing else "Click Place (Ctrl+RMB)",
            icon="CURSOR",
            depress=is_placing,
        )

        if session.cutters:
            row_view = step2.row(align=True)
            row_view.prop(settings, "preview_display_mode", expand=True)

            row_grid = step2.row(align=True)
            row_grid.prop(settings, "show_zone_grid", text="Zone Grid", icon="MESH_GRID", toggle=True)
            if settings.show_zone_grid:
                row_grid.prop(settings, "zone_grid_size", text="Size")
            step2.label(text="Tip: Click Gear icon on box for Radial Wheel & 3D Axis", icon="PREFERENCES")

            # Compact collapsible & scrollable Cutter Layers (UIList + Up/Down/Delete sidebar)
            list_box = step2.box()
            total_c = len(session.cutters)
            show_layers = bool(getattr(settings, "ui_show_cutter_layers", True))
            hdr = list_box.row(align=True)
            hdr.prop(
                settings,
                "ui_show_cutter_layers",
                text=f"Cutter Layers ({total_c}) — Top = Priority",
                icon="TRIA_DOWN" if show_layers else "TRIA_RIGHT",
                emboss=False,
            )
            if show_layers:
                items = getattr(settings, "cutter_layer_items", None)
                if items is not None:
                    cur_sig = [(it.name, it.zone, bool(it.muted), bool(it.rough)) for it in items]
                    want_sig = [(c.name, c.zone, bool(c.muted), bool(c.rough)) for c in session.cutters]
                    if cur_sig != want_sig:
                        _schedule_cutter_list_sync()

                list_row = list_box.row(align=True)
                if hasattr(list_row, "template_list"):
                    list_row.template_list(
                        "FDT_UL_cutter_layers",
                        "",
                        settings,
                        "cutter_layer_items",
                        settings,
                        "cutter_layer_index",
                        rows=min(5, max(2, total_c)),
                        maxrows=8,
                    )

                active_idx = int(getattr(settings, "cutter_layer_index", 0))
                if active_idx < 0 or active_idx >= total_c:
                    active_obj = context.active_object
                    active_idx = next(
                        (i for i, c in enumerate(session.cutters) if active_obj is not None and c.name == active_obj.name),
                        0,
                    )
                sel_entry = session.cutters[active_idx] if 0 <= active_idx < total_c else None

                side = list_row.column(align=True)
                col_up = side.row(align=True)
                col_up.enabled = sel_entry is not None and active_idx > 0
                op_up = col_up.operator("fdt.applebite_move_cutter", text="", icon="TRIA_UP")
                if sel_entry is not None:
                    op_up.cutter_name = sel_entry.name
                op_up.direction = "UP"

                col_dn = side.row(align=True)
                col_dn.enabled = sel_entry is not None and active_idx < (total_c - 1)
                op_dn = col_dn.operator("fdt.applebite_move_cutter", text="", icon="TRIA_DOWN")
                if sel_entry is not None:
                    op_dn.cutter_name = sel_entry.name
                op_dn.direction = "DOWN"

                col_del = side.row(align=True)
                col_del.enabled = sel_entry is not None
                op_del = col_del.operator("fdt.applebite_delete_cutter", text="", icon="X")
                if sel_entry is not None:
                    op_del.cutter_name = sel_entry.name

        # --- Step 3: Tune Cut & Apply ---
        if session.cutters:
            step3 = layout.box()
            step3.label(text="3. Tune & Apply", icon="CHECKMARK")

            step3.prop(settings, "enable_rough", text="Rough Cut Edge (All)")
            target_cutter, target_is_rough = apple_bite.get_active_rough_cutter(context, session)
            any_rough = settings.enable_rough or any(c.rough for c in session.cutters)
            if any_rough:
                rbox = step3.box()
                cr = getattr(target_cutter, "fdt_rough", None) if target_cutter is not None else None
                if cr is not None:
                    status = "" if target_is_rough else " (OFF)"
                    rbox.label(text=f"Cutter: {target_cutter.name}{status}", icon="MOD_DISPLACE")
                    rbox.prop(cr, "inward_depth", text="Inward Depth", slider=True)
                    rbox.prop(cr, "jaggedness", text="Jaggedness", slider=True)
                    rbox.prop(cr, "noise_scale", text="Noise Scale")
                    rbox.prop(cr, "detail", text="Detail (1-3)")
                    rbox.prop(cr, "seed", text="Seed")
                else:
                    rbox.label(text="Default Rough (New Cutters)", icon="MOD_DISPLACE")
                    rbox.prop(settings, "rough_inward_ratio", text="Inward Depth", slider=True)
                    rbox.prop(settings, "rough_jaggedness", text="Jaggedness", slider=True)
                    rbox.prop(settings, "rough_scale", text="Noise Scale")
                    rbox.prop(settings, "rough_subdiv", text="Detail (1-3)")
                    rbox.prop(settings, "rough_seed", text="Seed")

            draw_cut_material_row(step3, settings, text="Cut Material")
            row_uv = step3.row(align=True)
            row_uv.prop(settings, "texel_per_meter", text="Texel")
            row_uv.prop(settings, "texture_size", text="Tex Size")

            row_keep = step3.row(align=True)
            row_keep.prop(settings, "apply_keep_result", text="Main Mesh")
            row_keep.prop(settings, "apply_keep_pieces", text="Zone Pieces")

            step3.prop(settings, "apply_auto_fill_caps", text="Auto-Fill Cut Gaps", icon="FACESEL")
            step3.prop(settings, "apply_line_backfaces", text="Close See-Through Backfaces", icon="MOD_SOLIDIFY")

            summary = apple_bite.get_apply_preview_summary(context.scene)
            zone_str = ", ".join(
                f"{zone_registry.zone_token(z, context.scene)}({n})"
                for z, n in sorted(summary["zone_counts"].items())
            ) or "None"
            sum_box = step3.box()
            sum_box.label(
                text=f"Output: Main {summary['main_count']} | Zones: {zone_str}",
                icon="OUTLINER_COLLECTION",
            )

            row_apply = step3.row()
            row_apply.scale_y = 1.4
            row_apply.operator("fdt.applebite_apply", text="APPLY", icon="CHECKMARK")

        # Footer: Pause / Cancel
        row_end = layout.row(align=True)
        row_end.operator("fdt.applebite_exit", text="Pause / Exit", icon="PAUSE")
        row_end.operator("fdt.applebite_cancel", text="Cancel & Restore", icon="LOOP_BACK")


CLASSES = (FDT_OT_set_active_tool, FDT_UL_cutter_layers, FDT_PT_main_panel)

_HANDLER_ATTRS = ("load_post", "undo_post", "redo_post")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is not None:
        for attr in _HANDLER_ATTRS:
            hlist = getattr(handlers, attr, None)
            if hlist is not None and _on_cutter_list_handler not in hlist:
                hlist.append(_on_cutter_list_handler)


def unregister():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is not None:
        for attr in _HANDLER_ATTRS:
            hlist = getattr(handlers, attr, None)
            if hlist is not None and _on_cutter_list_handler in hlist:
                hlist.remove(_on_cutter_list_handler)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
