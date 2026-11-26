"""Step-by-Step Guided N-Panel for FullDestruc_Tool (English UI, Blender 5.2+)."""

from __future__ import annotations

import bpy

from ..core import apple_bite, rust_bridge, zone_registry
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
        for action, label, icon in (
            ("NEW", "New Zone", "ADD"),
            ("RENAME", "Rename", "GREASEPENCIL"),
            ("DELETE", "Delete", "TRASH"),
        ):
            op = zone_manage.operator("fdt.zone_manage", text=label, icon=icon)
            op.action = action
            op.zone_key = settings.active_zone

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

            # Live Cutter Layer List (Up/Down | Zone | Select | Rough | Mute | Delete)
            list_box = step2.box()
            list_box.label(text=f"Cutter Layers ({len(session.cutters)}) — Top = Priority:", icon="OUTLINER_OB_MESH")
            active_obj = context.active_object
            total_c = len(session.cutters)
            for idx, entry in enumerate(session.cutters):
                row = list_box.row(align=True)
                is_sel = active_obj is not None and active_obj.name == entry.name

                col_up = row.row(align=True)
                col_up.enabled = idx > 0
                op_up = col_up.operator("fdt.applebite_move_cutter", text="", icon="TRIA_UP")
                op_up.cutter_name = entry.name
                op_up.direction = "UP"

                col_dn = row.row(align=True)
                col_dn.enabled = idx < (total_c - 1)
                op_dn = col_dn.operator("fdt.applebite_move_cutter", text="", icon="TRIA_DOWN")
                op_dn.cutter_name = entry.name
                op_dn.direction = "DOWN"

                z_label = zone_registry.zone_token(entry.zone, context.scene)
                op_z = row.operator("fdt.applebite_cycle_zone", text=f"[{z_label[:12]}]")
                op_z.cutter_name = entry.name

                op_sel = row.operator(
                    "fdt.applebite_select_cutter",
                    text=entry.name,
                    depress=is_sel,
                )
                op_sel.cutter_name = entry.name

                op_rough = row.operator(
                    "fdt.applebite_toggle_rough",
                    text="",
                    icon="MOD_DISPLACE",
                    depress=entry.rough,
                )
                op_rough.cutter_name = entry.name

                op_mute = row.operator(
                    "fdt.applebite_toggle_mute",
                    text="",
                    icon="HIDE_ON" if entry.muted else "HIDE_OFF",
                    depress=entry.muted,
                )
                op_mute.cutter_name = entry.name

                op_del = row.operator("fdt.applebite_delete_cutter", text="", icon="X")
                op_del.cutter_name = entry.name

        # --- Step 3: Tune Cut & Apply ---
        if session.cutters:
            step3 = layout.box()
            step3.label(text="3. Tune & Apply", icon="CHECKMARK")

            step3.prop(settings, "enable_rough", text="Rough Cut Edge (All)")
            any_rough = settings.enable_rough or any(c.rough for c in session.cutters)
            if any_rough:
                rbox = step3.box()
                rbox.prop(settings, "rough_scale")
                rbox.prop(settings, "rough_strength")
                rbox.prop(settings, "rough_seed")

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


CLASSES = (FDT_OT_set_active_tool, FDT_PT_main_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
