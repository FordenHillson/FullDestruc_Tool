"""Minimal, Single-Screen Guided N-Panel with Popovers & Colored Icons (Blender 5.2+)."""

from __future__ import annotations

import bpy

from ..core import apple_bite, rust_bridge, zone_registry
from ..core.session import load_session
from ..ops import apple_bite_ops
from ..ops.material_picker import draw_cut_material_row
from . import icons
from .panel_rebar import draw as draw_rebar_section


# Zen UV-style left tool rail: (tool_id, custom_icon_name, blender_fallback_icon)
_TOOL_TABS: tuple[tuple[str, str, str], ...] = (
    ("APPLE_BITE", "tool_apple_bite", "MOD_BOOLEAN"),
    ("REBAR", "tool_rebar", "MOD_CURVE"),
    ("CLEANUP", "tool_cleanup", "BRUSH_DATA"),
    ("BATCH", "tool_batch", "MOD_ARRAY"),
)


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


def _draw_rough_sliders(layout: bpy.types.UILayout, context: bpy.types.Context, settings, session) -> None:
    """Shared Rough Cut Edge tuning block (used inside FDT_PT_popover_rough)."""
    target_cutter, target_is_rough = apple_bite.get_active_rough_cutter(context, session)
    cr = getattr(target_cutter, "fdt_rough", None) if target_cutter is not None else None
    if cr is not None:
        status = "" if target_is_rough else " (OFF)"
        layout.label(text=f"{target_cutter.name}{status}", **icons.icon_kw("rough_edge", "MOD_DISPLACE"))
        layout.prop(cr, "inward_depth", text="Inward Depth", slider=True)
        layout.prop(cr, "jaggedness", text="Jaggedness", slider=True)
        layout.prop(cr, "noise_scale", text="Noise Scale")
        layout.prop(cr, "detail", text="Detail (1-3)")
        layout.prop(cr, "seed", text="Seed")
    else:
        layout.label(text="Default (New Cutters)", **icons.icon_kw("rough_edge", "MOD_DISPLACE"))
        layout.prop(settings, "rough_inward_ratio", text="Inward Depth", slider=True)
        layout.prop(settings, "rough_jaggedness", text="Jaggedness", slider=True)
        layout.prop(settings, "rough_scale", text="Noise Scale")
        layout.prop(settings, "rough_subdiv", text="Detail (1-3)")
        layout.prop(settings, "rough_seed", text="Seed")


# ---------------------------------------------------------------------------
# Compact Popover Panels (keep N-Panel single-screen & clutter-free)
# ---------------------------------------------------------------------------


class FDT_PT_popover_settings(bpy.types.Panel):
    """Engine, GPU Fast Drag, Viewport Dock toggle & utility launchers."""

    bl_label = "Engine & Settings"
    bl_idname = "FDT_PT_popover_settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 13

    def draw(self, context):
        layout = self.layout
        settings = context.scene.fdt_settings

        layout.label(text="Boolean Engine", **icons.icon_kw("settings_gear", "PREFERENCES"))
        layout.prop(settings, "boolean_solver", text="")
        if settings.boolean_solver == "RUST":
            if rust_bridge.is_available():
                layout.prop(settings, "gpu_fast_drag", text="GPU Fast Drag (60+ FPS)", icon="SHADING_RENDERED")
                layout.label(text="Rust v1.5: CDT + GPU Ready", **icons.icon_kw("apply_green", "CHECKMARK"))
            else:
                layout.label(text="Rust DLL missing — Blender Exact", icon="ERROR")

        layout.separator()
        layout.prop(settings, "show_viewport_dock", text="3D Viewport Dock")
        layout.prop(settings, "cutter_size", text="Default Box Size (m)")

        layout.separator()
        row = layout.row(align=True)
        row.operator("fdt.quick_popup", text="Quick Tune", icon="PREFERENCES")
        row.operator("fdt.open_debug_window", text="Debug", icon="SYSTEM")


class FDT_PT_popover_zones(bpy.types.Panel):
    """Quick A-H Zone Grid + clearly labeled Zone Registry management."""

    bl_label = "Zones"
    bl_idname = "FDT_PT_popover_zones"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 13

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.fdt_settings

        layout.label(text="Quick Zones", **icons.icon_kw("zones_grid", "COLOR"))
        grid = layout.grid_flow(row_major=True, columns=4, even_columns=True, align=True)
        for zdef in zone_registry.load_zones(scene)[:8]:
            op = grid.operator(
                "fdt.applebite_set_zone",
                text=zdef.token[:8],
                depress=(settings.active_zone == zdef.key),
            )
            op.zone = zdef.key

        layout.separator()
        layout.label(text="Manage Zone Registry")
        row = layout.row(align=True)
        op_new = row.operator("fdt.zone_manage", text="New", **icons.icon_kw("plus_add", "ADD"))
        op_new.action = "NEW"
        op_new.zone_key = settings.active_zone

        op_edit = row.operator("fdt.zone_manage", text="Edit", **icons.icon_kw("rename_pen", "GREASEPENCIL"))
        op_edit.action = "RENAME"
        op_edit.zone_key = settings.active_zone

        op_del = row.operator("fdt.zone_manage", text="Delete", **icons.icon_kw("trash_red", "TRASH"))
        op_del.action = "DELETE"
        op_del.zone_key = settings.active_zone


class FDT_PT_popover_rough(bpy.types.Panel):
    """Rough Cut Edge toggle & fracture parameter sliders."""

    bl_label = "Rough Edge"
    bl_idname = "FDT_PT_popover_rough"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 13

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.fdt_settings
        session = load_session(scene)

        layout.prop(settings, "enable_rough", text="Rough Cut Edge (All)")
        layout.separator()
        _draw_rough_sliders(layout, context, settings, session)


class FDT_PT_popover_output(bpy.types.Panel):
    """UV Texel, Output Keep flags, Cap Fill, Backface Liner & Object Rename."""

    bl_label = "Output & UV"
    bl_idname = "FDT_PT_popover_output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 13

    def draw(self, context):
        layout = self.layout
        settings = context.scene.fdt_settings

        layout.label(text="Cut UV Scale", **icons.icon_kw("output_box", "UV"))
        row_uv = layout.row(align=True)
        row_uv.prop(settings, "texel_per_meter", text="Texel")
        row_uv.prop(settings, "texture_size", text="Size")

        layout.separator()
        layout.label(text="Apply Output")
        row_keep = layout.row(align=True)
        row_keep.prop(settings, "apply_keep_result", text="Main Mesh", toggle=True)
        row_keep.prop(settings, "apply_keep_pieces", text="Zone Pieces", toggle=True)

        layout.prop(settings, "apply_auto_fill_caps", text="Auto-Fill Cut Gaps", **icons.icon_kw("fill_cap", "FACESEL"))
        layout.prop(settings, "apply_line_backfaces", text="Close See-Through Backfaces", icon="MOD_SOLIDIFY")

        layout.separator()
        tools_row = layout.row(align=True)
        tools_row.operator("fdt.applebite_fill_cut_caps", text="Fill Cap", **icons.icon_kw("fill_cap", "FACESEL"))
        tools_row.operator("fdt.rename_popup", text="Rename Obj", **icons.icon_kw("rename_pen", "GREASEPENCIL"))


# ---------------------------------------------------------------------------
# Scrollable Cutter Layers UIList & Tool Switcher
# ---------------------------------------------------------------------------


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
                depress=bool(item.rough),
                emboss=bool(item.rough),
                **(icons.icon_kw("rough_edge", "MOD_DISPLACE") if item.rough else {"icon": "MOD_DISPLACE"}),
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


# ---------------------------------------------------------------------------
# Main Minimal N-Panel
# ---------------------------------------------------------------------------


class FDT_PT_main_panel(bpy.types.Panel):
    bl_label = "FullDestruc"
    bl_idname = "FDT_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FullDestruc"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.fdt_settings
        session = load_session(scene)

        # Zen UV-style left vertical tool rail with colored flat icons
        body = layout.split(factor=0.16)
        rail = body.column(align=True)
        detail = body.column()

        for tool_id, custom_ico, fallback_ico in _TOOL_TABS:
            btn_row = rail.row(align=True)
            btn_row.scale_y = 1.25
            op = btn_row.operator(
                "fdt.set_active_tool",
                text="",
                depress=(settings.ui_active_tool == tool_id),
                **icons.icon_kw(custom_ico, fallback_ico),
            )
            op.tool = tool_id

        active = settings.ui_active_tool
        if active == "APPLE_BITE":
            # Slim 2-icon header: Title + status | [❐ Window] [⚙ Engine/Settings]
            hdr = detail.row(align=True)
            if session is not None:
                valid_count = len(session.get_valid_cutters(include_muted=True))
                hdr.label(text=f"Apple Bite · {valid_count}", **icons.icon_kw("tool_apple_bite", "MOD_BOOLEAN"))
            else:
                hdr.label(text="Apple Bite", **icons.icon_kw("tool_apple_bite", "MOD_BOOLEAN"))
            hdr.operator("fdt.open_qt_window", text="", **icons.icon_kw("window_qt", "WINDOW"))
            hdr.popover(
                panel=FDT_PT_popover_settings.bl_idname,
                text="",
                **icons.icon_kw("settings_gear", "PREFERENCES"),
            )

            box_dst = detail.box()
            if session is None:
                self._draw_step1_idle(box_dst, context)
            else:
                self._draw_step2_and_step3_active(box_dst, context, settings, session)

        elif active == "REBAR":
            hdr = detail.row(align=True)
            hdr.label(text="Concrete Rebars", **icons.icon_kw("tool_rebar", "MOD_CURVE"))
            hdr.operator("fdt.open_qt_window", text="", **icons.icon_kw("window_qt", "WINDOW"))
            draw_rebar_section(detail, context, force_open=True)

        elif active == "CLEANUP":
            hdr = detail.row(align=True)
            hdr.label(text="Cleanup", **icons.icon_kw("tool_cleanup", "BRUSH_DATA"))
            box_clean = detail.box()
            box_clean.label(text="Find Texture / Del by Mat / Replace / Fix Slots", icon="INFO")

        elif active == "BATCH":
            hdr = detail.row(align=True)
            hdr.label(text="Batch", **icons.icon_kw("tool_batch", "MOD_ARRAY"))
            box_batch = detail.box()
            box_batch.label(text="Reserved for future Batch Pipeline", icon="TIME")

    def _draw_step1_idle(self, layout: bpy.types.UILayout, context: bpy.types.Context) -> None:
        info = apple_bite.detect_step1_action(context)
        mode = info["mode"]

        # Concise 1-line contextual status
        if mode == "NONE":
            layout.label(text="Select a mesh to start", icon="INFO")
        elif mode == "START":
            names = ", ".join(o.name for o in info["targets"][:2])
            if len(info["targets"]) > 2:
                names += f" +{len(info['targets']) - 2}"
            layout.label(text=f"Ready · {names}", **icons.icon_kw("apply_green", "CHECKMARK"))
        elif mode == "REEDIT":
            layout.label(text=f"Paused · {info['session_id']}", **icons.icon_kw("reedit_cyan", "LOOP_BACK"))
        elif mode == "BOTH":
            layout.label(text=f"Baked · {info['session_id']}", **icons.icon_kw("reedit_cyan", "LOOP_BACK"))

        # Primary / Round-trip Actions
        if mode == "BOTH" and info["session_id"]:
            # Clearly distinguish Re-edit (cyan) vs Cut Further (amber) to prevent accidental nested cuts
            pair = layout.row(align=True)
            pair.scale_y = 1.45
            op_re = pair.operator(
                "fdt.applebite_reedit",
                text="Re-edit Cut",
                **icons.icon_kw("reedit_cyan", "LOOP_BACK"),
            )
            op_re.session_id = info["session_id"]
            pair.operator(
                "fdt.applebite_smart_start",
                text="Cut Further",
                **icons.icon_kw("start_orange", "MOD_BOOLEAN"),
            )
        else:
            row = layout.row()
            row.scale_y = 1.55
            row.enabled = mode != "NONE"
            btn_ico = (
                icons.icon_kw("reedit_cyan", "LOOP_BACK")
                if mode == "REEDIT"
                else icons.icon_kw("start_orange", "MOD_BOOLEAN")
            )
            btn_txt = "Resume Cut" if mode == "REEDIT" else ("Start Cut" if mode == "START" else "Select Mesh")
            row.operator("fdt.applebite_smart_start", text=btn_txt, **btn_ico)

        # Destructive Reset (only on already-cut pieces)
        show_applied_tools = mode in ("BOTH", "REEDIT") and bool(info["session_id"])
        if show_applied_tools:
            row_reset = layout.row()
            row_reset.alert = True
            op_reset = row_reset.operator(
                "fdt.applebite_reset_original",
                text="Reset to Original",
                **icons.icon_kw("trash_red", "TRASH"),
            )
            op_reset.session_id = info["session_id"]

        # Compact Post-Cut Tools
        active_obj = context.active_object
        has_mesh = (active_obj is not None and active_obj.type == "MESH") or any(
            o.type == "MESH" for o in context.selected_objects
        )
        if show_applied_tools or has_mesh:
            settings = context.scene.fdt_settings
            layout.separator(factor=0.6)
            draw_cut_material_row(layout, settings, text="Mat")
            tools = layout.row(align=True)
            if show_applied_tools:
                sub_am = tools.row(align=True)
                sub_am.enabled = settings.cut_material is not None
                sub_am.operator("fdt.applebite_assign_cut_material", text="Apply Mat", icon="BRUSH_DATA")
            if has_mesh:
                fill_lbl = "Fill Cap" if context.mode != "EDIT_MESH" else "Fill Sel"
                tools.operator(
                    "fdt.applebite_fill_cut_caps",
                    text=fill_lbl,
                    **icons.icon_kw("fill_cap", "FACESEL"),
                )
                tools.operator(
                    "fdt.rename_popup",
                    text="Rename",
                    **icons.icon_kw("rename_pen", "GREASEPENCIL"),
                )

    def _draw_step2_and_step3_active(
        self,
        layout: bpy.types.UILayout,
        context: bpy.types.Context,
        settings,
        session,
    ) -> None:
        # Count ONLY valid existing cutters (handles manual X-key deletes cleanly)
        valid_all = session.get_valid_cutters(include_muted=True)
        valid_active = session.get_valid_cutters(include_muted=False)
        total_c = len(valid_all)
        active_c = len(valid_active)

        # 1. Clean 1-Row Zone Strip: [● Color Dot] [Zone Dropdown] [⊞ Zones Popover]
        z_row = layout.row(align=True)
        z_row.label(text="", **icons.get_zone_dot_icon_kw(settings.active_zone, context.scene))
        z_row.prop(settings, "active_zone", text="")
        z_row.popover(
            panel=FDT_PT_popover_zones.bl_idname,
            text="",
            **icons.icon_kw("zones_grid", "COLOR"),
        )

        # 2. Compact 2x2 Add Cutters Grid
        add_r1 = layout.row(align=True)
        add_r1.scale_y = 1.25
        add_r1.operator("fdt.applebite_add_box", text="+ Box", **icons.icon_kw("add_box", "MESH_CUBE"))
        add_r1.operator("fdt.applebite_draw_cutter", text="Draw", **icons.icon_kw("draw_pen", "GREASEPENCIL"))

        add_r2 = layout.row(align=True)
        is_placing = bool(apple_bite_ops._interactive_active)
        add_r2.operator(
            "fdt.applebite_interactive",
            text="Stop Click" if is_placing else "Click Place",
            depress=is_placing,
            **icons.icon_kw("click_place", "CURSOR"),
        )
        add_r2.operator("fdt.applebite_add_mesh", text="+ Mesh", **icons.icon_kw("use_mesh", "MOD_SOLIDIFY"))

        # Phase A: 0 valid cutters -> keep ultra-clean & guide next step
        if total_c == 0:
            layout.separator(factor=0.6)
            layout.label(text="Add a box or draw a cut in 3D View", icon="INFO")
            row_end = layout.row(align=True)
            row_end.operator("fdt.applebite_exit", text="Pause", **icons.icon_kw("pause_slate", "PAUSE"))
            row_end.operator("fdt.applebite_cancel", text="Cancel", **icons.icon_kw("cancel_red", "LOOP_BACK"))
            return

        # Phase B: >= 1 valid cutter
        layout.separator(factor=0.4)

        # 3. View Mode + Quick Grid / Rough Strip
        row_view = layout.row(align=True)
        row_view.prop(settings, "preview_display_mode", expand=True)

        row_quick = layout.row(align=True)
        row_quick.prop(
            settings,
            "show_zone_grid",
            text="Grid",
            toggle=True,
            **icons.icon_kw("zone_grid", "MESH_GRID"),
        )
        if settings.show_zone_grid:
            sub_sz = row_quick.row(align=True)
            sub_sz.scale_x = 0.65
            sub_sz.prop(settings, "zone_grid_size", text="")
        row_quick.prop(
            settings,
            "enable_rough",
            text="Rough",
            toggle=True,
            **icons.icon_kw("rough_edge", "MOD_DISPLACE"),
        )
        row_quick.popover(
            panel=FDT_PT_popover_rough.bl_idname,
            text="",
            **icons.icon_kw("settings_gear", "PREFERENCES"),
        )

        # 4. 1-Row Cut Material + Eyedropper + Output Popover
        mat_row = draw_cut_material_row(layout, settings, text="Mat")
        mat_row.popover(
            panel=FDT_PT_popover_output.bl_idname,
            text="",
            **icons.icon_kw("output_box", "PREFERENCES"),
        )

        # 5. Collapsible Cutter Layers (Collapsed by default for 1-screen fit)
        show_layers = bool(getattr(settings, "ui_show_cutter_layers", False))
        list_box = layout.box()
        hdr = list_box.row(align=True)
        hdr.prop(
            settings,
            "ui_show_cutter_layers",
            text=f"Layers ({total_c})",
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
                    rows=min(5, max(2, len(session.cutters))),
                    maxrows=8,
                )

            all_count = len(session.cutters)
            active_idx = int(getattr(settings, "cutter_layer_index", 0))
            if active_idx < 0 or active_idx >= all_count:
                active_obj = context.active_object
                active_idx = next(
                    (i for i, c in enumerate(session.cutters) if active_obj is not None and c.name == active_obj.name),
                    0,
                )
            sel_entry = session.cutters[active_idx] if 0 <= active_idx < all_count else None

            side = list_row.column(align=True)
            col_up = side.row(align=True)
            col_up.enabled = sel_entry is not None and active_idx > 0
            op_up = col_up.operator("fdt.applebite_move_cutter", text="", icon="TRIA_UP")
            if sel_entry is not None:
                op_up.cutter_name = sel_entry.name
            op_up.direction = "UP"

            col_dn = side.row(align=True)
            col_dn.enabled = sel_entry is not None and active_idx < (all_count - 1)
            op_dn = col_dn.operator("fdt.applebite_move_cutter", text="", icon="TRIA_DOWN")
            if sel_entry is not None:
                op_dn.cutter_name = sel_entry.name
            op_dn.direction = "DOWN"

            col_del = side.row(align=True)
            col_del.enabled = sel_entry is not None
            op_del = col_del.operator("fdt.applebite_delete_cutter", text="", **icons.icon_kw("trash_red", "X"))
            if sel_entry is not None:
                op_del.cutter_name = sel_entry.name

        # 6. Hero APPLY Section with Round-3 Safety Guards
        has_output = bool(settings.apply_keep_result or settings.apply_keep_pieces)
        can_apply = has_output and active_c > 0

        if not has_output:
            warn_row = layout.row()
            warn_row.alert = True
            warn_row.label(text="Output: None (enable in Output)", **icons.icon_kw("mat_alert_red", "ERROR"))
        elif active_c == 0:
            layout.label(text="All cutters muted — unmute to apply", icon="INFO")
        else:
            summary = apple_bite.get_apply_preview_summary(context.scene)
            zone_str = ", ".join(
                f"{zone_registry.zone_token(z, context.scene)}({n})"
                for z, n in sorted(summary["zone_counts"].items())
            ) or "None"
            layout.label(
                text=f"Main {summary['main_count']} · {zone_str}",
                icon="OUTLINER_COLLECTION",
            )

        row_apply = layout.row()
        row_apply.scale_y = 1.7
        row_apply.enabled = can_apply
        cut_word = "Cut" if active_c == 1 else "Cuts"
        apply_label = f"APPLY · {active_c} {cut_word}" if active_c > 0 else "APPLY"
        row_apply.operator(
            "fdt.applebite_apply",
            text=apply_label,
            depress=can_apply,
            **icons.icon_kw("apply_green", "CHECKMARK"),
        )

        # 7. Subtle Footer: Pause | Cancel
        row_end = layout.row(align=True)
        row_end.operator("fdt.applebite_exit", text="Pause", **icons.icon_kw("pause_slate", "PAUSE"))
        row_end.operator("fdt.applebite_cancel", text="Cancel", **icons.icon_kw("cancel_red", "LOOP_BACK"))


CLASSES = (
    FDT_PT_popover_settings,
    FDT_PT_popover_zones,
    FDT_PT_popover_rough,
    FDT_PT_popover_output,
    FDT_OT_set_active_tool,
    FDT_UL_cutter_layers,
    FDT_PT_main_panel,
)

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
