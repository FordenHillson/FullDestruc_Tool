"""Floating Pie Menu & Quick Popup under the mouse cursor (Alt+Shift+Q, Minimal UI)."""

from __future__ import annotations

import bpy

from ..core import apple_bite
from ..core.session import has_active_session, load_session
from ..ops.material_picker import draw_cut_material_row
from . import icons

_addon_keymaps: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


class FDT_MT_quick_pie(bpy.types.Menu):
    bl_idname = "FDT_MT_quick_pie"
    bl_label = "FullDestruc Quick Menu"

    def draw(self, context):
        pie = self.layout.menu_pie()
        active = has_active_session(context.scene)

        if active:
            pie.operator("fdt.applebite_exit", text="Pause", **icons.icon_kw("pause_slate", "PAUSE"))
        else:
            info = apple_bite.detect_step1_action(context)
            pie.operator("fdt.applebite_smart_start", text=info["button_label"], **icons.icon_kw("start_orange", "MOD_BOOLEAN"))

        pie.operator("fdt.applebite_apply", text="APPLY", **icons.icon_kw("apply_green", "CHECKMARK"))
        pie.operator("fdt.quick_popup", text="Quick Tune", **icons.icon_kw("settings_gear", "PREFERENCES"))
        pie.operator("fdt.applebite_draw_cutter", text="Draw Cutter", **icons.icon_kw("draw_pen", "GREASEPENCIL"))
        pie.operator("fdt.applebite_add_box", text="+ Box", **icons.icon_kw("add_box", "MESH_CUBE"))
        pie.operator("fdt.applebite_add_mesh", text="+ Mesh", **icons.icon_kw("use_mesh", "MOD_SOLIDIFY"))
        pie.operator("fdt.applebite_cancel", text="Cancel & Restore", **icons.icon_kw("cancel_red", "LOOP_BACK"))
        pie.operator("fdt.open_qt_window", text="Python Window", **icons.icon_kw("window_qt", "WINDOW"))


class FDT_OT_quick_popup(bpy.types.Operator):
    bl_idname = "fdt.quick_popup"
    bl_label = "FullDestruc Quick Tune"
    bl_description = "หน้าต่างปรับค่าด่วนใต้เมาส์ (Zone, Rough, Cut Material และ UV)"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.fdt_settings

        # 1-row Zone strip
        z_row = layout.row(align=True)
        z_row.label(text="", **icons.get_zone_dot_icon_kw(settings.active_zone, context.scene))
        z_row.prop(settings, "active_zone", text="")
        op_new = z_row.operator("fdt.zone_manage", text="", **icons.icon_kw("plus_add", "ADD"))
        op_new.action = "NEW"
        op_new.zone_key = settings.active_zone

        layout.prop(settings, "cutter_size", text="Box Size (m)")

        rough_row = layout.row(align=True)
        rough_row.prop(settings, "enable_rough", text="Rough Cut Edge", **icons.icon_kw("rough_edge", "MOD_DISPLACE"))
        session = load_session(context.scene)
        target_cutter, target_is_rough = apple_bite.get_active_rough_cutter(context, session)
        any_rough = settings.enable_rough or (session is not None and any(c.rough for c in session.cutters))
        if any_rough:
            box = layout.box()
            cr = getattr(target_cutter, "fdt_rough", None) if target_cutter is not None else None
            if cr is not None:
                status = "" if target_is_rough else " (OFF)"
                box.label(text=f"{target_cutter.name}{status}", **icons.icon_kw("rough_edge", "MOD_DISPLACE"))
                box.prop(cr, "inward_depth", text="Inward Depth", slider=True)
                box.prop(cr, "jaggedness", text="Jaggedness", slider=True)
                box.prop(cr, "noise_scale", text="Noise Scale")
                box.prop(cr, "detail", text="Detail (1-3)")
                box.prop(cr, "seed", text="Seed")
            else:
                box.prop(settings, "rough_inward_ratio", text="Inward Depth", slider=True)
                box.prop(settings, "rough_jaggedness", text="Jaggedness", slider=True)
                box.prop(settings, "rough_scale", text="Noise Scale")
                box.prop(settings, "rough_subdiv", text="Detail (1-3)")
                box.prop(settings, "rough_seed", text="Seed")

        draw_cut_material_row(layout, settings, text="Cut Mat")
        row = layout.row(align=True)
        row.prop(settings, "texel_per_meter", text="Texel")
        row.prop(settings, "texture_size", text="Tex Size")

    def execute(self, context):
        return {"FINISHED"}


CLASSES = (
    FDT_MT_quick_pie,
    FDT_OT_quick_popup,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm else None
    if kc is not None:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        kmi = km.keymap_items.new("wm.call_menu_pie", type="Q", value="PRESS", alt=True, shift=True)
        kmi.properties.name = FDT_MT_quick_pie.bl_idname
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
