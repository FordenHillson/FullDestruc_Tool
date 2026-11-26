"""Floating Pie Menu & Quick Popup under the mouse cursor (Alt+Shift+Q, English UI)."""

from __future__ import annotations

import bpy

from ..core import apple_bite
from ..core.session import has_active_session
from ..ops.material_picker import draw_cut_material_row

_addon_keymaps: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


class FDT_MT_quick_pie(bpy.types.Menu):
    bl_idname = "FDT_MT_quick_pie"
    bl_label = "FullDestruc Quick Menu"

    def draw(self, context):
        pie = self.layout.menu_pie()
        active = has_active_session(context.scene)

        if active:
            pie.operator("fdt.applebite_exit", text="Pause / Exit", icon="PAUSE")
        else:
            info = apple_bite.detect_step1_action(context)
            pie.operator("fdt.applebite_smart_start", text=info["button_label"], icon="MOD_BOOLEAN")

        pie.operator("fdt.applebite_apply", text="Apply", icon="CHECKMARK")
        pie.operator("fdt.quick_popup", text="Quick Settings", icon="PREFERENCES")
        pie.operator("fdt.applebite_interactive", text="Click Place (Ctrl+RMB)", icon="CURSOR")
        pie.operator("fdt.applebite_add_box", text="+ Add Box", icon="MESH_CUBE")
        pie.operator("fdt.applebite_add_mesh", text="+ Use Mesh", icon="MOD_SOLIDIFY")
        pie.operator("fdt.applebite_cancel", text="Cancel & Restore", icon="LOOP_BACK")
        pie.operator("fdt.open_qt_window", text="Python Window", icon="WINDOW")


class FDT_OT_quick_popup(bpy.types.Operator):
    bl_idname = "fdt.quick_popup"
    bl_label = "FullDestruc Quick Settings"
    bl_description = "Floating popup right under the mouse cursor for fast Zone & Apple Bite tweaks"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.fdt_settings
        layout.prop(settings, "active_zone", text="Zone (type to search)")
        row_zones = layout.row(align=True)
        for action, label, icon in (
            ("NEW", "New", "ADD"),
            ("RENAME", "Rename", "GREASEPENCIL"),
            ("DELETE", "Delete", "TRASH"),
        ):
            op = row_zones.operator("fdt.zone_manage", text=label, icon=icon)
            op.action = action
            op.zone_key = settings.active_zone
        layout.prop(settings, "cutter_size", text="Box Size (m)")
        layout.prop(settings, "enable_rough", text="Rough Cut Edge")
        from ..core.session import load_session
        session = load_session(context.scene)
        any_rough = settings.enable_rough or (session is not None and any(c.rough for c in session.cutters))
        if any_rough:
            box = layout.box()
            box.prop(settings, "rough_inward_ratio", text="Inward Depth", slider=True)
            box.prop(settings, "rough_scale", text="Noise Scale")
            box.prop(settings, "rough_subdiv", text="Detail (1-3)")
            box.prop(settings, "rough_seed", text="Seed")
        draw_cut_material_row(layout, settings, text="Cut Material")
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
