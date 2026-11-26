"""Zone picker / zone management operators shared by every UI surface."""

from __future__ import annotations

import bpy
from bpy.props import EnumProperty, FloatVectorProperty, StringProperty

from ..core import apple_bite, zone_registry

_BASE_ITEMS = [(letter, letter, f"Base zone {letter}") for letter in zone_registry.BUILTIN_ZONES]


def _zone_items(self, context):
    return zone_registry.iter_zone_items(getattr(context, "scene", None))


def _on_picker_zone_changed(self, context) -> None:
    cutter = bpy.data.objects.get(self.cutter_name)
    if cutter is None or context is None:
        return
    apple_bite.set_cutters_zone(context, [cutter], self.zone_key)


class FDT_OT_zone_picker(bpy.types.Operator):
    bl_idname = "fdt.zone_picker"
    bl_label = "Pick Zone"
    bl_description = "Search the Scene zone registry and assign the selected cutter"
    bl_options = {"REGISTER"}

    cutter_name: StringProperty(name="Cutter", default="")
    zone_key: EnumProperty(name="Zone", items=_zone_items, update=_on_picker_zone_changed)

    def invoke(self, context, event):
        scene = getattr(context, "scene", None)
        cutter = bpy.data.objects.get(self.cutter_name)
        if cutter is None:
            cutter = getattr(context, "active_object", None)
        if cutter is None or not bool(cutter.get(apple_bite.CUTTER_PROP, False)):
            self.report({"WARNING"}, "Select a cutter object first.")
            return {"CANCELLED"}
        self.cutter_name = cutter.name
        self.zone_key = zone_registry.resolve_zone(cutter.get(apple_bite.CUTTER_ZONE_PROP, "A"), scene).key
        return context.window_manager.invoke_popup(self, width=360)

    def draw(self, context):
        layout = self.layout
        scene = getattr(context, "scene", None)
        layout.label(text=f"Assign zone to: {self.cutter_name}", icon="COLOR")
        layout.prop(self, "zone_key", text="Zone (type to search)")
        layout.separator()
        row = layout.row(align=True)
        for action, label, icon in (
            ("NEW", "New", "ADD"),
            ("RENAME", "Rename", "GREASEPENCIL"),
            ("DELETE", "Delete", "TRASH"),
        ):
            op = row.operator("fdt.zone_manage", text=label, icon=icon)
            op.action = action
            op.zone_key = self.zone_key
        layout.label(text=f"Current token: {zone_registry.zone_token(self.zone_key, scene)}")

    def execute(self, context):
        cutter = bpy.data.objects.get(self.cutter_name)
        if cutter is None:
            return {"CANCELLED"}
        apple_bite.set_cutters_zone(context, [cutter], self.zone_key)
        return {"FINISHED"}


class FDT_OT_zone_manage(bpy.types.Operator):
    bl_idname = "fdt.zone_manage"
    bl_label = "Manage Zones"
    bl_description = "Create, rename or delete a zone in the Scene zone registry"
    bl_options = {"REGISTER"}

    action: EnumProperty(
        name="Action",
        items=[
            ("NEW", "New", "Create a custom zone"),
            ("RENAME", "Rename", "Rename token / base / kind / color of a zone"),
            ("DELETE", "Delete", "Delete a custom zone"),
        ],
        default="NEW",
    )
    zone_key: StringProperty(name="Zone Key", default="A")
    token: StringProperty(name="Zone Token", default="")
    base_zone: EnumProperty(name="Base Zone", items=_BASE_ITEMS, default="A")
    kind: EnumProperty(
        name="Kind",
        items=[("ID", "ID", "ID zone"), ("VIS", "VIS", "VIS zone")],
        default="ID",
    )
    color: FloatVectorProperty(
        name="Color",
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
        default=(1.0, 0.5, 0.12, 1.0),
    )

    def invoke(self, context, event):
        scene = getattr(context, "scene", None)
        if self.action in {"RENAME", "DELETE"}:
            zdef = zone_registry.find_zone(self.zone_key, scene) or zone_registry.resolve_zone(self.zone_key, scene)
            self.token = zdef.token
            self.base_zone = zdef.base
            self.kind = zdef.kind
            self.color = zdef.color
        else:
            kind, base = zone_registry.parse_token_hint(self.token)
            self.kind = kind
            self.base_zone = base
            if not self.token:
                self.token = f"{kind}-{base}"
            self.color = zone_registry.base_color(base)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        scene = getattr(context, "scene", None)
        if self.action == "DELETE":
            zdef = zone_registry.find_zone(self.zone_key, scene)
            nc, np = zone_registry.zone_usage(self.zone_key, scene)
            layout.label(text=f"Delete zone: {zdef.token if zdef else self.zone_key}", icon="ERROR")
            layout.label(text=f"In use by {nc} cutter(s) and {np} piece(s) — built-in A-Z cannot be deleted.")
            return
        layout.prop(self, "token", text="Zone Token")
        row = layout.row(align=True)
        row.prop(self, "base_zone", text="Base")
        row.prop(self, "kind", text="Kind")
        layout.prop(self, "color", text="Color")
        layout.label(text="Examples: ID-A#B, VIS-A#B, ID-A_VIS-A#B")

    def execute(self, context):
        scene = getattr(context, "scene", None)
        if self.action == "NEW":
            ok, msg, key = zone_registry.create_zone(self.token, self.base_zone, self.kind, self.color, scene)
            if ok:
                settings = getattr(scene, "fdt_settings", None)
                if settings is not None:
                    settings.active_zone = key
        elif self.action == "RENAME":
            ok, msg = zone_registry.update_zone(
                self.zone_key, token=self.token, base=self.base_zone, kind=self.kind, color=self.color, scene=scene
            )
        else:
            ok, msg = zone_registry.delete_zone(self.zone_key, scene)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        try:
            from .apple_bite_ops import _set_status

            _set_status(context, msg)
        except Exception:
            pass
        return {"FINISHED"} if ok else {"CANCELLED"}


CLASSES = (FDT_OT_zone_picker, FDT_OT_zone_manage)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
