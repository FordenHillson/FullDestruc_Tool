"""Blender Operators for user-driven object renaming (independent of Zone Registry)."""

from __future__ import annotations

import bpy

from ..core import rename_tool


def _tag_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        screen = getattr(win, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "OUTLINER", "PROPERTIES"}:
                area.tag_redraw()


class FDT_OT_rename_objects(bpy.types.Operator):
    bl_idname = "fdt.rename_objects"
    bl_label = "Rename Selected"
    bl_description = "เปลี่ยนชื่อ Object ที่เลือกตามข้อความที่กรอก (ไม่ผูกกับ Zone และซิงก์ Apple Bite session ให้อัตโนมัติ)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        if settings is None:
            self.report({"WARNING"}, "ไม่พบการตั้งค่า FullDestruc")
            return {"CANCELLED"}

        targets = rename_tool.get_target_objects(context)
        ok, msg, _pairs = rename_tool.rename_objects(
            context,
            targets,
            mode=str(getattr(settings, "rename_mode", "FULL") or "FULL"),
            text=str(getattr(settings, "rename_text", "") or ""),
            auto_number=bool(getattr(settings, "rename_auto_number", True)),
            start_index=int(getattr(settings, "rename_start_index", 1)),
            sync_mesh_data=bool(getattr(settings, "rename_sync_mesh", True)),
        )
        settings.last_status = msg
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        _tag_redraw()
        return {"FINISHED"} if ok else {"CANCELLED"}


class FDT_OT_rename_pick_active(bpy.types.Operator):
    bl_idname = "fdt.rename_pick_active"
    bl_label = "Copy Active Name"
    bl_description = "ดึงชื่อ (หรือรหัส ID) ของ Object ที่เลือกอยู่มาใส่ในช่องกรอกเพื่อแก้ต่อได้ทันที"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = getattr(context.scene, "fdt_settings", None)
        targets = rename_tool.get_target_objects(context)
        if settings is None or not targets:
            self.report({"WARNING"}, "กรุณาเลือก Object ก่อน")
            return {"CANCELLED"}

        ref = getattr(context, "active_object", None)
        if ref is None or ref not in targets:
            ref = targets[0]

        mode = str(getattr(settings, "rename_mode", "FULL") or "FULL").upper()
        if mode == "TAG":
            picked = rename_tool.extract_id_from_name(ref.name)
            if not picked:
                self.report({"INFO"}, f"'{ref.name}' ยังไม่มีรหัส _ID- — พิมพ์รหัสใหม่ได้เลย")
                return {"FINISHED"}
            settings.rename_text = picked
        else:
            settings.rename_text = ref.name

        _tag_redraw()
        return {"FINISHED"}


CLASSES = (FDT_OT_rename_objects, FDT_OT_rename_pick_active)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
