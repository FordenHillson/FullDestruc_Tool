"""Blender Operators for user-driven object renaming (independent of Zone Registry)."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

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


_POPUP_LOADING = False


def _on_popup_mode_changed(self, context) -> None:
    """Auto-adjust prefilled text when toggling Full Name <-> Replace ID inside the popup."""
    if _POPUP_LOADING or context is None:
        return
    targets = rename_tool.get_target_objects(context)
    if not targets:
        return
    ref = getattr(context, "active_object", None)
    if ref is None or ref not in targets:
        ref = targets[0]
    extracted = rename_tool.extract_id_from_name(ref.name)
    if self.mode == "TAG" and (not self.new_text or self.new_text == ref.name):
        self.new_text = extracted
    elif self.mode == "FULL" and (not self.new_text or self.new_text == extracted):
        self.new_text = ref.name


class FDT_OT_rename_popup(bpy.types.Operator):
    bl_idname = "fdt.rename_popup"
    bl_label = "Rename Objects"
    bl_description = "กรอกชื่อใหม่หรือรหัส ID เพื่อเปลี่ยนชื่อ Object ที่เลือก (ไม่ผูกกับ Zone)"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Mode",
        description="เลือกรูปแบบการเปลี่ยนชื่อ: พิมพ์ชื่อใหม่ทั้งหมด หรือเปลี่ยนเฉพาะส่วน ID ในชื่อเดิม",
        items=[
            ("FULL", "Full Name", "พิมพ์ชื่อใหม่ตรง ๆ (เลือกหลายชิ้นรันเลข _01, _02 อัตโนมัติ)"),
            ("TAG", "Replace ID", "เปลี่ยนเฉพาะส่วน _ID-... หรือ _FDST_ID-... ในชื่อเดิม (เช่น _ID-B_18 -> _ID-A_18)"),
        ],
        default="FULL",
        update=_on_popup_mode_changed,
    )
    new_text: StringProperty(
        name="New Name",
        description="พิมพ์ชื่อใหม่ หรือรหัส ID ที่ต้องการ (ไม่ผูกกับ Zone)",
        default="",
        maxlen=63,
    )
    auto_number: BoolProperty(
        name="Auto #",
        description="เติมเลขลำดับ _01, _02 ต่อท้ายอัตโนมัติเมื่อเลือกหลายชิ้น",
        default=True,
    )
    start_index: IntProperty(
        name="Start",
        description="เลขเริ่มต้นสำหรับรันเลขลำดับเมื่อเลือกหลายชิ้น",
        default=1,
        min=0,
        max=9999,
    )
    sync_mesh: BoolProperty(
        name="Sync Mesh",
        description="เปลี่ยนชื่อ Mesh Data ให้ตรงกับชื่อ Object ด้วย",
        default=True,
    )

    def invoke(self, context, event):
        global _POPUP_LOADING
        targets = rename_tool.get_target_objects(context)
        if not targets:
            self.report({"WARNING"}, "กรุณาเลือก Object ที่ต้องการเปลี่ยนชื่อก่อน")
            return {"CANCELLED"}

        ref = getattr(context, "active_object", None)
        if ref is None or ref not in targets:
            ref = targets[0]

        settings = getattr(getattr(context, "scene", None), "fdt_settings", None)
        _POPUP_LOADING = True
        try:
            if settings is not None:
                self.mode = str(getattr(settings, "rename_mode", "FULL") or "FULL")
                self.auto_number = bool(getattr(settings, "rename_auto_number", True))
                self.start_index = int(getattr(settings, "rename_start_index", 1))
                self.sync_mesh = bool(getattr(settings, "rename_sync_mesh", True))
            if self.mode == "TAG":
                self.new_text = rename_tool.extract_id_from_name(ref.name)
            else:
                self.new_text = ref.name
        finally:
            _POPUP_LOADING = False

        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        targets = rename_tool.get_target_objects(context)
        count = len(targets)

        if count == 1:
            layout.label(text=f"Object: {targets[0].name}", icon="OUTLINER_OB_MESH")
        else:
            layout.label(text=f"Selected: {count} objects", icon="OUTLINER_OB_MESH")

        row_mode = layout.row(align=True)
        row_mode.prop(self, "mode", expand=True)

        layout.prop(self, "new_text", text="New ID" if self.mode == "TAG" else "New Name")

        row_opts = layout.row(align=True)
        if self.mode == "FULL" and count > 1:
            row_opts.prop(self, "auto_number", text="Auto #")
            if self.auto_number:
                row_opts.prop(self, "start_index", text="Start")
        row_opts.prop(self, "sync_mesh", text="Sync Mesh")

        plan = rename_tool.build_target_names(
            targets,
            mode=self.mode,
            text=self.new_text,
            auto_number=self.auto_number,
            start_index=self.start_index,
        )
        if plan:
            prev_box = layout.box()
            for obj, desired in plan[:3]:
                prev_box.label(text=f"{obj.name}  ->  {desired}", icon="GREASEPENCIL")
            if len(plan) > 3:
                prev_box.label(text=f"... +{len(plan) - 3} more", icon="INFO")

    def execute(self, context):
        targets = rename_tool.get_target_objects(context)
        ok, msg, _pairs = rename_tool.rename_objects(
            context,
            targets,
            mode=self.mode,
            text=self.new_text,
            auto_number=self.auto_number,
            start_index=self.start_index,
            sync_mesh_data=self.sync_mesh,
        )
        settings = getattr(getattr(context, "scene", None), "fdt_settings", None)
        if settings is not None:
            settings.rename_mode = self.mode
            settings.rename_text = self.new_text
            settings.rename_auto_number = self.auto_number
            settings.rename_start_index = self.start_index
            settings.rename_sync_mesh = self.sync_mesh
            settings.last_status = msg
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        _tag_redraw()
        return {"FINISHED"} if ok else {"CANCELLED"}


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


CLASSES = (FDT_OT_rename_popup, FDT_OT_rename_objects, FDT_OT_rename_pick_active)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
