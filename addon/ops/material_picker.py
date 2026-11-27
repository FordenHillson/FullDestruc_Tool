"""Surface Material Picker (Eyedropper) for FullDestruc_Tool (inspired by Red_Panda, Blender 5.2+ safe)."""

from __future__ import annotations

import bpy
from bpy_extras import view3d_utils


def _pick_material_name_from_mouse(context: bpy.types.Context, event: bpy.types.Event) -> tuple[str | None, bpy.types.Object | None]:
    """Raycast under the mouse cursor and return (material_name, hit_obj).

    Returns only the material name as a plain string so no bpy reference is
    held across the deferred timer assignment (avoids EEVEE Next crashes).
    """
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None, None

    coord = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    try:
        depsgraph = context.evaluated_depsgraph_get()
        hit, _loc, _norm, face_index, hit_obj, _mat = context.scene.ray_cast(depsgraph, origin, direction)
    except Exception:
        return None, None

    if not hit or hit_obj is None or hit_obj.type != "MESH":
        return None, hit_obj

    # 1. Evaluated mesh first (accurate face_index when modifiers / GeoNodes exist)
    try:
        eval_obj = hit_obj.evaluated_get(depsgraph)
        if eval_obj is not None and eval_obj.data is not None:
            mesh = eval_obj.data
            if 0 <= face_index < len(mesh.polygons):
                mat_idx = mesh.polygons[face_index].material_index
                if 0 <= mat_idx < len(mesh.materials):
                    mat = mesh.materials[mat_idx]
                    if mat is not None:
                        return mat.name, hit_obj
    except Exception:
        pass

    # 2. Fallback to original mesh
    try:
        mesh = hit_obj.data
        if mesh is not None and 0 <= face_index < len(mesh.polygons):
            mat_idx = mesh.polygons[face_index].material_index
            if 0 <= mat_idx < len(mesh.materials):
                mat = mesh.materials[mat_idx]
                if mat is not None:
                    return mat.name, hit_obj
    except Exception:
        pass

    return None, hit_obj


class FDT_OT_pick_surface_material(bpy.types.Operator):
    bl_idname = "fdt.pick_surface_material"
    bl_label = "Pick Cut Material from Surface"
    bl_description = "Click any surface in the 3D Viewport to set its material as the Cut Material (ESC / Right-Click to cancel)"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "กรุณากดหลอดดูดจากหน้าต่าง 3D View")
            return {"CANCELLED"}

        context.window.cursor_modal_set("EYEDROPPER")
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "คลิกซ้ายบนผิวโมเดลเพื่อดูด Material (คลิกขวา / ESC เพื่อยกเลิก)")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            context.window.cursor_modal_restore()
            self.report({"INFO"}, "ยกเลิกการดูด Material แล้ว")
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            context.window.cursor_modal_restore()
            mat_name, hit_obj = _pick_material_name_from_mouse(context, event)
            if mat_name is None:
                target_name = hit_obj.name if hit_obj is not None else "พื้นผิว"
                self.report({"WARNING"}, f"ไม่พบ Material บน {target_name}")
                return {"CANCELLED"}

            # Assign via deferred timer (50 ms) so EEVEE Next never compiles during cursor-restore redraw
            picked_name = str(mat_name)

            def _deferred_assign():
                mat = bpy.data.materials.get(picked_name)
                if mat is None:
                    return None
                scene = getattr(bpy.context, "scene", None)
                settings = getattr(scene, "fdt_settings", None) if scene is not None else None
                if settings is not None:
                    try:
                        settings.cut_material = mat
                    except Exception as exc:
                        print(f"[FullDestruc_Tool] Deferred cut_material assign failed: {exc}")
                try:
                    for window in bpy.context.window_manager.windows:
                        for area in window.screen.areas:
                            area.tag_redraw()
                except Exception:
                    pass
                return None

            bpy.app.timers.register(_deferred_assign, first_interval=0.05)
            self.report({"INFO"}, f"เลือก Cut Material: {picked_name}")
            return {"FINISHED"}

        # Allow camera navigation while picking
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


def draw_cut_material_row(layout: bpy.types.UILayout, settings, text: str = "Cut Material") -> bpy.types.UILayout:
    """Draw Cut Material selector + Eyedropper button, highlighted red when unset (like Red_Panda)."""
    from ..ui import icons

    row = layout.row(align=True)
    if settings.cut_material is None:
        row.alert = True
    row.prop(settings, "cut_material", text=text)
    row.operator(
        FDT_OT_pick_surface_material.bl_idname,
        text="",
        **icons.icon_kw("eyedropper", "EYEDROPPER"),
    )
    return row


CLASSES = (FDT_OT_pick_surface_material,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
