"""Addon preferences for FullDestruc_Tool."""

from __future__ import annotations

import bpy
from bpy.props import StringProperty


class FDT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    svn_texture_path: StringProperty(
        name="SVN Texture Folder",
        description="Root folder used by Find Texture (Phase 4)",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="FullDestruc Tool Settings (Blender 5.2+)", icon="TOOL_SETTINGS")
        layout.prop(self, "svn_texture_path")
        box = layout.box()
        box.label(text="Quick Shortcut: Alt + Shift + Q (Floating Pie Menu)", icon="KEYINGSET")


CLASSES = (FDT_AddonPreferences,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
