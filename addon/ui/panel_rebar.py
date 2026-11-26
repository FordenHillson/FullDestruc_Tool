"""Concrete Rebars detail, drawn inside the main N-Panel's right-hand area.

The panel keeps four short collapsible sections:

1. **พื้นฐาน** — the only things most people need to touch.
2. **ลุคเหล็ก** — how the bars look.
3. **ขั้นสูง** — the original fine controls, kept but out of the way.
4. **ระบบและรายการชุด** — material, limits and the generator list.

Tool-level navigation (Apple Bite / Rebar / …) lives in the icon rail in
``panel_n.py``; this module only renders the Rebar detail.

Every icon lives in :data:`REBAR_ICONS` so a typo is caught by the headless
tests instead of showing up as an empty square in Blender.
"""

from __future__ import annotations

import bpy

from ..core import rebar_state as st
from ..ops import rebar_ops

# All names below are checked against Blender 5.2's UILayout icon enum in
# tests/test_rebar_ops.py.  Do not use guessed names such as OBJECT or MOVE.
REBAR_ICONS = {
    "section_open": "TRIA_DOWN",
    "section_closed": "TRIA_RIGHT",
    "basics": "SETTINGS",
    "look": "MOD_CURVE",
    "advanced": "MODIFIER",
    "system": "PREFERENCES",
    "target": "MATERIAL",
    "eyedropper": "EYEDROPPER",
    "label": "OUTLINER",
    "density": "MOD_ARRAY",
    "bar": "MOD_SOLIDIFY",
    "estimate": "INFO",
    "create": "ADD",
    "bake": "CHECKMARK",
    "rebuild": "FILE_REFRESH",
    "folder": "FILE_FOLDER",
    "sink": "MOD_SHRINKWRAP",
    "length": "ARROW_LEFTRIGHT",
    "bend": "MOD_CURVE",
    "bend_scale": "MOD_OCEAN",
    "seed": "MOD_NOISE",
    "dice": "FILE_REFRESH",
    "spacing": "GRID",
    "align_x": "AXIS_SIDE",
    "align_y": "AXIS_FRONT",
    "align_z": "AXIS_TOP",
    "offset": "ARROW_LEFTRIGHT",
    "angle": "QUESTION",
    "normal": "AXIS_TOP",
    "gravity": "MOD_DISPLACE",
    "avoid": "MOD_SCREW",
    "smooth": "MOD_SMOOTH",
    "steps": "MOD_SUBSURF",
    "max_bars": "MOD_ARRAY",
    "fast_drag": "AUTO",
    "live": "DOT",
    "baked": "CHECKMARK",
    "stale": "ERROR",
    "edit": "GREASEPENCIL",
    "faces": "FACESEL",
    "clear": "BRUSH_DATA",
    "delete": "TRASH",
    "status": "INFO",
    "debris": "MOD_PARTICLES",
    "debris_material": "MATERIAL",
    "debris_size": "ARROW_LEFTRIGHT",
    "debris_rough": "MOD_SMOOTH",
    "debris_cluster": "MOD_DISPLACE",
}


def _header(box, settings, prop_name: str, text: str, icon: str) -> None:
    """Draw a clickable section title (the property supplies its tooltip)."""
    row = box.row()
    row.prop(settings, prop_name, text=text, icon=icon, emboss=False)


def _draw_basics(box, context, settings) -> None:
    basic = box.box()
    _header(basic, settings, "ui_show_basics", "พื้นฐาน", REBAR_ICONS["basics"])
    if not settings.ui_show_basics:
        return

    target_row = basic.row(align=True)
    if settings.target_material is None:
        target_row.alert = True
    target_row.prop(settings, "target_material", text="Target Mat", icon=REBAR_ICONS["target"])
    pick = target_row.operator("fdt.rebar_pick_material", text="", icon=REBAR_ICONS["eyedropper"])
    pick.slot = "TARGET"

    basic.prop(settings, "rebar_label", text="Label", icon=REBAR_ICONS["label"])
    basic.prop(settings, "density_preset", text="Density", icon=REBAR_ICONS["density"])
    basic.prop(settings, "bar_size", text="Bar", icon=REBAR_ICONS["bar"])

    active = st.find_entry(settings.rebar_active_sid) if settings.rebar_active_sid else None
    obj = context.active_object
    if active is not None:
        estimated = settings.rebar_estimate
    elif obj is not None and obj.type == "MESH" and settings.target_material is not None:
        faces = rebar_ops.resolve_faces(context, obj, settings)
        estimated = rebar_ops.estimate_for(obj.data, faces, rebar_ops.settings_to_params(settings))
    else:
        estimated = 0
    capped = " · เต็มตามที่ตั้ง" if estimated > settings.max_bars else ""
    basic.label(text=f"คาดการณ์ ~{estimated:,} แท่ง{capped}", icon=REBAR_ICONS["estimate"])

    # Create comes first: it is the main action, even after other sets exist.
    if active is not None:
        basic.separator()
        create_row = basic.row(align=True)
        create_row.scale_y = 1.4
        create_row.enabled = bool(obj is not None and obj.type == "MESH")
        create_row.operator("fdt.rebar_create", text="Create Another Rebar", icon=REBAR_ICONS["create"])

    action_row = basic.row(align=True)
    action_row.scale_y = 1.3
    # Enable the row, never the operator: ``layout.operator()`` returns RNA
    # properties and setting ``.enabled`` on it raises, which used to abort
    # draw() before the other three section headers were added.
    action_row.enabled = bool(obj is not None and obj.type == "MESH")
    if active is not None:
        action_row.operator("fdt.rebar_bake", text="Bake", icon=REBAR_ICONS["bake"])
        if active.stale:
            action_row.operator("fdt.rebar_rebuild", text="Re-apply", icon=REBAR_ICONS["rebuild"])
    else:
        action_row.operator("fdt.rebar_create", text="Create Rebar", icon=REBAR_ICONS["create"])

    # The rebar selected in the viewport is also the target for everyday
    # edits.  Keeping these here avoids opening the far-away generator list.
    target, hint = rebar_ops.resolve_remove_target(context)
    quick_row = basic.row(align=True)
    quick_row.enabled = target is not None
    edit = quick_row.operator("fdt.rebar_reedit", text="Edit", icon=REBAR_ICONS["edit"])
    faces = quick_row.operator("fdt.rebar_select_faces", text="Faces", icon=REBAR_ICONS["faces"])
    clear = quick_row.operator("fdt.rebar_clear_faces", text="Clear", icon=REBAR_ICONS["clear"])
    if target is not None:
        edit.sid = target.sid
        faces.sid = target.sid
        clear.sid = target.sid

    # Debris sits directly under Edit / Faces / Clear, before the destructive
    # Remove Set action at the bottom of the button group.
    basic.separator()
    basic.prop(settings, "enable_debris", text="Concrete Debris", icon=REBAR_ICONS["debris"])

    # One big Remove button, aimed at the same selected rebar.  The label
    # above it always shows the exact target so there is no guessing.
    if target is not None:
        basic.label(
            text=f"จะลบ: {target.display_label} · {target.sid8}",
            icon=REBAR_ICONS["delete"],
        )
    else:
        basic.label(text=hint, icon=REBAR_ICONS["status"])
    remove_row = basic.row(align=True)
    remove_row.scale_y = 1.2
    remove_row.enabled = target is not None
    remove_row.operator("fdt.rebar_remove", text="Remove Set", icon=REBAR_ICONS["delete"])

    # Status / guidance lives after all buttons so nothing cuts the button
    # group in half.  The generator id stays on the "will delete" line above.
    if active is not None:
        if active.stale:
            footer = "ผิวเปลี่ยนแล้ว → Re-apply ให้เหล็กอยู่บนผิวใหม่ แล้ว Bake"
            footer_icon = REBAR_ICONS["stale"]
        elif active.baked:
            footer = "บันทึกแล้ว · แก้ต่อได้ด้วย Edit"
            footer_icon = REBAR_ICONS["baked"]
        else:
            footer = "ปรับค่าเห็นผลทันที · กด Bake เพื่อบันทึกชุด"
            footer_icon = REBAR_ICONS["live"]
        basic.label(text=footer, icon=footer_icon)


def _draw_look(box, settings) -> None:
    look = box.box()
    _header(look, settings, "ui_show_look", "ลุคเหล็ก", REBAR_ICONS["look"])
    if not settings.ui_show_look:
        return

    look.prop(settings, "sink_depth", text="Sink (mm)", icon=REBAR_ICONS["sink"])
    length_row = look.row(align=True)
    length_row.prop(settings, "min_length", text="Min (mm)", icon=REBAR_ICONS["length"])
    length_row.prop(settings, "max_length", text="Max (mm)", icon=REBAR_ICONS["length"])
    look.prop(settings, "bending_strength", text="Bend (mm)", icon=REBAR_ICONS["bend"])
    look.prop(settings, "bend_scale_mm", text="Bend Scale (mm)", icon=REBAR_ICONS["bend_scale"])
    seed_row = look.row(align=True)
    seed_row.prop(settings, "random_seed", text="Seed", icon=REBAR_ICONS["seed"])
    seed_row.operator("fdt.rebar_reroll_seed", text="", icon=REBAR_ICONS["dice"])


def _draw_advanced(box, settings) -> None:
    advanced = box.box()
    _header(advanced, settings, "ui_show_advanced", "ขั้นสูง", REBAR_ICONS["advanced"])
    if not settings.ui_show_advanced:
        return

    advanced.prop(settings, "diameter_mm", text="Diameter (mm)", icon=REBAR_ICONS["bar"])
    for prop_name, axis_icon in (
        ("spacing_x", REBAR_ICONS["align_x"]),
        ("spacing_y", REBAR_ICONS["align_y"]),
        ("spacing_z", REBAR_ICONS["align_z"]),
    ):
        advanced.prop(settings, prop_name, icon=axis_icon)

    align_row = advanced.row(align=True)
    align_row.prop(settings, "align_x", text="X", icon=REBAR_ICONS["align_x"], toggle=True)
    align_row.prop(settings, "align_y", text="Y", icon=REBAR_ICONS["align_y"], toggle=True)
    align_row.prop(settings, "align_z", text="Z", icon=REBAR_ICONS["align_z"], toggle=True)
    advanced.prop(settings, "grid_offset", text="Offset (mm)", icon=REBAR_ICONS["offset"])

    advanced.prop(settings, "angle_threshold", text="Angle", icon=REBAR_ICONS["angle"])
    bias_row = advanced.row(align=True)
    bias_row.prop(settings, "normal_bias", text="Normal", icon=REBAR_ICONS["normal"])
    bias_row.prop(settings, "gravity_bias", text="Gravity", icon=REBAR_ICONS["gravity"])
    bias_row.prop(settings, "avoidance_strength", text="Avoid", icon=REBAR_ICONS["avoid"])
    advanced.prop(settings, "smooth_radius", text="Smooth (mm)", icon=REBAR_ICONS["smooth"])
    advanced.prop(settings, "subdivisions", text="Smooth Steps", icon=REBAR_ICONS["steps"])

    if settings.enable_debris:
        debris = advanced.box()
        debris.label(text="Concrete Debris", icon=REBAR_ICONS["debris"])
        debris.prop(settings, "debris_density", text="Density", icon=REBAR_ICONS["debris"])
        size_row = debris.row(align=True)
        size_row.prop(settings, "debris_min_size", text="Min Size (mm)", icon=REBAR_ICONS["debris_size"])
        size_row.prop(settings, "debris_max_size", text="Max Size (mm)", icon=REBAR_ICONS["debris_size"])
        debris.prop(settings, "debris_irregularity", text="Rough", icon=REBAR_ICONS["debris_rough"])
        debris.prop(settings, "debris_concentration", text="Cluster", icon=REBAR_ICONS["debris_cluster"])
        mat_row = debris.row(align=True)
        mat_row.prop(settings, "debris_material", text="Debris Mat", icon=REBAR_ICONS["debris_material"])
        pick = mat_row.operator("fdt.rebar_pick_material", text="", icon=REBAR_ICONS["eyedropper"])
        pick.slot = "DEBRIS"


def _draw_system(box, context, settings) -> None:
    system = box.box()
    _header(system, settings, "ui_show_system", "ระบบและรายการชุด", REBAR_ICONS["system"])
    if not settings.ui_show_system:
        return

    rebar_row = system.row(align=True)
    rebar_row.prop(settings, "rebar_material", text="Rebar Mat", icon=REBAR_ICONS["target"])
    rebar_row.operator("fdt.rebar_load_material", text="", icon=REBAR_ICONS["folder"])
    pick = rebar_row.operator("fdt.rebar_pick_material", text="", icon=REBAR_ICONS["eyedropper"])
    pick.slot = "REBAR"

    system.prop(settings, "max_bars", text="Max Bars", icon=REBAR_ICONS["max_bars"])
    system.prop(settings, "fast_drag", text="Fast Drag", icon=REBAR_ICONS["fast_drag"])

    states = st.load_all_generators(context.scene)
    if states:
        system.separator()
        for state in states.values():
            entry = state.entry
            if entry.stale:
                state_icon = REBAR_ICONS["stale"]
            elif entry.baked:
                state_icon = REBAR_ICONS["baked"]
            else:
                state_icon = REBAR_ICONS["live"]
            system.label(text=entry.display_label, icon=state_icon)
            tag = f"  {entry.sid8}"
            if entry.stale:
                tag += "  · ผิวเปลี่ยนแล้ว"
            if not state.valid:
                tag += "  · หายไป"
            system.label(text=tag)

            ops_row = system.row(align=True)
            edit = ops_row.operator("fdt.rebar_reedit", text="Edit", icon=REBAR_ICONS["edit"])
            edit.sid = entry.sid
            faces = ops_row.operator("fdt.rebar_select_faces", text="Faces", icon=REBAR_ICONS["faces"])
            faces.sid = entry.sid
            clear = ops_row.operator("fdt.rebar_clear_faces", text="Clear", icon=REBAR_ICONS["clear"])
            clear.sid = entry.sid
            delete = ops_row.operator("fdt.rebar_delete", text="Delete", icon=REBAR_ICONS["delete"])
            delete.sid = entry.sid

    if settings.rebar_status:
        system.label(text=settings.rebar_status, icon=REBAR_ICONS["status"])


def draw(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
    *,
    force_open: bool = False,
) -> None:
    """Draw the Rebars detail.

    ``force_open`` is used by the main panel's tool rail: selecting the Rebar
    tab must show its settings straight away, without an extra expand step.
    The standalone header is kept for callers that draw the section inline.
    """
    settings = context.scene.fdt_rebar_settings
    box = layout.box()
    if not force_open:
        header = box.row()
        header.prop(
            settings,
            "ui_show_rebar",
            emboss=False,
            text="Concrete Rebars",
            icon=REBAR_ICONS["section_open"] if settings.ui_show_rebar else REBAR_ICONS["section_closed"],
        )
        if not settings.ui_show_rebar:
            return

    _draw_basics(box, context, settings)
    _draw_look(box, settings)
    _draw_advanced(box, settings)
    _draw_system(box, context, settings)
