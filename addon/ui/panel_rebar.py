"""Minimal Concrete Rebars detail, drawn inside the main N-Panel's right-hand area.

Uses concise English UI labels (matching Apple Bite, Viewport Dock & Qt Window)
while keeping all plain-language Thai hover tooltips in ``props.py``.
Every Blender fallback icon lives in :data:`REBAR_ICONS` so a typo is caught by
the headless tests instead of showing up as an empty square in Blender.
"""

from __future__ import annotations

import bpy

from ..core import rebar_state as st
from ..ops import rebar_ops
from . import icons

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


def _header(box, settings, prop_name: str, text: str) -> None:
    """Draw a clean collapsible section toggle header."""
    is_open = bool(getattr(settings, prop_name, False))
    row = box.row(align=True)
    row.prop(
        settings,
        prop_name,
        text=text,
        icon=REBAR_ICONS["section_open"] if is_open else REBAR_ICONS["section_closed"],
        emboss=False,
    )


def _draw_global_stale_banner(box, context, active) -> None:
    """Surface any stale rebar sets immediately (even when System & Sets is collapsed)."""
    states = st.load_all_generators(context.scene)
    stale_entries = [s.entry for s in states.values() if s.entry.stale]
    if not stale_entries:
        return
    # If the only stale set is already active, the Basics section shows its Re-apply button directly.
    if active is not None and len(stale_entries) == 1 and stale_entries[0].sid == active.sid:
        return
    warn = box.box()
    row = warn.row(align=True)
    row.alert = True
    n = len(stale_entries)
    word = "set" if n == 1 else "sets"
    row.label(text=f"{n} rebar {word} need Re-apply", **icons.icon_kw("stale_warn", REBAR_ICONS["stale"]))
    op_edit = row.operator("fdt.rebar_reedit", text="Fix", **icons.icon_kw("rebuild_amber", REBAR_ICONS["rebuild"]))
    op_edit.sid = stale_entries[0].sid


def _draw_basics(box, context, settings) -> None:
    basic = box.box()
    _header(basic, settings, "ui_show_basics", "Basics")
    if not settings.ui_show_basics:
        return

    # 1. Target Material + colored Eyedropper (red alert when unset)
    target_row = basic.row(align=True)
    if settings.target_material is None:
        target_row.alert = True
    target_row.prop(settings, "target_material", text="Mat", icon=REBAR_ICONS["target"])
    pick = target_row.operator(
        "fdt.rebar_pick_material",
        text="",
        **icons.icon_kw("eyedropper", REBAR_ICONS["eyedropper"]),
    )
    pick.slot = "TARGET"

    # 2. Compact 1-row Density + Bar Size
    preset_row = basic.row(align=True)
    preset_row.prop(settings, "density_preset", text="")
    preset_row.prop(settings, "bar_size", text="")

    # 3. Live bar count estimate + Debris toggle on a tight strip
    active = st.find_entry(settings.rebar_active_sid) if settings.rebar_active_sid else None
    obj = context.active_object
    if active is not None:
        estimated = settings.rebar_estimate
    elif obj is not None and obj.type == "MESH" and settings.target_material is not None:
        faces = rebar_ops.resolve_faces(context, obj, settings)
        estimated = rebar_ops.estimate_for(obj.data, faces, rebar_ops.settings_to_params(settings))
    else:
        estimated = 0

    meta_row = basic.row(align=True)
    if settings.target_material is None and active is None:
        meta_row.label(text="Pick Target Mat first", icon=REBAR_ICONS["estimate"])
    else:
        capped = " (cap)" if estimated > settings.max_bars else ""
        meta_row.label(text=f"~{estimated:,} bars{capped}", icon=REBAR_ICONS["estimate"])
    meta_row.prop(
        settings,
        "enable_debris",
        text="Debris",
        toggle=True,
        **icons.icon_kw("debris", REBAR_ICONS["debris"]),
    )

    # 4. Primary Action Buttons (Create always precedes Bake when active)
    has_mesh = bool(obj is not None and obj.type == "MESH")
    if active is not None:
        create_row = basic.row(align=True)
        create_row.scale_y = 1.25
        create_row.enabled = has_mesh
        create_row.operator(
            "fdt.rebar_create",
            text="Create Another Rebar",
            **icons.icon_kw("plus_add", REBAR_ICONS["create"]),
        )

    action_row = basic.row(align=True)
    action_row.scale_y = 1.65
    action_row.enabled = has_mesh if active is not None else (has_mesh and settings.target_material is not None)
    if active is not None:
        action_row.operator(
            "fdt.rebar_bake",
            text="Bake",
            depress=True,
            **icons.icon_kw("apply_green", REBAR_ICONS["bake"]),
        )
        if active.stale:
            action_row.operator(
                "fdt.rebar_rebuild",
                text="Re-apply",
                **icons.icon_kw("rebuild_amber", REBAR_ICONS["rebuild"]),
            )
    else:
        action_row.operator(
            "fdt.rebar_create",
            text="Create Rebar",
            depress=action_row.enabled,
            **icons.icon_kw("start_orange", REBAR_ICONS["create"]),
        )

    # 5. Selected/Active Set Controls (shown ONLY when a rebar target exists -> clean Idle!)
    target, _hint = rebar_ops.resolve_remove_target(context)
    if target is not None:
        basic.separator(factor=0.5)
        basic.label(
            text=f"Target: {target.display_label} · {target.sid8}",
            icon=REBAR_ICONS["label"],
        )
        quick_row = basic.row(align=True)
        edit = quick_row.operator("fdt.rebar_reedit", text="Edit", **icons.icon_kw("rename_pen", REBAR_ICONS["edit"]))
        faces_op = quick_row.operator("fdt.rebar_select_faces", text="Faces", icon=REBAR_ICONS["faces"])
        clear = quick_row.operator("fdt.rebar_clear_faces", text="Clear", icon=REBAR_ICONS["clear"])
        edit.sid = target.sid
        faces_op.sid = target.sid
        clear.sid = target.sid

        remove_row = basic.row(align=True)
        remove_row.scale_y = 1.15
        remove_row.operator(
            "fdt.rebar_remove",
            text="Remove Set",
            **icons.icon_kw("trash_red", REBAR_ICONS["delete"]),
        )

    # 6. Concise Status Footer (after all buttons)
    if active is not None:
        if active.stale:
            basic.label(
                text="Host changed · Re-apply then Bake",
                **icons.icon_kw("stale_warn", REBAR_ICONS["stale"]),
            )
        elif active.baked:
            basic.label(
                text="Baked · click Edit to adjust",
                **icons.icon_kw("apply_green", REBAR_ICONS["baked"]),
            )
        else:
            basic.label(text="Live preview · press Bake to commit", icon=REBAR_ICONS["live"])


def _draw_look(box, settings) -> None:
    look = box.box()
    _header(look, settings, "ui_show_look", "Look")
    if not settings.ui_show_look:
        return

    look.prop(settings, "sink_depth", text="Sink (mm)", icon=REBAR_ICONS["sink"])
    length_row = look.row(align=True)
    length_row.prop(settings, "min_length", text="Min", icon=REBAR_ICONS["length"])
    length_row.prop(settings, "max_length", text="Max", icon=REBAR_ICONS["length"])
    look.prop(settings, "bending_strength", text="Bend (mm)", icon=REBAR_ICONS["bend"])
    look.prop(settings, "bend_scale_mm", text="Bend Scale", icon=REBAR_ICONS["bend_scale"])
    seed_row = look.row(align=True)
    seed_row.prop(settings, "random_seed", text="Seed", icon=REBAR_ICONS["seed"])
    seed_row.operator("fdt.rebar_reroll_seed", text="", icon=REBAR_ICONS["dice"])
    look.prop(settings, "rebar_label", text="Label", icon=REBAR_ICONS["label"])


def _draw_advanced(box, settings) -> None:
    advanced = box.box()
    _header(advanced, settings, "ui_show_advanced", "Advanced")
    if not settings.ui_show_advanced:
        return

    advanced.prop(settings, "diameter_mm", text="Diameter (mm)", icon=REBAR_ICONS["bar"])
    sp_row = advanced.row(align=True)
    sp_row.prop(settings, "spacing_x", text="X")
    sp_row.prop(settings, "spacing_y", text="Y")
    sp_row.prop(settings, "spacing_z", text="Z")

    align_row = advanced.row(align=True)
    align_row.prop(settings, "align_x", text="X", icon=REBAR_ICONS["align_x"], toggle=True)
    align_row.prop(settings, "align_y", text="Y", icon=REBAR_ICONS["align_y"], toggle=True)
    align_row.prop(settings, "align_z", text="Z", icon=REBAR_ICONS["align_z"], toggle=True)
    advanced.prop(settings, "grid_offset", text="Offset (mm)", icon=REBAR_ICONS["offset"])

    advanced.prop(settings, "angle_threshold", text="Angle", icon=REBAR_ICONS["angle"])
    bias_row = advanced.row(align=True)
    bias_row.prop(settings, "normal_bias", text="Norm")
    bias_row.prop(settings, "gravity_bias", text="Grav")
    bias_row.prop(settings, "avoidance_strength", text="Avoid")
    smooth_row = advanced.row(align=True)
    smooth_row.prop(settings, "smooth_radius", text="Smooth")
    smooth_row.prop(settings, "subdivisions", text="Steps")

    if settings.enable_debris:
        debris = advanced.box()
        debris.label(text="Concrete Debris", **icons.icon_kw("debris", REBAR_ICONS["debris"]))
        debris.prop(settings, "debris_density", text="Amount", slider=True)
        size_row = debris.row(align=True)
        size_row.prop(settings, "debris_min_size", text="Min")
        size_row.prop(settings, "debris_max_size", text="Max")
        shape_row = debris.row(align=True)
        shape_row.prop(settings, "debris_irregularity", text="Rough", slider=True)
        shape_row.prop(settings, "debris_concentration", text="Cluster", slider=True)
        mat_row = debris.row(align=True)
        mat_row.prop(settings, "debris_material", text="Mat", icon=REBAR_ICONS["debris_material"])
        pick = mat_row.operator(
            "fdt.rebar_pick_material",
            text="",
            **icons.icon_kw("eyedropper", REBAR_ICONS["eyedropper"]),
        )
        pick.slot = "DEBRIS"


def _draw_system(box, context, settings) -> None:
    system = box.box()
    states = st.load_all_generators(context.scene)
    title = f"System & Sets ({len(states)})" if states else "System & Sets"
    _header(system, settings, "ui_show_system", title)
    if not settings.ui_show_system:
        return

    rebar_row = system.row(align=True)
    rebar_row.prop(settings, "rebar_material", text="Bar Mat", icon=REBAR_ICONS["target"])
    rebar_row.operator("fdt.rebar_load_material", text="", icon=REBAR_ICONS["folder"])
    pick = rebar_row.operator(
        "fdt.rebar_pick_material",
        text="",
        **icons.icon_kw("eyedropper", REBAR_ICONS["eyedropper"]),
    )
    pick.slot = "REBAR"

    lim_row = system.row(align=True)
    lim_row.prop(settings, "max_bars", text="Max")
    lim_row.prop(settings, "fast_drag", text="Fast Drag", toggle=True)

    if states:
        system.separator(factor=0.5)
        for state in states.values():
            entry = state.entry
            if entry.stale:
                ico_kw = icons.icon_kw("stale_warn", REBAR_ICONS["stale"])
                tag = f"{entry.display_label} · {entry.sid8} (Stale)"
            elif entry.baked:
                ico_kw = icons.icon_kw("apply_green", REBAR_ICONS["baked"])
                tag = f"{entry.display_label} · {entry.sid8}"
            else:
                ico_kw = {"icon": REBAR_ICONS["live"]}
                tag = f"{entry.display_label} · {entry.sid8} (Live)"
            if not state.valid:
                tag += " (Missing)"
            system.label(text=tag, **ico_kw)

            ops_row = system.row(align=True)
            edit = ops_row.operator("fdt.rebar_reedit", text="Edit", icon=REBAR_ICONS["edit"])
            edit.sid = entry.sid
            faces = ops_row.operator("fdt.rebar_select_faces", text="Faces", icon=REBAR_ICONS["faces"])
            faces.sid = entry.sid
            clear = ops_row.operator("fdt.rebar_clear_faces", text="Clear", icon=REBAR_ICONS["clear"])
            clear.sid = entry.sid
            delete = ops_row.operator(
                "fdt.rebar_delete",
                text="",
                **icons.icon_kw("trash_red", REBAR_ICONS["delete"]),
            )
            delete.sid = entry.sid

    if settings.rebar_status:
        system.label(text=settings.rebar_status, icon=REBAR_ICONS["status"])


def draw(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
    *,
    force_open: bool = False,
) -> None:
    """Draw the Rebars detail inside the main panel's right-hand area."""
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

    active = st.find_entry(settings.rebar_active_sid) if settings.rebar_active_sid else None
    _draw_global_stale_banner(box, context, active)
    _draw_basics(box, context, settings)
    _draw_look(box, settings)
    _draw_advanced(box, settings)
    _draw_system(box, context, settings)
