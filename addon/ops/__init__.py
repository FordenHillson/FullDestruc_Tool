"""Operator registration for FullDestruc_Tool."""

from . import apple_bite_ops, material_picker, rebar_ops, zone_ops


def register():
    material_picker.register()
    apple_bite_ops.register()
    zone_ops.register()
    rebar_ops.register()


def unregister():
    rebar_ops.unregister()
    zone_ops.unregister()
    apple_bite_ops.unregister()
    material_picker.unregister()
