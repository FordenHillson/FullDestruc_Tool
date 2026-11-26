"""Protected Vault shield so Zone/Visibility tools never touch working backups."""

from __future__ import annotations

import bpy

PROTECTED_PROP = "fdt_protected"
VAULT_PREFIX = "FDT_Vault_"


def mark_protected(datablock) -> None:
    """Tag a collection or object so visibility tools always skip it."""
    if datablock is not None:
        datablock[PROTECTED_PROP] = True


def unmark_protected(datablock) -> None:
    """Remove protection tag when restoring an object back to the scene."""
    if datablock is not None and PROTECTED_PROP in datablock:
        del datablock[PROTECTED_PROP]


def is_protected_collection(col: bpy.types.Collection | None) -> bool:
    """Return True if this collection is a protected backup vault."""
    if col is None:
        return False
    if bool(col.get(PROTECTED_PROP, False)):
        return True
    return col.name.startswith(VAULT_PREFIX)


def is_protected_object(obj: bpy.types.Object | None) -> bool:
    """Return True if this object is tagged protected or lives inside a protected vault."""
    if obj is None:
        return False
    if bool(obj.get(PROTECTED_PROP, False)):
        return True
    for col in obj.users_collection:
        if is_protected_collection(col):
            return True
    return False


def filter_unprotected_objects(objects) -> list[bpy.types.Object]:
    """Return only objects that visibility/zone tools are allowed to control."""
    return [obj for obj in objects if not is_protected_object(obj)]


def is_protected_layer_collection(lc: bpy.types.LayerCollection | None) -> bool:
    """Return True if a LayerCollection wraps a protected collection."""
    if lc is None:
        return False
    return is_protected_collection(lc.collection)
