"""Protected Original Vault and safe temporary evaluation for Apply."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import bpy

from .zone_shield import VAULT_PREFIX, mark_protected, unmark_protected


def find_layer_collection_with_ancestors(
    root_lc: bpy.types.LayerCollection,
    target_col: bpy.types.Collection,
) -> list[bpy.types.LayerCollection] | None:
    """Return [root, ..., target_lc] path if target_col is inside root_lc."""
    if root_lc.collection == target_col:
        return [root_lc]
    for child in root_lc.children:
        path = find_layer_collection_with_ancestors(child, target_col)
        if path is not None:
            return [root_lc] + path
    return None


def find_layer_collection(
    view_layer: bpy.types.ViewLayer,
    target_col: bpy.types.Collection | None,
) -> bpy.types.LayerCollection | None:
    """Find the LayerCollection corresponding to target_col."""
    if target_col is None:
        return None
    path = find_layer_collection_with_ancestors(view_layer.layer_collection, target_col)
    return path[-1] if path else None


def get_or_create_collection(
    name: str,
    parent: bpy.types.Collection | None = None,
    color_tag: str = "NONE",
) -> bpy.types.Collection:
    """Get an existing collection by name or create and link it under parent."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if parent is None:
        parent = bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    if color_tag and hasattr(col, "color_tag"):
        col.color_tag = color_tag
    return col


def move_object_to_collection(
    obj: bpy.types.Object,
    target_col: bpy.types.Collection,
) -> None:
    """Move obj exclusively into target_col without leaving stray links."""
    for old_col in list(obj.users_collection):
        if old_col != target_col:
            old_col.objects.unlink(obj)
    if obj.name not in target_col.objects:
        target_col.objects.link(obj)


def set_collection_hidden(
    view_layer: bpy.types.ViewLayer,
    col: bpy.types.Collection | None,
    hidden: bool,
) -> None:
    """Hide or show ONLY this single collection (never touches child collections)."""
    if col is None:
        return
    col.hide_viewport = hidden
    col.hide_render = hidden
    lc = find_layer_collection(view_layer, col)
    if lc is not None:
        lc.hide_viewport = hidden


def set_collection_excluded(
    view_layer: bpy.types.ViewLayer,
    col: bpy.types.Collection | None,
    excluded: bool,
) -> None:
    """Exclude (uncheck in Outliner) + hide/disable render, or fully restore a collection."""
    if col is None:
        return
    col.hide_viewport = excluded
    col.hide_render = excluded
    lc = find_layer_collection(view_layer, col)
    if lc is not None:
        lc.exclude = excluded
        lc.hide_viewport = excluded


def create_vault_collection(
    view_layer: bpy.types.ViewLayer,
    session_id: str,
    parent_col: bpy.types.Collection,
) -> bpy.types.Collection:
    """Create a protected, hidden Vault collection for original source meshes."""
    vault_name = f"{VAULT_PREFIX}{session_id}"
    vault_col = get_or_create_collection(vault_name, parent=parent_col, color_tag="COLOR_01")
    mark_protected(vault_col)
    set_collection_hidden(view_layer, vault_col, True)
    return vault_col


def store_originals_in_vault(
    view_layer: bpy.types.ViewLayer,
    vault_col: bpy.types.Collection,
    objects: Iterable[bpy.types.Object],
) -> list[bpy.types.Object]:
    """Move original objects into the protected vault and keep them hidden safely."""
    stored: list[bpy.types.Object] = []
    mark_protected(vault_col)
    for obj in objects:
        mark_protected(obj)
        move_object_to_collection(obj, vault_col)
        stored.append(obj)
    set_collection_hidden(view_layer, vault_col, True)
    return stored


def restore_originals_from_vault(
    scene: bpy.types.Scene,
    vault_col: bpy.types.Collection | None,
) -> list[bpy.types.Object]:
    """Restore original objects back to the main scene collection and unmark them."""
    restored: list[bpy.types.Object] = []
    if vault_col is None:
        return restored
    for obj in list(vault_col.objects):
        unmark_protected(obj)
        move_object_to_collection(obj, scene.collection)
        obj.hide_viewport = False
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass
        restored.append(obj)
    return restored


@contextmanager
def temporary_vault_evaluation(
    context: bpy.types.Context,
    collections: Iterable[bpy.types.Collection | None],
):
    """Temporarily un-exclude and un-hide collections (plus ancestors) for depsgraph baking.

    Restores exact original exclude/hide_viewport states in `finally`, so Apply
    always succeeds even when the user disabled or hid the Original Vault.
    """
    view_layer = context.view_layer
    saved_col_hide: dict[bpy.types.Collection, bool] = {}
    saved_lc_state: dict[bpy.types.LayerCollection, tuple[bool, bool]] = {}

    for col in collections:
        if col is None:
            continue
        path = find_layer_collection_with_ancestors(view_layer.layer_collection, col)
        if not path:
            continue
        for lc in path:
            c = lc.collection
            if c not in saved_col_hide:
                saved_col_hide[c] = c.hide_viewport
            if lc not in saved_lc_state:
                saved_lc_state[lc] = (lc.exclude, lc.hide_viewport)

    try:
        for lc in saved_lc_state:
            lc.exclude = False
            lc.hide_viewport = False
        for col in saved_col_hide:
            col.hide_viewport = False
        view_layer.update()
        yield
    finally:
        for col, old_hide in saved_col_hide.items():
            try:
                col.hide_viewport = old_hide
            except ReferenceError:
                pass
        for lc, (old_exclude, old_lc_hide) in saved_lc_state.items():
            try:
                lc.hide_viewport = old_lc_hide
                lc.exclude = old_exclude
            except ReferenceError:
                pass
        view_layer.update()
