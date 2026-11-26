r"""One-shot builder for addon/resources/rebar_material.blend.

Usage (from the repo root):

    blender.exe --background \
      "C:\Users\saetanpee\AppData\Roaming\Blender Foundation\Blender\5.2\scripts\addons\Red_Panda\Blender_Resourece\Material.blend" \
      --python scripts\build_rebar_asset.py

The source is only needed once.  The output contains just
``ST_Metal_Rebar_Ruin_01`` with its EBT custom properties and packed textures,
so the addon can append the material offline on any machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy

MATERIAL_NAME = "ST_Metal_Rebar_Ruin_01"
RESOURCE_NAME = "{1ED917F76283BA69}Assets/_SharedMaterials/Metal/ST_Metal_Rebar_Ruin_01.emat"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "addon" / "resources" / "rebar_material.blend"


def _output_path() -> Path:
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        if extra:
            return Path(extra[0]).resolve()
    return DEFAULT_OUTPUT


def _resource_name(mat) -> str:
    direct = mat.get("ebt_resource_name") or getattr(mat, "ebt_resource_name", "")
    if direct:
        return str(direct)
    group = mat.get("ebt")
    if group is not None:
        try:
            return str(group.get("resource_name", ""))
        except Exception:
            return ""
    return ""


def _find_material():
    for mat in bpy.data.materials:
        if mat.name == MATERIAL_NAME or _resource_name(mat) == RESOURCE_NAME:
            return mat
    return None


def _pack_images(mat) -> tuple[int, int]:
    """Pack every image reachable from the material, descending node groups.

    Returns ``(packed, missing)``.  Source .tif files often live outside the
    repository; missing files are reported but never block the build — a live
    Workbench connection refreshes them through EBT later.
    """
    packed = 0
    missing = 0
    seen: set[int] = set()

    def walk(tree) -> None:
        nonlocal packed, missing
        if tree is None or id(tree) in seen:
            return
        seen.add(id(tree))
        for node in tree.nodes:
            image = getattr(node, "image", None)
            if image is not None and image.source == "FILE":
                try:
                    if image.packed_file is None:
                        image.pack()
                    packed += 1
                except Exception:
                    missing += 1
            walk(getattr(node, "node_tree", None))

    if mat.use_nodes:
        walk(mat.node_tree)
    return packed, missing


def main() -> int:
    mat = _find_material()
    if mat is None:
        names = ", ".join(m.name for m in bpy.data.materials[:12])
        print(f"[build_rebar_asset] ERROR: {MATERIAL_NAME} not found. Materials: {names}")
        return 1

    print(f"[build_rebar_asset] Source material: {mat.name}")
    print(f"[build_rebar_asset] EBT resource: {_resource_name(mat) or '<none>'}")
    packed, missing = _pack_images(mat)
    print(f"[build_rebar_asset] Packed {packed} image(s), {missing} unavailable")

    # Keep exactly this material so the appended .blend stays tiny.
    for other in list(bpy.data.materials):
        if other is not mat:
            bpy.data.materials.remove(other, do_unlink=True)
    mat.use_fake_user = True
    mat.name = MATERIAL_NAME

    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_recursive=True)

    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    print(f"[build_rebar_asset] Saved {output} ({os.path.getsize(output)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
