# FullDestruc Tool

Fast boolean cutting and building zone manager for **Blender 5.2+**.

Splits architectural meshes into many addressable destruction pieces ("Apple Bite"),
manages ID/VIS zones, and generates Red-Panda-style concrete rebar with optional
debris — all previewed in real time through a Rust boolean engine and GPU pipeline.

## Features

- **Apple Bite** — multi-piece boolean cutting with layered gear-wheel picker,
  per-piece `Piece Only` view, auto face capping and backface-liner fix.
- **Zone Manager** — registry of built-in (`A`–`Z`) and custom zones (`z_0001`, …);
  stable keys, editable tokens, delete guards. See `CONTEXT.md` for the glossary
  and invariants.
- **Rebar Generator** — seeded per-bar bend/twist splines, face/BVH generation,
  optional concrete debris, manual bake.
- **Rust engine** — mesh booleans, BVH, retriangulation and GPU packing in
  `addon/rust_engine` (exposed via FFI, see `addon/core/rust_bridge.py`).
- **Three UI surfaces** — N-panel tool rail, floating menus/gear wheel, and a
  standalone PySide6 window, all driven by the same operators.

## Requirements

- Blender **5.2** or newer (Python 3.10+)
- Rust toolchain — only if you want to rebuild the engine (a prebuilt
  `addon/bin/fdt_bool_engine.dll` is committed)

## Install

1. Copy the `addon/` folder to Blender's addons directory, or run:

   ```powershell
   .\scripts\sync_to_blender.ps1   # copies addon/ -> Blender 5.2 addons dir
   ```

2. Restart Blender and enable **FullDestruc Tool**.
   The engine DLL in `addon/bin/` ships with the addon; after rebuilding it,
   run `.\scripts\restore_rust_engine.ps1` to deploy + verify it (SHA256 check,
   export check, minimum ABI version).

## Building the Rust engine

```powershell
cd addon/rust_engine
.\build.ps1     # cargo build --release, deploys to addon/bin/ (hot-swaps if Blender holds the DLL)
```

## Tests

Headless Blender runner covering Apple Bite and all rebar modules:

```powershell
blender.exe --factory-startup --background --python tests\run_tests.py
```

## Repository layout

```
addon/               Blender addon (manifest: blender_manifest.toml)
  core/              brain: cutting, zones, rebar, GPU preview, Rust bridge
  ops/               operators shared by all UI surfaces
  ui/                N-panel, Qt window, gear wheel, overlays
  bin/               prebuilt fdt_bool_engine.dll
  rust_engine/       Rust crate (src/, build.ps1)
tests/               headless Blender test suite
scripts/             build/sync/restore helpers
docs/                knowledge base (Obsidian vault), ADRs
CONTEXT.md           shared glossary + invariants for the zone/rebar system
```

## License

GPL-3.0-or-later
