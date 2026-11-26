# FullDestruc_Tool — Context & Glossary

This file is the shared vocabulary for the zone system. When code, docs, or a UI label says “zone”, use the terms below exactly.

## Glossary

| Term | Meaning | Stored / shown |
| --- | --- | --- |
| **Zone Definition** | One entry in the Scene-level zone registry. Contains `key`, `token`, `kind`, `base_zone`, `color`, `builtin`. | Scene ID property `fdt_zone_registry_json` |
| **Zone Key** | Stable identity of a zone. Built-ins are `A`…`Z`; customs are `z_0001`, `z_0002`, … in creation order. Never changes on rename. | `fdt_cutter_zone`, `fdt_piece_zone`, `session.cutters[].zone`, `fdt_settings.active_zone` |
| **Zone Token** | Editable user-facing label. Examples: `A`, `ID-A#B`, `VIS-A#B`, `ID-A_VIS-A#B`, free text. Uniqueness is case-insensitive, max 40 chars. | Dropdown labels, cutter/piece names, HUD badges |
| **Base Zone** | A–Z fallback for a custom token. Always selected explicitly by the user in the manage dialog; regex only pre-fills the field. | `Zone Definition.base_zone` |
| **Custom Zone** | Any non-built-in Zone Definition. Can be created, renamed and deleted. Its key starts with `z_`. | Registry JSON |
| **Rebar Generator** | One set of demolition-style rebars. Its identity is a stable `sid`, never its Outliner name. | `fdt_rebar_registry_json` (Scene), `fdt_rebar_id` (Object) |
| **Rebar Label** | User-editable cosmetic Outliner label, e.g. `ID-A_VIS-!B#A`. Renaming it detaches nothing. | `Object.name` / registry `label` |
| **Rebar Tag** | Per-host integer group marking which faces belong to a generator. | `fdt_rebar_group` (INT face attribute on host mesh) |
| **Rebar Buffer** | Scene-level live parameter buffer shown in mm in the N-Panel; generators store metres. | `Scene.fdt_rebar_settings` |
| **Rebar Bake** | Deliberate commit step: freezes params, tags the generator baked and moves it to its collection. No auto-bake timer exists. | `fdt.rebar_bake` |
| **Concrete Debris** | Optional Red Panda-style concrete chunks clinging to the exposed bars. Lives inside the same generator mesh, never a separate object. | `enable_debris` + `debris_*` in `Scene.fdt_rebar_settings` |
| **Bend Spec** | The per-bar random shape of the exposed rebar: dominant bend direction, lateral wobble direction, amplitude (Bend Amount), wavelength (Bend Scale), three wave phases, two wave scales and the signed tip twist. Drawn from the bar's seeded RNG, so a fixed seed reproduces every bar's shape exactly. | `rebar_spline.BendSpec` (in memory; frozen into geometry at Bake) |
| **Bend Scale** | Wavelength of the bend wave in metres; the spacing of the bend rhythm along the bar. UI is “Bend Scale (mm)”, 50–1000, default 200. | `bend_scale` (m, generator params) / `bend_scale_mm` (Scene buffer) |
| **Twist** | Signed axial rotation at the bar tip in radians, ramped linearly from 0 at the concrete surface. Has no slider of its own: `twist_limit = min(π, bend_amount / 0.32 · π)`, drawn per bar from the seeded RNG. | 3rd element of the spline tuple `(points, subdivisions, twist)`; frame rotation at bake |

## Invariants

1. **One registry, all surfaces.** N-Panel, floating menu, Qt window, gear wheel and operators all read `addon/core/zone_registry.py`. No UI hard-codes A–H/A–Z for zone identity.
2. **Key is identity, token is cosmetic.** Renaming a token must not detach cutters or pieces: their ID properties and session entries keep the same key.
3. **Built-ins are permanent fallback.** A–Z can be renamed/recolored but never deleted. Missing keys fall back to their base zone when the stored value is a structured token, then to `A` for arbitrary text.
4. **Creation order is registry order.** Built-ins `A`…`Z` first, then customs in creation order. Gear “Change Zone” cycles in this order and wraps.
5. **Color rule.** A custom zone copies its base zone color at creation. Changing `base_zone` later does **not** overwrite a user-edited color.
6. **Delete guard.** Deleting a custom zone is blocked when any cutter or piece references its key; the message reports both counts.
7. **Name safety.** Generated object names use the sanitized **token**, preserving `#`, `-`, `_`, and are truncated to 63 characters. `_FDST_ID-[A-Za-z0-9_#-]+` recognizes them.
8. **Kind is metadata.** `ID` / `VIS` are stored for future condition work; there is no behavior difference yet.
9. **Shared operators.** `fdt.zone_picker` is the searchable popup used by the wheel. `fdt.zone_manage` provides New / Rename / Delete everywhere. Qt calls the same operators.
10. **Rebar identity is `sid`.** The Outliner label is cosmetic; lookups, edits, deletes and stale-marking use `sid` plus `fdt_rebar_id`.
11. **Rebar faces are a tagged set.** Generation uses `selected faces ∩ faces whose material matches Target Material`, falling back to all matching faces when nothing is selected. The face-tag attribute is the authority, not polygon indices.
12. **Rebar commits only on Bake.** Property changes only schedule a debounced live preview (GPU lines during a fast drag); no timer ever bakes.
13. **Rebar stale is manual.** An Apple Bite re-cut marks affected generators `stale` only; the user must press Re-apply then Bake. Nothing auto-heals them.

14. **Rebar defaults mirror Red Panda.** Fresh buffers use 400 mm spacing, 50 mm sink, 100–500 mm random length and Ø40 mm bars (`bar_size=CUSTOM`, equal to RP `bevel_depth=0.02` radius); the friendly density dropdown starts at `SPARSE`.
15. **Rebar delete guard.** A generator cannot be deleted while its face tags remain on the host; Clear Faces must run first.
16. **Debris is part of the set.** Concrete debris is appended to the generator's single mesh on face-material slot 1. There is no second object, no name-prefix cleanup and no separate identity — Bake, Re-apply, delete guard and `sid` authority all apply unchanged.
17. **Debris is off by default.** It follows Red Panda's defaults (density 0.5, 5–40 mm, irregularity 0.4, concentration 0.7) but only appears when the user explicitly enables `Concrete Debris`.
18. **Rebar shape is per-bar reproducible.** Each bar draws its bend/twist from its own seeded stream keyed by `(seed, index)` — a *string* seed, because string seeds are SHA-512-mixed inside the random module and stay deterministic across processes and Python versions (int tuples are rejected by newer CPython). A fixed seed reproduces the identical set, and bar *n*'s shape never depends on the bars before it.
19. **Twist is derived from Bend Amount.** Allowed axial twist grows linearly with Bend Amount and saturates at a half turn (π rad) at 320 mm of bend; a zero-bend bar is perfectly straight and untwisted. There is no independent twist parameter.

## Code map

- `addon/core/zone_registry.py` — model, JSON persistence, validation, parsing, lookup/fallback, enum items, usage counts.
- `addon/core/apple_bite.py` — resolves keys for cutters/pieces, naming, cycle, colors.
- `addon/ops/zone_ops.py` — shared picker + management operators.
- `addon/props.py` — `active_zone` dynamic enum backed by the registry.
- `addon/ui/*` — every zone picker renders registry tokens/keys.
- `tests/test_apple_bite.py` — tests 18–19 cover registry behavior.
- `addon/core/rebar_spline.py`, `rebar_generate.py`, `rebar_debris.py`, `rebar_state.py`, `rebar_overlay.py` — rebar geometry, face/BVH generation, concrete debris, registry and GPU-line preview.
- `addon/ops/rebar_ops.py` — rebar operators, mm↔m conversion and material resolution.
- `addon/ui/panel_n.py` — main N-Panel with a Zen UV-style tool rail on the left (Apple Bite / Rebar / Cleanup / Batch icons) and the selected tool's detail on the right. Switching tools goes through `fdt.set_active_tool` → `fdt_settings.ui_active_tool`. The boolean Engine selector belongs to the Apple Bite detail only; it is not drawn above the other tool tabs.
- `addon/ui/panel_rebar.py` — Rebar detail rendered inside that right-hand area. Its `draw(..., force_open=True)` variant skips the standalone "Concrete Rebars" header so the rail tab opens straight into the settings.
- `tests/test_rebar*.py` — rebar geometry, generation, state and operator tests.

See also: `docs/adr/0001-zone-registry.md`, `docs/FullDestruc/02_Tools/Zone_Manager.md`.
