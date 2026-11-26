# ADR 0001 — Scene-level Zone Registry with Stable Keys

- **Status:** Accepted
- **Date:** 2026-11-24
- **Deciders:** FullDestruc_Tool maintainer

## Context

Zone identity used to be a single character duplicated across many places:

- `FDT_SceneSettings.active_zone` was a closed `A`–`Z` enum.
- Cutter/piece ID properties stored that same character.
- Session JSON stored it again.
- Object names embedded it (`_ID-A_01`).
- Wheel sub-rings, N-Panel quick buttons, Qt combos and docs all assumed one letter.
- Colors came from the first character, with `I`–`Z` wrapped onto eight colors.

Real Enfusion destruction data uses longer tokens such as `ID-A#B`, `VIS-A#B` and `ID-A_VIS-A#B`. If we simply replaced the letter stored on objects with the editable token, a later rename would silently detach cutters and pieces from their session metadata and break old files. Changing that after users have saved scenes would be expensive and surprising.

## Decision

Introduce a **scene-level zone registry** (`addon/core/zone_registry.py`) as the single source of truth.

- A `Zone Definition` has `key`, `token`, `kind`, `base_zone`, `color`, `builtin`.
- `key` is stable and is the only value stored on cutters, pieces, session entries and `active_zone`:
  - built-ins: `A`…`Z`
  - customs: `z_0001`, `z_0002`, … in creation order
- `token` is the editable display label and may contain `#`, `-`, `_` and free text. It is case-insensitively unique, max 40 chars.
- Built-in A–Z entries are restored on load, may be renamed/recolored, and can never be deleted. They guarantee a fallback for old files.
- A custom zone's `base_zone` is chosen explicitly in the manage dialog; typed tokens only pre-fill the dialog. The copied color is user-overridable and does not follow later base changes.
- Missing keys resolve to: structured token base → `A`. Arbitrary unknown text falls straight to `A`.
- All zone pickers use registry enum items. The wheel's `Zone` slice opens `fdt.zone_picker`, a searchable popup with New / Rename / Delete actions backed by `fdt.zone_manage`.
- Generated object names use the sanitized token (preserving `#`, `-`, `_`) and still respect Blender's 63-character limit.
- `kind` (`ID` / `VIS`) is stored now for future condition logic; it has no behavior difference in this change.

## Consequences

### Positive

- Renaming a token never breaks references; only the label changes.
- Old scenes continue to work: a one-character key is a valid built-in key.
- One code path drives every UI, so adding/renaming/deleting a zone is consistent.
- Long real-world tokens are visible in every picker and summary.

### Negative / follow-up

- Registry JSON lives on the Scene, so separate scenes have separate custom zones. This is intentional for now; a project-wide registry would need a different persistence layer.
- Deleting a custom zone must scan objects and session entries. This is cheap at current session sizes but should be revisited if cutter counts grow dramatically.
- `kind` is inert until the validation/condition phase consumes it.

## Alternatives considered

1. **Use the token as the stored key.** Rejected: rename would detach data and old names would need a migration.
2. **Keep A–Z plus a parallel suffix table.** Rejected: two sources of truth, same detachment risk.
3. **Hard-code more letters or a fixed list of special tokens.** Rejected: cannot represent user-defined Enfusion tokens.
