"""Generate crisp 64x64 anti-aliased flat colored PNG icons for FullDestruc_Tool.

Run from the repository root:
    python3 scripts/generate_icons.py

All icons are drawn at 4x supersampling (256x256) and downsampled with LANCZOS
to 64x64 RGBA PNGs under ``addon/resources/icons/``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "addon" / "resources" / "icons"

SS = 4  # 4x supersampling
SIZE = 64
CANVAS = SIZE * SS

# Unified Semantic Palette (RGBA)
EMERALD = (46, 212, 122, 255)
EMERALD_DARK = (24, 152, 84, 255)
AMBER = (245, 158, 11, 255)
AMBER_LIGHT = (253, 198, 92, 255)
ROSE = (244, 63, 94, 255)
PURPLE = (168, 85, 247, 255)
CYAN = (56, 189, 248, 255)
TEAL = (20, 184, 166, 255)
RED = (239, 68, 68, 255)
SLATE = (168, 180, 200, 255)
WHITE = (248, 250, 252, 255)
DARK_INK = (18, 22, 28, 255)

# 8 hue buckets for dynamic Zone color dot icons (matched by Euclidean RGB distance)
ZONE_DOT_COLORS: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("zone_c0", (245, 130, 32, 255)),   # Orange (Zone A)
    ("zone_c1", (56, 176, 255, 255)),   # Sky Blue (Zone B)
    ("zone_c2", (46, 212, 122, 255)),   # Emerald Green (Zone C)
    ("zone_c3", (244, 72, 92, 255)),    # Coral Red (Zone D)
    ("zone_c4", (176, 98, 250, 255)),   # Purple (Zone E)
    ("zone_c5", (245, 208, 48, 255)),   # Yellow (Zone F)
    ("zone_c6", (34, 211, 208, 255)),   # Cyan/Teal (Zone G)
    ("zone_c7", (244, 114, 182, 255)),  # Pink (Zone H)
)


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _save(img: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    out.save(OUT_DIR / f"{name}.png", format="PNG", optimize=True)


def _circle(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, **kw) -> None:
    d.ellipse((cx - r, cy - r, cx + r, cy + r), **kw)


def _line_round(
    d: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    if len(pts) < 2:
        return
    d.line(pts, fill=fill, width=width, joint="curve")
    r = width * 0.5
    for x, y in pts:
        _circle(d, x, y, r, fill=fill)


# ---------------------------------------------------------------------------
# Icon Drawers
# ---------------------------------------------------------------------------


def draw_apply_green() -> Image.Image:
    img, d = _new_canvas()
    # Vivid emerald solid rounded badge with crisp white checkmark
    _circle(d, 128, 128, 106, fill=EMERALD)
    _circle(d, 128, 128, 106, outline=(134, 245, 184, 180), width=10)
    _line_round(d, [(78, 132), (114, 168), (182, 92)], fill=WHITE, width=26)
    return img


def draw_start_orange() -> Image.Image:
    img, d = _new_canvas()
    _circle(d, 128, 128, 104, fill=AMBER)
    d.polygon([(102, 78), (182, 128), (102, 178)], fill=WHITE)
    return img


def draw_reedit_cyan() -> Image.Image:
    img, d = _new_canvas()
    # Arc from 300 deg (bottom-right) clockwise through bottom & left to 215 deg? In PIL: 0=3 o'clock, CW.
    # Draw 270 deg arc from 0 deg (3 o'clock) CW to 270 deg (12 o'clock), arrow pointing left at 12 o'clock
    d.arc((40, 40, 216, 216), start=340, end=235, fill=CYAN, width=24)
    # Arrowhead at top-left (around 235 deg) pointing CCW/down-left
    d.polygon([(38, 108), (94, 44), (104, 106)], fill=CYAN)
    return img


def draw_pause_slate() -> Image.Image:
    img, d = _new_canvas()
    d.rounded_rectangle((62, 48, 106, 208), radius=16, fill=SLATE)
    d.rounded_rectangle((150, 48, 194, 208), radius=16, fill=SLATE)
    return img


def draw_cancel_red() -> Image.Image:
    img, d = _new_canvas()
    _circle(d, 128, 128, 102, outline=RED, width=20)
    _line_round(d, [(86, 86), (170, 170)], fill=RED, width=22)
    _line_round(d, [(170, 86), (86, 170)], fill=RED, width=22)
    return img


def draw_trash_red() -> Image.Image:
    img, d = _new_canvas()
    # Lid
    _line_round(d, [(50, 74), (206, 74)], fill=RED, width=20)
    d.rounded_rectangle((96, 42, 160, 74), radius=10, outline=RED, width=16)
    # Body
    d.polygon([(66, 92), (190, 92), (176, 214), (80, 214)], outline=RED, width=18)
    _line_round(d, [(108, 120), (112, 186)], fill=RED, width=14)
    _line_round(d, [(148, 120), (144, 186)], fill=RED, width=14)
    return img


def draw_tool_apple_bite() -> Image.Image:
    img, d = _new_canvas()
    # Isometric cube with top-right bite cut out
    d.rounded_rectangle((36, 44, 184, 212), radius=24, fill=AMBER)
    # Cut bite circle punched out in dark/transparent
    _circle(d, 184, 86, 54, fill=(0, 0, 0, 0))
    # Floating detached piece
    d.rounded_rectangle((164, 48, 224, 108), radius=14, fill=AMBER_LIGHT)
    # Crisp inner cut line
    _line_round(d, [(72, 140), (128, 140)], fill=DARK_INK, width=14)
    return img


def draw_tool_rebar() -> Image.Image:
    img, d = _new_canvas()
    # Concrete base bar on left
    d.rounded_rectangle((28, 44, 78, 212), radius=14, fill=SLATE)
    # 3 protruding bent cyan rebar rods
    _line_round(d, [(78, 78), (156, 74), (220, 56)], fill=CYAN, width=20)
    _line_round(d, [(78, 128), (168, 134), (224, 152)], fill=CYAN, width=20)
    _line_round(d, [(78, 178), (148, 182), (204, 206)], fill=CYAN, width=20)
    return img


def draw_tool_cleanup() -> Image.Image:
    img, d = _new_canvas()
    # Four-point sparkle star + smaller accent star
    d.polygon(
        [(128, 28), (154, 102), (228, 128), (154, 154), (128, 228), (102, 154), (28, 128), (102, 102)],
        fill=TEAL,
    )
    _circle(d, 194, 62, 18, fill=CYAN)
    return img


def draw_tool_batch() -> Image.Image:
    img, d = _new_canvas()
    for y in (52, 114, 176):
        d.rounded_rectangle((40, y, 216, y + 34), radius=12, fill=SLATE)
    return img


def draw_add_box() -> Image.Image:
    img, d = _new_canvas()
    # Clean outlined cube
    d.rounded_rectangle((36, 56, 178, 198), radius=20, outline=AMBER, width=20)
    # Plus badge top-right
    _circle(d, 188, 68, 44, fill=AMBER)
    _line_round(d, [(166, 68), (210, 68)], fill=DARK_INK, width=16)
    _line_round(d, [(188, 46), (188, 90)], fill=DARK_INK, width=16)
    return img


def draw_use_mesh() -> Image.Image:
    img, d = _new_canvas()
    # Isometric wireframe hex/cube
    pts = [(128, 32), (212, 80), (212, 176), (128, 224), (44, 176), (44, 80)]
    d.polygon(pts, outline=AMBER, width=18)
    _line_round(d, [(128, 128), (128, 224)], fill=AMBER, width=16)
    _line_round(d, [(128, 128), (44, 80)], fill=AMBER, width=16)
    _line_round(d, [(128, 128), (212, 80)], fill=AMBER, width=16)
    return img


def draw_draw_pen() -> Image.Image:
    img, d = _new_canvas()
    # Diagonal pen body
    d.polygon([(52, 204), (72, 136), (172, 36), (220, 84), (120, 184)], fill=ROSE)
    # Tip cut highlight
    _line_round(d, [(72, 136), (120, 184)], fill=DARK_INK, width=14)
    # Freehand curve stroke underneath
    _line_round(d, [(40, 218), (132, 218), (216, 206)], fill=(251, 146, 168, 255), width=16)
    return img


def draw_click_place() -> Image.Image:
    img, d = _new_canvas()
    _circle(d, 128, 128, 66, outline=CYAN, width=18)
    _circle(d, 128, 128, 22, fill=CYAN)
    _line_round(d, [(128, 26), (128, 66)], fill=CYAN, width=18)
    _line_round(d, [(128, 190), (128, 230)], fill=CYAN, width=18)
    _line_round(d, [(26, 128), (66, 128)], fill=CYAN, width=18)
    _line_round(d, [(190, 128), (230, 128)], fill=CYAN, width=18)
    return img


def draw_rough_edge() -> Image.Image:
    img, d = _new_canvas()
    _line_round(
        d,
        [(28, 146), (68, 72), (112, 184), (156, 68), (196, 168), (228, 114)],
        fill=PURPLE,
        width=24,
    )
    return img


def draw_view_eye() -> Image.Image:
    img, d = _new_canvas()
    # Almond eye outline
    pts_top = [(26, 128), (82, 74), (174, 74), (230, 128)]
    pts_bot = [(26, 128), (82, 182), (174, 182), (230, 128)]
    _line_round(d, pts_top, fill=CYAN, width=20)
    _line_round(d, pts_bot, fill=CYAN, width=20)
    _circle(d, 128, 128, 34, fill=CYAN)
    _circle(d, 128, 128, 14, fill=DARK_INK)
    return img


def draw_zone_grid() -> Image.Image:
    img, d = _new_canvas()
    for pos in (60, 128, 196):
        _line_round(d, [(pos, 36), (pos, 220)], fill=CYAN, width=16)
        _line_round(d, [(36, pos), (220, pos)], fill=CYAN, width=16)
    return img


def draw_layers() -> Image.Image:
    img, d = _new_canvas()
    # Top diamond
    d.polygon([(128, 36), (220, 88), (128, 140), (36, 88)], fill=SLATE)
    # Middle chevron
    _line_round(d, [(40, 134), (128, 182), (216, 134)], fill=CYAN, width=18)
    # Bottom chevron
    _line_round(d, [(40, 178), (128, 224), (216, 178)], fill=SLATE, width=18)
    return img


def draw_settings_gear() -> Image.Image:
    img, d = _new_canvas()
    cx, cy = 128.0, 128.0
    for i in range(8):
        ang = i * (math.pi / 4.0)
        x0 = cx + math.cos(ang) * 66
        y0 = cy + math.sin(ang) * 66
        x1 = cx + math.cos(ang) * 98
        y1 = cy + math.sin(ang) * 98
        _line_round(d, [(x0, y0), (x1, y1)], fill=SLATE, width=26)
    _circle(d, cx, cy, 68, fill=SLATE)
    _circle(d, cx, cy, 32, fill=(0, 0, 0, 0))
    return img


def draw_window_qt() -> Image.Image:
    img, d = _new_canvas()
    d.rounded_rectangle((32, 42, 224, 214), radius=20, outline=SLATE, width=18)
    d.rectangle((38, 48, 218, 92), fill=SLATE)
    _circle(d, 68, 70, 10, fill=DARK_INK)
    _circle(d, 98, 70, 10, fill=DARK_INK)
    return img


def draw_eyedropper() -> Image.Image:
    img, d = _new_canvas()
    # Bulb
    _circle(d, 190, 66, 36, fill=TEAL)
    # Collar
    _line_round(d, [(140, 74), (182, 116)], fill=WHITE, width=20)
    # Tube
    _line_round(d, [(154, 102), (68, 188)], fill=TEAL, width=28)
    # Tip
    d.polygon([(40, 216), (62, 172), (84, 194)], fill=CYAN)
    return img


def draw_mat_alert_red() -> Image.Image:
    img, d = _new_canvas()
    # Alert triangle in red with white exclamation mark
    d.polygon([(128, 30), (228, 214), (28, 214)], fill=RED)
    _line_round(d, [(128, 96), (128, 152)], fill=WHITE, width=22)
    _circle(d, 128, 186, 13, fill=WHITE)
    return img


def draw_fill_cap() -> Image.Image:
    img, d = _new_canvas()
    d.rounded_rectangle((42, 42, 214, 214), radius=24, fill=(20, 184, 166, 120), outline=TEAL, width=20)
    _line_round(d, [(86, 132), (118, 164), (174, 96)], fill=WHITE, width=20)
    return img


def draw_rename_pen() -> Image.Image:
    img, d = _new_canvas()
    d.polygon([(46, 210), (66, 146), (168, 44), (212, 88), (110, 190)], fill=SLATE)
    _line_round(d, [(66, 146), (110, 190)], fill=DARK_INK, width=14)
    return img


def draw_plus_add() -> Image.Image:
    img, d = _new_canvas()
    _line_round(d, [(48, 128), (208, 128)], fill=EMERALD, width=28)
    _line_round(d, [(128, 48), (128, 208)], fill=EMERALD, width=28)
    return img


def draw_zones_grid() -> Image.Image:
    img, d = _new_canvas()
    d.rounded_rectangle((36, 36, 114, 114), radius=16, fill=AMBER)
    d.rounded_rectangle((142, 36, 220, 114), radius=16, fill=CYAN)
    d.rounded_rectangle((36, 142, 114, 220), radius=16, fill=EMERALD)
    d.rounded_rectangle((142, 142, 220, 220), radius=16, fill=PURPLE)
    return img


def draw_output_box() -> Image.Image:
    img, d = _new_canvas()
    # Tray bottom
    _line_round(d, [(42, 150), (42, 208), (214, 208), (214, 150)], fill=TEAL, width=20)
    # Upward export arrow
    _line_round(d, [(128, 166), (128, 48)], fill=CYAN, width=22)
    _line_round(d, [(82, 92), (128, 46), (174, 92)], fill=CYAN, width=22)
    return img


def draw_debris() -> Image.Image:
    img, d = _new_canvas()
    d.polygon([(44, 132), (96, 68), (154, 108), (124, 184), (58, 176)], fill=AMBER)
    d.polygon([(162, 60), (212, 74), (202, 124), (160, 110)], fill=AMBER_LIGHT)
    d.polygon([(154, 154), (216, 150), (196, 212), (144, 198)], fill=SLATE)
    return img


def draw_stale_warn() -> Image.Image:
    img, d = _new_canvas()
    d.polygon([(128, 30), (228, 214), (28, 214)], fill=AMBER)
    _line_round(d, [(128, 96), (128, 152)], fill=DARK_INK, width=22)
    _circle(d, 128, 186, 13, fill=DARK_INK)
    return img


def draw_rebuild_amber() -> Image.Image:
    img, d = _new_canvas()
    # Clockwise arc from 305 deg (top-right) -> bottom -> left -> top (270 deg), arrowhead at top-right pointing CW/down-right
    d.arc((40, 40, 216, 216), start=305, end=200, fill=AMBER, width=24)
    d.polygon([(218, 108), (162, 44), (152, 106)], fill=AMBER)
    return img


def draw_zone_dot(color: tuple[int, int, int, int]) -> Image.Image:
    img, d = _new_canvas()
    _circle(d, 128, 128, 86, fill=color)
    _circle(d, 128, 128, 86, outline=(255, 255, 255, 190), width=14)
    return img


GENERATORS = {
    "apply_green": draw_apply_green,
    "start_orange": draw_start_orange,
    "reedit_cyan": draw_reedit_cyan,
    "pause_slate": draw_pause_slate,
    "cancel_red": draw_cancel_red,
    "trash_red": draw_trash_red,
    "tool_apple_bite": draw_tool_apple_bite,
    "tool_rebar": draw_tool_rebar,
    "tool_cleanup": draw_tool_cleanup,
    "tool_batch": draw_tool_batch,
    "add_box": draw_add_box,
    "use_mesh": draw_use_mesh,
    "draw_pen": draw_draw_pen,
    "click_place": draw_click_place,
    "rough_edge": draw_rough_edge,
    "view_eye": draw_view_eye,
    "zone_grid": draw_zone_grid,
    "layers": draw_layers,
    "settings_gear": draw_settings_gear,
    "window_qt": draw_window_qt,
    "eyedropper": draw_eyedropper,
    "mat_alert_red": draw_mat_alert_red,
    "fill_cap": draw_fill_cap,
    "rename_pen": draw_rename_pen,
    "plus_add": draw_plus_add,
    "zones_grid": draw_zones_grid,
    "output_box": draw_output_box,
    "debris": draw_debris,
    "stale_warn": draw_stale_warn,
    "rebuild_amber": draw_rebuild_amber,
}


def main() -> None:
    count = 0
    for name, fn in GENERATORS.items():
        _save(fn(), name)
        count += 1
    for zname, zcol in ZONE_DOT_COLORS:
        _save(draw_zone_dot(zcol), zname)
        count += 1
    print(f"Generated {count} icons in {OUT_DIR}")


if __name__ == "__main__":
    main()
