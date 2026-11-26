"""Scene properties for FullDestruc_Tool."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .core import apple_bite, rebar_state

ZONE_ITEMS = [(chr(c), f"Zone {chr(c)}", f"Assign cutters to Zone {chr(c)}") for c in range(ord("A"), ord("Z") + 1)]

_BAR_SIZE_ITEMS = [
    ("RB6", "RB6", "เส้นเหล็กหนา 6 มม."),
    ("RB9", "RB9", "เส้นเหล็กหนา 9 มม."),
    ("DB10", "DB10", "เส้นเหล็กหนา 10 มม."),
    ("DB12", "DB12", "เส้นเหล็กหนา 12 มม. ค่าที่ใช้บ่อย"),
    ("DB16", "DB16", "เส้นเหล็กหนา 16 มม."),
    ("DB20", "DB20", "เส้นเหล็กหนา 20 มม."),
    ("DB25", "DB25", "เส้นเหล็กหนา 25 มม."),
    ("CUSTOM", "Custom", "พิมพ์ความหนาเองเป็นมิลลิเมตร — ค่าเริ่มต้น 40 มม. ตาม Red Panda"),
]

# One friendly control replaces the three per-axis spacing fields for normal use.
DENSITY_SPACING_MM = {
    "FINE": 75.0,
    "STANDARD": 150.0,
    "COARSE": 250.0,
    "SPARSE": 400.0,
}
_DENSITY_ITEMS = [
    ("FINE", "Fine · 75 mm", "วางถี่ เหมาะกับผิวเล็ก ๆ"),
    ("STANDARD", "Standard · 150 mm", "ระยะที่ใช้ได้ทั่วไป"),
    ("COARSE", "Coarse · 250 mm", "วางห่าง เหล็กจะดูโปร่ง"),
    ("SPARSE", "Sparse · 400 mm", "วางห่างมาก ได้แท่งน้อย"),
    ("CUSTOM", "Custom", "กำหนดระยะเองในส่วนขั้นสูง"),
]
_LOADING_DENSITY = False


def _zone_items(self, context):
    """Dynamic zone list from the Scene registry: identifier=key, label=token."""
    from .core import zone_registry

    scene = getattr(context, "scene", None) if context is not None else None
    return zone_registry.iter_zone_items(scene)


def _on_live_setting_changed(self, context):
    if context is not None and context.scene is not None:
        apple_bite.rebuild_session(context)


def _on_global_rough_changed(self, context):
    if context is not None and context.scene is not None:
        apple_bite.set_all_cutters_rough(context, bool(self.enable_rough))


def _on_rebar_setting_changed(self, context):
    """Schedule a rebar preview — never build geometry inside a callback."""
    from .ops import rebar_ops

    rebar_ops.request_rebuild()


def _on_rebar_bar_size_changed(self, context):
    """Preset diameters stay in sync with the manual diameter field."""
    size = str(getattr(self, "bar_size", "CUSTOM") or "CUSTOM")
    if size in rebar_state.BAR_SIZES:
        try:
            self.diameter_mm = rebar_state.BAR_SIZES[size]
        except Exception:
            pass
    _on_rebar_setting_changed(self, context)


def _on_rebar_density_changed(self, context):
    """One dropdown fills all three spacing fields; no geometry work here."""
    global _LOADING_DENSITY
    preset = str(getattr(self, "density_preset", "SPARSE") or "SPARSE")
    spacing = DENSITY_SPACING_MM.get(preset)
    if spacing is None:
        _on_rebar_setting_changed(self, context)
        return
    _LOADING_DENSITY = True
    try:
        self.spacing_x = spacing
        self.spacing_y = spacing
        self.spacing_z = spacing
    finally:
        _LOADING_DENSITY = False
    _on_rebar_setting_changed(self, context)


def _on_rebar_spacing_changed(self, context):
    """Editing the detailed fields switches the friendly dropdown to Custom."""
    if not _LOADING_DENSITY:
        try:
            if str(getattr(self, "density_preset", "")) != "CUSTOM":
                self.density_preset = "CUSTOM"
        except Exception:
            pass
    _on_rebar_setting_changed(self, context)


def _on_preview_display_changed(self, context):
    """Instant 0ms display-mode & Zone Grid update (zero Rust/Boolean recompute)."""
    from .core import gpu_preview
    from .core.session import load_session

    disp_mode = str(getattr(self, "preview_display_mode", "ALL") or "ALL")
    show_grid = bool(getattr(self, "show_zone_grid", True))
    grid_size = float(getattr(self, "zone_grid_size", 0.35) or 0.35)
    gpu_preview.set_display_options(
        display_mode=disp_mode,
        show_grid=show_grid,
        grid_size=grid_size,
    )

    if context is None or context.scene is None:
        return

    session = load_session(context.scene)
    cols = session.get_collections() if session is not None else {}
    res_col = cols.get("results")
    pcs_col = cols.get("pieces")

    # MAIN_WIRE: only the removed/cut pieces become wireframe; everything else stays solid.
    want_piece_display = "WIRE" if disp_mode == "MAIN_WIRE" else "TEXTURED"
    if pcs_col is not None:
        for o in pcs_col.objects:
            if o.type == "MESH" and o.display_type != want_piece_display:
                try:
                    o.display_type = want_piece_display
                except Exception:
                    pass

    if not gpu_preview.is_active():
        hide_res = disp_mode == "PIECES"
        hide_pcs = disp_mode == "MAIN"
        if res_col is not None:
            for o in res_col.objects:
                if o.type == "MESH" and bool(o.hide_get()) != hide_res:
                    try:
                        o.hide_set(hide_res)
                    except Exception:
                        pass
        if pcs_col is not None:
            sel_zones = gpu_preview.get_selected_cutter_zones() if disp_mode == "PIECES" else set()
            for o in pcs_col.objects:
                if o.type == "MESH":
                    if disp_mode == "PIECES" and sel_zones:
                        pz = str(o.get("fdt_piece_zone", "A") or "A").strip() or "A"
                        want_hide = pz not in sel_zones
                    else:
                        want_hide = hide_pcs
                    if bool(o.hide_get()) != want_hide:
                        try:
                            o.hide_set(want_hide)
                        except Exception:
                            pass
        # Tag viewport redraw for direct-mesh Zone Grid overlay
        wm = getattr(bpy.context, "window_manager", None)
        if wm is not None:
            for win in wm.windows:
                if win.screen is None:
                    continue
                for area in win.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()


class FDT_SceneSettings(bpy.types.PropertyGroup):
    # Zone selection for new/selected cutters
    active_zone: EnumProperty(
        name="Active Zone",
        description="Zone tagged onto new cutters — from the Scene zone registry (type to search)",
        items=_zone_items,
        default=0,  # Blender requires an index default for dynamic EnumProperty items
    )

    # Cutter sizing
    cutter_size: FloatProperty(
        name="Cutter Size",
        description="Default size (meters) of newly spawned box cutters",
        default=2.0,
        min=0.05,
        max=50.0,
    )

    # Rough edge (Edge-band inward Geometry Nodes displacement)
    enable_rough: BoolProperty(
        name="Rough Edge (All)",
        description="Toggle rough noise edge on all cutters (each cutter can also be toggled individually)",
        default=False,
        update=_on_global_rough_changed,
    )
    rough_inward_ratio: FloatProperty(
        name="Inward Ratio",
        description="Proportion of outer shell band eating inward from the outer surface (0% = off, 100% = full depth)",
        subtype="FACTOR",
        default=0.35,
        min=0.0,
        max=1.0,
        update=_on_live_setting_changed,
    )
    rough_carve_amount: FloatProperty(
        name="Carve Amount",
        description="White-noise carve threshold (0% = none, 50% = balanced white chunks carved out, 100% = heavy erosion)",
        subtype="FACTOR",
        default=0.50,
        min=0.0,
        max=1.0,
        update=_on_live_setting_changed,
    )
    rough_strength: FloatProperty(
        name="Rough Strength",
        description="Deprecated legacy displacement depth kept for backward compatibility",
        default=0.08,
        min=0.0,
        max=2.0,
        update=_on_live_setting_changed,
    )
    rough_scale: FloatProperty(
        name="Noise Scale",
        description="Frequency / chunk size of 4D carve-out noise",
        default=3.0,
        min=0.1,
        max=50.0,
        update=_on_live_setting_changed,
    )
    rough_detail: FloatProperty(
        name="Noise Detail",
        description="Jaggedness and fine detail of rough cut noise (controls Noise Detail & Roughness)",
        subtype="FACTOR",
        default=0.5,
        min=0.0,
        max=1.0,
        update=_on_live_setting_changed,
    )
    rough_subdiv: IntProperty(
        name="Resolution",
        description="Volume grid resolution level for white-noise carve chunks (WYSIWYG in live preview & apply, 1-5)",
        default=3,
        min=1,
        max=5,
        update=_on_live_setting_changed,
    )
    rough_voxel_size: FloatProperty(
        name="Voxel Size",
        description="Deprecated legacy resolution parameter kept for backward compatibility",
        default=0.06,
        min=0.01,
        max=0.5,
        update=_on_live_setting_changed,
    )
    rough_seed: FloatProperty(
        name="Seed",
        description="Random seed for rough cut variation",
        default=0.0,
        update=_on_live_setting_changed,
    )

    # Cut face Material & Texel UV
    cut_material: PointerProperty(
        name="Cut Material",
        description="Material automatically assigned to cut faces",
        type=bpy.types.Material,
        update=_on_live_setting_changed,
    )
    texel_per_meter: FloatProperty(
        name="Texel / Meter",
        description="Target texel density for cut-face UV mapping",
        default=512.0,
        min=16.0,
        max=4096.0,
        update=_on_live_setting_changed,
    )
    texture_size: FloatProperty(
        name="Texture Size",
        description="Texture resolution in pixels for UV scaling",
        default=1024.0,
        min=64.0,
        max=8192.0,
        update=_on_live_setting_changed,
    )

    # Boolean solver & Apply options
    boolean_solver: EnumProperty(
        name="Boolean Engine",
        description="Boolean solver used during live preview and Apply",
        items=[
            ("RUST", "Rust Engine", "Ultra-fast multi-threaded Rust Boolean engine (preserves UVs & Quads)"),
            ("EXACT", "Blender Exact", "Blender built-in Exact boolean solver"),
            ("FLOAT", "Blender Float (Fast)", "Blender built-in Fast boolean solver"),
        ],
        default="RUST",
        update=_on_live_setting_changed,
    )
    gpu_fast_drag: BoolProperty(
        name="GPU Fast Drag (60+ FPS)",
        description="Render multi-material Textured GPU preview continuously at 60-144+ FPS with zero switching (commits real mesh on Apply/Exit)",
        default=True,
        update=_on_live_setting_changed,
    )
    apply_keep_result: BoolProperty(
        name="Keep Cut Main Mesh",
        description="Keep the main mesh with holes cut out on Apply",
        default=True,
    )
    apply_keep_pieces: BoolProperty(
        name="Keep Cut Pieces (Zones)",
        description="Keep separated cut pieces tagged with their Zone IDs on Apply",
        default=True,
    )
    apply_auto_fill_caps: BoolProperty(
        name="Auto-Fill Cut Gaps",
        description="Automatically seal remaining unfilled solid cavities strictly on cut planes during Apply (never closes windows or hollow rooms)",
        default=True,
    )
    apply_line_backfaces: BoolProperty(
        name="Close See-Through Backfaces",
        description="Automatically add welded inner liner faces with Cut Material + Box UV behind single-sided walls/attics exposed by the cut (fixes game Backface Culling see-through)",
        default=True,
    )

    # UI Section toggles
    ui_show_full_dst: BoolProperty(name="Full dst", default=True)
    ui_show_cleanup: BoolProperty(name="Cleanup", default=False)
    ui_show_batch: BoolProperty(name="Batch", default=False)

    # Zen UV-style tool rail: which tool's detail is shown on the right.
    ui_active_tool: EnumProperty(
        name="Active Tool",
        description="เลือกเครื่องมือที่แสดงรายละเอียดด้านขวา — เช่น Apple Bite สำหรับตัดผิว หรือ Rebar สำหรับใส่เหล็ก",
        items=[
            ("APPLE_BITE", "Apple Bite", "ตัดผิวและแบ่งโซน", "MOD_BOOLEAN", 0),
            ("REBAR", "Rebar", "ใส่เหล็กโผล่จากผิวคอนกรีต", "MOD_CURVE", 1),
            ("CLEANUP", "Cleanup", "จัดการวัสดุและเก็บงาน", "BRUSH_DATA", 2),
            ("BATCH", "Batch", "งานหลายชิ้น (ยังไม่เปิด)", "MOD_ARRAY", 3),
        ],
        default="APPLE_BITE",
    )

    # Live Zone Grid GPU Overlay & Zero-Lag Preview View Filter
    show_zone_grid: BoolProperty(
        name="Zone Grid",
        description="Project a zone-colored grid overlay onto mesh areas overlapped by active cutters",
        default=True,
        update=_on_preview_display_changed,
    )
    zone_grid_size: FloatProperty(
        name="Grid Size",
        description="World-space cell size (meters) of the projected Zone Grid overlay",
        default=0.35,
        min=0.05,
        max=5.0,
        update=_on_preview_display_changed,
    )
    preview_display_mode: EnumProperty(
        name="Preview View",
        description="Choose what to show in the 3D Viewport while moving/editing cutters (zero lag)",
        items=[
            ("ALL", "All", "Show main building + cut piece(s) + cutters"),
            ("PIECES", "Piece Only", "Show ONLY the cut piece(s) + cutter box (hide main building)"),
            ("MAIN", "Hole Only", "Show ONLY the main building with the cut hole + cutter box"),
            ("MAIN_WIRE", "Hole + Wire", "Main building solid + the removed/cut pieces shown as wireframe in zone color"),
        ],
        default="ALL",
        update=_on_preview_display_changed,
    )

    # Reserved status message
    last_status: StringProperty(name="Status", default="Ready")


class FDT_RebarSettings(bpy.types.PropertyGroup):
    """Concrete Rebars live buffer.

    RNA values are kept in **millimetres** so the N-Panel shows the numbers the
    artist expects.  Conversion to metres happens once, in
    ``ops.rebar_ops.settings_to_params()``.

    Every ``description`` below is written for a person, not a programmer:
    everyday words, one short sentence, then a concrete "เช่น ..." example.
    """

    # --- Section headers (these only open/close parts of the panel) ---
    ui_show_rebar: BoolProperty(
        name="Concrete Rebars",
        description="เปิดหรือยุบแผงใส่เหล็กทั้งหมด",
        default=False,
    )
    ui_show_basics: BoolProperty(
        name="พื้นฐาน",
        description="ส่วนที่ต้องตั้งก่อนเสมอ ผิวที่จะใส่เหล็ก ความหนา และระยะห่าง",
        default=True,
    )
    ui_show_look: BoolProperty(
        name="ลุคเหล็ก",
        description="ส่วนปรับลักษณะเหล็ก เช่น ความโค้ง ความยาว และการโผล่",
        default=True,
    )
    ui_show_advanced: BoolProperty(
        name="ขั้นสูง",
        description="ค่าละเอียดสำหรับคนที่อยากคุมเอง ไม่ต้องเปิดก็ได้",
        default=False,
    )
    ui_show_system: BoolProperty(
        name="ระบบและรายการชุด",
        description="ผิวของเส้นเหล็ก จำนวนสูงสุด และรายการชุดที่เคยสร้าง",
        default=False,
    )

    # --- Basics ---
    target_material: PointerProperty(
        name="Target Mat",
        description=(
            "ผิวแบบไหนที่อยากให้เหล็กโผล่ — ระบบจะเลือกเฉพาะแผ่นที่ใช้วัสดุนี้ "
            "ถ้าไม่ได้เลือกแผ่นไว้ก็จะใช้ทุกแผ่นของวัสดุนี้"
        ),
        type=bpy.types.Material,
        update=_on_rebar_setting_changed,
    )
    rebar_label: StringProperty(
        name="Label",
        description=(
            "ชื่อเรียกชุดนี้ในหน้าตัดรายการ ตั้งเป็น ID-A_VIS-!B#A ก็ได้ "
            "เปลี่ยนชื่อไม่ทำให้เหล็กหาย เพราะระบบจำด้วยรหัสประจำชุด"
        ),
        default="",
        maxlen=120,
    )
    density_preset: EnumProperty(
        name="Density",
        description=(
            "บอกว่าจะวางเหล็กถี่หรือห่างบนผิว เป็นมิลลิเมตร — "
            "ลองเทียบตะปูที่ตอกห่าง 40 ซม. กับห่าง 7.5 ซม. "
            "อันหลังจะได้ตะปูมากกว่าประมาณสี่เท่า"
        ),
        items=_DENSITY_ITEMS,
        default="SPARSE",
        update=_on_rebar_density_changed,
    )
    bar_size: EnumProperty(
        name="Bar",
        description="ขนาดเส้นเหล็กที่ใช้บ่อย — ตัวเลือกนี้จะเปลี่ยนความหนาให้โดยอัตโนมัติ ค่าตั้งต้น 40 มม. ตาม Red Panda",
        items=_BAR_SIZE_ITEMS,
        default="CUSTOM",
        update=_on_rebar_bar_size_changed,
    )

    # --- Look ---
    sink_depth: FloatProperty(
        name="Sink (mm)",
        description=(
            "ตอกลงไปในผิวก่อนปล่อยให้โผล่กี่มิลลิเมตร — "
            "50 คือจม 5 ซม.; 0 คือเริ่มที่ผิวเลย"
        ),
        default=50.0,
        min=0.0,
        update=_on_rebar_setting_changed,
    )
    min_length: FloatProperty(
        name="Min (mm)",
        description=(
            "ความยาวสั้นสุดของเหล็กแต่ละแท่งเป็นมิลลิเมตร ระบบจะสุ่มให้ไม่เท่ากัน — "
            "100 คือสั้นสุด 10 ซม."
        ),
        default=100.0,
        min=0.0,
        update=_on_rebar_setting_changed,
    )
    max_length: FloatProperty(
        name="Max (mm)",
        description=(
            "ความยาวยาวสุดของเหล็กแต่ละแท่งเป็นมิลลิเมตร ระบบจะสุ่มให้ไม่เท่ากัน — "
            "500 คือยาวสุด 50 ซม."
        ),
        default=500.0,
        min=0.0,
        update=_on_rebar_setting_changed,
    )
    bending_strength: FloatProperty(
        name="Bend (mm)",
        description=(
            "ดัดเหล็กไปข้าง ๆ กี่มิลลิลิเมตร — 0 คือตรง; 30 คือปลายงอนนอย ๆ เหมือนเหล็กท่ีถูกเศษคอนกรีตทาบคา่ยังกำหนดเงื่่่อนขั่่นการบิด: ยิ่่งดัดมาก ปลายยิ่่งบิดขั่่นเกลียว"
        ),
        default=30.0,
        min=0.0,
        update=_on_rebar_setting_changed,
    )
    bend_scale_mm: FloatProperty(
        name="Bend Scale (mm)",
        description=(
            "ระยะระหว่่างจังหวะดัดย่อย — ยิ่งเล็กลายแกว่่งถี่ ยิ่งใหญ่โค้งสะอาด เชิงเดียว 400 คือโค้งเรียบพอดีสำหรับเหล็กยาว 10–50 ซม."
        ),
        default=400.0,
        min=50.0,
        max=1000.0,
        update=_on_rebar_setting_changed,
    )
    random_seed: IntProperty(
        name="Seed",
        description=(
            "เลขเริ่มของการสุ่ม — ใช้เลขเดิมได้ลายเดิม "
            "เปลี่ยนเลขได้ลายใหม่โดยอย่างอื่นไม่เปลี่ยน"
        ),
        default=0,
        update=_on_rebar_setting_changed,
    )

    # --- Concrete debris (RP-style, off by default) ---
    enable_debris: BoolProperty(
        name="Concrete Debris",
        description=(
            "ให้มีก้อนคอนกรีตเล็ก ๆ ติดอยู่กับเหล็กที่โผล่มา — "
            "เปิดคือเห็นเศษแตก ๆ ติดแท่งเหล็ก เช่น เศษคอนกรีตหลุดติดปลายเหล็ก; "
            "ปิดแล้วเหล็กเปล่า"
        ),
        default=False,
        update=_on_rebar_setting_changed,
    )
    debris_density: FloatProperty(
        name="Debris Amount",
        description=(
            "ความหนาแน่นของก้อนเศษตามแท่งเหล็ก — "
            "0.5 คือพอดีแบบ Red Panda; 1 คือเยอะสุด เช่น เหล็ก 1 เมตรจะมีเศษราว 15 ก้อน"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
        update=_on_rebar_setting_changed,
    )
    debris_min_size: FloatProperty(
        name="Debris Min (mm)",
        description=(
            "ขนาดก้อนเศษที่เล็กสุดเป็นมิลลิเมตร — "
            "5 คือเม็ดเล็ก ๆ; 20 คือก้อนใหญ่ขึ้นเห็นชัด"
        ),
        default=5.0,
        min=0.1,
        max=200.0,
        update=_on_rebar_setting_changed,
    )
    debris_max_size: FloatProperty(
        name="Debris Max (mm)",
        description=(
            "ขนาดก้อนเศษที่ใหญ่สุดเป็นมิลลิเมตร — "
            "40 คือก้อนเศษราว 4 ซม. ที่ติดปลายเหล็กได้"
        ),
        default=40.0,
        min=0.1,
        max=200.0,
        update=_on_rebar_setting_changed,
    )
    debris_irregularity: FloatProperty(
        name="Debris Rough",
        description=(
            "ความขรุขระของก้อนเศษ — "
            "0 คือก้อนเรียบ; 1 คือแตกเป็นเหลี่ยมไม่เป็นรูป"
        ),
        default=0.4,
        min=0.0,
        max=1.0,
        update=_on_rebar_setting_changed,
    )
    debris_concentration: FloatProperty(
        name="Debris Cluster",
        description=(
            "ให้ก้อนเศษอยู่ใกล้ปากรอยแตกมากแค่ไหน — "
            "0 คือกระจายตามแท่งเหล็ก; 1 คือมารวมกันตรงผิวคอนกรีต"
        ),
        default=0.7,
        min=0.0,
        max=1.0,
        update=_on_rebar_setting_changed,
    )

    # --- Advanced ---
    diameter_mm: FloatProperty(
        name="Diameter (mm)",
        description=(
            "ความหนาของเส้นเหล็กเป็นมิลลิเมตร — "
            "ค่าเริ่มต้น 40 มม. เท่ากับเหล็กใน Red Panda; "
            "เส้นหนา 12 มม. จะดูเล็กลง ไม่ต้องสนใจชื่อย่อข้างช่อง"
        ),
        default=40.0,
        min=0.1,
        max=200.0,
        update=_on_rebar_setting_changed,
    )
    align_x: BoolProperty(
        name="Align X",
        description="ให้ระบบลองเสียบเหล็กตามด้านนี้ไหม — ผิวราบใช้สองด้านพอ ถ้าเป็นก้อนหนาให้เปิดทั้งสามด้าน",
        default=True,
        update=_on_rebar_setting_changed,
    )
    align_y: BoolProperty(
        name="Align Y",
        description="ให้ระบบลองเสียบเหล็กตามด้านนี้ไหม — ผิวราบใช้สองด้านพอ ถ้าเป็นก้อนหนาให้เปิดทั้งสามด้าน",
        default=True,
        update=_on_rebar_setting_changed,
    )
    align_z: BoolProperty(
        name="Align Z",
        description="ให้ระบบลองเสียบเหล็กตามด้านนี้ไหม — ผิวราบใช้สองด้านพอ ถ้าเป็นก้อนหนาให้เปิดทั้งสามด้าน",
        default=True,
        update=_on_rebar_setting_changed,
    )
    spacing_x: FloatProperty(
        name="Spacing X (mm)",
        description="ระยะห่างตามด้านนี้เป็นมิลลิเมตร — ถ้าปรับเอง ตัวเลือกระยะห่างด้านบนจะกลายเป็นกำหนดเอง",
        default=400.0,
        min=1.0,
        update=_on_rebar_spacing_changed,
    )
    spacing_y: FloatProperty(
        name="Spacing Y (mm)",
        description="ระยะห่างตามด้านนี้เป็นมิลลิเมตร — ถ้าปรับเอง ตัวเลือกระยะห่างด้านบนจะกลายเป็นกำหนดเอง",
        default=400.0,
        min=1.0,
        update=_on_rebar_spacing_changed,
    )
    spacing_z: FloatProperty(
        name="Spacing Z (mm)",
        description="ระยะห่างตามด้านนี้เป็นมิลลิเมตร — ถ้าปรับเอง ตัวเลือกระยะห่างด้านบนจะกลายเป็นกำหนดเอง",
        default=400.0,
        min=1.0,
        update=_on_rebar_spacing_changed,
    )
    grid_offset: FloatVectorProperty(
        name="Offset (mm)",
        description="เลื่อนตำแหน่งเริ่มของเหล็กทั้งชุดเป็นมิลลิเมตร — ใช้ตอนเหล็กไปทับรอยต่อหรือขอบที่ไม่อยากให้ทับ",
        subtype="TRANSLATION",
        size=3,
        default=(0.0, 0.0, 0.0),
        update=_on_rebar_setting_changed,
    )
    angle_threshold: FloatProperty(
        name="Angle",
        description=(
            "ถ้าระบบลองเสียบแล้วเฉียดผิวเกินไปจะไม่ทำ — "
            "0.5 ใช้ได้ทั่วไป; 0.8 เข้มงวดขึ้น เหล็กจะดันออกมาตรง ๆ แทน"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
        update=_on_rebar_setting_changed,
    )
    normal_bias: FloatProperty(
        name="Normal",
        description="น้ำหนักตอนดัดให้ไปตามทิศที่ผิวหัน — 0 คือไม่มีผล; 1 คือพอเหมาะ",
        default=0.5,
        min=0.0,
        max=10.0,
        update=_on_rebar_setting_changed,
    )
    gravity_bias: FloatProperty(
        name="Gravity",
        description="น้ำหนักตอนดัดให้ปลายเหล็กตกลงด้านล่าง — 0 คือไม่มีผล; 1 คือพอเหมาะ",
        default=0.5,
        min=0.0,
        max=10.0,
        update=_on_rebar_setting_changed,
    )
    avoidance_strength: FloatProperty(
        name="Avoid",
        description="น้ำหนักตอนดัดให้หลบของข้าง ๆ — 0 คือไม่มีผล; 1 คือพอเหมาะ; 2 คือหลบแรงมาก",
        default=0.5,
        min=0.0,
        max=2.0,
        update=_on_rebar_setting_changed,
    )
    smooth_radius: FloatProperty(
        name="Smooth (mm)",
        description=(
            "เอาผิวรอบ ๆ มาร่วมตัดสินใจทิศทางเป็นมิลลิเมตร — "
            "300 คือใช้บริเวณราว 30 ซม. ช่วยไม่ให้เหล็กกระโดดตามรอยร้าวเล็ก ๆ"
        ),
        default=300.0,
        min=1.0,
        update=_on_rebar_setting_changed,
    )
    subdivisions: IntProperty(
        name="Smooth Steps",
        description=(
            "ความเนียนตอนดัดเหล็ก — 6 เนียนพอดีสำหรับโค้งดัด; ยิ่งสูงยิ่งเนียนแต่ช้าขึ้่น อันนี้ไม่ใช่จำนวนแท่งนะ"
        ),
        default=6,
        min=1,
        max=12,
        update=_on_rebar_setting_changed,
    )

    # --- System ---
    rebar_material: PointerProperty(
        name="Rebar Mat",
        description="สีหรือผิวของเส้นเหล็กที่จะโผล่มา — ปกติระบบหาให้เองจากไฟล์ที่แนบมากับตัวโปรแกรม",
        type=bpy.types.Material,
        update=_on_rebar_setting_changed,
    )
    debris_material: PointerProperty(
        name="Debris Mat",
        description=(
            "สีหรือผิวของก้อนเศษคอนกรีต — "
            "เช่น ถ้าไม่เลือก ระบบจะใช้ Target Material เดิมกับก้อนเศษ"
        ),
        type=bpy.types.Material,
        update=_on_rebar_setting_changed,
    )
    max_bars: IntProperty(
        name="Max Bars",
        description=(
            "กันไม่ให้เหล็กเพียบจนเครื่องช้า — ตั้ง 5,000 ระบบก็จะทำไม่เกิน 5,000 แท่ง "
            "และจะบอกว่าเต็มตามที่ตั้ง"
        ),
        default=5000,
        min=1,
        max=200000,
        update=_on_rebar_setting_changed,
    )
    fast_drag: BoolProperty(
        name="Fast Drag",
        description="ตอนลากแถบเลื่อน ให้เห็นเส้นสีฟ้าอย่างเดียวไปก่อน แล้วค่อยทำเหล็กจริงตอนหยุดลาก",
        default=True,
        update=_on_rebar_setting_changed,
    )

    # Status fields are plain data, deliberately without update callbacks.
    rebar_active_sid: StringProperty(name="Active Generator", default="")
    rebar_estimate: IntProperty(name="Estimated Bars", default=0)
    rebar_status: StringProperty(name="Rebar Status", default="")


CLASSES = (FDT_SceneSettings, FDT_RebarSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fdt_settings = PointerProperty(type=FDT_SceneSettings)
    bpy.types.Scene.fdt_rebar_settings = PointerProperty(type=FDT_RebarSettings)


def unregister():
    if hasattr(bpy.types.Scene, "fdt_rebar_settings"):
        del bpy.types.Scene.fdt_rebar_settings
    if hasattr(bpy.types.Scene, "fdt_settings"):
        del bpy.types.Scene.fdt_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
