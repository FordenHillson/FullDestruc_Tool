---
title: ข้อควรระวังสำหรับ Blender 5.2
tags:
  - blender-5-2
  - lessons-learned
---

# ข้อควรระวังสำหรับ Blender 5.2

กลับไปที่ [[00_Home]]

> [!warning] จุดที่เปลี่ยนจาก Blender รุ่นเก่า (ต้องจำไว้ทุกครั้งที่เขียนโค้ดใหม่)

1. **เส้นขอบ Freestyle (`freestyle_edge`)**:
   - ใน Blender 5.x ไม่มี `edge.use_freestyle_mark` แล้ว
   - ต้องเขียนผ่าน Mesh Attribute ชื่อ `"freestyle_edge"` ชนิด `'BOOLEAN'` บนโดเมน `'EDGE'` แทน
2. **Smooth by Angle**:
   - ไม่มี `mesh.use_auto_smooth` แล้ว ให้ใช้ `bpy.ops.object.shade_smooth_by_angle()` หรือโหนด/ค่าปกติของ 5.x แทน
3. **การสลับ Context (`temp_override`)**:
   - ใช้ `with context.temp_override(...):` แทนการส่ง dict แบบเก่า
4. **ห้ามเปิด Modal Operator ค้างไว้ตลอดเวลา**:
   - ถ้ามี Modal ค้างไว้ จะกด `Reload Scripts` ไม่ได้ (Blender จะขึ้นเตือน `Can't reload with running modal operators`)
   - ดังนั้นหน้าต่างแยก PySide6 จะใช้ `bpy.app.timers` แทน Modal และโหมด Apple Bite จะมีปุ่มออกโหมดที่เคลียร์ตัวเอง 100%
5. **PySide6 ใน Blender 5.2**:
   - มีติดตั้งพร้อมใช้อยู่แล้วใน `C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe`
6. **การส่งค่าเข้าช่อง Geometry Nodes Modifier (`NodesModifier`)**:
   - ใน Blender 5.2 ยกเลิกการเขียนแบบเก่า `mod["Socket_1"] = value` แล้ว (จะขึ้น `TypeError: id properties not supported for this type`)
   - ต้องเขียนผ่าน `mod.properties.inputs[identifier]["value"] = value` แทน (รวมไว้ในฟังก์ชัน `set_modifier_socket()` ใน `core/geonodes.py` แล้ว)
7. **`gpu.types.GPUTexture` กับ Fallback Texture**:
   - การสร้าง `GPUTexture((1, 1), format="RGBA8", data=buf)` ใน Blender 5.2 **ต้องส่ง `Buffer("FLOAT", 4, [1.0, 1.0, 1.0, 1.0])`** เท่านั้น (ถ้าส่ง `"UBYTE"` จะ Error แล้วคืนค่า `None` ทำให้ Shader 4 ชั้นวาดตึกหายทั้งหลัง!)
8. **ห้ามเรียก `gpu.texture.from_image` ในโหมด `--background`**:
   - จะทำให้ Headless Blender พังทันที (Segfault) — ต้องครอบด้วย `if not bpy.app.background:` เสมอ
9. **Custom Split Normals & Sharp Edges หลังตัดโมเดล**:
   - อ่านค่า Normal เดิมรายมุมด้วย `mesh.corner_normals.foreach_get("vector", arr)` (เร็วมาก ~0.04 ms)
   - กู้คืนหลังแก้ BMesh ด้วย `mesh.normals_split_custom_set(nrm_3)` และเก็บไว้ใน `bmesh.loops.layers.float_vector` (`fdt_nrm`) ซึ่งไม่หายตอน `dissolve_verts`
10. **Edit Mode: `obj.data` และ `obj.bound_box` จะค้างค่าเก่า**:
    - ระหว่างที่ผู้ใช้อยู่ใน Edit Mode ของกล่องตัด ข้อมูลใน `obj.data.vertices` และ `obj.bound_box` ยังไม่อัปเดต
    - ต้องอ่านพิกัดสดจาก `bmesh.from_edit_mesh(obj.data)` (ห้ามเรียก `.free()` กับ Edit BMesh!) เพื่อให้ Rust และ GPU Preview อัปเดตตามทุกเฟรม
11. **`POST_VIEW` GPU Preview วาดทับจุดกลางหน้า (`*`) ของ Edit Mode**:
    - Draw handler ชนิด `POST_VIEW` วาดหลัง Overlay ของ Blender ทำให้ตัวตึกบังจุดกลางหน้า `*` ของกล่องตัด
    - แก้ด้วยการวาด Edit Mode Overlay (`depth_test = NONE`) ทับให้เองใน `gpu_preview.py` เฉพาะกล่องที่กำลังอยู่ใน Edit Mode
12. **โมเดลตึกเกมหลายชุดใช้สเกล `0.01` (หน่วย Local เป็นเซนติเมตร)**:
    - ห้ามใช้ค่าความคลาดเคลื่อน (Tolerance) แบบตายตัวเป็นเมตร ต้องคำนวณตามเส้นทแยงมุมของโมเดล (`max(0.008, diag * 4e-4)`) เพื่อรองรับทั้งโมเดลสเกลเมตรและสเกลเซนติเมตร
13. **ฟอนต์ `blf` บนจอ 3D Viewport**:
    - `font_id = 0` ของ Blender ใช้ภาษาอังกฤษสั้นๆ ชัดๆ สำหรับป้ายบนจอ 3D (กันปัญหาฟอนต์ขึ้นสี่เหลี่ยม) และใช้ภาษาไทยเข้าใจง่ายในแถบสถานะ / `self.report`
