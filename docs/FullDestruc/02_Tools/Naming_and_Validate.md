---
title: Naming and Validate (เฟส 2)
tags:
  - tool
  - phase-2
  - naming
---

# FDST Set Name ID / Validate Name / Merge (แผนเฟส 2)

กลับไปที่ [[00_Home]] | ดูบั๊กเดิมที่ต้องแก้: [[สรุปบั๊กเดิมจาก_Red_Panda]]

## เป้าหมาย
1. **FDST Set Name ID**: ติดรหัสโซน `ID-` และเงื่อนไข `VIS-` ได้ไวและคุมความยาวชื่อไม่เกิน 63 ตัวอักษร
2. **FDST Validate Name (แก้บั๊ก `Fix socket`)**:
   - ทำงานได้ทุกโหมด (`Smart`, `Collection`, `Object`)
   - แยกชิ้น `socket_` ออกมาก่อนเปลี่ยนชื่อหน้า เพื่อไม่ให้คำว่า `socket_` หาย
   - รักษา `_FDST_Base` และซิงก์ `EBT_SocketName` ให้ตรงกับชื่อจริงใน Blender
3. **Merge Same Condition**: รวมชิ้นที่มีเงื่อนไข `LOD` / `ID` / `VIS` ตรงกันอย่างปลอดภัย ไม่พังกับชิ้นที่ซ่อนอยู่

## รูปแบบโซนที่รองรับแล้ว (ใช้กับ Apple Bite / Zone Registry)

Token ที่ผู้ใช้เลือกได้มาจาก [[Zone_Manager]] และอาจเป็นได้ทั้ง:

- `A` … `Z` (built-in)
- `ID-A#B`, `VIS-A#B`, `ID-A_VIS-A#B` (special zone ของ Enfusion)
- `A#B` หรือข้อความอิสระที่ผู้ใช้ตั้งเอง

กฎการตั้งชื่อที่ใช้งานจริงใน `addon/core/apple_bite.py`:

- regex สำหรับลบ/reread ป้ายเดิม: `_FDST_ID-[A-Za-z0-9_#-]+` (ยอมรับ `#`, `-`, `_`)
- ป้ายที่สร้างใหม่: `_FDST_ID-<token>` โดย token ถูก sanitize ให้เหลือ `[A-Za-z0-9_#-]` (อักขระอื่นกลายเป็น `_`)
- ชื่อ-object ทั้งหมดยังตัดให้ไม่เกิน **63 ตัวอักษร** เหมือนเดิม
- `A1` / `A2` ในไฟล์ต้นฉบับเป็น **decal** ไม่ใช่ zone — อย่าเอามาใช้เป็น base zone

Property บน cutter/piece เก็บ **Zone Key** (`A`…`Z` หรือ `z_0001`) ไม่ใช่ token ดังนั้นการเปลี่ยนชื่อ token จะไม่ทำให้ชิ้นหลุดจาก session
