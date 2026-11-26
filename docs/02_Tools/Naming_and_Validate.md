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
   - **กฎชื่อโซนของ Enfusion (จดไว้กันลืม)**:
     - `A1`, `A2` คือพวก **Decal** ไม่ใช่โซนพิเศษ
     - โซนพิเศษจริง ๆ คือแบบมีเงื่อนไขผสม เช่น `ID-A_VIS-A#B` และ `VIS-A#B`
2. **FDST Validate Name (แก้บั๊ก `Fix socket`)**:
   - ทำงานได้ทุกโหมด (`Smart`, `Collection`, `Object`)
   - แยกชิ้น `socket_` ออกมาก่อนเปลี่ยนชื่อหน้า เพื่อไม่ให้คำว่า `socket_` หาย
   - รักษา `_FDST_Base` และซิงก์ `EBT_SocketName` ให้ตรงกับชื่อจริงใน Blender
3. **Merge Same Condition**: รวมชิ้นที่มีเงื่อนไข `LOD` / `ID` / `VIS` ตรงกันอย่างปลอดภัย ไม่พังกับชิ้นที่ซ่อนอยู่
