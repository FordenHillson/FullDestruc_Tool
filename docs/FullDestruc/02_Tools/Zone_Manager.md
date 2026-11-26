---
title: Zone Manager (Scene Zone Registry)
tags:
  - tool
  - zone-manager
  - registry
updated: 2026-11-24
---

# Zone Manager (Scene Zone Registry)

กลับไปที่ [[00_Home]] | แนวคิดและคำศัพท์: [[CONTEXT]] | ADR: [[0001-zone-registry]] | เกราะต้นฉบับ: [[ระบบกล่องเก็บต้นฉบับลับ]]

## แหล่งความจริงเดียว

โซนทั้งหมดมาจาก `addon/core/zone_registry.py` และเก็บเป็น JSON บน Scene ที่คีย์ `fdt_zone_registry_json` ทุกหน้าตา (N-Panel, เมนูลอย, Qt, วงล้อเฟือง, HUD) อ่านชุดเดียวกัน

Zone Definition แต่ละตัวมี:

| ฟิลด์ | ความหมาย |
| --- | --- |
| `key` | รหัสคงที่ที่เก็บจริงใน `fdt_cutter_zone` / `fdt_piece_zone` / session — `A`…`Z` หรือ `z_0001`… |
| `token` | ชื่อที่ผู้ใช้แก้ได้ เช่น `ID-A#B`, `VIS-A#B`, `ID-A_VIS-A#B` |
| `kind` | `ID` หรือ `VIS` (เก็บไว้ก่อน ยังไม่มีผลต่อพฤติกรรม) |
| `base_zone` | ตัวอักษร A–Z ที่เลือกเองในหน้าต่างสำหรับเป็น fallback |
| `color` | RGBA ที่ก๊อปปี้จาก base ตอนสร้าง แก้ได้ในภายหลัง |
| `builtin` | `true` สำหรับ A–Z (ลบไม่ได้) |

> **กฎเหล็ก:** `base_zone` มาจากการเลือกของผู้ใช้ใน dialog เสมอ regex ใน token เป็นแค่ตัวเติมค่าเริ่มต้นให้ฟอร์ม

## การใช้งาน

### สร้าง / เปลี่ยนชื่อ / ลบ
- ทุกที่ที่มี dropdown zone จะมีปุ่ม `New`, `Rename`, `Delete` เรียก `fdt.zone_manage` ตัวเดียวกัน
- วงล้อเฟือง: คลิกช่อง `Zone` → ปิดวงล้อ → เปิด `fdt.zone_picker` แบบค้นหาได้ → เลือก zone หรือกด New/Rename/Delete
- N-Panel: quick buttons แสดง token ของ 8 รายการแรก + dropdown แบบพิมพ์ค้นหา + ปุ่มจัดการ
- Qt window: combo editable แบบ `MatchContains`, ปุ่ม `+ New Zone` / `Rename` / `Delete`

### Validation
- token ต้องไม่ว่าง ยาวไม่เกิน 40 ตัวอักษร และไม่ซ้ำกับตัวอื่น (ไม่สนพิมพ์ใหญ่/เล็ก)
- built-in A–Z เปลี่ยน token/base_zone/color ได้ แต่ **ลบไม่ได้** เพราะเป็น fallback ของไฟล์เก่า
- ลบ custom zone ไม่ได้ถ้ามี cutter หรือ piece อ้างอิง key อยู่ — ข้อความจะบอกจำนวน cutter และ piece ทั้งสองอย่าง

### Fallback
- key หาย → ถ้าเป็น structured token เช่น `VIS-A#B` ให้ตกไปที่ base `A` → ถ้าไม่ใช่ structured text ให้ตกไปที่ `A`
- key ของ built-in ยังใช้ตัวอักษรเดิมได้เสมอ แม้จะเปลี่ยน token แล้ว

### ลำดับ cycling
- ปุ่ม Gear `Change Zone` และปุ่มรูปโซนในรายการ cutter วนตามลำดับ registry: `A`…`Z` แล้วตามด้วย custom ตามลำดับที่สร้าง แล้วกลับมาที่ `A`
- การ cycle เปลี่ยนชื่อ cutter ให้มี token ใหม่ (เช่น `_ID-B_01`) แต่ property ยังเก็บ key เสมอ และอัปเดต `fdt_piece_cutter` ของชิ้นที่เกี่ยวข้องให้ตาม

## โน้ตที่เกี่ยวข้อง
- [[Apple_Bite_v2]]
- [[Naming_and_Validate]]
- [[CONTEXT]]
