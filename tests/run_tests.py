r"""Headless Blender 5.2 test runner for FullDestruc_Tool.

Usage:
    blender.exe --factory-startup --background --python D:\backup\github\FullDestruc_Tool\tests\run_tests.py
"""

from __future__ import annotations

import os
import sys
import traceback

import bpy

ADDONS_DIR = r"C:\Users\saetanpee\AppData\Roaming\Blender Foundation\Blender\5.2\scripts\addons"
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

for path in (ADDONS_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import FullDestruc_Tool as fdt

fdt.register()

import test_apple_bite
import test_rebar
import test_rebar_debris
import test_rebar_generate
import test_rebar_state
import test_rebar_ops

SUITES = (
    ("apple_bite", test_apple_bite),
    ("rebar", test_rebar),
    ("rebar_debris", test_rebar_debris),
    ("rebar_generate", test_rebar_generate),
    ("rebar_state", test_rebar_state),
    ("rebar_ops", test_rebar_ops),
)


def main() -> int:
    passed = 0
    failed = 0
    print("=" * 60)
    print("FullDestruc_Tool Automated Test Suite (Blender 5.2)")
    print("=" * 60)
    for fn in [fn for _name, suite in SUITES for fn in suite.ALL_TESTS]:
        name = fn.__name__
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
