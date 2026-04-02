# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Distill — standalone macOS .app bundle."""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
ROOT = Path(SPECPATH)

# ── Hidden imports ───────────────────────────────────────────────────────────
# litellm and instructor dynamically import providers
hidden = (
    collect_submodules("litellm")
    + collect_submodules("instructor")
    + collect_submodules("tiktoken")
    + collect_submodules("starlette")
    + collect_submodules("uvicorn")
    + collect_submodules("httpx")
    + collect_submodules("pydantic")
    + collect_submodules("pydantic_core")
    + [
        "claude_agent_sdk",
        "anyio",
        "anyio._backends._asyncio",
        "sniffio",
        "h11",
        "httpcore",
        "sse_starlette",
        "python_multipart",
    ]
)

# ── Data files ───────────────────────────────────────────────────────────────
datas = [
    # Static web UI
    (str(ROOT / "distill" / "web"), "distill/web"),
]
# tiktoken needs its encoding data files
datas += collect_data_files("tiktoken")
datas += collect_data_files("litellm")
datas += collect_data_files("pydantic")
# certifi CA bundle for HTTPS
datas += collect_data_files("certifi")

# ── Excludes ─────────────────────────────────────────────────────────────────
excludes = [
    "output",
    "docs",
    "matplotlib",
    "scipy",
    "numpy",
    "PIL",
    "tkinter",
    "test",
    "unittest",
    "pytest",
]

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Distill",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Distill",
)

app = BUNDLE(
    coll,
    name="Distill.app",
    icon=str(ROOT / "AppIcon.icns"),
    bundle_identifier="com.kenyi.distill",
    info_plist={
        "CFBundleName": "Distill",
        "CFBundleDisplayName": "Distill",
        "CFBundleShortVersionString": "0.3.0",
        "CFBundleVersion": "0.3.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
