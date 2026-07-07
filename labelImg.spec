# -*- mode: python ; coding: utf-8 -*-
import os

# 경로를 SPECPATH(이 spec 파일 위치) 기준으로 고정 — 다른 작업 디렉터리에서
# 빌드해도 libs 패키지가 번들에서 빠지지 않는다(빠지면 실행 시
# "No module named 'libs'"로 즉사).
a = Analysis(
    [os.path.join(SPECPATH, 'labelImg.py')],
    pathex=[os.path.join(SPECPATH, 'libs'), SPECPATH],
    binaries=[],
    datas=[(os.path.join(SPECPATH, 'data'), 'data')],
    hiddenimports=['xml', 'xml.etree', 'xml.etree.ElementTree', 'lxml.etree'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='labelImg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
