# -*- mode: python ; coding: utf-8 -*-
import os

# 경로를 SPECPATH(이 spec 파일 위치) 기준으로 고정 — 다른 작업 디렉터리에서
# 빌드해도 libs 패키지가 번들에서 빠지지 않는다(빠지면 실행 시
# "No module named 'libs'"로 즉사).

binaries = []
datas = [(os.path.join(SPECPATH, 'data'), 'data')]
hiddenimports = ['xml', 'xml.etree', 'xml.etree.ElementTree', 'lxml.etree']

# onnxruntime ships native artifacts -- onnxruntime_pybind11_state*.pyd, its
# DLLs, the onnxruntime/capi data payload -- that PyInstaller's static import
# analysis does not reliably discover on its own; a plain hiddenimports entry
# is not enough (see release notes / FORK_CHANGES.md: the released exe used to
# contain the AI *code* but not this *runtime*, so build_backend() always hit
# MissingDependency inside the frozen app).
#
# collect_all('onnxruntime') was tried first and BREAKS THE BUILD: it also
# returns ~168 hiddenimports covering every onnxruntime submodule, including
# onnxruntime.transformers.torch_onnx_export_helper (`import torch`) and
# onnxruntime.quantization (needs the `onnx` package, not installed). Forcing
# those two onto PyInstaller's static analysis makes it run pyinstaller-hooks-
# contrib's torch hook, which shells out to an isolated subprocess to collect
# torch's own submodules -- and that subprocess dies (exit code 3) on a plain
# `.[ai]` install with no working torch build, aborting the whole exe build.
# None of this is needed: this fork only ever calls plain CPU inference
# (`libs/inference/yolo_onnx.py`), never the training/quantization/transformer-
# export helpers, so we collect exactly what inference needs instead --
# collect_dynamic_libs() for the onnxruntime.dll / onnxruntime_providers_
# shared.dll pair, collect_data_files() for the onnxruntime/capi data payload
# (LICENSE, datasets/*.onnx fixtures, etc.) -- and declare only the handful of
# hiddenimports inference actually touches. numpy is imported LAZILY inside
# yolo_onnx.py (see that module's docstring), so PyInstaller's static source
# scan never sees it either; it needs its own hiddenimport for the same reason
# onnxruntime's native extension does. PyInstaller's built-in numpy hook
# handles numpy's own native bits once it knows numpy is reachable, so no
# collect_all/collect_dynamic_libs call is needed for numpy itself.
#
# Guarded on whether the package is importable in THIS build environment, not
# just on collect_dynamic_libs()/collect_data_files() being importable
# (PyInstaller is always present while running this spec) -- a contributor
# building from a base install (`pip install pyinstaller pyqt5 lxml`, no
# `.[ai]` extra) must still get a working exe, just one where the AI menu is
# disabled at runtime (libs/inference/registry.py degrades to None), not a
# PyInstaller failure.
excludes = []
try:
    __import__('onnxruntime')
except ImportError:
    pass
else:
    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

    binaries += collect_dynamic_libs('onnxruntime')
    datas += collect_data_files('onnxruntime')
    hiddenimports += [
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        'onnxruntime.capi.onnxruntime_pybind11_state',
    ]
    try:
        __import__('numpy')
    except ImportError:
        pass
    else:
        hiddenimports += ['numpy']
    # Heavy/optional onnxruntime extras this fork never imports -- keep them
    # out even if some future hiddenimport (or a contributor's local hook)
    # would otherwise drag them in. torch in particular is what broke
    # collect_all() above; excluding it here is what makes that impossible
    # again regardless of which hiddenimports end up in the list.
    excludes += [
        'torch',
        'onnxruntime.training',
        'onnxruntime.quantization',
        'onnxruntime.transformers',
    ]

a = Analysis(
    [os.path.join(SPECPATH, 'labelImg.py')],
    pathex=[os.path.join(SPECPATH, 'libs'), SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
