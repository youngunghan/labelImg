#!/usr/bin/env python
# -*- coding: utf8 -*-
"""labelImg inference core -- model-facing types, the backend ABC, the registry.

Two invariants define this package:

**Zero runtime dependencies.**  Importing ``libs.inference`` (or anything under
it, except a concrete heavy backend) must pull in nothing but the standard
library: no PyQt5, no numpy, no onnxruntime, no cv2, and nothing from the app
itself (``libs.constants``, ``libs.settings``, ``labelImg``).  That is what
makes the AI layer unit-testable with no deps installed, keeps labelImg usable
as a plain annotation tool on a machine without the AI extras, and lets the GUI
wire this in later (Phase 1c) without inheriting an import cycle.

**No Qt in the vocabulary.**  Backends speak ``Detection`` / ``Mask`` /
``SegPrompt`` / ``Prediction`` (see ``types``), always in original image pixels.
Converting those to ``Shape`` objects on a canvas is the UI's job, in the UI's
layer.

Re-exports below are lazy (PEP 562 module ``__getattr__``): the names are
resolved on first attribute access, so ``import libs.inference`` costs nothing
and -- once heavy backends land here -- cannot drag onnxruntime into a process
that only wanted to look at a dataclass.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # types
    'Detection',
    'Mask',
    'SegPrompt',
    'Prediction',
    'least_confidence',
    # backend ABC
    'ModelBackend',
    'MissingDependency',
    # registry
    'build_backend',
    'available_backends',
    'register_backend',
    'DEFAULT_BACKEND',
]

# name -> submodule it lives in.  Concrete backends (stub, and later yolo_onnx /
# mobile_sam) are intentionally absent: they are reached through the registry,
# which imports them inside its factories.
_LAZY_EXPORTS = {
    'Detection': 'types',
    'Mask': 'types',
    'SegPrompt': 'types',
    'Prediction': 'types',
    'least_confidence': 'types',
    'ModelBackend': 'backend',
    'MissingDependency': 'backend',
    'build_backend': 'registry',
    'available_backends': 'registry',
    'register_backend': 'registry',
    'DEFAULT_BACKEND': 'registry',
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError('module %r has no attribute %r' % (__name__, name))

    from importlib import import_module

    value = getattr(import_module('.' + module_name, __name__), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
