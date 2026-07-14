#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Name -> backend construction, with missing optional deps treated as normal.

The registry is the single place that knows which backends exist and how to
build one from plain config.  Two rules shape it:

1. **Config is a plain mapping**, not the app's ``Settings`` object.  The
   inference package must stay importable and testable with no app, no Qt and
   no settings file, so the caller (Phase 1c's AssistController) is the one that
   translates ``Settings`` into a dict.

2. **A missing optional dependency is not an error.**  ``build_backend`` returns
   ``None`` when the requested backend cannot be constructed, and the caller
   disables the AI actions and points the user at installing the optional
   extras from this checkout (see the hints in ``libs/assist/controller.py``)
   rather than a PyPI install of this package. labelImg must keep working as a
   plain annotation tool on a machine with no onnxruntime -- an ImportError
   escaping from here would take the app down at startup for users who never
   asked for AI.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from .backend import MissingDependency, ModelBackend

__all__ = [
    'build_backend',
    'available_backends',
    'register_backend',
    'DEFAULT_BACKEND',
]

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = None

# A factory takes the config mapping and returns a constructed backend.  It may
# raise MissingDependency / ImportError -- build_backend absorbs both.
BackendFactory = Callable[[Mapping[str, Any]], ModelBackend]


def _build_stub(config: Mapping[str, Any]) -> ModelBackend:
    # Imported inside the factory, not at module import: the same discipline the
    # heavy backends will need (a top-level `import onnxruntime` in this module
    # would break the zero-dependency guarantee for everyone).
    from .stub import StubBackend

    return StubBackend(
        class_names=config.get('class_names'),
        num_detections=config.get('num_detections', 2),
        conf_threshold=config.get('conf_threshold', 0.0),
    )


def _build_yolo_onnx(config: Mapping[str, Any]) -> ModelBackend:
    # Same discipline, and now it matters: `libs.inference.yolo_onnx` is itself
    # import-cheap (its onnxruntime/numpy imports live in the constructor), but
    # importing it from here rather than at module scope keeps even that off the
    # path of a process that only wanted `Detection`.
    #
    # Every value is passed as-is, `None` included: the backend owns its defaults
    # (one place, not two), so `None` means "unset" rather than "0".
    from .yolo_onnx import YoloOnnxBackend

    return YoloOnnxBackend(
        model_path=config.get('model_path'),
        conf_threshold=config.get('conf_threshold'),
        iou_threshold=config.get('iou_threshold'),
        max_detections=config.get('max_detections'),
        input_size=config.get('input_size'),
        providers=config.get('providers'),
        class_names=config.get('class_names'),
        layout=config.get('layout'),
    )


# Built-in table.  `mobile_sam` registers here in a later phase; like yolo_onnx,
# its factory will do the heavy imports inside the function body and raise
# MissingDependency when onnxruntime/numpy are absent -- which build_backend
# turns into "AI disabled", not a crash.
_BACKENDS: Dict[str, BackendFactory] = {
    'stub': _build_stub,
    'yolo_onnx': _build_yolo_onnx,
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Add/replace a backend factory.

    This is also the merge point for third-party backends (see the entry-point
    seam in ``available_backends`` / below).
    """
    if not name:
        raise ValueError('backend name must be a non-empty string')
    _BACKENDS[name] = factory


# --------------------------------------------------------------------------
# SEAM: third-party backend discovery (NOT implemented yet -- later phase).
#
# When we open this up to plugins, discovery goes exactly here: iterate
#
#     importlib.metadata.entry_points(group='labelimg.model_backends')
#
# and merge each entry point's loaded factory into the table below the built-ins
# (built-ins win on name collision, so a broken plugin cannot shadow `stub`).
# Loading an entry point must be wrapped in the same MissingDependency/ImportError
# tolerance as build_backend: a plugin whose deps are absent is skipped and
# logged, never fatal.  Kept out of the current phase deliberately -- it needs a
# trust/versioning story of its own.
# --------------------------------------------------------------------------


def available_backends() -> List[str]:
    """Names that ``build_backend`` will recognise, sorted.

    "Recognised" is not "constructible": a name listed here can still yield
    ``None`` if its optional dependencies are missing on this machine.
    """
    return sorted(_BACKENDS)


def build_backend(config: Optional[Mapping[str, Any]] = None) -> Optional[ModelBackend]:
    """Construct the backend named by ``config['backend']``, or ``None``.

    config: plain mapping, e.g.
        {'backend': 'stub', 'model_path': ..., 'conf_threshold': 0.25}
    Unknown keys are ignored, so callers can pass a superset.

    Returns ``None`` -- never raises -- when no backend was requested, the
    backend is unknown, its optional dependencies are missing, or its
    construction fails (e.g. the model file on ``model_path`` is absent or
    corrupt).  Every one of those is a "run labelImg without AI" situation, not
    a crash.

    ``DEFAULT_BACKEND`` is ``None``: a fresh install with no
    ``SETTING_MODEL_BACKEND`` configured must not silently light up an AI
    feature nobody asked for (``stub`` fabricates detections from the image's
    dimensions -- see ``libs/inference/stub.py`` -- and a user could easily
    mistake those for a real model's output). ``stub`` stays fully registered
    and fully functional as an explicit, opt-in choice; it is just no longer
    what an unconfigured caller gets.
    """
    config = config or {}
    name = config.get('backend') or DEFAULT_BACKEND

    if name is None:
        # No backend requested -- distinct from a typo'd/unknown name below,
        # which is a configuration mistake worth a warning. This is not one:
        # it is the ordinary, expected state of an unconfigured install.
        return None

    factory = _BACKENDS.get(name)
    if factory is None:
        logger.warning('Unknown inference backend %r; available: %s',
                       name, ', '.join(available_backends()))
        return None

    try:
        return factory(config)
    except MissingDependency as exc:
        # Expected on a machine without the AI extras -- info, not a traceback.
        logger.info("Inference backend %r unavailable: %s "
                    "(install from your checkout: pip install -e \".[ai]\" -- "
                    "this fork isn't on PyPI)",
                    name, exc)
        return None
    except ImportError as exc:
        # A backend that forgot to convert its import failure; same outcome.
        logger.info('Inference backend %r could not be imported: %s', name, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - deliberate: never crash the app
        # Bad model path, unreadable weights, nonsense config...  The AI feature
        # is optional; degrading to "disabled" always beats killing the editor.
        # exc_info: unlike the two expected branches above, anything landing here
        # may be an actual bug in a factory (a typo, a bad signature). Without the
        # traceback the log line is indistinguishable from "extras not installed"
        # and the bug is undebuggable -- while the app still degrades to
        # "AI disabled" exactly as designed.
        logger.warning('Inference backend %r failed to initialise: %s', name, exc,
                       exc_info=True)
        return None
