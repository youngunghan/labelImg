#!/usr/bin/env python
# -*- coding: utf8 -*-
"""The plugin seam: every model the app can use hides behind ``ModelBackend``.

Adding a model means adding a subclass here and one line in ``registry.py`` --
never a branch in the UI.  Backends may need heavy optional deps
(onnxruntime, numpy, torch); this module must not, so the interface can be
imported and subclassed with the standard library alone.

Capability flags rather than ``hasattr`` sniffing: the UI decides whether to
enable the "auto-annotate" / "smart select" actions by *asking* the backend
what it supports, and a backend that claims nothing simply produces a greyed
out action instead of a crash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .types import Detection, Mask, SegPrompt

__all__ = ['MissingDependency', 'ModelBackend']


class MissingDependency(Exception):
    """An optional dependency of a backend is not installed.

    Backends raise this from their constructor (or from a lazy import inside
    it) instead of letting ``ImportError`` escape, so the registry can tell
    "this model needs ``pip install labelImg[ai]``" apart from a genuine bug and
    degrade to "AI actions disabled" rather than taking the whole app down.
    """


class ModelBackend(ABC):
    """A model, reduced to what labelImg needs from it.

    Implementations own *all* pre/post-processing.  In particular they must
    invert whatever resize/letterbox they applied and return coordinates in
    ORIGINAL image pixels (see ``libs/inference/types.py``); no caller is
    allowed to "fix up" a backend's coordinates.
    """

    # Human-facing identifier, also the key used in the registry table.
    name: str = 'base'

    # Capability flags -- the UI reads these to enable/disable its AI actions.
    supports_detection: bool = False
    supports_segmentation: bool = False

    # Class names in class-id order.  Backends map ids to names themselves so
    # that the rest of the app only ever sees names.
    class_names: List[str] = []

    @abstractmethod
    def predict(self, image: Any) -> List[Detection]:
        """Detect objects in one image.

        image: HxWx3 RGB uint8 array-like (numpy array in practice; the type is
        left open so this module needs no numpy).
        Returns detections in ORIGINAL image-pixel space, already filtered by
        whatever confidence/NMS settings the backend was configured with.
        """

    def segment(self, image: Any, prompts: SegPrompt, embedding: Any = None) -> Mask:
        """Segment one object from user prompts (points/box).

        ``embedding`` is an optional cached image encoding: interactive
        segmenters (SAM-like) split into an expensive per-image encode and a
        cheap per-click decode, and the caller may pass back a previous
        ``embed()`` result to keep clicking at interactive speed.

        Optional: default raises, and ``supports_segmentation`` stays False.
        """
        raise NotImplementedError(
            '%s does not support segmentation' % type(self).__name__)

    def embed(self, image: Any) -> Any:
        """Compute the reusable per-image encoding used by ``segment``.

        Optional; opaque to callers -- they only pass the value back in.
        """
        raise NotImplementedError(
            '%s does not support embeddings' % type(self).__name__)

    def close(self) -> None:
        """Release any session/handle the backend holds.

        No-op by default; a real backend (an ONNX session, a CUDA context) can
        override it so the app can drop a model without exiting.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return '<%s name=%r classes=%d>' % (
            type(self).__name__, self.name, len(self.class_names))


# Type alias used by the registry; kept here so backends and registry agree.
OptionalBackend = Optional[ModelBackend]
