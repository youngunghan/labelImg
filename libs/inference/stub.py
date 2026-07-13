#!/usr/bin/env python
# -*- coding: utf8 -*-
"""A fake model with no dependencies -- the substrate the assist tests run on.

Why this exists: the whole point of the inference package is that the app can be
tested without a model.  ``StubBackend`` gives every layer above it (registry
now, AssistController/UI later) something real to talk to that

  * needs nothing but the standard library (no numpy, no onnxruntime), and
  * is **deterministic**: same image size and same config -> byte-identical
    detections, every run, on every machine.

Determinism is not cosmetic.  A stub that returned random boxes would force
every test above it to assert loosely ("some boxes appeared"), which is exactly
the kind of test that keeps passing while the coordinate contract rots.  Because
the boxes here are a pure function of the image dimensions, tests can assert the
*exact* numbers, and any layer that silently rescales or transposes coordinates
is caught immediately.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from .backend import ModelBackend
from .types import Detection

__all__ = ['StubBackend', 'image_size']

DEFAULT_CLASS_NAMES = ['person', 'face']
DEFAULT_NUM_DETECTIONS = 2


def image_size(image: Any) -> Tuple[int, int]:
    """Return ``(height, width)`` of an HxWx3 array-like, without numpy.

    Accepts anything the real backends will see: a numpy array (``.shape``), or
    a plain nested sequence of rows (what a test hands in).  Duck-typed on
    purpose -- importing numpy here just to read two integers would defeat the
    zero-dependency rule.
    """
    shape = getattr(image, 'shape', None)
    if shape is not None:
        if len(shape) < 2:
            raise ValueError('expected an HxW(xC) image, got shape %r' % (tuple(shape),))
        return int(shape[0]), int(shape[1])

    # Nested-sequence fallback: len() is the height, len(row 0) is the width.
    try:
        height = len(image)
        width = len(image[0]) if height else 0
    except (TypeError, IndexError, KeyError):
        raise ValueError('unsupported image type: %r' % type(image))
    return int(height), int(width)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StubBackend(ModelBackend):
    """Deterministic, dependency-free fake detector.

    Detection *i* of *n* is a box centred at ``((i+1)/(n+1) * W, (i+1)/(n+1) * H)``
    with half-extents of 10% of the image, clamped to the image rectangle.  So
    the boxes march down the diagonal, scale with the image, and are always
    inside bounds -- i.e. they satisfy the same coordinate contract a real
    backend must satisfy, and a caller that mishandles that contract will
    produce visibly wrong numbers rather than plausible ones.

    Scores descend 0.90, 0.80, 0.70, ... (floored at 0.05) so that ordering,
    top-k and confidence-threshold logic have something non-degenerate to chew
    on.  Labels cycle through ``class_names``.
    """

    name = 'stub'
    supports_detection = True
    supports_segmentation = False

    def __init__(self,
                 class_names: Optional[Sequence[str]] = None,
                 num_detections: int = DEFAULT_NUM_DETECTIONS,
                 conf_threshold: float = 0.0,
                 **_ignored: Any) -> None:
        """``**_ignored`` swallows unrelated config keys (``model_path``, ...)
        so the registry can hand the stub the same config dict a real backend
        would get, and tests can flip one key without special-casing.
        """
        if num_detections < 0:
            raise ValueError('num_detections must be >= 0')

        # None means "unset, use the defaults" (that is what the registry passes
        # when the key is absent).  An explicit empty list is a different thing
        # -- a backend with no classes cannot label anything -- so it is an error
        # rather than a silent fallback.
        if class_names is None:
            self.class_names = list(DEFAULT_CLASS_NAMES)
        else:
            self.class_names = list(class_names)
            if not self.class_names:
                raise ValueError('class_names must not be empty')
        self.num_detections = int(num_detections)
        self.conf_threshold = float(conf_threshold)

    def predict(self, image: Any) -> List[Detection]:
        height, width = image_size(image)
        n = self.num_detections
        detections: List[Detection] = []

        for i in range(n):
            # Pure function of (i, n, W, H) -- no clocks, no RNG, no hashing.
            frac = (i + 1) / float(n + 1)
            cx = width * frac
            cy = height * frac
            half_w = width * 0.1
            half_h = height * 0.1

            box = (
                _clamp(cx - half_w, 0.0, float(width)),
                _clamp(cy - half_h, 0.0, float(height)),
                _clamp(cx + half_w, 0.0, float(width)),
                _clamp(cy + half_h, 0.0, float(height)),
            )

            score = max(0.05, 0.9 - 0.1 * i)
            if score < self.conf_threshold:
                # Real backends drop sub-threshold boxes; the stub does too, so
                # threshold plumbing is exercised end to end.
                continue

            class_id = i % len(self.class_names)
            detections.append(Detection(
                label=self.class_names[class_id],
                box=box,
                score=score,
                class_id=class_id,
            ))

        return detections
