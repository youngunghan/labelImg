#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Value types exchanged between model backends and the rest of labelImg.

This module is the *AI seam*: these dataclasses are the only vocabulary the
application speaks to models with.  They deliberately expose no Qt objects and
no ``libs.shape.Shape`` -- a backend must never need to know that the caller is
a GUI, and the GUI must never need to know which model produced a box.

COORDINATE CONTRACT (load-bearing -- do not weaken)
--------------------------------------------------
``Detection.box`` is ``(x1, y1, x2, y2)`` with ``x1 <= x2`` / ``y1 <= y2``, in
**ORIGINAL image pixels**: the same coordinate space that the existing
annotation readers (``PascalVocReader``, ``YoloReader``, ``CreateMLReader``)
produce, i.e. the pixel grid of the file on disk -- *not* the network input
size, *not* the zoomed/scaled canvas, *not* normalised 0..1.

That is what lets the UI treat a ``Detection`` exactly like reader output: the
corner points ``[(x1,y1), (x2,y1), (x2,y2), (x1,y2)]`` can be handed to the
canvas with no further arithmetic.

Consequence: undoing any letterbox padding / resize / normalisation a model
needed internally is the **backend's** job, never the caller's.  A backend that
returns 640x640-space boxes has violated this contract.

``Mask.polygon`` follows the same rule (original image pixels), as do the
``SegPrompt`` points and box, which are what the *user clicked* mapped back to
original image pixels.

Zero runtime dependencies: standard library only (see ``libs/inference``
package docstring for why).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    'Detection',
    'Mask',
    'SegPrompt',
    'Prediction',
    'least_confidence',
]

# (x1, y1, x2, y2) in original image pixels.
Box = Tuple[float, float, float, float]
# (x, y) in original image pixels.
Point = Tuple[float, float]
# (x, y, label) where label is 1 for foreground, 0 for background.
PromptPoint = Tuple[float, float, int]


@dataclass(frozen=True)
class Detection:
    """One predicted bounding box.

    Frozen: a detection is a fact reported by a model.  If the user edits it,
    it stops being a detection and becomes a ``Shape`` -- the conversion is the
    UI layer's job (Phase 1c), and immutability here keeps the two from being
    silently confused.

    ``label`` is the class *name*, already mapped from the raw class id by the
    backend, because the rest of the app only ever deals in names (that is what
    goes into the label list and into the annotation files).  ``class_id`` is
    kept only as provenance for backends that have one.
    """

    label: str
    box: Box  # (x1, y1, x2, y2), ORIGINAL image pixels -- see module docstring
    score: float  # 0..1
    class_id: Optional[int] = None


@dataclass(frozen=True)
class Mask:
    """One segmentation result, as an exterior contour in original image pixels.

    A polygon (not a bitmap) because that is what an annotation tool can put on
    a canvas and write to a file; rasterisation, if ever needed, belongs to the
    consumer.  ``score`` defaults to 1.0 since interactive segmenters (SAM-like)
    return a single mask for an explicit user prompt rather than a ranked list.
    """

    polygon: List[Point]  # exterior contour, ORIGINAL image pixels
    score: float = 1.0


@dataclass
class SegPrompt:
    """User guidance for an interactive segmenter.

    Mutable on purpose: the user builds a prompt up click by click (add a
    foreground point, add a background point, drag a box), so this is the one
    type that is edited in place while the interaction is live.
    """

    points: List[PromptPoint] = field(default_factory=list)  # (x, y, 1=fg / 0=bg)
    box: Optional[Box] = None


@dataclass
class Prediction:
    """Everything a backend produced for one image.

    Mutable so that later stages can annotate it without rebuilding it: in
    particular ``uncertainty`` is filled in by the active-learning pass (a later
    phase), which scores an already-computed prediction rather than re-running
    the model.
    """

    image_path: str
    detections: List[Detection] = field(default_factory=list)
    uncertainty: Optional[float] = None  # None = not scored yet


def least_confidence(detections: Sequence[Detection], top_k: Optional[int] = None) -> float:
    """Least-confidence uncertainty of a prediction, in ``0.0 .. 1.0``.

    ``1 - mean(top-k detection scores)``: the more confident the model's best
    boxes are, the less this image is worth a human's time.  Pure function --
    no state, no I/O -- so active learning (a later phase) can call it over a
    whole dataset cheaply and so it can be tested without a model.

    ``top_k=None`` (default) uses every detection.  Averaging the *top* k rather
    than all of them keeps a long tail of junk low-score boxes from making a
    confidently-labelled image look uncertain.

    Empty / no detections deliberately returns **1.0 (maximum uncertainty)**:
    "the model found nothing" is exactly the case a human most needs to look at
    (either the image is genuinely empty, or the model is blind to it), so it
    must sort to the *front* of a review queue, not the back.  Returning 0.0
    here would silently hide every failure case.

    Scores are clamped to 0..1 before averaging so that a misbehaving backend
    cannot push the uncertainty outside its documented range.
    """
    if not detections:
        return 1.0

    scores = sorted((min(1.0, max(0.0, float(d.score))) for d in detections), reverse=True)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError('top_k must be a positive integer or None')
        scores = scores[:top_k]

    mean_score = sum(scores) / len(scores)
    return 1.0 - mean_score
