#!/usr/bin/env python
# -*- coding: utf8 -*-
"""YOLO-family ONNX detector -- the first real model behind the backend seam.

Nothing above this module changes because a real network now runs underneath it:
the app still asks the registry for a backend and gets back ``Detection``s in
ORIGINAL image pixels (see ``libs/inference/types.py``).  Everything a model
needed internally -- letterboxing, NCHW, 0..1 floats, cx/cy/w/h, class ids --
is undone here, which is the deal ``ModelBackend`` makes with its callers.

TWO RULES SHAPE THIS FILE.

**Dependencies are lazy.**  ``onnxruntime`` and ``numpy`` are imported inside
the constructor / methods, never at module scope, so ``import
libs.inference.yolo_onnx`` -- and therefore ``libs.inference`` and the registry
table that names it -- still costs nothing on a base install.  When either dep
is absent the constructor raises ``MissingDependency``, ``build_backend``
returns None, and the user gets "pip install labelImg[ai]" instead of a dead
editor.

**The geometry is pure Python.**  ``letterbox_params`` / ``inverse_letterbox``
/ ``nms`` / ``decode_output`` / ``postprocess`` use no numpy at all: getting the
letterbox inversion wrong shifts or scales *every* box the model ever produces,
and that class of bug is only ever caught by tests that can run everywhere --
including the base install, where numpy does not exist.  So only the two steps
that genuinely need an array (building the input tensor, ``session.run``) touch
numpy; the raw output is turned into plain Python lists immediately (``tolist``)
and the whole decode -> NMS -> inverse path below that line is dependency-free
and unit-testable with nested lists.

THE LETTERBOX CONTRACT
----------------------
Preprocessing scales the image by a single factor (aspect preserved) and centres
it in an ``in_w x in_h`` canvas, padding the leftover with grey::

    model_x = orig_x * scale + pad_x
    model_y = orig_y * scale + pad_y

so the inverse -- which is what a detection needs -- is subtract, divide, clip::

    orig_x = (model_x - pad_x) / scale   clipped to [0, orig_w]
    orig_y = (model_y - pad_y) / scale   clipped to [0, orig_h]

Only ONE scale exists (never one per axis): that is the whole point of padding.
A backend that used ``in_w / orig_w`` and ``in_h / orig_h`` separately would
return boxes that look almost right and are wrong on every non-square image.

The pads are WHOLE PIXELS, and that is not a rounding detail -- it is the
contract.  The forward map is not an abstraction: it is a real image pasted into
a real array at a real row, and an array has no row 1.5.  When the leftover is
odd (a 1000x995 image into a 640x640 net leaves 3 rows: an ideal pad of 1.5) the
paste must pick a row, so the *ideal* centre and the *actual* offset differ by
half a pixel -- and every box comes back biased by ``0.5 / scale`` original
pixels if the inverse subtracts the ideal one.  So there is no ideal one: both
directions take their pads from ``_letterbox_geometry`` below, which floors the
split to a whole pixel and gives the odd row to the bottom/right.  Preprocessing
and inversion cannot disagree because they no longer compute it twice.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

from .backend import MissingDependency, ModelBackend
from .types import Detection

__all__ = [
    'YoloOnnxBackend',
    'letterbox_params',
    'inverse_letterbox',
    'nms',
    'detect_layout',
    'decode_output',
    'postprocess',
    'LAYOUT_V8',
    'LAYOUT_V5',
]

logger = logging.getLogger(__name__)

# Fallback when the model does not pin its input size (dynamic axes).
DEFAULT_INPUT_SIZE = 640

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 300

# A detector emits one candidate per anchor (8400 for a 640px v8 model, 25200
# for v5) and the app deliberately asks for conf_threshold=0.0 (see
# AssistController._build_backend) so that the confidence slider can re-filter
# without re-running the model. There is DELIBERATELY NO FLOOR here: a floor
# would silently break that contract (a user dragging the slider below the
# floor would see nothing new, and the UI has no way to tell them the model
# never even kept the box). This was measured, not assumed: decoding the full
# candidate set at conf_threshold=0.0 costs ~34ms versus ~10ms at 0.05 for a
# stock (1, 84, 8400) v8 tensor -- ~24ms, negligible next to actual model
# inference. MAX_NMS_CANDIDATES below is the real, cheap safety valve against a
# noisy image producing a wall of near-zero anchors.
MAX_NMS_CANDIDATES = 1000

# Only guard against candidate-count blowup: a noisy image (or the conf_threshold
# =0.0 the app always asks for) can leave thousands of decoded candidates. Only
# the best MAX_NMS_CANDIDATES go into NMS, which bounds its cost; they are the
# highest-scoring ones, i.e. exactly the boxes the user would have kept anyway.

# Ultralytics' letterbox fill.  Any constant works (the model never sees an
# object there); matching the training-time value keeps the padded strip from
# looking like a feature.
PAD_VALUE = 114

# CPU by default: labelImg is a desktop tool that must run offline on whatever
# machine the annotator has, and a GPU provider that is merely *installed* but
# misconfigured fails at session creation rather than falling back.
DEFAULT_PROVIDERS = ('CPUExecutionProvider',)

# The two Ultralytics export layouts this backend decodes.
LAYOUT_V8 = 'v8'  # (1, 4+nc, N): no objectness, per-class scores
LAYOUT_V5 = 'v5'  # (1, N, 5+nc): objectness * per-class score


# ---------------------------------------------------------------------------
# Geometry -- pure Python, no numpy.  See the module docstring's contract.
# ---------------------------------------------------------------------------

def _letterbox_geometry(orig_w: float, orig_h: float,
                        in_w: float, in_h: float) -> Tuple[float, int, int, float, float]:
    """``(scale, new_w, new_h, pad_x, pad_y)`` -- the ONE source of letterbox truth.

    Both directions of the transform come from here: ``_letterbox`` pastes a
    ``new_w x new_h`` resize at ``(pad_x, pad_y)``, and ``letterbox_params`` hands
    the very same ``scale`` / ``pad_x`` / ``pad_y`` to ``inverse_letterbox``.  That
    is the whole reason this function exists rather than each side deriving its own
    numbers: when they were computed twice, the paste rounded ``pad_y`` to an int
    and the inverse did not, and an odd pad silently biased every box by half a
    model pixel (see the module docstring).  A single return value cannot disagree
    with itself.

    Everything is a whole number of pixels because the canvas is an array:

    * ``new_w`` / ``new_h`` -- the scaled extent, clamped to at least 1px (a
      sliver of a panorama must still paste *something*) and never larger than
      the canvas it has to fit inside.
    * ``pad_x`` / ``pad_y`` -- the left / top offset, the even split FLOORED to a
      whole pixel, so an odd leftover gives the extra row/column to the
      bottom/right.  Exactly one of the two is 0 unless the aspect ratios already
      match (then both are).  They are returned as floats only because the
      inverse does float arithmetic with them.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError('image size must be positive, got %rx%r' % (orig_w, orig_h))
    if in_w <= 0 or in_h <= 0:
        raise ValueError('model input size must be positive, got %rx%r' % (in_w, in_h))

    canvas_w, canvas_h = int(in_w), int(in_h)
    scale = min(float(in_w) / float(orig_w), float(in_h) / float(orig_h))
    new_w = min(max(1, int(round(orig_w * scale))), canvas_w)
    new_h = min(max(1, int(round(orig_h * scale))), canvas_h)
    pad_x = float((canvas_w - new_w) // 2)
    pad_y = float((canvas_h - new_h) // 2)
    return scale, new_w, new_h, pad_x, pad_y


def letterbox_params(orig_w: float, orig_h: float,
                     in_w: float, in_h: float) -> Tuple[float, float, float]:
    """``(scale, pad_x, pad_y)`` for fitting ``orig`` into ``in`` with padding.

    ``scale`` is the single (aspect-preserving) factor that makes the image fit
    inside the network input; the image is then centred to the nearest whole
    pixel, and ``pad_x`` / ``pad_y`` are the left / top offsets in model-input
    pixels -- the exact offsets the preprocessing pastes at, not an idealised
    centre.  This is what ``inverse_letterbox`` must be given; see
    ``_letterbox_geometry``.
    """
    scale, _new_w, _new_h, pad_x, pad_y = _letterbox_geometry(orig_w, orig_h, in_w, in_h)
    return scale, pad_x, pad_y


def inverse_letterbox(box_xyxy: Sequence[float], scale: float,
                      pad_x: float, pad_y: float,
                      orig_w: float, orig_h: float) -> Tuple[float, float, float, float]:
    """Map a box from MODEL-INPUT space back to ORIGINAL image pixels.

    Undo the pad (a translation), then the scale, then clip to the image rect:
    a model happily predicts boxes that stick out into the grey padding, and a
    detection that escapes the image would put a shape outside the canvas and
    write a negative coordinate into an annotation file.

    Clipping is the last step on purpose -- clipping before un-padding would
    clip against the wrong rectangle.  The corners are re-ordered at the end so
    the ``x1 <= x2`` / ``y1 <= y2`` half of the coordinate contract holds even
    if a decoder handed us a box with a negative width.
    """
    if scale <= 0:
        raise ValueError('scale must be positive, got %r' % (scale,))

    x1, y1, x2, y2 = (float(v) for v in box_xyxy)

    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = min(max(x1, 0.0), float(orig_w))
    x2 = min(max(x2, 0.0), float(orig_w))
    y1 = min(max(y1, 0.0), float(orig_h))
    y2 = min(max(y2, 0.0), float(orig_h))
    return x1, y1, x2, y2


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:  # two zero-area boxes: no overlap to speak of
        return 0.0
    return inter / union


def nms(boxes: Sequence[Sequence[float]], scores: Sequence[float],
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        class_ids: Optional[Sequence[int]] = None,
        max_detections: Optional[int] = None) -> List[int]:
    """Greedy non-maximum suppression; returns the indices that survive.

    **Class-aware**: a box only suppresses another box of the SAME class (that
    is what ``class_ids`` is for, and the backend always passes it).  An
    annotation tool must be able to put a "person" and the "backpack" they are
    wearing on the same pixels -- a class-agnostic NMS would silently delete one
    of them, and the user would have to redraw by hand what the model already
    found.  ``class_ids=None`` degrades to plain class-agnostic NMS.

    Kept indices come back highest-score-first; ties break on the lower index,
    so the result is deterministic (the same tensor always yields the same
    boxes, which is what lets the tests pin exact numbers).
    """
    if len(boxes) != len(scores):
        raise ValueError('boxes and scores must have the same length (%d vs %d)'
                         % (len(boxes), len(scores)))
    if class_ids is not None and len(class_ids) != len(boxes):
        raise ValueError('class_ids must have the same length as boxes')

    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))

    kept: List[int] = []
    for candidate in order:
        if max_detections is not None and len(kept) >= max_detections:
            break
        suppressed = False
        for winner in kept:
            if class_ids is not None and class_ids[winner] != class_ids[candidate]:
                continue  # different class: they are allowed to overlap
            if _iou(boxes[candidate], boxes[winner]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Raw output decoding -- pure Python, works on nested lists.
# ---------------------------------------------------------------------------

def _shape_of(tensor: Any) -> Tuple[int, ...]:
    """Shape of a nested sequence (or of anything exposing ``.shape``)."""
    shape = getattr(tensor, 'shape', None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)

    dims: List[int] = []
    node = tensor
    while isinstance(node, (list, tuple)):
        dims.append(len(node))
        if not node:
            break
        node = node[0]
    return tuple(dims)


def detect_layout(shape: Sequence[int], num_classes: Optional[int] = None) -> str:
    """Which Ultralytics export layout an output tensor of ``shape`` is.

    The two layouts are distinguishable from the shape alone because they put
    the channel axis on opposite sides::

        v8: (1, 4+nc, N)   channels first, no objectness
        v5: (1, N, 5+nc)   anchors first, objectness at index 4

    Evidence, in order:

    1. **nc consistency** (strong) -- when the model declares its class names,
       only one reading of the shape can produce that many classes.  This beats
       the geometry below, because it is the model talking about itself.
    2. **the anchor axis is never the short one** (fallback) -- a detector always
       has at least as many anchors as it has channels (8400 vs 84, 25200 vs 85);
       a "YOLOv5 output" with 4 anchors and 8395 classes is not a thing.  So the
       reading whose N would be shorter than its own channel count is discarded.

    Fails SAFE: anything it cannot pin down -- a square output (where both
    readings survive and nothing chooses between them), batch > 1, an end-to-end
    export that already applied its own NMS, a shape neither reading explains --
    raises ``ValueError`` naming the shape.  Guessing would not crash; it would
    quietly emit boxes decoded off the wrong axis, which is worse than a disabled
    model because it looks like it worked.
    """
    dims = tuple(int(d) for d in shape)
    if len(dims) == 2:  # some exporters squeeze the batch axis away
        dims = (1,) + dims
    if len(dims) != 3 or dims[0] != 1:
        raise ValueError(
            'unsupported model output shape %r: expected (1, 4+nc, N) [YOLOv8] '
            'or (1, N, 5+nc) [YOLOv5]. An end-to-end export that already applies '
            'NMS is not supported -- re-export without it.' % (tuple(shape),))

    _, d1, d2 = dims
    v8_nc = d1 - 4  # (1, 4+nc, N): channels are d1, anchors are d2
    v5_nc = d2 - 5  # (1, N, 5+nc): anchors are d1, channels are d2

    if num_classes:
        v8_match = v8_nc == num_classes
        v5_match = v5_nc == num_classes
        if v8_match and not v5_match:
            return LAYOUT_V8
        if v5_match and not v8_match:
            return LAYOUT_V5
        if not v8_match and not v5_match:
            # The class count is what is wrong (a stale classes.txt), not the
            # shape: a bad names file must not disable a decodable model, so drop
            # the evidence and let class_name() paper over any extra ids.
            logger.warning(
                'Declared class count %d matches neither reading of model output '
                'shape %r (v8 would be %d classes, v5 %d); ignoring it and going '
                'by the shape alone -- the class names are probably stale.',
                num_classes, tuple(shape), v8_nc, v5_nc)
        # (both match: only possible for shapes like (1, nc+4, nc+5) -- fall
        # through to the geometry, which separates them cleanly)

    v8_ok = v8_nc >= 1 and d2 >= d1
    v5_ok = v5_nc >= 1 and d1 >= d2

    if v8_ok and not v5_ok:
        return LAYOUT_V8
    if v5_ok and not v8_ok:
        return LAYOUT_V5
    if v8_ok and v5_ok:
        raise ValueError(
            'ambiguous model output shape %r: it reads as YOLOv8 (nc=%d) and as '
            'YOLOv5 (nc=%d), and the two axes are the same length so neither is '
            "identifiable as the anchor axis. Set the 'layout' config key "
            "('v8' or 'v5') explicitly." % (tuple(shape), v8_nc, v5_nc))

    raise ValueError(
        'cannot decode model output shape %r as YOLOv8 (1, 4+nc, N) or YOLOv5 '
        '(1, N, 5+nc): neither reading yields a plausible (class count, anchor '
        'count) pair.' % (tuple(shape),))


def decode_output(raw: Any,
                  layout: Optional[str] = None,
                  num_classes: Optional[int] = None,
                  conf_threshold: float = DEFAULT_CONF_THRESHOLD
                  ) -> Tuple[List[Tuple[float, float, float, float]],
                             List[float], List[int]]:
    """Raw model output -> ``(boxes_xyxy, scores, class_ids)`` in MODEL-INPUT space.

    Boxes are still letterboxed here (``inverse_letterbox`` is a separate step);
    they are converted from the network's ``cx, cy, w, h`` to the corner form the
    rest of the app speaks, and filtered by ``conf_threshold``.

    Accepts a numpy array or plain nested lists: an array is converted with
    ``tolist()`` on the first line, so everything below is dependency-free (and
    the tests can drive it with literals).
    """
    if hasattr(raw, 'tolist'):
        raw = raw.tolist()

    shape = _shape_of(raw)
    if len(shape) == 2:  # squeezed batch axis; put it back so indexing is uniform
        raw = [raw]
        shape = (1,) + shape

    layout = layout or detect_layout(shape, num_classes)
    batch = raw[0]

    if layout == LAYOUT_V8:
        # (1, 4+nc, N): channel-major, so transpose to one vector per anchor.
        # zip() does that at C speed -- a Python double loop over 8400x84 floats
        # would be the slowest thing in the whole pipeline.
        num_classes = shape[1] - 4
        anchors: Any = zip(*batch)
        has_objectness = False
    elif layout == LAYOUT_V5:
        num_classes = shape[2] - 5
        anchors = batch
        has_objectness = True
    else:
        raise ValueError('unknown layout %r (expected %r or %r)'
                         % (layout, LAYOUT_V8, LAYOUT_V5))

    if num_classes < 1:
        raise ValueError('model output shape %r has no class scores under layout %r'
                         % (shape, layout))

    first_class = 5 if has_objectness else 4

    boxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    class_ids: List[int] = []

    for vector in anchors:
        objectness = 1.0
        if has_objectness:
            objectness = vector[4]
            # v5 scores are objectness * class score, and a class score is <= 1,
            # so a sub-threshold objectness can never clear the bar -- skip
            # before touching the class scores at all (that is most anchors, on
            # most images, and it is what makes this loop affordable in Python).
            if objectness < conf_threshold:
                continue

        class_scores = vector[first_class:first_class + num_classes]
        best = max(class_scores)
        score = best * objectness
        if score < conf_threshold:
            continue

        cx, cy, w, h = vector[0], vector[1], vector[2], vector[3]
        half_w = w / 2.0
        half_h = h / 2.0
        boxes.append((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
        scores.append(float(score))
        class_ids.append(class_scores.index(best))

    return boxes, scores, class_ids


def class_name(class_names: Sequence[str], class_id: int) -> str:
    """Name for a class id, with the generic fallback.

    A model whose names we could not resolve (no metadata, no ``classes.txt``)
    still has to produce *some* label, and ``class_7`` is honest about what it
    knows.  Also covers a names list that is shorter than the model's real class
    count -- a mislabelled box is worse than an obviously generic one.
    """
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return 'class_%d' % class_id


def postprocess(raw: Any,
                orig_w: float, orig_h: float,
                in_w: float, in_h: float,
                class_names: Optional[Sequence[str]] = None,
                layout: Optional[str] = None,
                conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                max_detections: int = DEFAULT_MAX_DETECTIONS,
                num_classes: Optional[int] = None) -> List[Detection]:
    """The whole dependency-free half of the backend: raw tensor -> ``Detection``s.

    decode -> confidence filter -> NMS -> ``inverse_letterbox``, in that order.
    NMS runs in MODEL-INPUT space (before the inverse) because that is where the
    model's own IoU statistics live, and because a uniform scale makes IoU
    invariant anyway -- but the letterbox must be undone before the boxes leave
    this function, since ``Detection.box`` is in ORIGINAL image pixels.

    ``predict`` is just this function with a session in front of it, which is why
    every interesting failure mode is testable without onnxruntime.
    """
    names = list(class_names or [])
    # How many classes the model *should* have, as evidence for the layout
    # autodetection: if the names came from the model's own metadata, their count
    # is the model's class count.  detect_layout ignores it when it fits neither
    # reading (a stale classes.txt), so a wrong names file cannot disable a model.
    if num_classes is None and names:
        num_classes = len(names)

    boxes, scores, class_ids = decode_output(
        raw, layout=layout, num_classes=num_classes, conf_threshold=conf_threshold)

    if len(boxes) > MAX_NMS_CANDIDATES:
        # Keep the best; see MAX_NMS_CANDIDATES.
        best = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))[:MAX_NMS_CANDIDATES]
        boxes = [boxes[i] for i in best]
        scores = [scores[i] for i in best]
        class_ids = [class_ids[i] for i in best]

    kept = nms(boxes, scores, iou_threshold=iou_threshold, class_ids=class_ids,
               max_detections=max_detections)

    scale, pad_x, pad_y = letterbox_params(orig_w, orig_h, in_w, in_h)

    detections: List[Detection] = []
    for i in kept:  # nms returns highest-score-first, so this list is sorted too
        box = inverse_letterbox(boxes[i], scale, pad_x, pad_y, orig_w, orig_h)
        detections.append(Detection(
            label=class_name(names, class_ids[i]),
            box=box,
            score=scores[i],
            class_id=class_ids[i],
        ))
    return detections


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

def parse_names_metadata(value: str) -> List[str]:
    """Class names out of an ONNX ``metadata_props`` entry.

    Ultralytics stores ``names`` as the *repr of a Python dict* --
    ``{0: 'person', 1: 'bicycle'}`` -- which is neither JSON nor a list, so it
    gets ``ast.literal_eval`` (never ``eval``: this string comes out of a file
    the user downloaded).  JSON objects and plain lists are accepted too, because
    other exporters write those and the cost of tolerating them is one branch.

    Defensive by design: any junk in there returns ``[]`` (fall through to the
    next source) rather than raising -- a malformed metadata blob must not be
    able to stop a working model from loading.  Keys are treated as class ids, so
    a sparse dict yields generic names in the gaps.
    """
    if not value:
        return []

    parsed: Any = None
    for loads in (json.loads, ast.literal_eval):
        try:
            parsed = loads(value)
            break
        except Exception:  # noqa: BLE001 - untrusted metadata; try the next reader
            continue

    if isinstance(parsed, (list, tuple)):
        return [str(name) for name in parsed]

    if isinstance(parsed, dict):
        try:
            ids = {int(key): str(name) for key, name in parsed.items()}
        except (TypeError, ValueError):
            logger.warning('Model metadata "names" has non-integer class ids; ignoring it')
            return []
        if not ids:
            return []
        return [ids.get(i, 'class_%d' % i) for i in range(max(ids) + 1)]

    logger.warning('Model metadata "names" is not a list or dict; ignoring it')
    return []


def load_classes_txt(path: str) -> List[str]:
    """Class names from a sibling ``classes.txt`` (one per line, blank lines skipped).

    Same file format labelImg already uses for its predefined classes, so a user
    who has a YOLO dataset can point at the file they already have.
    """
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        logger.info('Could not read class names from %s: %s', path, exc)
        return []


def generic_class_names(num_classes: int) -> List[str]:
    return ['class_%d' % i for i in range(max(0, int(num_classes)))]


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

def _require(module_name: str) -> Any:
    """Import an optional dependency, or raise ``MissingDependency``.

    ``MissingDependency`` (not ImportError) is what tells the registry that this
    is a "the extras are not installed" machine rather than a bug, so it logs an
    info line and returns None instead of a traceback -- and labelImg keeps
    working as a plain annotation tool.
    """
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise MissingDependency(
            "the 'yolo_onnx' backend needs %s, which is not installed "
            "(pip install labelImg[ai])" % module_name) from exc


class YoloOnnxBackend(ModelBackend):
    """Ultralytics-style YOLO detector (v5 / v8 ONNX exports) on onnxruntime.

    Owns one session for its whole life: creating an ``InferenceSession`` costs
    hundreds of milliseconds (graph load, provider init, memory arena) and doing
    it per image would make auto-labelling a folder unusable.  The session is
    only ever touched from the inference worker thread -- ``InferenceService``
    runs a single worker precisely so that two ``Run`` calls cannot race on it.

    No weights ship with labelImg (they are AGPL-3.0; the app is MIT) -- the user
    supplies a ``.onnx`` file through the model-path setting.  See
    ``data/models/README.md``.
    """

    name = 'yolo_onnx'
    supports_detection = True
    supports_segmentation = False

    def __init__(self,
                 model_path: Optional[str] = None,
                 conf_threshold: Optional[float] = None,
                 iou_threshold: Optional[float] = None,
                 max_detections: Optional[int] = None,
                 input_size: Optional[Any] = None,
                 providers: Optional[Sequence[str]] = None,
                 class_names: Optional[Sequence[str]] = None,
                 layout: Optional[str] = None,
                 session: Optional[Any] = None,
                 **_ignored: Any) -> None:
        """Every argument is optional and ``None`` means "use the default", so the
        registry can hand this the same flat config dict every other backend gets.

        ``session`` is an injection seam: pass an already-built session (the tests
        pass a fake one) and neither onnxruntime nor a model file is needed.
        ``layout`` overrides the autodetection for an export it cannot pin down.
        """
        # numpy is needed by every predict() call (input tensor), so demand it up
        # front: a backend that constructs happily and then fails on the first
        # image is a worse experience than one the UI never offers.
        self._np = _require('numpy')

        # No floor: see the comment above MAX_NMS_CANDIDATES. conf_threshold=0.0
        # (what AssistController always passes) must reach postprocess() as 0.0.
        self.conf_threshold = (DEFAULT_CONF_THRESHOLD if conf_threshold is None
                               else float(conf_threshold))
        self.iou_threshold = (DEFAULT_IOU_THRESHOLD if iou_threshold is None
                              else float(iou_threshold))
        self.max_detections = (DEFAULT_MAX_DETECTIONS if max_detections is None
                               else int(max_detections))
        if self.max_detections < 1:
            raise ValueError('max_detections must be >= 1')
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError('iou_threshold must be in 0..1')

        self.layout = layout
        self.model_path = model_path

        if session is not None:
            self._session = session
        else:
            onnxruntime = _require('onnxruntime')
            self.model_path = self._resolve_model_path(model_path)
            self._session = self._create_session(onnxruntime, self.model_path, providers)

        self._input_name = self._session.get_inputs()[0].name
        self._input_dtype = self._numpy_input_dtype()
        self.input_width, self.input_height = self._resolve_input_size(input_size)

        # Class names, best source first (see the three helpers above).  An
        # explicit config list wins over anything in the file: it is the only
        # source the user typed on purpose.
        self.class_names: List[str] = list(class_names or []) or self._resolve_class_names()

    # -- construction helpers ----------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        """MissingDependency, not an exception, for "no model configured yet".

        An unset or absent model path is the same *situation* as absent extras --
        the AI feature is not set up on this machine -- so it gets the same
        treatment: the registry logs one info line, returns None, and the UI
        greys the AI actions out.  Reserving the noisy warning+traceback path for
        genuinely broken models (corrupt file, bad graph) keeps a real bug
        visible in the log instead of buried under everyone's empty setting.
        """
        if not model_path:
            raise MissingDependency(
                "the 'yolo_onnx' backend has no model file configured "
                '(set the model path in the settings)')
        if not os.path.isfile(model_path):
            raise MissingDependency(
                "the 'yolo_onnx' model file does not exist: %s" % model_path)
        return model_path

    @staticmethod
    def _create_session(onnxruntime: Any, model_path: str,
                        providers: Optional[Sequence[str]]) -> Any:
        """One session, CPU unless the config asked for something else.

        A provider that onnxruntime does not have compiled in makes
        ``InferenceSession`` raise, which would take the AI feature down for a
        typo in a setting; unknown names are dropped (with a warning) and CPU is
        always left as the last resort, because it is the one that always works.
        """
        requested = list(providers or DEFAULT_PROVIDERS)
        available = list(onnxruntime.get_available_providers())
        usable = [p for p in requested if p in available]

        for name in requested:
            if name not in available:
                logger.warning('onnxruntime provider %r is not available here; '
                               'available: %s', name, ', '.join(available))
        if not usable:
            usable = ['CPUExecutionProvider']

        options = onnxruntime.SessionOptions()
        # The app already parallelises at the image level (and the worker is
        # single-threaded by design); let ORT use the machine's cores for one
        # image rather than fighting it.
        return onnxruntime.InferenceSession(model_path, sess_options=options,
                                            providers=usable)

    def _numpy_input_dtype(self) -> Any:
        """float32, unless the model was exported in half precision."""
        declared = getattr(self._session.get_inputs()[0], 'type', None)
        if declared == 'tensor(float16)':
            return self._np.float16
        return self._np.float32

    def _resolve_input_size(self, input_size: Optional[Any]) -> Tuple[int, int]:
        """``(width, height)`` of the network input.

        The MODEL'S OWN declared shape wins on any axis where it is pinned (the
        usual case: ``[1, 3, 640, 640]``) -- a session built for a fixed input
        shape cannot ``session.run()`` at any other size, so letting a config
        value override it used to build a backend that failed on every single
        call.  If ``input_size`` disagrees with a pinned axis, that is logged as
        a warning and the model still wins: this must never produce a session
        that cannot run.

        Only an axis the model leaves dynamic (a string, or missing -- what
        onnxruntime reports for a symbolic dimension) is actually decided by the
        config, because then the letterbox still has to know the canvas it is
        padding into and the model does not say. ``input_size`` (or 640) is the
        fallback for exactly those axes, never a general override.
        """
        shape = list(getattr(self._session.get_inputs()[0], 'shape', []) or [])
        model_height = shape[2] if len(shape) == 4 else None
        model_width = shape[3] if len(shape) == 4 else None
        model_height = model_height if isinstance(model_height, int) and model_height >= 1 else None
        model_width = model_width if isinstance(model_width, int) and model_width >= 1 else None

        configured_width = configured_height = None
        if input_size is not None:
            configured_width, configured_height = self._parse_input_size(input_size)

        width = model_width if model_width is not None else (
            configured_width if configured_width is not None else DEFAULT_INPUT_SIZE)
        height = model_height if model_height is not None else (
            configured_height if configured_height is not None else DEFAULT_INPUT_SIZE)

        if (input_size is not None and (model_width is not None or model_height is not None)
                and (configured_width, configured_height) != (width, height)):
            logger.warning(
                "configured input_size %r conflicts with the model's own static "
                "input shape (width=%r, height=%r); using the model's shape -- a "
                'session built for a fixed input size cannot run at any other size',
                input_size, model_width, model_height)

        return width, height

    @staticmethod
    def _parse_input_size(input_size: Any) -> Tuple[int, int]:
        """Normalise a config ``input_size`` (an int, or an explicit ``(w, h)``)."""
        if isinstance(input_size, (list, tuple)):
            width, height = int(input_size[0]), int(input_size[1])
        else:
            width = height = int(input_size)
        if width < 1 or height < 1:
            raise ValueError('input_size must be positive, got %r' % (input_size,))
        return width, height

    def _resolve_class_names(self) -> List[str]:
        """Metadata -> sibling ``classes.txt`` -> generic, first hit wins.

        Metadata comes first because it is *inside* the model the boxes came from
        and therefore cannot drift from it; a ``classes.txt`` next to the file is
        the user's own answer and comes next; ``class_0..`` is what is left when
        nobody said anything (and is filled in lazily, from the output shape, on
        the first run -- see ``_ensure_class_names``).
        """
        names = self._names_from_metadata()
        if names:
            return names

        if self.model_path:
            sibling = os.path.join(os.path.dirname(os.path.abspath(self.model_path)),
                                   'classes.txt')
            if os.path.isfile(sibling):
                names = load_classes_txt(sibling)
                if names:
                    logger.info('Class names read from %s', sibling)
                    return names

        # Generic names need the class count, which is only certain once an
        # output shape is known; if the model declares one statically, use it.
        num_classes = self._num_classes_from_output_shape()
        if num_classes:
            return generic_class_names(num_classes)
        return []

    def _names_from_metadata(self) -> List[str]:
        try:
            meta = self._session.get_modelmeta()
            raw = dict(getattr(meta, 'custom_metadata_map', None) or {})
        except Exception as exc:  # noqa: BLE001 - metadata is a nice-to-have
            logger.info('Could not read ONNX model metadata: %s', exc)
            return []

        names = parse_names_metadata(raw.get('names', ''))
        if names:
            logger.info('Class names read from ONNX model metadata (%d classes)',
                        len(names))
        return names

    def _num_classes_from_output_shape(self) -> Optional[int]:
        """Class count from the (statically declared) output shape, if it has one."""
        try:
            shape = list(getattr(self._session.get_outputs()[0], 'shape', []) or [])
        except Exception:  # noqa: BLE001 - a session that cannot describe itself
            return None
        if not all(isinstance(dim, int) for dim in shape):
            return None  # dynamic axes: wait for a real output tensor
        try:
            layout = detect_layout(shape, num_classes=None) if self.layout is None else self.layout
        except ValueError:
            return None
        if layout == LAYOUT_V8:
            return shape[1] - 4 if len(shape) == 3 else None
        return shape[2] - 5 if len(shape) == 3 else None

    def _ensure_class_names(self, shape: Sequence[int]) -> None:
        """Fill in generic names once a real output tensor has revealed nc.

        Only reachable for a model with dynamic output axes AND no names anywhere;
        without this the labels would be ``class_0..`` anyway (``class_name``
        falls back), but ``class_names`` would stay empty and the UI would think
        the model has no classes to offer.
        """
        if self.class_names:
            return
        try:
            layout = self.layout or detect_layout(shape)
        except ValueError:
            return
        num_classes = shape[1] - 4 if layout == LAYOUT_V8 else shape[2] - 5
        if num_classes >= 1:
            self.class_names = generic_class_names(num_classes)

    # -- inference ---------------------------------------------------------

    def _as_array(self, image: Any) -> Any:
        """The input image as an HxWx3 numpy array, or a clear failure.

        A ``RawImage`` (``libs.inference.service``) means the UI thread had no
        numpy when it converted the QImage -- i.e. the base install.  This
        backend cannot run there at all, so say so with ``MissingDependency``
        rather than crashing on a missing array method three frames deeper.
        """
        np = self._np
        data = getattr(image, 'data', None)
        if not isinstance(image, np.ndarray) and isinstance(data, (bytes, bytearray)):
            raise MissingDependency(
                "the 'yolo_onnx' backend received a numpy-free RawImage; it needs "
                'numpy and onnxruntime (pip install labelImg[ai])')

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('expected an HxWx3 RGB image, got shape %r'
                             % (tuple(array.shape),))
        return array

    def _letterbox(self, image: Any) -> Any:
        """Resize (aspect-preserving, bilinear) + centre-pad into the model canvas.

        Bilinear, not nearest: nearest-neighbour aliases thin structures away and
        costs real detection accuracy, and doing it here in numpy is what keeps
        opencv out of the dependency list (labelImg only needs cv2 once masks
        land, in a later phase).  The gather is done on uint8 and only the four
        sampled corners are widened to float, so the temporaries stay the size of
        the *output*, not of a 4K input.

        The paste offsets are NOT recomputed here: they come from
        ``_letterbox_geometry``, the same call ``letterbox_params`` makes for the
        inverse.  Deriving them a second time is precisely how the two sides drift
        apart (they once did, by half a pixel on any odd pad), so this method is
        deliberately given no arithmetic of its own to get wrong.
        """
        np = self._np
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        in_w, in_h = self.input_width, self.input_height

        _scale, new_w, new_h, pad_x, pad_y = _letterbox_geometry(orig_w, orig_h, in_w, in_h)

        # Half-pixel centres: sample the source at the middle of each destination
        # pixel, which is what every reference resize does (and what keeps a 2x
        # downscale from drifting half a pixel to the left).
        src_x = (np.arange(new_w, dtype=np.float32) + 0.5) * (orig_w / float(new_w)) - 0.5
        src_y = (np.arange(new_h, dtype=np.float32) + 0.5) * (orig_h / float(new_h)) - 0.5
        src_x = np.clip(src_x, 0.0, orig_w - 1.0)
        src_y = np.clip(src_y, 0.0, orig_h - 1.0)

        x0 = np.floor(src_x).astype(np.int64)
        y0 = np.floor(src_y).astype(np.int64)
        x1 = np.minimum(x0 + 1, orig_w - 1)
        y1 = np.minimum(y0 + 1, orig_h - 1)
        wx = (src_x - x0).reshape(1, new_w, 1)
        wy = (src_y - y0).reshape(new_h, 1, 1)

        top_left = image[np.ix_(y0, x0)].astype(np.float32)
        top_right = image[np.ix_(y0, x1)].astype(np.float32)
        bottom_left = image[np.ix_(y1, x0)].astype(np.float32)
        bottom_right = image[np.ix_(y1, x1)].astype(np.float32)

        top = top_left + (top_right - top_left) * wx
        bottom = bottom_left + (bottom_right - bottom_left) * wx
        resized = top + (bottom - top) * wy

        canvas = np.full((in_h, in_w, 3), float(PAD_VALUE), dtype=np.float32)
        # Exact, not rounded: the pads are already whole pixels, and int() here
        # is only a type cast. Rounding at this line was the half-pixel bias.
        left = int(pad_x)
        top_offset = int(pad_y)
        canvas[top_offset:top_offset + new_h, left:left + new_w] = resized

        # NCHW, 0..1 -- the layout and range every Ultralytics export expects.
        tensor = canvas.transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor, dtype=self._input_dtype)

    def predict(self, image: Any) -> List[Detection]:
        """Detect objects in one image; boxes come back in ORIGINAL image pixels.

        Runs on the inference worker thread (see ``libs/inference/service.py``).
        Raising here is safe -- the service catches it and emits
        ``predictionFailed`` -- so a model that turns out to be undecodable
        reports itself instead of taking the editor with it.
        """
        array = self._as_array(image)
        orig_h, orig_w = int(array.shape[0]), int(array.shape[1])

        tensor = self._letterbox(array)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            raise ValueError('the model produced no output tensor')

        raw = outputs[0]
        # Plain Python from here on: everything below is the dependency-free,
        # unit-tested half of this backend.
        if hasattr(raw, 'tolist'):
            raw = raw.tolist()
        shape = _shape_of(raw)
        self._ensure_class_names(shape if len(shape) == 3 else (1,) + shape)

        return postprocess(
            raw,
            orig_w=orig_w, orig_h=orig_h,
            in_w=self.input_width, in_h=self.input_height,
            class_names=self.class_names,
            layout=self.layout,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )

    def close(self) -> None:
        """Drop the session (and the model's memory arena) without exiting."""
        self._session = None
