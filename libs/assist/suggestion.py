#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Detection -> Shape.  The single seam between the model layer and the canvas.

Pure adapter: no MainWindow, no canvas, no settings.  Keeping it that way is
what lets the coordinate contract be checked in isolation -- and it is the whole
reason ``Shape`` needs no import from ``libs.inference`` and vice versa.

COORDINATES: ``Detection.box`` is ``(x1, y1, x2, y2)`` in ORIGINAL image pixels
(see ``libs/inference/types.py``), which is *exactly* the space the existing
readers (Pascal VOC / YOLO / CreateML / COCO) hand to ``MainWindow.load_labels``.
So the four corners go straight onto the canvas: no zoom factor, no canvas
scale, no letterbox arithmetic.  Any scaling math appearing in this file is a
bug -- either here or in the backend that produced the box.
"""

try:
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QColor
except ImportError:  # pragma: no cover - the app's PyQt4 fallback path
    from PyQt4.QtCore import QPointF
    from PyQt4.QtGui import QColor

from libs.shape import Shape

__all__ = [
    'PROVISIONAL_LINE_COLOR',
    'PROVISIONAL_FILL_COLOR',
    'detection_to_shape',
    'detections_to_shapes',
    'style_as_committed',
]

# Amber, and unlike anything generate_color_by_text() produces for a real label:
# a suggestion must be unmistakable at a glance. The outline is opaque, the fill
# deliberately faint so a suggestion never hides the pixels the user is judging
# it on. Shape.paint() adds the dashed outline for provisional shapes.
PROVISIONAL_LINE_COLOR = QColor(255, 170, 0, 255)
PROVISIONAL_FILL_COLOR = QColor(255, 170, 0, 40)


def detection_to_shape(detection):
    """One ``Detection`` -> one closed, provisional, rectangular ``Shape``."""
    x1, y1, x2, y2 = detection.box

    shape = Shape(label=detection.label, shape_type=Shape.RECT)
    # Corner order matches what the readers produce (TL, TR, BR, BL), so a
    # suggestion behaves like any other box under move/resize.
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        shape.add_point(QPointF(float(x), float(y)))
    shape.close()

    shape.provisional = True
    shape.confidence = float(detection.score)
    shape.line_color = PROVISIONAL_LINE_COLOR
    shape.fill_color = PROVISIONAL_FILL_COLOR
    shape.difficult = False
    return shape


def detections_to_shapes(detections):
    return [detection_to_shape(d) for d in detections]


def style_as_committed(shape, color):
    """Turn an accepted suggestion into an ordinary box.

    Clearing ``provisional`` is what actually promotes it (it is the flag the
    save filter and the painter read); repainting it in the label's usual colour
    is what stops the user from having to guess which of the boxes on screen
    have been accepted.  ``confidence`` is kept as provenance -- it costs
    nothing and no writer looks at it.
    """
    shape.provisional = False
    shape.line_color = color
    shape.fill_color = color
    return shape
