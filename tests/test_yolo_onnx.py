#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Unit tests for the YOLO/ONNX backend (libs/inference/yolo_onnx.py).

MOST OF THIS FILE RUNS ON THE BASE INSTALL -- no numpy, no onnxruntime -- and
that is the point. The dangerous part of an ONNX detector is not the session, it
is the *geometry*: an off-by-one in the letterbox inversion shifts or scales
every box the model will ever produce, and the boxes still look plausible. So
the geometry (letterbox / inverse / NMS / decode) is pure Python and is tested
here with literals, on every machine, including the ones the AI extras were
never installed on.

Only two groups need the extras, and both are skip-guarded:
  * TestPredictWithFakeSession  -- needs numpy (the input tensor), not onnxruntime.
  * TestRealOnnxModel           -- needs numpy AND onnxruntime; builds a real
                                   (tiny, hand-serialised) ONNX file and runs it.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libs.inference.backend import MissingDependency, ModelBackend
from libs.inference.registry import available_backends, build_backend
from libs.inference.types import Detection
from libs.inference.yolo_onnx import (DEFAULT_IOU_THRESHOLD, LAYOUT_V5,
                                      LAYOUT_V8, MIN_CONF_THRESHOLD,
                                      YoloOnnxBackend, class_name,
                                      decode_output, detect_layout,
                                      generic_class_names, inverse_letterbox,
                                      letterbox_params, load_classes_txt, nms,
                                      parse_names_metadata, postprocess)


def _importable(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


HAS_NUMPY = _importable('numpy')
HAS_ONNXRUNTIME = _importable('onnxruntime')

EPS = 1e-6


def forward_letterbox(box, scale, pad_x, pad_y):
    """The forward map, written HERE and not imported.

    inverse_letterbox is only trustworthy if it inverts something; importing the
    forward map from the module under test would just prove it is
    self-consistent. This is the transform the preprocessing performs, stated
    independently: scale about the origin, then translate by the pad.
    """
    x1, y1, x2, y2 = box
    return (x1 * scale + pad_x, y1 * scale + pad_y,
            x2 * scale + pad_x, y2 * scale + pad_y)


# ---------------------------------------------------------------------------
# Synthetic model outputs (plain nested lists -- no numpy).
# ---------------------------------------------------------------------------

def v8_tensor(anchors, num_classes, num_anchors):
    """(1, 4+nc, N) -- channel-major, no objectness.

    `anchors` is {index: (cx, cy, w, h, [class scores])}; everything else is 0,
    which is what an unfired anchor looks like (and must be filtered out).
    """
    channels = 4 + num_classes
    rows = [[0.0] * num_anchors for _ in range(channels)]
    for index, (cx, cy, w, h, scores) in anchors.items():
        for channel, value in enumerate((cx, cy, w, h) + tuple(scores)):
            rows[channel][index] = float(value)
    return [rows]


def v5_tensor(anchors, num_classes, num_anchors):
    """(1, N, 5+nc) -- anchor-major, objectness at index 4."""
    rows = [[0.0] * (5 + num_classes) for _ in range(num_anchors)]
    for index, (cx, cy, w, h, obj, scores) in anchors.items():
        rows[index] = [float(v) for v in (cx, cy, w, h, obj) + tuple(scores)]
    return [rows]


# The same scene in both layouts. Image 1280x720 -> 640x640 input:
# scale 0.5, pad_x 0, pad_y 140 (letterboxed top and bottom).
SCENE_IMAGE = (1280, 720)   # (w, h)
SCENE_INPUT = (640, 640)    # (w, h)
SCENE_CLASSES = ['cat', 'dog']

# cat @ (80,170,120,230) model space -> (160,60,240,180) original
CAT_BOX_ORIG = (160.0, 60.0, 240.0, 180.0)
# dog @ (350,250,450,350) model space -> (700,220,900,420) original
DOG_BOX_ORIG = (700.0, 220.0, 900.0, 420.0)


class TestLetterboxParams(unittest.TestCase):
    """scale + centred padding, for every aspect ratio and several input sizes."""

    def assert_params(self, orig, model, expected):
        scale, pad_x, pad_y = letterbox_params(orig[0], orig[1], model[0], model[1])
        self.assertAlmostEqual(expected[0], scale, delta=EPS)
        self.assertAlmostEqual(expected[1], pad_x, delta=EPS)
        self.assertAlmostEqual(expected[2], pad_y, delta=EPS)

    def test_square_image_into_square_input_needs_no_padding(self):
        self.assert_params((640, 640), (640, 640), (1.0, 0.0, 0.0))
        self.assert_params((1280, 1280), (640, 640), (0.5, 0.0, 0.0))
        self.assert_params((320, 320), (640, 640), (2.0, 0.0, 0.0))  # upscaling too

    def test_wide_image_pads_top_and_bottom(self):
        # 1280x720 -> 0.5 -> 640x360, leaving 280px of height split evenly.
        self.assert_params((1280, 720), (640, 640), (0.5, 0.0, 140.0))
        # 1280x720 -> 320x320: scale 0.25 -> 320x180 -> pad_y 70.
        self.assert_params((1280, 720), (320, 320), (0.25, 0.0, 70.0))
        # 1280x720 -> 416x416: scale 0.325 -> 416x234 -> pad_y 91.
        self.assert_params((1280, 720), (416, 416), (0.325, 0.0, 91.0))

    def test_tall_image_pads_left_and_right(self):
        self.assert_params((720, 1280), (640, 640), (0.5, 140.0, 0.0))
        self.assert_params((720, 1280), (320, 320), (0.25, 70.0, 0.0))

    def test_non_square_model_input(self):
        # A 800x400 image into a 640x320 canvas: the aspect ratios match, so a
        # single scale fits it exactly and nothing is padded.
        self.assert_params((800, 400), (640, 320), (0.8, 0.0, 0.0))
        # A square image into that same canvas is limited by the height.
        self.assert_params((400, 400), (640, 320), (0.8, 160.0, 0.0))

    def test_padding_is_centred_and_only_one_axis_is_padded(self):
        for orig in ((1280, 720), (720, 1280), (500, 500), (65, 33), (37, 2000)):
            for model in ((640, 640), (320, 320), (416, 416), (640, 384)):
                scale, pad_x, pad_y = letterbox_params(orig[0], orig[1], model[0], model[1])
                new_w = round(orig[0] * scale)
                new_h = round(orig[1] * scale)
                self.assertGreaterEqual(pad_x, 0.0)
                self.assertGreaterEqual(pad_y, 0.0)
                # Centred: the two pads plus the scaled image fill the canvas.
                self.assertAlmostEqual(model[0], new_w + 2 * pad_x, delta=EPS)
                self.assertAlmostEqual(model[1], new_h + 2 * pad_y, delta=EPS)
                # The scaled image fits, and touches at least one edge.
                self.assertLessEqual(new_w, model[0])
                self.assertLessEqual(new_h, model[1])
                self.assertTrue(pad_x < 1.0 or pad_y < 1.0,
                                'both axes padded: the scale is not maximal')

    def test_degenerate_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            letterbox_params(0, 100, 640, 640)
        with self.assertRaises(ValueError):
            letterbox_params(100, -1, 640, 640)
        with self.assertRaises(ValueError):
            letterbox_params(100, 100, 0, 640)


class TestInverseLetterbox(unittest.TestCase):
    """Design risk #2: the inverse must return EXACTLY where the box came from."""

    def assert_round_trip(self, orig, model, boxes):
        scale, pad_x, pad_y = letterbox_params(orig[0], orig[1], model[0], model[1])
        for box in boxes:
            in_model_space = forward_letterbox(box, scale, pad_x, pad_y)
            back = inverse_letterbox(in_model_space, scale, pad_x, pad_y, orig[0], orig[1])
            for expected, actual in zip(box, back):
                self.assertAlmostEqual(
                    expected, actual, delta=1e-4,
                    msg='%r -> %r -> %r (image %r, input %r)'
                        % (box, in_model_space, back, orig, model))

    def test_round_trip_wide_image_pads_y(self):
        w, h = 1280, 720
        self.assert_round_trip((w, h), (640, 640), [
            (0.0, 0.0, float(w), float(h)),      # the whole image
            (0.0, 0.0, 10.0, 10.0),              # touching the origin
            (float(w) - 10, float(h) - 10, float(w), float(h)),  # touching w/h
            (0.0, 300.0, 40.0, 420.0),           # touching the left edge only
            (100.5, 200.25, 340.75, 500.125),    # interior, non-integer
        ])

    def test_round_trip_tall_image_pads_x(self):
        w, h = 720, 1280
        self.assert_round_trip((w, h), (640, 640), [
            (0.0, 0.0, float(w), float(h)),
            (0.0, 0.0, 1.0, 1.0),
            (float(w) - 1, float(h) - 1, float(w), float(h)),
            (300.0, 0.0, 420.0, 40.0),           # touching the top edge only
            (12.5, 900.5, 700.5, 1279.5),
        ])

    def test_round_trip_square_and_odd_sizes_and_input_sizes(self):
        for orig in ((500, 500), (65, 33), (37, 2000), (1, 1)):
            for model in ((640, 640), (320, 320), (416, 416), (640, 384)):
                w, h = float(orig[0]), float(orig[1])
                self.assert_round_trip(orig, model, [
                    (0.0, 0.0, w, h),
                    (0.0, 0.0, w / 2.0, h / 2.0),
                    (w / 2.0, h / 2.0, w, h),
                ])

    def test_box_predicted_into_the_padding_is_clipped_to_the_image(self):
        # 1280x720 -> 640x640: scale 0.5, pad_y 140. A model box that runs off
        # the left edge and up into the grey strip must come back clipped.
        scale, pad_x, pad_y = letterbox_params(1280, 720, 640, 640)
        box = inverse_letterbox((-20.0, 120.0, 40.0, 180.0), scale, pad_x, pad_y, 1280, 720)
        self.assertEqual((0.0, 0.0, 80.0, 80.0), box)

    def test_box_running_off_the_far_edge_is_clipped(self):
        scale, pad_x, pad_y = letterbox_params(1280, 720, 640, 640)
        # x2 beyond the right edge (640 -> 1280) and y2 into the bottom pad.
        box = inverse_letterbox((600.0, 480.0, 700.0, 560.0), scale, pad_x, pad_y, 1280, 720)
        self.assertEqual((1200.0, 680.0, 1280.0, 720.0), box)

    def test_clipping_cannot_produce_an_inverted_box(self):
        scale, pad_x, pad_y = letterbox_params(1280, 720, 640, 640)
        # Entirely inside the padding: collapses onto the image edge, still valid.
        x1, y1, x2, y2 = inverse_letterbox((10.0, 0.0, 50.0, 100.0),
                                           scale, pad_x, pad_y, 1280, 720)
        self.assertLessEqual(x1, x2)
        self.assertLessEqual(y1, y2)
        self.assertEqual(0.0, y1)
        self.assertEqual(0.0, y2)

    def test_a_reversed_box_is_re_ordered_not_propagated(self):
        box = inverse_letterbox((300.0, 400.0, 100.0, 200.0), 1.0, 0.0, 0.0, 640, 640)
        self.assertEqual((100.0, 200.0, 300.0, 400.0), box)

    def test_zero_or_negative_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            inverse_letterbox((0.0, 0.0, 1.0, 1.0), 0.0, 0.0, 0.0, 10, 10)


class TestNms(unittest.TestCase):

    def test_heavily_overlapping_boxes_are_suppressed_and_the_best_wins(self):
        boxes = [(0, 0, 100, 100), (5, 5, 105, 105), (2, 2, 98, 98)]
        scores = [0.6, 0.9, 0.7]
        kept = nms(boxes, scores, iou_threshold=0.45, class_ids=[0, 0, 0])
        self.assertEqual([1], kept, 'only the highest-scoring box survives')

    def test_distinct_boxes_all_survive(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 110), (200, 0, 210, 10)]
        scores = [0.5, 0.9, 0.7]
        kept = nms(boxes, scores, iou_threshold=0.45, class_ids=[0, 0, 0])
        self.assertEqual([1, 2, 0], kept, 'no overlap: everything is kept, best first')

    def test_kept_indices_come_back_highest_score_first(self):
        boxes = [(0, 0, 10, 10), (50, 50, 60, 60), (100, 100, 110, 110)]
        kept = nms(boxes, [0.1, 0.8, 0.4], iou_threshold=0.5)
        self.assertEqual([1, 2, 0], kept)

    def test_suppression_is_class_aware(self):
        # A person and the backpack they are wearing occupy the same pixels. A
        # class-agnostic NMS would delete one of them; this one must not.
        boxes = [(0, 0, 100, 100), (0, 0, 100, 100)]
        scores = [0.9, 0.8]
        self.assertEqual([0, 1], nms(boxes, scores, 0.45, class_ids=[0, 1]),
                         'different classes must not suppress each other')
        self.assertEqual([0], nms(boxes, scores, 0.45, class_ids=[0, 0]),
                         'same class, identical box: one must go')

    def test_class_ids_none_is_class_agnostic(self):
        boxes = [(0, 0, 100, 100), (0, 0, 100, 100)]
        self.assertEqual([0], nms(boxes, [0.9, 0.8], 0.45))

    def test_threshold_governs_what_counts_as_a_duplicate(self):
        # IoU of exactly 0.25: kept at 0.45, suppressed at 0.1.
        boxes = [(0, 0, 100, 100), (0, 80, 100, 180)]  # inter 2000, union 18000 -> 0.111
        self.assertEqual([0, 1], nms(boxes, [0.9, 0.8], 0.45, class_ids=[0, 0]))
        self.assertEqual([0], nms(boxes, [0.9, 0.8], 0.05, class_ids=[0, 0]))

    def test_iou_at_exactly_the_threshold_is_kept(self):
        # Suppression is strictly greater-than, so a box AT the threshold stays.
        boxes = [(0, 0, 100, 100), (0, 50, 100, 150)]  # inter 5000, union 15000 -> 1/3
        self.assertEqual([0, 1], nms(boxes, [0.9, 0.8], 1.0 / 3.0, class_ids=[0, 0]))

    def test_ties_break_deterministically_on_the_lower_index(self):
        boxes = [(0, 0, 10, 10), (200, 200, 210, 210)]
        self.assertEqual([0, 1], nms(boxes, [0.5, 0.5], 0.45, class_ids=[0, 0]))

    def test_max_detections_caps_the_result(self):
        boxes = [(i * 100, 0, i * 100 + 10, 10) for i in range(5)]
        scores = [0.1 * (i + 1) for i in range(5)]
        kept = nms(boxes, scores, 0.45, class_ids=[0] * 5, max_detections=2)
        self.assertEqual([4, 3], kept)

    def test_empty_input_and_length_mismatch(self):
        self.assertEqual([], nms([], [], 0.45))
        with self.assertRaises(ValueError):
            nms([(0, 0, 1, 1)], [0.5, 0.5], 0.45)
        with self.assertRaises(ValueError):
            nms([(0, 0, 1, 1)], [0.5], 0.45, class_ids=[0, 1])

    def test_zero_area_boxes_do_not_divide_by_zero(self):
        self.assertEqual([0, 1], nms([(5, 5, 5, 5), (5, 5, 5, 5)], [0.9, 0.8], 0.45,
                                     class_ids=[0, 0]))


class TestDetectLayout(unittest.TestCase):

    def test_v8_shape(self):
        self.assertEqual(LAYOUT_V8, detect_layout((1, 84, 8400)))
        self.assertEqual(LAYOUT_V8, detect_layout((1, 5, 8400)))       # nc=1
        self.assertEqual(LAYOUT_V8, detect_layout((1, 84, 8400), num_classes=80))

    def test_v5_shape(self):
        self.assertEqual(LAYOUT_V5, detect_layout((1, 25200, 85)))
        self.assertEqual(LAYOUT_V5, detect_layout((1, 25200, 85), num_classes=80))

    def test_a_squeezed_batch_axis_is_tolerated(self):
        self.assertEqual(LAYOUT_V8, detect_layout((84, 8400)))

    def test_class_count_disambiguates_a_square_output(self):
        # (1, 7, 7): reads as v8/nc=3 or v5/nc=2, and neither axis is longer, so
        # the geometry has nothing to say. The model's own class count does.
        with self.assertRaises(ValueError):
            detect_layout((1, 7, 7))
        self.assertEqual(LAYOUT_V8, detect_layout((1, 7, 7), num_classes=3))
        self.assertEqual(LAYOUT_V5, detect_layout((1, 7, 7), num_classes=2))

    def test_the_class_count_outranks_the_geometry(self):
        # A v5 output with more classes than anchors: the geometry alone would
        # read it as v8, the declared class count says otherwise and wins.
        self.assertEqual(LAYOUT_V5, detect_layout((1, 300, 1005), num_classes=1000))
        self.assertEqual(LAYOUT_V8, detect_layout((1, 300, 1005)))

    def test_a_stale_class_count_does_not_disable_a_decodable_model(self):
        # classes.txt says 5 classes, the model has 80. The shape is unambiguous;
        # the names are what is wrong, so decoding must still work.
        self.assertEqual(LAYOUT_V8, detect_layout((1, 84, 8400), num_classes=5))

    def test_ambiguous_and_undecodable_shapes_raise(self):
        with self.assertRaises(ValueError):
            detect_layout((1, 84, 84))          # square: no anchor axis
        with self.assertRaises(ValueError):
            # v8 with no class scores; reading it as v5 would mean 4 anchors and
            # 8395 classes, which is not a detector -- so this must not decode.
            detect_layout((1, 4, 8400))
        with self.assertRaises(ValueError):
            detect_layout((1, 300, 6, 2))       # not a detector output
        with self.assertRaises(ValueError):
            detect_layout((2, 84, 8400))        # batch > 1
        with self.assertRaises(ValueError):
            detect_layout(())

    def test_the_error_names_the_shape(self):
        with self.assertRaises(ValueError) as caught:
            detect_layout((1, 84, 84))
        self.assertIn('(1, 84, 84)', str(caught.exception))


class TestDecodeOutput(unittest.TestCase):
    """cx,cy,w,h -> corners, threshold, and the objectness rule -- in MODEL space."""

    def test_v8_decodes_boxes_and_takes_the_best_class(self):
        raw = v8_tensor({0: (100, 200, 40, 60, (0.9, 0.1)),
                         1: (400, 300, 100, 100, (0.2, 0.8))}, 2, 20)
        boxes, scores, class_ids = decode_output(raw, conf_threshold=0.25)

        self.assertEqual([(80.0, 170.0, 120.0, 230.0), (350.0, 250.0, 450.0, 350.0)], boxes)
        self.assertEqual([0, 1], class_ids)
        for expected, actual in zip([0.9, 0.8], scores):
            self.assertAlmostEqual(expected, actual, delta=EPS)

    def test_v8_has_no_objectness_channel(self):
        # If index 4 were read as objectness, this box (class score 0.9) would
        # come back with score 0.9*0.9 -- or vanish. It must be 0.9.
        raw = v8_tensor({3: (10, 10, 4, 4, (0.9, 0.0))}, 2, 20)
        _, scores, _ = decode_output(raw, conf_threshold=0.25)
        self.assertEqual(1, len(scores))
        self.assertAlmostEqual(0.9, scores[0], delta=EPS)

    def test_v5_multiplies_objectness_by_the_class_score(self):
        raw = v5_tensor({0: (100, 200, 40, 60, 0.9, (1.0, 0.0)),
                         5: (400, 300, 100, 100, 0.5, (0.0, 0.8))}, 2, 20)
        boxes, scores, class_ids = decode_output(raw, conf_threshold=0.25)

        self.assertEqual([(80.0, 170.0, 120.0, 230.0), (350.0, 250.0, 450.0, 350.0)], boxes)
        self.assertEqual([0, 1], class_ids)
        self.assertAlmostEqual(0.9, scores[0], delta=EPS)
        self.assertAlmostEqual(0.4, scores[1], delta=EPS, msg='0.5 objectness * 0.8 class')

    def test_confidence_threshold_drops_weak_anchors(self):
        raw = v8_tensor({0: (100, 200, 40, 60, (0.9, 0.0)),
                         1: (10, 10, 4, 4, (0.2, 0.0))}, 2, 20)
        self.assertEqual(1, len(decode_output(raw, conf_threshold=0.25)[0]))
        self.assertEqual(2, len(decode_output(raw, conf_threshold=0.1)[0]))

    def test_v5_objectness_alone_can_drop_an_anchor(self):
        # High class score, but the model says there is probably nothing there.
        raw = v5_tensor({0: (100, 200, 40, 60, 0.2, (1.0, 0.0))}, 2, 20)
        self.assertEqual([], decode_output(raw, conf_threshold=0.25)[0])

    def test_unfired_anchors_are_not_detections(self):
        self.assertEqual(([], [], []), decode_output(v8_tensor({}, 2, 20), conf_threshold=0.25))
        self.assertEqual(([], [], []), decode_output(v5_tensor({}, 2, 20), conf_threshold=0.25))

    def test_layout_can_be_forced_when_the_shape_is_ambiguous(self):
        raw = v8_tensor({0: (100, 200, 40, 60, (0.9, 0.1, 0.0))}, 3, 7)  # (1, 7, 7)
        with self.assertRaises(ValueError):
            decode_output(raw, conf_threshold=0.25)
        boxes, _, _ = decode_output(raw, layout=LAYOUT_V8, conf_threshold=0.25)
        self.assertEqual([(80.0, 170.0, 120.0, 230.0)], boxes)

    def test_an_undecodable_shape_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            decode_output([[[0.0] * 84] * 84], conf_threshold=0.25)  # (1, 84, 84)


class TestPostprocess(unittest.TestCase):
    """The whole dependency-free pipeline: raw tensor -> Detections in ORIGINAL pixels."""

    def assert_detection(self, detection, label, box, score, class_id):
        self.assertIsInstance(detection, Detection)
        self.assertEqual(label, detection.label)
        self.assertEqual(class_id, detection.class_id)
        self.assertAlmostEqual(score, detection.score, delta=EPS)
        for expected, actual in zip(box, detection.box):
            self.assertAlmostEqual(expected, actual, delta=1e-4,
                                   msg='box %r != %r' % (detection.box, box))

    def scene_v8(self):
        # Two cats on top of each other (one must be suppressed), a dog on the
        # same pixels as the surviving cat (must NOT be suppressed: different
        # class), a dog elsewhere, and one sub-threshold anchor.
        return v8_tensor({
            0: (100, 200, 40, 60, (0.9, 0.0)),    # cat 0.90
            1: (104, 204, 40, 60, (0.7, 0.0)),    # cat 0.70, IoU 0.72 with #0
            2: (100, 200, 40, 60, (0.0, 0.85)),   # dog 0.85, same box as #0
            3: (400, 300, 100, 100, (0.0, 0.8)),  # dog 0.80, elsewhere
            4: (500, 500, 20, 20, (0.02, 0.01)),  # noise
        }, 2, 20)

    def scene_v5(self):
        return v5_tensor({
            0: (100, 200, 40, 60, 0.9, (1.0, 0.0)),
            1: (104, 204, 40, 60, 0.7, (1.0, 0.0)),
            2: (100, 200, 40, 60, 0.85, (0.0, 1.0)),
            3: (400, 300, 100, 100, 1.0, (0.0, 0.8)),
            4: (500, 500, 20, 20, 0.02, (1.0, 1.0)),
        }, 2, 20)

    def run_scene(self, raw):
        return postprocess(
            raw,
            orig_w=SCENE_IMAGE[0], orig_h=SCENE_IMAGE[1],
            in_w=SCENE_INPUT[0], in_h=SCENE_INPUT[1],
            class_names=SCENE_CLASSES,
            conf_threshold=0.25,
            iou_threshold=DEFAULT_IOU_THRESHOLD)

    def test_v8_scene_end_to_end(self):
        detections = self.run_scene(self.scene_v8())

        self.assertEqual(3, len(detections), 'the duplicate cat and the noise must go')
        self.assert_detection(detections[0], 'cat', CAT_BOX_ORIG, 0.9, 0)
        self.assert_detection(detections[1], 'dog', CAT_BOX_ORIG, 0.85, 1)
        self.assert_detection(detections[2], 'dog', DOG_BOX_ORIG, 0.8, 1)

    def test_v5_scene_end_to_end(self):
        detections = self.run_scene(self.scene_v5())

        self.assertEqual(3, len(detections))
        self.assert_detection(detections[0], 'cat', CAT_BOX_ORIG, 0.9, 0)
        self.assert_detection(detections[1], 'dog', CAT_BOX_ORIG, 0.85, 1)
        self.assert_detection(detections[2], 'dog', DOG_BOX_ORIG, 0.8, 1)

    def test_both_layouts_agree(self):
        self.assertEqual(self.run_scene(self.scene_v8()), self.run_scene(self.scene_v5()))

    def test_boxes_land_in_original_pixels_not_model_pixels(self):
        # The whole coordinate contract in one assertion: the model saw a 640x640
        # letterboxed canvas, the user sees a 1280x720 image.
        for detection in self.run_scene(self.scene_v8()):
            x1, y1, x2, y2 = detection.box
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, float(SCENE_IMAGE[0]))
            self.assertLessEqual(y2, float(SCENE_IMAGE[1]))
        # ...and at least one box is outside the 640x640 the model worked in.
        self.assertTrue(any(d.box[2] > 640 for d in self.run_scene(self.scene_v8())))

    def test_a_box_predicted_into_the_padding_comes_back_clipped(self):
        # cx=10, w=60 -> x1 = -20 (left of the image); cy=150,h=60 -> y1=120,
        # which is above the image top once the 140px pad is removed.
        raw = v8_tensor({0: (10, 150, 60, 60, (0.9, 0.0))}, 2, 20)
        detections = self.run_scene(raw)

        self.assertEqual(1, len(detections))
        self.assert_detection(detections[0], 'cat', (0.0, 0.0, 80.0, 80.0), 0.9, 0)

    def test_unknown_class_ids_fall_back_to_generic_names(self):
        raw = v8_tensor({0: (100, 200, 40, 60, (0.0, 0.0, 0.9))}, 3, 20)
        detections = postprocess(raw, 1280, 720, 640, 640,
                                 class_names=['cat', 'dog'],  # stale: model has 3
                                 conf_threshold=0.25)
        self.assertEqual(1, len(detections))
        self.assertEqual('class_2', detections[0].label)
        self.assertEqual(2, detections[0].class_id)

    def test_no_class_names_at_all_still_labels_every_box(self):
        raw = v8_tensor({0: (100, 200, 40, 60, (0.9, 0.0))}, 2, 20)
        detections = postprocess(raw, 1280, 720, 640, 640, conf_threshold=0.25)
        self.assertEqual(['class_0'], [d.label for d in detections])

    def test_max_detections_caps_the_output(self):
        raw = v8_tensor({i: (i * 30 + 20, 300, 10, 10, (0.5 + i * 0.01, 0.0))
                         for i in range(10)}, 2, 20)
        detections = postprocess(raw, 1280, 720, 640, 640, class_names=SCENE_CLASSES,
                                 conf_threshold=0.25, max_detections=3)
        self.assertEqual(3, len(detections))
        scores = [d.score for d in detections]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestClassNames(unittest.TestCase):

    def test_ultralytics_metadata_is_a_python_dict_repr(self):
        self.assertEqual(['person', 'bicycle', 'car'],
                         parse_names_metadata("{0: 'person', 1: 'bicycle', 2: 'car'}"))

    def test_json_and_list_metadata_are_accepted_too(self):
        self.assertEqual(['a', 'b'], parse_names_metadata('{"0": "a", "1": "b"}'))
        self.assertEqual(['a', 'b'], parse_names_metadata("['a', 'b']"))

    def test_sparse_ids_get_generic_names_in_the_gaps(self):
        self.assertEqual(['class_0', 'cat', 'class_2', 'dog'],
                         parse_names_metadata("{1: 'cat', 3: 'dog'}"))

    def test_junk_metadata_is_ignored_not_fatal(self):
        # A malformed blob must not stop a working model from loading.
        for junk in ('', 'not python at all', '{0: ', '42', "{'a': 'b'}"):
            self.assertEqual([], parse_names_metadata(junk))

    def test_metadata_is_never_eval_ed(self):
        # ast.literal_eval, not eval: this string came out of a downloaded file.
        self.assertEqual([], parse_names_metadata("__import__('os').getcwd()"))

    def test_classes_txt_is_one_name_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'classes.txt')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('cat\ndog\n\n  bird  \n')
            self.assertEqual(['cat', 'dog', 'bird'], load_classes_txt(path))

    def test_missing_classes_txt_is_empty_not_fatal(self):
        self.assertEqual([], load_classes_txt('/no/such/classes.txt'))

    def test_generic_names(self):
        self.assertEqual(['class_0', 'class_1'], generic_class_names(2))
        self.assertEqual([], generic_class_names(0))

    def test_class_name_falls_back_outside_the_list(self):
        self.assertEqual('cat', class_name(['cat', 'dog'], 0))
        self.assertEqual('class_7', class_name(['cat', 'dog'], 7))
        self.assertEqual('class_0', class_name([], 0))


class TestRegistryDegradation(unittest.TestCase):
    """build_backend must answer None -- never raise -- on every setup failure."""

    def test_the_backend_is_registered(self):
        self.assertIn('yolo_onnx', available_backends())

    def test_no_model_path_yields_none(self):
        self.assertIsNone(build_backend({'backend': 'yolo_onnx'}))
        self.assertIsNone(build_backend({'backend': 'yolo_onnx', 'model_path': ''}))

    def test_bogus_model_path_yields_none(self):
        self.assertIsNone(build_backend({
            'backend': 'yolo_onnx',
            'model_path': os.path.join(REPO_ROOT, 'no', 'such', 'model.onnx'),
        }))

    def test_a_directory_is_not_a_model(self):
        self.assertIsNone(build_backend({'backend': 'yolo_onnx', 'model_path': REPO_ROOT}))

    def test_missing_numpy_yields_none(self):
        # sys.modules['numpy'] = None makes `import numpy` raise ImportError --
        # exactly what the base install looks like.
        with mock.patch.dict(sys.modules, {'numpy': None}):
            self.assertIsNone(build_backend({
                'backend': 'yolo_onnx',
                'model_path': os.path.join(REPO_ROOT, 'setup.py'),  # exists, so the
            }))                                                     # dep is the only issue

    def test_missing_onnxruntime_yields_none(self):
        with mock.patch.dict(sys.modules, {'onnxruntime': None}):
            self.assertIsNone(build_backend({
                'backend': 'yolo_onnx',
                'model_path': os.path.join(REPO_ROOT, 'setup.py'),
            }))

    def test_a_corrupt_model_file_yields_none(self):
        # Not a MissingDependency -- a real failure, which build_backend still
        # has to absorb (the editor must open).
        if not (HAS_NUMPY and HAS_ONNXRUNTIME):
            self.skipTest('needs numpy + onnxruntime to get as far as the session')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.onnx')
            with open(path, 'wb') as handle:
                handle.write(b'this is not an onnx graph')
            self.assertIsNone(build_backend({'backend': 'yolo_onnx', 'model_path': path}))


class TestZeroDependencyImport(unittest.TestCase):
    """The backend module itself must not drag onnxruntime into a base install."""

    HEAVY_MODULES = ('numpy', 'onnxruntime', 'cv2', 'PyQt5', 'torch')

    def run_probe(self, source):
        proc = subprocess.run([sys.executable, '-c', source],
                              cwd=REPO_ROOT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        self.assertEqual(0, proc.returncode, proc.stderr)
        return proc.stdout.strip()

    def test_importing_the_backend_module_imports_nothing_heavy(self):
        # The geometry has to be importable and testable with no deps at all --
        # a module-level `import onnxruntime` here is exactly the regression that
        # would take the AI menu (and these tests) away from the base install.
        source = (
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import libs.inference.yolo_onnx as backend\n'
            'assert backend.letterbox_params(1280, 720, 640, 640) == (0.5, 0.0, 140.0)\n'
            'print(",".join(m for m in %r if m in sys.modules))\n'
            % (REPO_ROOT, self.HEAVY_MODULES)
        )
        self.assertEqual('', self.run_probe(source),
                         'importing libs.inference.yolo_onnx pulled in heavy modules')

    def test_the_registry_names_the_backend_without_importing_it(self):
        # available_backends() is a table lookup: listing yolo_onnx must not cost
        # an import of it, let alone of onnxruntime.
        source = (
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import libs.inference\n'
            'assert "yolo_onnx" in libs.inference.available_backends()\n'
            'print("libs.inference.yolo_onnx" in sys.modules,'
            ' ",".join(m for m in %r if m in sys.modules))\n'
            % (REPO_ROOT, self.HEAVY_MODULES)
        )
        self.assertEqual('False', self.run_probe(source))


@unittest.skipUnless(HAS_NUMPY, 'needs numpy (pip install labelImg[ai])')
class TestPredictWithFakeSession(unittest.TestCase):
    """The real predict() path -- letterbox, feed, decode -- with a fake session.

    Needs numpy (predict builds an input tensor) but NOT onnxruntime: the session
    is the one thing being faked, so everything around it is testable without the
    heavyweight dependency.
    """

    class FakeNode(object):
        def __init__(self, name, shape, type_='tensor(float)'):
            self.name = name
            self.shape = list(shape)
            self.type = type_

    class FakeMeta(object):
        def __init__(self, metadata):
            self.custom_metadata_map = dict(metadata)

    class FakeSession(object):
        """Enough of onnxruntime.InferenceSession for the backend to run."""

        def __init__(self, output, input_shape=(1, 3, 640, 640),
                     output_shape=(), metadata=None, input_type='tensor(float)'):
            self._output = output
            self._input = TestPredictWithFakeSession.FakeNode('images', input_shape,
                                                              input_type)
            self._output_node = TestPredictWithFakeSession.FakeNode('output0',
                                                                    output_shape)
            self._metadata = metadata or {}
            self.feeds = []

        def get_inputs(self):
            return [self._input]

        def get_outputs(self):
            return [self._output_node]

        def get_modelmeta(self):
            return TestPredictWithFakeSession.FakeMeta(self._metadata)

        def run(self, output_names, feeds):
            self.feeds.append(feeds)
            return [self._output]

    def scene_tensor(self):
        return v8_tensor({
            0: (100, 200, 40, 60, (0.9, 0.0)),
            1: (104, 204, 40, 60, (0.7, 0.0)),    # duplicate cat
            2: (400, 300, 100, 100, (0.0, 0.8)),
        }, 2, 20)

    def image(self, width, height, value=200):
        import numpy as np
        return np.full((height, width, 3), value, dtype=np.uint8)

    def build(self, session=None, **config):
        config.setdefault('conf_threshold', 0.25)
        return YoloOnnxBackend(session=session or self.FakeSession(self.scene_tensor()),
                               **config)

    def test_it_is_a_model_backend_that_detects(self):
        backend = self.build()
        self.assertIsInstance(backend, ModelBackend)
        self.assertEqual('yolo_onnx', backend.name)
        self.assertTrue(backend.supports_detection)
        self.assertFalse(backend.supports_segmentation)

    def test_predict_returns_original_image_coordinates(self):
        backend = self.build(class_names=SCENE_CLASSES)
        detections = backend.predict(self.image(*SCENE_IMAGE))

        self.assertEqual(2, len(detections), 'the duplicate cat must be suppressed')
        self.assertEqual('cat', detections[0].label)
        self.assertEqual('dog', detections[1].label)
        for expected, actual in zip(CAT_BOX_ORIG, detections[0].box):
            self.assertAlmostEqual(expected, actual, delta=1e-4)
        for expected, actual in zip(DOG_BOX_ORIG, detections[1].box):
            self.assertAlmostEqual(expected, actual, delta=1e-4)

    def test_the_same_boxes_land_differently_on_a_differently_sized_image(self):
        # The proof that the inverse really uses the image size: an identical
        # tensor on a square image maps to different original pixels.
        backend = self.build(class_names=SCENE_CLASSES)
        detections = backend.predict(self.image(640, 640))
        # scale 1.0, no padding: model pixels ARE original pixels here.
        self.assertEqual((80.0, 170.0, 120.0, 230.0), detections[0].box)

    def test_the_input_tensor_is_nchw_normalised_and_letterboxed(self):
        import numpy as np
        session = self.FakeSession(self.scene_tensor())
        backend = self.build(session=session)
        backend.predict(self.image(1280, 720, value=255))

        tensor = session.feeds[0]['images']
        self.assertEqual((1, 3, 640, 640), tuple(tensor.shape))
        self.assertEqual(np.float32, tensor.dtype)
        # Image content is white (1.0); the padded strip is the grey 114/255.
        self.assertAlmostEqual(1.0, float(tensor[0, 0, 320, 320]), delta=1e-3)
        self.assertAlmostEqual(114.0 / 255.0, float(tensor[0, 0, 0, 0]), delta=1e-3)
        self.assertAlmostEqual(114.0 / 255.0, float(tensor[0, 0, 639, 639]), delta=1e-3)
        # 720*0.5 = 360 rows of image, centred: rows 140..499.
        self.assertAlmostEqual(1.0, float(tensor[0, 0, 140, 0]), delta=1e-3)
        self.assertAlmostEqual(114.0 / 255.0, float(tensor[0, 0, 139, 0]), delta=1e-3)

    def test_the_session_is_reused_across_images(self):
        session = self.FakeSession(self.scene_tensor())
        backend = self.build(session=session)
        backend.predict(self.image(640, 480))
        backend.predict(self.image(800, 600))
        self.assertEqual(2, len(session.feeds), 'one run per image, one session')

    def test_the_input_size_comes_from_the_model(self):
        session = self.FakeSession(self.scene_tensor(), input_shape=(1, 3, 320, 416))
        backend = self.build(session=session)
        self.assertEqual((416, 320), (backend.input_width, backend.input_height))

    def test_a_dynamic_input_size_falls_back_to_the_config_then_to_640(self):
        dynamic = (1, 3, 'height', 'width')
        self.assertEqual(
            (640, 640),
            (lambda b: (b.input_width, b.input_height))(
                self.build(session=self.FakeSession(self.scene_tensor(), dynamic))))
        backend = self.build(session=self.FakeSession(self.scene_tensor(), dynamic),
                             input_size=320)
        self.assertEqual((320, 320), (backend.input_width, backend.input_height))

    def test_class_names_come_from_the_model_metadata_first(self):
        session = self.FakeSession(self.scene_tensor(),
                                   metadata={'names': "{0: 'cat', 1: 'dog'}"})
        backend = self.build(session=session)
        self.assertEqual(['cat', 'dog'], backend.class_names)
        self.assertEqual('cat', backend.predict(self.image(640, 640))[0].label)

    def test_class_names_fall_back_to_a_sibling_classes_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'classes.txt'), 'w', encoding='utf-8') as handle:
                handle.write('kitten\npuppy\n')
            backend = self.build(session=self.FakeSession(self.scene_tensor()),
                                 model_path=os.path.join(tmp, 'model.onnx'))
            self.assertEqual(['kitten', 'puppy'], backend.class_names)

    def test_metadata_wins_over_classes_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'classes.txt'), 'w', encoding='utf-8') as handle:
                handle.write('wrong\nalso_wrong\n')
            session = self.FakeSession(self.scene_tensor(),
                                       metadata={'names': "{0: 'cat', 1: 'dog'}"})
            backend = self.build(session=session,
                                 model_path=os.path.join(tmp, 'model.onnx'))
            self.assertEqual(['cat', 'dog'], backend.class_names)

    def test_class_names_default_to_generic_from_the_output_shape(self):
        session = self.FakeSession(self.scene_tensor(), output_shape=(1, 6, 20))
        self.assertEqual(['class_0', 'class_1'], self.build(session=session).class_names)

    def test_generic_class_names_are_filled_in_after_the_first_run(self):
        # Dynamic output shape, no metadata: nc is unknown until a tensor arrives.
        session = self.FakeSession(self.scene_tensor(), output_shape=(1, 'anchors', 'nc'))
        backend = self.build(session=session)
        self.assertEqual([], backend.class_names)
        detections = backend.predict(self.image(640, 640))
        self.assertEqual(['class_0', 'class_1'], backend.class_names)
        self.assertEqual('class_0', detections[0].label)

    def test_an_explicit_class_names_config_wins_over_everything(self):
        session = self.FakeSession(self.scene_tensor(),
                                   metadata={'names': "{0: 'cat', 1: 'dog'}"})
        backend = self.build(session=session, class_names=['kitten', 'puppy'])
        self.assertEqual(['kitten', 'puppy'], backend.class_names)

    def test_the_confidence_threshold_has_a_floor(self):
        # The app asks for 0.0 (it re-filters in the UI), but a real detector
        # emits thousands of near-zero anchors; taking 0.0 literally would feed
        # all of them to a pure-Python NMS and bury the canvas.
        self.assertEqual(MIN_CONF_THRESHOLD, self.build(conf_threshold=0.0).conf_threshold)
        self.assertEqual(0.6, self.build(conf_threshold=0.6).conf_threshold)

    def test_a_raw_image_carrier_reports_a_missing_dependency(self):
        class RawImageLike(object):  # libs.inference.service.RawImage
            shape = (720, 1280, 3)
            data = b'\x00' * 12

        with self.assertRaises(MissingDependency):
            self.build().predict(RawImageLike())

    def test_a_non_rgb_image_is_rejected_clearly(self):
        import numpy as np
        with self.assertRaises(ValueError):
            self.build().predict(np.zeros((10, 10), dtype=np.uint8))

    def test_an_undecodable_output_raises_instead_of_inventing_boxes(self):
        session = self.FakeSession([[[0.0] * 84 for _ in range(84)]])  # (1, 84, 84)
        with self.assertRaises(ValueError):
            self.build(session=session).predict(self.image(640, 640))

    def test_the_layout_override_reaches_the_decoder(self):
        # (1, 7, 7): square, so ambiguous on shape alone, and no names to help.
        raw = v8_tensor({0: (100, 200, 40, 60, (0.9, 0.1, 0.0))}, 3, 7)
        with self.assertRaises(ValueError):
            self.build(session=self.FakeSession(raw)).predict(self.image(640, 640))

        backend = self.build(session=self.FakeSession(raw), layout=LAYOUT_V8)
        detections = backend.predict(self.image(640, 640))
        self.assertEqual([(80.0, 170.0, 120.0, 230.0)], [d.box for d in detections])

    def test_missing_numpy_is_a_missing_dependency_not_an_import_error(self):
        with mock.patch.dict(sys.modules, {'numpy': None}):
            with self.assertRaises(MissingDependency):
                self.build()

    def test_close_releases_the_session(self):
        backend = self.build()
        backend.close()
        self.assertIsNone(backend._session)


# ---------------------------------------------------------------------------
# A real ONNX model, serialised by hand.
#
# onnxruntime can RUN a model but cannot BUILD one, and the `onnx` package (which
# can) is deliberately NOT in the [ai] extra -- adding a dependency just to write
# a test would be the tail wagging the dog. So the few hundred bytes of protobuf
# are emitted here: a graph with one Constant node whose value is a v8-layout
# detection tensor, plus the input tensor the backend must feed and the
# Ultralytics-style `names` metadata it must parse.
#
# This is the only test that exercises the real onnxruntime session: provider
# selection, the feeds dict, the input dtype, get_modelmeta() and get_inputs().
# ---------------------------------------------------------------------------

def _varint(value):
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        out.append(chunk | 0x80 if value else chunk)
        if not value:
            return bytes(out)


def _pb_uint(field, value):
    return _varint(field << 3) + _varint(value)               # wire type 0


def _pb_bytes(field, payload):
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload  # wire type 2


def _pb_str(field, text):
    return _pb_bytes(field, text.encode('utf-8'))


def _pb_tensor(dims, values):
    """TensorProto: dims (1), data_type=FLOAT (2), raw_data (9, little-endian)."""
    body = b''.join(_pb_uint(1, d) for d in dims)
    body += _pb_uint(2, 1)
    body += _pb_bytes(9, struct.pack('<%df' % len(values), *values))
    return body


def _pb_value_info(name, dims):
    """ValueInfoProto -> TypeProto -> Tensor{elem_type, shape{dim{dim_value}}}."""
    shape = b''.join(_pb_bytes(1, _pb_uint(1, d)) for d in dims)
    tensor_type = _pb_uint(1, 1) + _pb_bytes(2, shape)
    return _pb_str(1, name) + _pb_bytes(2, _pb_bytes(1, tensor_type))


def build_onnx_model(input_shape, output_dims, output_values, metadata):
    """A minimal, valid ONNX ModelProto as bytes."""
    # AttributeProto: name (1), t (5), type=TENSOR=4 (20).
    attribute = (_pb_str(1, 'value')
                 + _pb_bytes(5, _pb_tensor(output_dims, output_values))
                 + _pb_uint(20, 4))
    # NodeProto: output (2), name (3), op_type (4), attribute (5).
    node = (_pb_str(2, 'output0') + _pb_str(3, 'detections')
            + _pb_str(4, 'Constant') + _pb_bytes(5, attribute))
    # GraphProto: node (1), name (2), input (11), output (12).
    graph = (_pb_bytes(1, node) + _pb_str(2, 'tiny_yolo')
             + _pb_bytes(11, _pb_value_info('images', input_shape))
             + _pb_bytes(12, _pb_value_info('output0', output_dims)))
    # ModelProto: ir_version (1), producer_name (2), graph (7), opset_import (8),
    # metadata_props (14).
    model = (_pb_uint(1, 8)
             + _pb_str(2, 'labelImg-tests')
             + _pb_bytes(8, _pb_str(1, '') + _pb_uint(2, 13))
             + _pb_bytes(7, graph))
    for key, value in metadata.items():
        model += _pb_bytes(14, _pb_str(1, key) + _pb_str(2, value))
    return model


@unittest.skipUnless(HAS_NUMPY and HAS_ONNXRUNTIME,
                     'needs numpy + onnxruntime (pip install labelImg[ai])')
class TestRealOnnxModel(unittest.TestCase):
    """One end-to-end run through a real onnxruntime session."""

    # A 64x64 network, 2 classes, 20 anchors -> output (1, 6, 20), v8 layout.
    INPUT_SHAPE = (1, 3, 64, 64)
    OUTPUT_DIMS = (1, 6, 20)

    def flat_output(self):
        # One cat at model-space (24, 28, 40, 36) -- cx 32, cy 32, w 16, h 8.
        rows = v8_tensor({0: (32, 32, 16, 8, (0.9, 0.1))}, 2, 20)[0]
        return [value for row in rows for value in row]

    def write_model(self, directory):
        path = os.path.join(directory, 'tiny_yolo.onnx')
        with open(path, 'wb') as handle:
            handle.write(build_onnx_model(
                self.INPUT_SHAPE, self.OUTPUT_DIMS, self.flat_output(),
                {'names': "{0: 'cat', 1: 'dog'}"}))
        return path

    def test_predict_through_a_real_session_lands_in_original_pixels(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            backend = build_backend({
                'backend': 'yolo_onnx',
                'model_path': self.write_model(tmp),
                'conf_threshold': 0.25,
            })
            self.assertIsNotNone(backend, 'the registry failed to build a real model')

            # Input size and class names both read out of the real model file.
            self.assertEqual((64, 64), (backend.input_width, backend.input_height))
            self.assertEqual(['cat', 'dog'], backend.class_names)

            # A 128x64 image: scale 0.5, pad_x 0, pad_y 16.
            image = np.zeros((64, 128, 3), dtype=np.uint8)
            detections = backend.predict(image)

            self.assertEqual(1, len(detections))
            detection = detections[0]
            self.assertEqual('cat', detection.label)
            self.assertEqual(0, detection.class_id)
            self.assertAlmostEqual(0.9, detection.score, delta=1e-6)
            # (24, 28, 40, 36) model space -> ((24-0)/0.5, (28-16)/0.5, ...)
            for expected, actual in zip((48.0, 24.0, 80.0, 40.0), detection.box):
                self.assertAlmostEqual(expected, actual, delta=1e-3)

            backend.close()

    def test_an_unavailable_provider_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = build_backend({
                'backend': 'yolo_onnx',
                'model_path': self.write_model(tmp),
                'providers': ['NoSuchExecutionProvider'],
            })
            self.assertIsNotNone(backend, 'a bad provider must not disable the model')
            self.assertEqual(1, len(backend.predict(
                __import__('numpy').zeros((64, 64, 3), dtype=__import__('numpy').uint8))))
            backend.close()


if __name__ == '__main__':
    unittest.main()
