#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Unit tests for the dependency-free inference core (libs/inference).

Plain unittest, mirroring tests/test_io.py: no Qt, no numpy, no onnxruntime.
If any test here ever needs one of those, the decoupling has been broken and
that is precisely what these tests exist to catch.
"""

import dataclasses
import os
import re
import subprocess
import sys
import unittest

# The package is imported as `libs.inference`, so the repo root (not libs/) is
# what has to be importable -- unlike tests/test_io.py, which imports the flat
# modules inside libs/ directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libs.inference.backend import MissingDependency, ModelBackend
from libs.inference.registry import (available_backends, build_backend,
                                     register_backend)
from libs.inference import registry as registry_module
from libs.inference.stub import StubBackend, image_size
from libs.inference.types import (Detection, Mask, Prediction, SegPrompt,
                                  least_confidence)


class FakeImage(object):
    """Numpy-free stand-in for an HxWx3 array: only `.shape` is ever read."""

    def __init__(self, height, width, channels=3):
        self.shape = (height, width, channels)


class TestTypesContract(unittest.TestCase):

    def test_detection_is_frozen_and_has_defaults(self):
        det = Detection(label='person', box=(10.0, 20.0, 30.0, 40.0), score=0.9)
        self.assertEqual('person', det.label)
        self.assertEqual((10.0, 20.0, 30.0, 40.0), det.box)
        self.assertEqual(0.9, det.score)
        self.assertIsNone(det.class_id, 'class_id must default to None')

        with self.assertRaises(dataclasses.FrozenInstanceError):
            det.label = 'face'
        with self.assertRaises(dataclasses.FrozenInstanceError):
            det.box = (0.0, 0.0, 1.0, 1.0)

    def test_detection_equality_is_by_value(self):
        # Determinism assertions elsewhere rely on this.
        a = Detection('face', (1.0, 2.0, 3.0, 4.0), 0.5, 1)
        b = Detection('face', (1.0, 2.0, 3.0, 4.0), 0.5, 1)
        self.assertEqual(a, b)

    def test_mask_is_frozen_and_score_defaults_to_one(self):
        mask = Mask(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        self.assertEqual(1.0, mask.score)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            mask.score = 0.5

    def test_seg_prompt_defaults_are_not_shared(self):
        # A mutable default done wrong (points=[]) would share one list across
        # every prompt; field(default_factory=list) is what prevents that.
        first = SegPrompt()
        second = SegPrompt()
        self.assertEqual([], first.points)
        self.assertIsNone(first.box)

        first.points.append((5.0, 6.0, 1))
        self.assertEqual([], second.points, 'SegPrompt.points leaked between instances')

    def test_prediction_defaults_and_mutability(self):
        pred = Prediction(image_path='tests/test.512.512.bmp')
        self.assertEqual([], pred.detections)
        self.assertIsNone(pred.uncertainty, 'uncertainty is unset until scored')

        # Mutable on purpose: active learning scores an existing Prediction.
        pred.uncertainty = 0.25
        pred.detections.append(Detection('person', (1.0, 1.0, 2.0, 2.0), 0.8))
        self.assertEqual(0.25, pred.uncertainty)
        self.assertEqual(1, len(pred.detections))
        self.assertEqual([], Prediction(image_path='other.bmp').detections)


class TestLeastConfidence(unittest.TestCase):

    @staticmethod
    def _dets(*scores):
        return [Detection('person', (0.0, 0.0, 1.0, 1.0), s) for s in scores]

    def test_empty_detections_are_maximally_uncertain(self):
        # "The model found nothing" must sort to the FRONT of a review queue.
        self.assertEqual(1.0, least_confidence([]))

    def test_higher_scores_mean_lower_uncertainty(self):
        confident = least_confidence(self._dets(0.9, 0.9))
        unsure = least_confidence(self._dets(0.4, 0.4))
        self.assertLess(confident, unsure)
        self.assertAlmostEqual(0.1, confident)
        self.assertAlmostEqual(0.6, unsure)

    def test_monotonic_in_score(self):
        values = [least_confidence(self._dets(s)) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(values, sorted(values, reverse=True))
        for value in values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_perfect_and_zero_scores_hit_the_bounds(self):
        self.assertAlmostEqual(0.0, least_confidence(self._dets(1.0, 1.0)))
        self.assertAlmostEqual(1.0, least_confidence(self._dets(0.0)))

    def test_top_k_averages_only_the_best_scores(self):
        dets = self._dets(0.8, 0.6, 0.1, 0.1)
        self.assertAlmostEqual(1.0 - 0.4, least_confidence(dets))          # all four
        self.assertAlmostEqual(1.0 - 0.7, least_confidence(dets, top_k=2))  # 0.8, 0.6
        self.assertAlmostEqual(1.0 - 0.8, least_confidence(dets, top_k=1))
        # A tail of junk boxes must not drown a confident top-k.
        self.assertLess(least_confidence(dets, top_k=2), least_confidence(dets))

    def test_top_k_larger_than_available_uses_everything(self):
        self.assertAlmostEqual(least_confidence(self._dets(0.5, 0.5), top_k=10),
                               least_confidence(self._dets(0.5, 0.5)))

    def test_out_of_range_scores_are_clamped(self):
        self.assertAlmostEqual(0.0, least_confidence(self._dets(1.5)))
        self.assertAlmostEqual(1.0, least_confidence(self._dets(-0.5)))

    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            least_confidence(self._dets(0.5), top_k=0)


class TestStubBackend(unittest.TestCase):

    def test_image_size_reads_shape_and_nested_sequences(self):
        self.assertEqual((512, 640), image_size(FakeImage(512, 640)))
        self.assertEqual((2, 3), image_size([[(0, 0, 0)] * 3, [(0, 0, 0)] * 3]))
        with self.assertRaises(ValueError):
            image_size(object())

    def test_is_a_model_backend_and_declares_detection(self):
        backend = StubBackend()
        self.assertIsInstance(backend, ModelBackend)
        self.assertEqual('stub', backend.name)
        self.assertTrue(backend.supports_detection)
        self.assertFalse(backend.supports_segmentation)

    def test_predict_is_deterministic(self):
        backend = StubBackend()
        image = FakeImage(480, 640)
        first = backend.predict(image)
        second = backend.predict(image)
        self.assertEqual(first, second, 'StubBackend must be deterministic')
        # A fresh instance with the same config must agree too.
        self.assertEqual(first, StubBackend().predict(FakeImage(480, 640)))

    def test_boxes_are_inside_the_image_bounds_and_well_formed(self):
        for height, width in ((480, 640), (512, 512), (1, 1), (2000, 37)):
            dets = StubBackend(num_detections=5).predict(FakeImage(height, width))
            for det in dets:
                x1, y1, x2, y2 = det.box
                self.assertLessEqual(x1, x2, 'box must satisfy x1 <= x2')
                self.assertLessEqual(y1, y2, 'box must satisfy y1 <= y2')
                self.assertGreaterEqual(x1, 0.0)
                self.assertGreaterEqual(y1, 0.0)
                self.assertLessEqual(x2, float(width), 'box escapes image width')
                self.assertLessEqual(y2, float(height), 'box escapes image height')
                self.assertGreaterEqual(det.score, 0.0)
                self.assertLessEqual(det.score, 1.0)

    def test_exact_boxes_for_a_known_image(self):
        # Pinning the numbers: any layer above that rescales or transposes
        # coordinates will now fail loudly instead of looking plausible.
        dets = StubBackend(num_detections=1).predict(FakeImage(100, 200))
        self.assertEqual(1, len(dets))
        # centre (100, 50), half extents (20, 10)
        self.assertEqual((80.0, 40.0, 120.0, 60.0), dets[0].box)
        self.assertAlmostEqual(0.9, dets[0].score)
        self.assertEqual('person', dets[0].label)
        self.assertEqual(0, dets[0].class_id)

    def test_configurable_class_names_and_count(self):
        backend = StubBackend(class_names=['cat', 'dog', 'bird'], num_detections=4)
        dets = backend.predict(FakeImage(400, 400))
        self.assertEqual(4, len(dets))
        self.assertEqual(['cat', 'dog', 'bird', 'cat'], [d.label for d in dets])
        self.assertEqual([0, 1, 2, 0], [d.class_id for d in dets])
        # Labels are names taken from class_names, in class-id order.
        for det in dets:
            self.assertEqual(backend.class_names[det.class_id], det.label)

    def test_zero_detections_is_supported(self):
        self.assertEqual([], StubBackend(num_detections=0).predict(FakeImage(64, 64)))

    def test_scores_descend_so_ordering_is_testable(self):
        scores = [d.score for d in StubBackend(num_detections=4).predict(FakeImage(300, 300))]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(scores), len(set(scores)))

    def test_conf_threshold_filters_detections(self):
        backend = StubBackend(num_detections=4, conf_threshold=0.75)
        dets = backend.predict(FakeImage(300, 300))
        self.assertEqual(2, len(dets), 'only the 0.9 and 0.8 boxes survive')
        for det in dets:
            self.assertGreaterEqual(det.score, 0.75)

    def test_segmentation_and_embedding_are_not_implemented(self):
        backend = StubBackend()
        with self.assertRaises(NotImplementedError):
            backend.segment(FakeImage(64, 64), SegPrompt(points=[(1.0, 1.0, 1)]))
        with self.assertRaises(NotImplementedError):
            backend.embed(FakeImage(64, 64))

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            StubBackend(num_detections=-1)
        with self.assertRaises(ValueError):
            StubBackend(class_names=[])


class TestRegistry(unittest.TestCase):

    def setUp(self):
        # register_backend mutates a module-level table; snapshot and restore so
        # tests cannot leak into each other (or into the rest of the suite).
        self._saved_backends = dict(registry_module._BACKENDS)

    def tearDown(self):
        registry_module._BACKENDS.clear()
        registry_module._BACKENDS.update(self._saved_backends)

    def test_build_stub_backend(self):
        backend = build_backend({'backend': 'stub'})
        self.assertIsInstance(backend, StubBackend)
        self.assertTrue(backend.supports_detection)

    def test_config_is_a_plain_mapping_and_extra_keys_are_ignored(self):
        backend = build_backend({
            'backend': 'stub',
            'model_path': 'C:/nonexistent/model.onnx',  # meaningless to the stub
            'conf_threshold': 0.85,
            'num_detections': 3,
            'class_names': ['car'],
            'some_future_key': 42,
        })
        self.assertIsInstance(backend, StubBackend)
        self.assertEqual(0.85, backend.conf_threshold)
        self.assertEqual(['car'], backend.class_names)
        # conf_threshold 0.85 keeps only the 0.9 box out of 3.
        self.assertEqual(1, len(backend.predict(FakeImage(200, 200))))

    def test_empty_or_missing_config_falls_back_to_default_backend(self):
        self.assertIsInstance(build_backend(), StubBackend)
        self.assertIsInstance(build_backend({}), StubBackend)

    def test_unknown_backend_returns_none_without_raising(self):
        self.assertIsNone(build_backend({'backend': 'no_such_model'}))

    def test_missing_dependency_yields_none(self):
        # This is the whole point: a user without the AI extras must get a
        # disabled feature, never a crash.
        def _needs_onnx(config):
            raise MissingDependency('onnxruntime is not installed')

        register_backend('fake_onnx', _needs_onnx)
        self.assertIn('fake_onnx', available_backends())
        self.assertIsNone(build_backend({'backend': 'fake_onnx'}))

    def test_import_error_yields_none(self):
        def _bad_import(config):
            raise ImportError('No module named onnxruntime')

        register_backend('fake_import', _bad_import)
        self.assertIsNone(build_backend({'backend': 'fake_import'}))

    def test_arbitrary_construction_failure_yields_none(self):
        # Bad model path / corrupt weights: still "AI disabled", not "app dead".
        def _explodes(config):
            raise RuntimeError('failed to load weights')

        register_backend('fake_broken', _explodes)
        self.assertIsNone(build_backend({'backend': 'fake_broken'}))

    def test_available_backends_lists_names(self):
        names = available_backends()
        self.assertIn('stub', names)
        self.assertEqual(names, sorted(names))

    def test_register_backend_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            register_backend('', lambda config: StubBackend())


class TestZeroDependencyImport(unittest.TestCase):

    HEAVY_MODULES = ('numpy', 'onnxruntime', 'cv2', 'PyQt5', 'torch', 'lxml')

    def test_package_imports_without_optional_deps(self):
        # Must run in a subprocess: other tests in this suite (test_qt.py) import
        # PyQt5 into *this* interpreter, so inspecting sys.modules here would
        # prove nothing.  A clean interpreter is the only honest check that
        # `import libs.inference` drags in nothing heavy.
        source = (
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import libs.inference\n'
            'from libs.inference import Detection, ModelBackend, build_backend\n'
            'assert build_backend({"backend": "stub"}) is not None\n'
            'heavy = [m for m in %r if m in sys.modules]\n'
            'print(",".join(heavy))\n' % (REPO_ROOT, self.HEAVY_MODULES)
        )
        proc = subprocess.run([sys.executable, '-c', source],
                              cwd=REPO_ROOT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        self.assertEqual(0, proc.returncode,
                         'importing libs.inference failed:\n%s' % proc.stderr)
        self.assertEqual('', proc.stdout.strip(),
                         'import libs.inference pulled in heavy modules: %s'
                         % proc.stdout.strip())

    def test_lazy_reexports_resolve(self):
        import libs.inference as inference

        for name in inference.__all__:
            self.assertTrue(hasattr(inference, name), 'missing re-export: %s' % name)
        self.assertIs(inference.Detection, Detection)
        self.assertIs(inference.ModelBackend, ModelBackend)
        self.assertIs(inference.MissingDependency, MissingDependency)
        self.assertIs(inference.build_backend, build_backend)
        with self.assertRaises(AttributeError):
            inference.no_such_symbol

    def test_stub_backend_is_not_imported_eagerly(self):
        # The registry reaches concrete backends through its factories; a
        # top-level import of a backend here is how onnxruntime would eventually
        # sneak into every process.
        source = (
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import libs.inference\n'
            'print("libs.inference.stub" in sys.modules)\n' % (REPO_ROOT,)
        )
        proc = subprocess.run([sys.executable, '-c', source],
                              cwd=REPO_ROOT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual('False', proc.stdout.strip(),
                         'importing the package must not import concrete backends')


class TestDeclaredPythonSupport(unittest.TestCase):
    """This package is 3.7+ and the metadata has to say so.

    libs/inference uses `from __future__ import annotations` (3.7) and
    dataclasses (3.7), and labelImg.py imports libs.inference.service /
    libs.assist.controller unconditionally at module top. On a 3.3-3.6
    interpreter that is not "the AI feature is unavailable", it is
    `import labelImg` raising SyntaxError/ImportError -- the whole editor, dead.
    setup.py promised >=3.0.0 and classified 3.3-3.6, so the package would have
    installed happily onto an interpreter it can no longer run on.
    """

    def setUp(self):
        with open(os.path.join(REPO_ROOT, 'setup.py'), 'r', encoding='utf-8') as file:
            self.setup_py = file.read()

    def test_requires_python_is_at_least_3_7(self):
        match = re.search(r"^REQUIRES_PYTHON\s*=\s*'([^']+)'", self.setup_py, re.M)
        self.assertIsNotNone(match, 'REQUIRES_PYTHON is gone from setup.py')
        spec = match.group(1).strip()
        self.assertTrue(spec.startswith('>='), 'unexpected specifier: %s' % spec)
        version = tuple(int(part) for part in spec[2:].strip().split('.'))
        self.assertGreaterEqual(version, (3, 7),
                                'setup.py still promises Python %s, which cannot '
                                'import libs.inference' % spec)

    def test_no_classifier_below_3_7(self):
        classifiers = re.findall(
            r"'Programming Language :: Python :: (\d+)\.(\d+)'", self.setup_py)
        self.assertTrue(classifiers, 'no versioned Python classifiers found')
        for major, minor in classifiers:
            self.assertGreaterEqual(
                (int(major), int(minor)), (3, 7),
                'setup.py still classifies Python %s.%s' % (major, minor))

    def test_the_3_7_only_syntax_that_forces_the_floor_is_really_there(self):
        # Keeps the two tests above honest: if this ever stops being true, the
        # floor can be revisited deliberately instead of by accident.
        types_py = os.path.join(REPO_ROOT, 'libs', 'inference', 'types.py')
        with open(types_py, 'r', encoding='utf-8') as file:
            source = file.read()
        self.assertIn('from __future__ import annotations', source)
        self.assertIn('from dataclasses import', source)


if __name__ == '__main__':
    unittest.main()
