import os
import sys
import tempfile
import unittest
from unittest import mock
from xml.etree import ElementTree

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from PyQt5.QtWidgets import QApplication

# QApplication is a process-wide singleton; create it once at import so every
# test module (and get_main_app, which reuses an existing instance) shares it.
APP = QApplication.instance() or QApplication([])

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QImage

from labelImg import get_main_app
from libs.assist.suggestion import detection_to_shape
from libs.inference.service import RawImage, SynchronousExecutor, to_model_image
from libs.inference.stub import StubBackend, image_size
from libs.inference.types import Detection
from libs.shape import Shape

IMAGE_SIZE = 64


class CountingStub(StubBackend):
    """StubBackend that records how often the model was actually run.

    The point of the confidence threshold is that it re-filters what is already
    on screen; a test that only counted boxes could not tell that apart from a
    silent second inference run.
    """

    def __init__(self, **kwargs):
        super(CountingStub, self).__init__(**kwargs)
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return super(CountingStub, self).predict(image)


class AssistTestCase(unittest.TestCase):
    """Drives a real MainWindow headless (same idiom as tests/test_classify.py),
    with StubBackend + a synchronous executor injected: no model, no
    onnxruntime, no event-loop timing races."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        for stem in ('a', 'b'):
            img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
            img.fill(0xffffffff)
            img.save(os.path.join(self.dir, stem + '.png'))
        self.win = None
        self.launch()

    def launch(self, backend=None):
        """Open the directory the way the CLI does. expanduser is patched for the
        construction window so Settings() reads a fresh pickle from tmp — the
        developer's real ~/.labelImgSettings.pkl must steer neither construction
        nor teardown."""
        if self.win is not None:
            self.win.close()
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

        # Inference runs inline on this thread, so an assertion may follow the
        # call directly instead of racing a worker.
        self.win.inference_service.set_executor(SynchronousExecutor())
        self.backend = backend if backend is not None else CountingStub()
        self.win.assist.set_backend(self.backend)
        # Everything the annotation writers need, without a file dialog.
        self.win.default_save_dir = self.dir

    def tearDown(self):
        # closeEvent -> may_continue() pops a MODAL "unsaved changes" dialog when
        # the window is dirty, which would hang a headless run forever. Accepting
        # or rejecting suggestions legitimately leaves the document dirty, so drop
        # the flag rather than the assertion that produced it.
        self.win.dirty = False
        self.win.close()
        self.tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def path(self, name):
        return os.path.join(self.dir, name)

    def add_real_box(self, label='dog'):
        """A committed (non-provisional) box, added the way the app adds one."""
        shape = Shape(label=label)
        for x, y in ((1, 1), (9, 1), (9, 9), (1, 9)):
            shape.add_point(QPointF(x, y))
        shape.close()
        self.win.canvas.load_shapes(self.win.canvas.shapes + [shape])
        self.win.add_label(shape)
        return shape

    def saved_labels(self):
        """Save through the app's normal path and read back what hit the disk."""
        self.win.save_file()
        xml_path = os.path.splitext(self.win.file_path)[0] + '.xml'
        self.assertTrue(os.path.isfile(xml_path), 'nothing was saved')
        root = ElementTree.parse(xml_path).getroot()
        return [obj.find('name').text for obj in root.findall('object')]


class TestAutoLabel(AssistTestCase):

    def test_auto_label_creates_registered_provisional_shapes(self):
        self.win.assist.auto_label_image()

        shapes = self.win.canvas.shapes
        self.assertEqual(2, len(shapes))  # StubBackend: 2 detections
        self.assertEqual(['person', 'face'], [s.label for s in shapes])
        self.assertTrue(all(s.provisional for s in shapes))
        self.assertEqual([0.9, 0.8], [round(s.confidence, 2) for s in shapes])

        # Registered via add_label, not dropped straight onto the canvas: canvas
        # selection resolves shapes through shapes_to_items, and remove_label
        # does not guard that lookup.
        self.assertEqual(2, self.win.label_list.count())
        for shape in shapes:
            self.assertIn(shape, self.win.shapes_to_items)
        self.assertEqual([], self.errors)

    def test_boxes_land_in_original_image_pixels(self):
        # StubBackend detection 0 of 2 on a 64x64 image: centre (64/3, 64/3),
        # half-extents 10% of the image. If anything applied a canvas scale or a
        # zoom factor on the way in, these numbers move.
        self.win.assist.auto_label_image()

        centre = IMAGE_SIZE / 3.0
        half = IMAGE_SIZE * 0.1
        expected = [(centre - half, centre - half), (centre + half, centre - half),
                    (centre + half, centre + half), (centre - half, centre + half)]
        points = [(p.x(), p.y()) for p in self.win.canvas.shapes[0].points]
        for (ex, ey), (ax, ay) in zip(expected, points):
            self.assertAlmostEqual(ex, ax, places=4)
            self.assertAlmostEqual(ey, ay, places=4)

    def test_rerunning_replaces_instead_of_stacking(self):
        self.win.assist.auto_label_image()
        self.win.assist.auto_label_image()

        self.assertEqual(2, len(self.win.canvas.shapes))
        self.assertEqual(2, self.win.label_list.count())
        self.assertEqual(2, self.backend.calls)

    def test_suggestions_do_not_dirty_the_document(self):
        # Nothing was written and nothing saveable changed: an untouched file
        # must not start looking unsaved just because a model had an opinion.
        self.assertFalse(self.win.dirty)
        self.win.assist.auto_label_image()
        self.assertFalse(self.win.dirty)


class TestSaveFilter(AssistTestCase):

    def test_provisional_shapes_are_excluded_from_a_save(self):
        self.add_real_box('dog')
        self.win.assist.auto_label_image()
        self.assertEqual(3, len(self.win.canvas.shapes))  # 1 real + 2 suggestions

        # The single choke point (save_labels) is the only thing standing between
        # a model's guess and the annotation file.
        self.assertEqual(['dog'], self.saved_labels())

    def test_accepted_suggestions_are_saved(self):
        self.win.assist.auto_label_image()
        self.win.assist.accept_all()

        self.assertTrue(all(not s.provisional for s in self.win.canvas.shapes))
        self.assertTrue(self.win.dirty)
        self.assertEqual(['person', 'face'], self.saved_labels())
        self.assertEqual(2, self.win.label_list.count())

    def test_rejected_suggestions_are_gone_and_the_label_list_stays_consistent(self):
        self.add_real_box('dog')
        self.win.assist.auto_label_image()

        self.win.assist.reject_all()  # must not KeyError in remove_label

        self.assertEqual(1, len(self.win.canvas.shapes))
        self.assertEqual(1, self.win.label_list.count())
        self.assertEqual(1, len(self.win.shapes_to_items))
        self.assertEqual(1, len(self.win.items_to_shapes))
        self.assertEqual(['dog'], self.saved_labels())

        # The two maps still agree, so a later delete cannot blow up.
        for shape, item in self.win.shapes_to_items.items():
            self.assertIs(shape, self.win.items_to_shapes[item])

    def test_reject_of_every_shape_leaves_an_empty_but_usable_state(self):
        self.win.assist.auto_label_image()
        self.win.assist.reject_all()

        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())
        self.assertEqual({}, self.win.shapes_to_items)
        self.assertTrue(self.win.no_shapes())
        # A second reject is a no-op, not a crash.
        self.assertEqual(0, self.win.assist.reject_all())


class TestShapeCopy(unittest.TestCase):
    """Shape.copy() is a hand-rolled field whitelist: a field it forgets is
    silently dropped, and a dropped `provisional` turns a duplicated suggestion
    into a box that gets SAVED."""

    def test_copy_preserves_provisional_confidence_and_type(self):
        original = detection_to_shape(Detection(label='person', box=(1, 2, 3, 4), score=0.75))

        clone = original.copy()

        self.assertTrue(clone.provisional)
        self.assertEqual(0.75, clone.confidence)
        self.assertEqual(Shape.RECT, clone.shape_type)
        self.assertEqual([(p.x(), p.y()) for p in original.points],
                         [(p.x(), p.y()) for p in clone.points])

    def test_copy_of_a_committed_shape_stays_committed(self):
        shape = Shape(label='dog')
        clone = shape.copy()

        self.assertFalse(clone.provisional)
        self.assertIsNone(clone.confidence)


class TestThreshold(AssistTestCase):

    def test_threshold_filters_without_re_running_the_model(self):
        self.win.assist.set_threshold(0.0)
        self.win.assist.auto_label_image()
        self.assertEqual(2, len(self.win.canvas.shapes))
        self.assertEqual(1, self.backend.calls)

        # Stub scores are 0.9 and 0.8: raising the bar past 0.8 hides the second.
        self.win.assist.set_threshold(0.85)
        self.assertEqual(1, len(self.win.canvas.shapes))
        self.assertEqual('person', self.win.canvas.shapes[0].label)
        self.assertEqual(1, self.win.label_list.count())

        # Lowering it brings the box back — from the detections already in hand.
        self.win.assist.set_threshold(0.1)
        self.assertEqual(2, len(self.win.canvas.shapes))
        self.assertEqual(2, self.win.label_list.count())

        self.assertEqual(1, self.backend.calls, 'the threshold re-ran the model')

    def test_threshold_change_does_not_dirty_the_document(self):
        self.win.assist.auto_label_image()
        self.assertFalse(self.win.dirty)

        self.win.assist.set_threshold(0.95)  # hides both suggestions

        self.assertEqual([], self.win.canvas.shapes)
        self.assertFalse(self.win.dirty, 'a view filter is not a document edit')
        self.assertFalse(self.win.actions.save.isEnabled())

    def test_threshold_never_hides_an_accepted_box(self):
        self.win.assist.set_threshold(0.0)
        self.win.assist.auto_label_image()
        self.win.assist.accept_all()

        self.win.assist.set_threshold(1.0)

        # They are the user's boxes now; the model's score has no say over them.
        self.assertEqual(2, len(self.win.canvas.shapes))

    def test_threshold_is_clamped_and_survives_a_corrupt_setting(self):
        self.win.assist.set_threshold(5.0)
        self.assertEqual(1.0, self.win.assist.threshold)
        self.win.assist.set_threshold(-1.0)
        self.assertEqual(0.0, self.win.assist.threshold)
        self.win.assist.set_threshold('not a number')
        self.assertEqual(0.5, self.win.assist.threshold)


class TestStaleResults(AssistTestCase):

    def test_result_for_another_image_is_dropped(self):
        # The classic race: inference for image A finishes after the user has
        # already moved to image B. Injecting A's boxes into B would write them
        # into B's annotation file on the next save.
        detections = [Detection(label='person', box=(1, 1, 5, 5), score=0.99)]

        self.win.inference_service.predictionReady.emit(self.path('b.png'), detections)

        self.assertEqual(self.path('a.png'), self.win.file_path)
        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())

    def test_result_for_the_current_image_is_accepted(self):
        detections = [Detection(label='person', box=(1, 1, 5, 5), score=0.99)]

        self.win.inference_service.predictionReady.emit(self.win.file_path, detections)

        self.assertEqual(1, len(self.win.canvas.shapes))

    def test_navigating_away_mid_inference_drops_the_result(self):
        # Same race, driven through the real code path: the executor "runs" the
        # job only after the app has moved on to the next image.
        deferred = []
        self.win.inference_service.set_executor(
            type('Deferred', (), {'submit': lambda _s, job: deferred.append(job),
                                  'wait_for_done': lambda _s, m=0: True})())

        self.win.assist.auto_label_image()  # queued against a.png
        self.win.open_next_image()
        self.assertEqual(self.path('b.png'), self.win.file_path)

        deferred[0]()  # a.png's result finally arrives

        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())


class TestNoBackend(AssistTestCase):

    def launch(self, backend=None):
        # build_backend returns None when the optional deps are missing; that is
        # a normal outcome, not an error, and the app must stay usable.
        with mock.patch('libs.assist.controller.build_backend', return_value=None):
            super(TestNoBackend, self).launch(backend=backend)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))
        self.win = None
        with mock.patch('libs.assist.controller.build_backend', return_value=None):
            with mock.patch('os.path.expanduser', return_value=self.tmp.name):
                self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def test_ai_actions_are_disabled_with_a_hint(self):
        self.assertFalse(self.win.assist.is_available())
        # An image IS loaded, so toggle_actions has already run and enabled every
        # onLoadActive action — the controller has to win that argument.
        self.assertTrue(self.win.file_path)
        for action in self.win.assist_actions:
            self.assertFalse(action.isEnabled(), action.text())
            self.assertIn('labelImg[ai]', action.toolTip())

    def test_auto_label_without_a_backend_does_not_crash(self):
        self.assertFalse(self.win.assist.auto_label_image())
        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual([], self.errors)

    def test_the_rest_of_the_app_still_works(self):
        # The whole point of build_backend returning None: labelImg keeps working
        # as a plain annotation tool.
        self.win.default_save_dir = self.dir
        shape = Shape(label='dog')
        for x, y in ((1, 1), (9, 1), (9, 9), (1, 9)):
            shape.add_point(QPointF(x, y))
        shape.close()
        self.win.canvas.load_shapes([shape])
        self.win.add_label(shape)
        self.win.save_file()

        xml_path = os.path.splitext(self.win.file_path)[0] + '.xml'
        self.assertTrue(os.path.isfile(xml_path))


class TestImageCarrier(unittest.TestCase):
    """numpy is OPTIONAL: the base install has neither numpy nor onnxruntime, so
    the UI-thread conversion has to produce something a backend can read either
    way — and the worker must never be handed a QImage."""

    def make_image(self, width, height):
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0xffff0000)  # opaque red
        return image

    def test_numpy_path_yields_an_hwc_array(self):
        array = to_model_image(self.make_image(65, 33))  # odd width => padded rows

        self.assertEqual((33, 65, 3), tuple(array.shape))
        self.assertEqual((33, 65), image_size(array))
        # Row padding stripped, channels in RGB order.
        self.assertEqual([255, 0, 0], list(array[0][0]))
        self.assertEqual([255, 0, 0], list(array[32][64]))

    def test_carrier_without_numpy_still_drives_a_backend(self):
        # sys.modules['numpy'] = None makes `import numpy` raise ImportError,
        # which is exactly what the base install looks like.
        with mock.patch.dict(sys.modules, {'numpy': None}):
            carrier = to_model_image(self.make_image(65, 33))

        self.assertIsInstance(carrier, RawImage)
        self.assertEqual((33, 65, 3), carrier.shape)
        self.assertEqual((33, 65), image_size(carrier))
        # Tightly packed: Qt pads a 65px RGB888 row to 196 bytes, not 195.
        self.assertEqual(65 * 33 * 3, len(carrier.data))
        self.assertEqual(b'\xff\x00\x00', carrier.data[:3])
        self.assertEqual(b'\xff\x00\x00', carrier.data[-3:])

        # And a real backend can consume it — no numpy anywhere in this path.
        detections = StubBackend().predict(carrier)
        self.assertEqual(2, len(detections))
        self.assertEqual('person', detections[0].label)

    def test_null_image_is_not_dispatched(self):
        self.assertIsNone(to_model_image(QImage()))
        self.assertIsNone(to_model_image(None))


if __name__ == '__main__':
    unittest.main()
