import glob
import inspect
import os
import pickle
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
from libs.assist import controller as assist_controller
from libs.assist.suggestion import detection_to_shape
from libs.constants import SETTING_MODEL_BACKEND
from libs.inference import backend as inference_backend
from libs.inference import registry as inference_registry
from libs.inference import yolo_onnx as inference_yolo_onnx
from libs.inference.service import RawImage, SynchronousExecutor, to_model_image
from libs.inference.stub import StubBackend, image_size
from libs.inference.types import Detection
from libs.shape import Shape

IMAGE_SIZE = 64


def _importable(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


# numpy ships with the [ai] extra, not with the base install -- and the core CI
# job installs pyqt5+lxml ONLY. A test that asserts the numpy carrier therefore
# has to say so, or it fails on exactly the install this package promises to keep
# working. (The no-numpy half of the contract is covered below, unguarded.)
HAS_NUMPY = _importable('numpy')


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
        # No SETTING_MODEL_BACKEND was ever written to settings, so this is the
        # "nothing configured" cause specifically (DEFAULT_BACKEND is None) --
        # not "a backend was named but failed to build". The hint text must
        # match: still updated (not left asserting the pre-fix string), because
        # the two causes now say different things (see
        # AssistController._unavailable_hint).
        self.assertIsNone(self.win.assist.backend_name)
        # An image IS loaded, so toggle_actions has already run and enabled every
        # onLoadActive action — the controller has to win that argument.
        self.assertTrue(self.win.file_path)
        for action in self.win.assist_actions:
            self.assertFalse(action.isEnabled(), action.text())
            tooltip = action.toolTip()
            self.assertIn('pip install -e ".[ai]"', tooltip)
            self.assertIn('No model backend configured', tooltip)

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


class TestBackendConfiguredButUnavailable(AssistTestCase):
    """SETTING_MODEL_BACKEND names something real (e.g. 'yolo_onnx'), but
    build_backend still answers None -- missing extras, or extras present with
    a bad/missing SETTING_MODEL_PATH. This is a DIFFERENT cause than a fresh
    install with nothing configured (TestNoBackend), and must not be reported
    with the same "nothing configured" hint: that would be actively misleading
    to a user who already picked a backend."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))

        # Pre-seed the settings pickle so the real Settings().load() call inside
        # MainWindow.__init__ sees an explicit backend choice, exactly like a
        # user who configured one -- rather than mocking backend_name directly.
        settings_path = os.path.join(self.tmp.name, '.labelImgSettings.pkl')
        with open(settings_path, 'wb') as handle:
            pickle.dump({SETTING_MODEL_BACKEND: 'yolo_onnx'}, handle,
                       pickle.HIGHEST_PROTOCOL)

        self.win = None
        with mock.patch('libs.assist.controller.build_backend', return_value=None):
            with mock.patch('os.path.expanduser', return_value=self.tmp.name):
                self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def test_hint_names_the_configured_backend_not_a_missing_config(self):
        self.assertEqual('yolo_onnx', self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        for action in self.win.assist_actions:
            self.assertFalse(action.isEnabled(), action.text())
            tooltip = action.toolTip()
            self.assertIn('pip install -e ".[ai]"', tooltip)
            self.assertIn('yolo_onnx', tooltip, 'hint must name the backend that failed')
            self.assertNotIn('No model backend configured', tooltip,
                             'a configured-but-broken backend is not the same as '
                             'nothing being configured')


class TestLegacyPersistedStubIsTreatedAsUnset(unittest.TestCase):
    """REGRESSION: a settings pickle with SETTING_MODEL_BACKEND == 'stub' --
    exactly what an earlier build of this branch (when DEFAULT_BACKEND was
    still 'stub', before the fix in libs/inference/registry.py) would have
    left behind the first time a user ran labelImg and closed it even once
    -- must NOT build a StubBackend on the next launch. There is no
    settings-picker UI and no doc ever tells a user to write 'stub' by hand
    (see AssistController._LEGACY_IMPLICIT_DEFAULT_BACKEND), so a persisted
    'stub' can only be that old implicit default leaking through; treating
    it as a deliberate choice would keep resurrecting fabricated
    (image-dimension-derived) detections for exactly the users the
    DEFAULT_BACKEND=None fix was meant to protect.

    Unmocked, like TestDefaultConstructionHasNoBackend: this drives the real
    AssistController.__init__ -> build_backend -> registry path.
    StubBackend itself needs no optional dependency, so nothing stops it
    from actually being built here if the read-time guard regresses -- a
    mocked build_backend could not tell that apart from the fix working."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))

        # Pre-seed the settings pickle exactly the way an earlier run of
        # this branch would have left it: SETTING_MODEL_BACKEND == 'stub',
        # written by the old unconditional closeEvent persist while
        # DEFAULT_BACKEND was still 'stub' (mirrors
        # TestBackendConfiguredButUnavailable's real-Settings().load() idiom
        # above, rather than mocking backend_name directly).
        settings_path = os.path.join(self.tmp.name, '.labelImgSettings.pkl')
        with open(settings_path, 'wb') as handle:
            pickle.dump({SETTING_MODEL_BACKEND: 'stub'}, handle,
                       pickle.HIGHEST_PROTOCOL)

        self.win = None
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def tearDown(self):
        self.win.dirty = False
        self.win.close()
        self.tmp.cleanup()

    def test_persisted_stub_does_not_build_a_backend(self):
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        self.assertIsNone(self.win.inference_service.backend())

    def test_persisted_stub_leaves_ai_actions_disabled_with_the_fresh_install_hint(self):
        # Same hint as a genuinely fresh install (TestNoBackend), not the
        # "configured but broken" one (TestBackendConfiguredButUnavailable)
        # -- a treated-as-unset backend must read exactly like nothing was
        # ever configured, not like a real backend that failed to build.
        for action in self.win.assist_actions:
            self.assertFalse(action.isEnabled(), action.text())
            tooltip = action.toolTip()
            self.assertIn('No model backend configured', tooltip)


class TestUnconfiguredBackendIsNotPersistedOnClose(unittest.TestCase):
    """REGRESSION: MainWindow.closeEvent used to write
    ``settings[SETTING_MODEL_BACKEND] = self.assist.backend_name``
    unconditionally on every save/close -- so a fresh install, whose
    backend_name is None only because nothing was ever configured, still
    got an explicit SETTING_MODEL_BACKEND entry written to the pickle on
    every close. That unconditional write is exactly how DEFAULT_BACKEND
    being 'stub' (before the fix in libs/inference/registry.py) turned into
    a STICKY, explicit 'stub' setting the very first time anyone ran an
    earlier build of this branch and closed it even once.

    This drives the actual persist-on-close path (closeEvent, via the real
    QWidget.close()) end to end -- launch a fresh install, close it, and
    inspect what actually landed in the settings pickle -- rather than
    asserting anything about closeEvent's behaviour from memory. Deliberately
    does NOT override ``win.settings.path`` the way the other test cases in
    this module do: the whole point here is the same file round-tripping
    through load -> save -> load, exactly like a real user's
    ``~/.labelImgSettings.pkl``."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))
        self.settings_path = os.path.join(self.tmp.name, '.labelImgSettings.pkl')

    def tearDown(self):
        self.tmp.cleanup()

    def _launch(self):
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            app, win = get_main_app([sys.argv[0], self.dir])
        win.error_message = lambda title, msg: None
        return app, win

    def test_closing_a_fresh_install_does_not_write_a_backend_name(self):
        app, win = self._launch()
        self.assertIsNone(win.assist.backend_name)  # fresh-install precondition

        win.dirty = False
        win.close()  # runs closeEvent -- the actual persist-on-close path

        self.assertTrue(os.path.isfile(self.settings_path),
                        'closeEvent did not save a settings pickle at all')
        with open(self.settings_path, 'rb') as handle:
            saved = pickle.load(handle)
        self.assertNotIn(SETTING_MODEL_BACKEND, saved,
                         'closeEvent persisted a backend name nobody configured')

    def test_relaunch_after_that_close_still_has_no_backend(self):
        # The full round trip named in the regression: launch -> close ->
        # relaunch must still come up with AI disabled, not resurrect a
        # backend from whatever closeEvent just wrote to the pickle.
        app, win = self._launch()
        win.dirty = False
        win.close()

        app2, win2 = self._launch()
        try:
            self.assertIsNone(win2.assist.backend_name)
            self.assertFalse(win2.assist.is_available())
        finally:
            win2.dirty = False
            win2.close()


class TestNoStaleUpstreamPyPIInstruction(unittest.TestCase):
    """Regression guard: this fork is not published to PyPI, so the bare
    upstream form of the install command (the exact string in
    BAD_INSTRUCTION below) silently fetches the unrelated upstream
    HumanSignal package (which has none of this fork's AI code) instead of
    installing this fork's extras. Every user-facing runtime string in the AI
    seam (the disabled-action tooltips, the registry's log line, and the
    backend's MissingDependency messages) must instead point at installing
    from this checkout -- never at the bare package name.

    The scan below covers the whole repo surface the wrong instruction could
    plausibly reappear on (libs/, tests/, setup.py) -- not just the AI seam
    modules -- because it has previously reappeared OUTSIDE that seam too:
    as a `unittest.skipUnless` reason string in tests/test_assist.py and
    tests/test_yolo_onnx.py, and as a comment in setup.py.

    ALLOWLIST_SELF_REFERENCE (below) is the only exemption, and it is
    narrow and explicit on purpose: the BAD_INSTRUCTION assignment line
    itself necessarily spells out the exact banned string -- that IS what
    makes it the source of truth this whole guard checks against -- but it
    is not an install instruction a user could ever copy-paste and run.
    Every OTHER occurrence found so far has been a genuine bug (a real
    instruction someone could execute), not a documented "do NOT run this"
    anti-pattern example, so nothing else is allowlisted. If a future
    occurrence is a legitimate warned-against example, extend
    ALLOWLIST_SELF_REFERENCE with an equally explicit, visibly-commented
    entry then -- do not broaden it preemptively."""

    BAD_INSTRUCTION = 'pip install labelImg[ai]'

    REPO_ROOT = os.path.abspath(os.path.join(dir_name, '..'))

    SCAN_GLOBS = ('libs/**/*.py', 'tests/**/*.py', 'setup.py')

    # (relative/path.py, exact stripped line text) pairs exempted from the
    # repo-wide scan below -- see ALLOWLIST_SELF_REFERENCE note in the class
    # docstring. Matched on the FULL stripped line, not a substring, so it
    # cannot accidentally swallow a real, different offending line that
    # merely happens to share a file with this one.
    # Built via %r (not retyped as a literal) so THIS line does not itself
    # recreate the exact banned string in source and require its own entry.
    ALLOWLIST_SELF_REFERENCE = frozenset([
        ('tests/test_assist.py', 'BAD_INSTRUCTION = %r' % BAD_INSTRUCTION),
    ])

    def _repo_py_files(self):
        paths = []
        for pattern in self.SCAN_GLOBS:
            paths.extend(glob.glob(os.path.join(self.REPO_ROOT, pattern), recursive=True))
        return sorted(set(paths))

    def test_hint_constants_do_not_name_the_wrong_package(self):
        self.assertNotIn(self.BAD_INSTRUCTION,
                         assist_controller.NO_BACKEND_CONFIGURED_HINT)
        self.assertNotIn(self.BAD_INSTRUCTION,
                         assist_controller.BACKEND_UNAVAILABLE_HINT)

    def test_no_source_in_the_ai_seam_names_the_wrong_package(self):
        # Belt-and-suspenders over the constants check above: scan the full
        # source (including log lines, exception messages, and docstrings) of
        # every module in the "AI disabled" reporting path, so a new call site
        # cannot reintroduce the bare PyPI form even outside the two hints.
        modules = (assist_controller, inference_registry, inference_backend,
                   inference_yolo_onnx)
        for module in modules:
            source = inspect.getsource(module)
            self.assertNotIn(self.BAD_INSTRUCTION, source,
                             '%s still tells the user to run %r' %
                             (module.__name__, self.BAD_INSTRUCTION))

    def test_no_file_in_the_repo_names_the_wrong_package(self):
        # Wider than the AI-seam-only check above: the bad instruction has
        # reappeared as a test skip-reason string and a setup.py comment,
        # neither of which is reachable via inspect.getsource() on the AI
        # seam modules. Scans raw text across libs/**, tests/**, and
        # setup.py so it also catches non-code prose (comments, docstrings,
        # string literals) anywhere in that surface.
        failures = []
        for path in self._repo_py_files():
            rel = os.path.relpath(path, self.REPO_ROOT).replace(os.sep, '/')
            with open(path, 'r', encoding='utf-8') as handle:
                for lineno, line in enumerate(handle, start=1):
                    if self.BAD_INSTRUCTION not in line:
                        continue
                    if (rel, line.strip()) in self.ALLOWLIST_SELF_REFERENCE:
                        continue
                    failures.append('%s:%d: %s' % (rel, lineno, line.strip()))
        if failures:
            self.fail(
                '%d file(s) still tell the user to run %r (this fork is not '
                'on PyPI under this name, so that command would silently '
                'fetch the unrelated upstream package):\n%s'
                % (len(failures), self.BAD_INSTRUCTION, '\n'.join(failures)))


class TestToolTipRestoredWhenBackendBecomesAvailable(AssistTestCase):
    """Regression: refresh_actions used to overwrite an action's toolTip and
    statusTip with the disabled-state hint while the backend was unavailable,
    but never cleared either once the backend became available again -- the
    available branch fell back to `action.statusTip()`, which was the very
    hint this same method had just stamped over the original tip. An ENABLED
    action could therefore keep showing "No model backend configured" forever.

    AssistTestCase.launch() already drives exactly the transition that
    triggers this: a fresh install has no default backend (DEFAULT_BACKEND is
    None), so create_actions()'s first refresh_actions() stamps the hint over
    every action; launch() then injects a real backend via set_backend(),
    which must restore each action's own tip -- not leave the hint sitting in
    statusTip() -- once refresh_actions runs again.
    """

    def test_status_tip_and_tooltip_are_clean_once_available(self):
        self.assertTrue(self.win.assist.is_available())

        expected_tips = (
            (self.win.assist.action_auto,
             'Run the model on this image and show its boxes as suggestions'),
            (self.win.assist.action_accept,
             'Turn every suggestion on this image into a real box'),
            (self.win.assist.action_reject,
             'Discard every suggestion on this image'),
        )
        for action, expected_tip in expected_tips:
            self.assertNotIn('No model backend configured', action.statusTip())
            self.assertNotIn('No model backend configured', action.toolTip())
            self.assertEqual(expected_tip, action.statusTip())
            self.assertEqual(expected_tip, action.toolTip())

        # The threshold action never had a tip of its own (it is built by hand
        # in _create_threshold_action, not through new_action's `tip` kwarg):
        # once available it must fall back to its text, not to a leftover hint.
        threshold = self.win.assist.action_threshold
        self.assertEqual('', threshold.statusTip())
        self.assertNotIn('No model backend configured', threshold.toolTip())
        self.assertEqual('Confidence Threshold', threshold.toolTip())

    def test_hint_reappears_if_the_backend_is_dropped_again(self):
        # The other direction: available -> unavailable must still show the
        # hint (this half already worked; guard it so the fix does not flip
        # the bug instead of removing it).
        self.win.assist.set_backend(None)

        self.assertFalse(self.win.assist.is_available())
        for action in self.win.assist_actions:
            self.assertIn('No model backend configured', action.statusTip())
            self.assertIn('No model backend configured', action.toolTip())


class TestDefaultConstructionHasNoBackend(unittest.TestCase):
    """REGRESSION, end-to-end and unmocked: a fresh install (no settings file,
    SETTING_MODEL_BACKEND never set) must come up with AI disabled.

    Unlike TestNoBackend (which patches libs.assist.controller.build_backend to
    return None directly) and the AssistTestCase idiom (whose launch() always
    calls self.win.assist.set_backend(...) right after construction, overriding
    whatever the constructor built), this test does neither: it drives the real
    AssistController.__init__ -> build_backend -> registry path with nothing
    stubbed out. On a machine with numpy+onnxruntime installed (this one), the
    only thing that can make is_available() False here is DEFAULT_BACKEND being
    None -- if that regresses back to 'stub', this test builds a real
    StubBackend and fails.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))

        self.win = None
        # No mock.patch on build_backend anywhere in this setUp: the settings
        # pickle simply does not exist at the patched expanduser() target, so
        # Settings().load() leaves self.data == {} and SETTING_MODEL_BACKEND is
        # genuinely unset -- exactly a fresh install.
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def tearDown(self):
        self.win.dirty = False
        self.win.close()
        self.tmp.cleanup()

    def test_fresh_install_has_no_backend_and_ai_actions_disabled(self):
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        self.assertIsNone(self.win.inference_service.backend())
        for action in self.win.assist_actions:
            self.assertFalse(action.isEnabled(), action.text())


class TestOutOfBandRemoval(AssistTestCase):
    """A suggestion can leave the canvas WITHOUT the controller: "Delete RectBox"
    is the ordinary way to get rid of one box, and it goes nowhere near
    reject_all. The controller used to keep the dead Shape in its detection ->
    Shape map, and the next threshold change handed it back to
    canvas.delete_selected() -> list.remove(x) with x already gone: ValueError."""

    def delete_by_hand(self, shape):
        """Exactly what the Delete RectBox action does (MainWindow.actions.delete)."""
        self.win.canvas.select_shape(shape)
        self.win.delete_selected_shape()

    def test_delete_a_suggestion_then_raise_the_threshold(self):
        self.win.assist.set_threshold(0.0)
        self.win.assist.auto_label_image()  # person 0.9, face 0.8
        face = self.win.canvas.shapes[1]

        self.delete_by_hand(face)

        # Pushes `face` below the bar: the controller tries to remove a shape the
        # canvas no longer has. This raised ValueError before the fix.
        self.win.assist.set_threshold(0.85)

        self.assertEqual(['person'], [s.label for s in self.win.canvas.shapes])
        self.assertEqual(1, self.win.label_list.count())
        self.assertEqual(1, len(self.win.shapes_to_items))

    def test_a_hand_deleted_suggestion_is_not_resurrected(self):
        self.win.assist.set_threshold(0.0)
        self.win.assist.auto_label_image()
        self.delete_by_hand(self.win.canvas.shapes[1])  # face (0.8)

        # 0.7 still leaves face above the bar — but the user deleted it, so it
        # must not come back from the detections the controller is still holding.
        self.win.assist.set_threshold(0.7)

        self.assertEqual(['person'], [s.label for s in self.win.canvas.shapes])
        self.assertEqual(1, self.win.label_list.count())
        self.assertEqual(1, self.backend.calls, 'the threshold re-ran the model')

    def test_reject_all_after_a_hand_delete(self):
        self.win.assist.auto_label_image()
        self.delete_by_hand(self.win.canvas.shapes[1])

        self.assertEqual(1, self.win.assist.reject_all())  # the dead one is not iterated

        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())
        self.assertEqual({}, self.win.shapes_to_items)

    def test_accept_all_after_a_hand_delete(self):
        self.win.assist.auto_label_image()
        self.delete_by_hand(self.win.canvas.shapes[1])

        self.assertEqual(1, self.win.assist.accept_all())

        self.assertEqual(['person'], self.saved_labels())
        # And the threshold no longer has anything to filter — least of all a
        # shape that is gone.
        self.win.assist.set_threshold(1.0)
        self.assertEqual(1, len(self.win.canvas.shapes))

    def test_deleting_the_last_suggestion_disables_the_bulk_actions(self):
        self.win.assist.auto_label_image()
        self.assertTrue(self.win.assist.action_reject.isEnabled())

        for shape in list(self.win.canvas.shapes):
            self.delete_by_hand(shape)

        self.assertFalse(self.win.assist.action_accept.isEnabled())
        self.assertFalse(self.win.assist.action_reject.isEnabled())


class TestDuplicatedSuggestion(AssistTestCase):
    """Ctrl+D on a suggestion: Shape.copy() faithfully clones `provisional`, so
    the duplicate is a suggestion too — but it has no detection behind it. It was
    tracked nowhere, which made it unreachable for Accept All / Reject All: a
    dashed box that could not be saved and could not be resolved."""

    def duplicate(self, shape):
        self.win.canvas.select_shape(shape)
        self.win.copy_selected_shape()  # Ctrl+D
        return self.win.canvas.shapes[-1]

    def test_the_duplicate_is_a_tracked_suggestion(self):
        self.win.assist.auto_label_image()
        clone = self.duplicate(self.win.canvas.shapes[0])

        self.assertTrue(clone.provisional)
        self.assertIn(clone, self.win.assist.provisional_shapes())
        self.assertEqual([], self.saved_labels(), 'a suggestion is not data yet')

    def test_reject_all_removes_a_duplicated_suggestion(self):
        self.win.assist.auto_label_image()
        self.duplicate(self.win.canvas.shapes[0])
        self.assertEqual(3, len(self.win.canvas.shapes))

        self.assertEqual(3, self.win.assist.reject_all())  # was 2: the clone was orphaned

        self.assertEqual([], self.win.canvas.shapes, 'the duplicate outlived Reject All')
        self.assertEqual(0, self.win.label_list.count())
        self.assertEqual({}, self.win.shapes_to_items)

    def test_accept_all_commits_a_duplicated_suggestion(self):
        self.win.assist.auto_label_image()
        self.duplicate(self.win.canvas.shapes[0])

        self.assertEqual(3, self.win.assist.accept_all())

        self.assertTrue(all(not s.provisional for s in self.win.canvas.shapes))
        self.assertEqual(['person', 'face', 'person'], self.saved_labels())


class TestImportCocoOntoALabeledImage(AssistTestCase):
    """File > Import COCO... loads annotations without going through load_file(),
    so nothing reset the label state first: load_labels appended its items while
    canvas.load_shapes replaced the canvas wholesale. The list then held items
    whose shapes were not on the canvas — selecting or deleting one is
    canvas.delete_selected() -> list.remove(x) -> ValueError, and a save silently
    dropped them."""

    def write_dataset(self, label='cat', box=(10, 10, 20, 20)):
        from libs.coco_io import COCOWriter
        dataset = os.path.join(self.dir, 'annotations.json')
        writer = COCOWriter('photos', 'a.png', (IMAGE_SIZE, IMAGE_SIZE, 3),
                            local_img_path=self.path('a.png'))
        writer.add_bnd_box(box[0], box[1], box[2], box[3], label, 0)
        writer.save(class_list=[label], target_file=dataset)
        return dataset

    def import_coco(self, dataset):
        fake_dialog = mock.Mock()
        fake_dialog.getOpenFileName.return_value = (dataset, '')
        with mock.patch('labelImg.QFileDialog', fake_dialog):
            self.win.import_coco_dialog()

    def test_import_replaces_the_label_state_instead_of_appending(self):
        dataset = self.write_dataset()
        self.add_real_box('dog')  # the image already has a label

        self.import_coco(dataset)

        self.assertEqual(['cat'], [s.label for s in self.win.canvas.shapes])
        self.assertEqual(1, self.win.label_list.count())
        self.assertEqual('cat', self.win.label_list.item(0).text())
        self.assertEqual(1, len(self.win.shapes_to_items))
        self.assertEqual(1, len(self.win.items_to_shapes))
        # the two maps agree, and every item's shape is really on the canvas
        for shape, item in self.win.shapes_to_items.items():
            self.assertIs(shape, self.win.items_to_shapes[item])
            self.assertIn(shape, self.win.canvas.shapes)

    def test_deleting_an_item_after_an_import_does_not_crash(self):
        dataset = self.write_dataset()
        self.add_real_box('dog')

        self.import_coco(dataset)

        # The crash repro: walk the label list and delete each item's shape.
        for index in range(self.win.label_list.count()):
            shape = self.win.items_to_shapes[self.win.label_list.item(index)]
            self.win.canvas.select_shape(shape)
            self.win.delete_selected_shape()  # ValueError on a stale item

        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())

    def test_import_over_live_suggestions_leaves_no_stale_tracking(self):
        dataset = self.write_dataset()
        self.win.assist.auto_label_image()  # suggestions on the canvas...

        self.import_coco(dataset)  # ...and load_labels replaces the canvas

        self.assertEqual(['cat'], [s.label for s in self.win.canvas.shapes])
        # The controller must have let go of the shapes that canvas no longer has.
        self.assertEqual([], self.win.assist.provisional_shapes())
        self.win.assist.set_threshold(0.95)  # used to delete a shape that is gone
        self.assertEqual(['cat'], [s.label for s in self.win.canvas.shapes])


class TestImageCarrier(unittest.TestCase):
    """numpy is OPTIONAL: the base install has neither numpy nor onnxruntime, so
    the UI-thread conversion has to produce something a backend can read either
    way — and the worker must never be handed a QImage."""

    def make_image(self, width, height):
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0xffff0000)  # opaque red
        return image

    @unittest.skipUnless(HAS_NUMPY, 'needs numpy (pip install -e ".[ai]")')
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
