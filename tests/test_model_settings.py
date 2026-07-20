#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Tests for the in-app "Model Settings..." dialog (libs/assist/settings_dialog.py)
and the logic behind it, AssistController.apply_model_settings
(libs/assist/controller.py).

SAFETY: this suite never calls ``exec_()`` on ``ModelSettingsDialog`` and never
lets a real ``MainWindow.error_message`` reach the screen -- every test drives
``apply_model_settings`` directly (a plain method, no Qt event loop involved)
the way the dialog's OK button does, and ``error_message`` is always stubbed
to record calls instead of showing a ``QMessageBox``. This mirrors
``tests/test_assist.py``'s ``AssistTestCase`` idiom exactly.

Runs on BOTH the base install (pyqt5+lxml only) and the full ``[ai]`` install:
path-validation and "no backend" tests need neither numpy nor onnxruntime and
always run; the one test that builds a real backend
(``TestApplySucceedsWithARealModel``) is skip-guarded like
``tests/test_yolo_onnx.py``'s ``TestRealOnnxModel``; the runtime-missing test
simulates an absent onnxruntime via ``sys.modules`` patching, so it exercises
the SAME code path regardless of what is actually installed here.
"""

import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

dir_name = os.path.abspath(os.path.dirname(__file__))
if dir_name not in sys.path:
    sys.path.insert(0, dir_name)
REPO_ROOT = os.path.abspath(os.path.join(dir_name, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt5.QtWidgets import QApplication

# QApplication is a process-wide singleton; create it once at import so every
# test module (and get_main_app, which reuses an existing instance) shares it.
APP = QApplication.instance() or QApplication([])

from PyQt5.QtGui import QImage

from labelImg import get_main_app
from libs.assist.controller import (AVAILABLE_UI_BACKENDS,
                                    RUNTIME_MISSING_HINT, ModelSettingsError)
from libs.assist.settings_dialog import ModelSettingsDialog
from libs.constants import SETTING_MODEL_BACKEND, SETTING_MODEL_PATH
from libs.inference.backend import ModelBackend
from libs.inference.service import SynchronousExecutor
from libs.inference.stub import StubBackend
from test_active_learning import _deferred_executor
from test_yolo_onnx import build_onnx_model, v8_tensor

IMAGE_SIZE = 64


def _importable(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


HAS_NUMPY = _importable('numpy')
HAS_ONNXRUNTIME = _importable('onnxruntime')


class _RaisingBackend(ModelBackend):
    """predict() always raises -- used to force InferenceService's job()
    closure down its predictionFailed path (rather than predictionReady's)
    under a deferred executor, so a test can exercise
    AssistController.on_prediction_failed's interactive generation guard
    specifically."""

    name = 'raising'
    supports_detection = True
    supports_segmentation = False

    def predict(self, image):
        raise RuntimeError('synthetic failure for the interactive '
                           'generation-guard regression test')


class ModelSettingsTestCase(unittest.TestCase):
    """A fresh-install MainWindow: no settings pickle, no backend configured,
    one image open (so the availability-gated AI actions have something to
    turn on). Mirrors tests/test_assist.py's AssistTestCase/
    TestDefaultConstructionHasNoBackend idiom -- real MainWindow, headless,
    error_message stubbed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))

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

    # -- helpers -------------------------------------------------------

    def persisted(self):
        """Read back exactly what settings.save() wrote to disk -- proof the
        choice was persisted IMMEDIATELY (not left for closeEvent)."""
        with open(self.win.settings.path, 'rb') as handle:
            return pickle.load(handle)

    def gated_actions(self):
        """Every AI action EXCEPT action_model_settings -- the one action
        that must stay enabled/disabled independently of backend
        availability (see AssistController.create_actions)."""
        return [a for a in self.win.assist_actions
               if a is not self.win.assist.action_model_settings]


class TestModelSettingsMenuAction(ModelSettingsTestCase):
    """The menu action itself: always present, always enabled, never gated
    the way the other AI actions are."""

    def test_action_model_settings_is_first_and_always_enabled(self):
        self.assertIs(self.win.assist.action_model_settings, self.win.assist_actions[0])
        self.assertTrue(self.win.assist.action_model_settings.isEnabled())
        self.assertFalse(self.win.assist.is_available())  # fresh install: nothing configured


class TestStubIsNotExposed(ModelSettingsTestCase):
    """HARD DESIGN CONSTRAINT: 'stub' must never be a selectable UI choice --
    see AVAILABLE_UI_BACKENDS's docstring and
    AssistController._LEGACY_IMPLICIT_DEFAULT_BACKEND. Offering it here would
    let a user pick a setting that the very next launch silently treats as
    unset."""

    def test_stub_is_not_in_the_available_ui_backends(self):
        self.assertNotIn('stub', AVAILABLE_UI_BACKENDS)

    def test_stub_is_not_a_dialog_choice(self):
        offered = [backend_name for _label, backend_name in
                  ModelSettingsDialog.BACKEND_CHOICES]
        self.assertNotIn('stub', offered)

    def test_apply_model_settings_refuses_stub_explicitly(self):
        # Defensive: even if something bypassed the dialog and called the
        # controller method directly with 'stub', it must still be refused,
        # not quietly build a StubBackend.
        with self.assertRaises(ModelSettingsError):
            self.win.assist.apply_model_settings('stub', '/does/not/matter.onnx')
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())

    def test_a_legacy_persisted_stub_is_still_treated_as_unset(self):
        # REGRESSION GUARD (post-dialog): the read-time guard in
        # AssistController.__init__ (_LEGACY_IMPLICIT_DEFAULT_BACKEND) must
        # keep working after this change -- a pickle with an explicit
        # SETTING_MODEL_BACKEND == 'stub' (what an old build's unconditional
        # closeEvent write could have left behind) is still read back as
        # unset, not as a deliberate choice. Full end-to-end coverage of this
        # already exists in tests/test_assist.py
        # (TestLegacyPersistedStubIsTreatedAsUnset); this is a light,
        # same-suite confirmation that adding the dialog did not disturb it.
        with open(self.win.settings.path, 'wb') as handle:
            pickle.dump({SETTING_MODEL_BACKEND: 'stub'}, handle, pickle.HIGHEST_PROTOCOL)

        tmp2 = tempfile.TemporaryDirectory()
        try:
            dir2 = os.path.join(tmp2.name, 'photos')
            os.makedirs(dir2)
            img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
            img.fill(0xffffffff)
            img.save(os.path.join(dir2, 'a.png'))
            settings_path = os.path.join(self.tmp.name, '.labelImgSettings.pkl')
            with open(settings_path, 'wb') as handle:
                pickle.dump({SETTING_MODEL_BACKEND: 'stub'}, handle, pickle.HIGHEST_PROTOCOL)

            with mock.patch('os.path.expanduser', return_value=self.tmp.name):
                app2, win2 = get_main_app([sys.argv[0], dir2])
            try:
                self.assertIsNone(win2.assist.backend_name)
                self.assertFalse(win2.assist.is_available())
            finally:
                win2.dirty = False
                win2.close()
        finally:
            tmp2.cleanup()


class TestApplyNoBackend(ModelSettingsTestCase):
    """Selecting "사용 안 함" (backend_name=None)."""

    def setUp(self):
        super(TestApplyNoBackend, self).setUp()
        # Start from a CONFIGURED-and-available state, so turning it off is
        # an observable transition rather than a no-op.
        self.win.assist.backend_name = 'yolo_onnx'
        self.win.assist.model_path = '/previously/configured/model.onnx'
        self.win.assist.set_backend(StubBackend())
        self.assertTrue(self.win.assist.is_available())

    def test_none_clears_the_backend_and_disables_ai_actions(self):
        self.win.assist.apply_model_settings(None, '')

        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        for action in self.gated_actions():
            self.assertFalse(action.isEnabled(), action.text())
        self.assertEqual([], self.errors, 'turning AI off must never show an error dialog')

    def test_none_removes_the_persisted_backend_key(self):
        self.win.assist.apply_model_settings(None, '')

        self.assertNotIn(SETTING_MODEL_BACKEND, self.win.settings.data)
        # Persisted to DISK immediately -- not deferred to closeEvent.
        self.assertNotIn(SETTING_MODEL_BACKEND, self.persisted())

    def test_none_also_removes_the_now_stale_persisted_model_path_key(self):
        # SETTING_MODEL_PATH means nothing without SETTING_MODEL_BACKEND to
        # pair it with (functionally inert), but leaving it behind is
        # untidy and could confuse anything reading the pickle directly --
        # both keys are dropped together.
        self.win.assist.apply_model_settings(None, '')

        self.assertNotIn(SETTING_MODEL_PATH, self.win.settings.data)
        self.assertNotIn(SETTING_MODEL_PATH, self.persisted())


class TestBadPath(ModelSettingsTestCase):
    """Validation failures that are NOT "runtime missing" -- these must run
    identically regardless of whether onnxruntime is installed here, because
    the path is checked BEFORE the runtime is ever probed."""

    def test_empty_path_is_rejected(self):
        with self.assertRaises(ModelSettingsError):
            self.win.assist.apply_model_settings('yolo_onnx', '')
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())

    def test_missing_file_is_rejected(self):
        missing = os.path.join(self.tmp.name, 'does-not-exist.onnx')
        with self.assertRaises(ModelSettingsError) as ctx:
            self.win.assist.apply_model_settings('yolo_onnx', missing)
        self.assertIn('not found', str(ctx.exception).lower())
        self.assertIsNone(self.win.assist.backend_name)

    def test_non_onnx_extension_is_rejected(self):
        not_onnx = os.path.join(self.tmp.name, 'model.txt')
        with open(not_onnx, 'w') as handle:
            handle.write('not a model')
        with self.assertRaises(ModelSettingsError) as ctx:
            self.win.assist.apply_model_settings('yolo_onnx', not_onnx)
        self.assertIn('.onnx', str(ctx.exception))
        self.assertIsNone(self.win.assist.backend_name)

    def test_bad_path_does_not_enable_ai_actions_or_persist_anything(self):
        missing = os.path.join(self.tmp.name, 'nope.onnx')
        with self.assertRaises(ModelSettingsError):
            self.win.assist.apply_model_settings('yolo_onnx', missing)

        self.assertFalse(self.win.assist.is_available())
        for action in self.gated_actions():
            self.assertFalse(action.isEnabled(), action.text())
        self.assertNotIn(SETTING_MODEL_BACKEND, self.win.settings.data)
        self.assertNotIn(SETTING_MODEL_PATH, self.win.settings.data)
        # Never shows a modal itself -- the DIALOG decides how to surface
        # ModelSettingsError, apply_model_settings only raises.
        self.assertEqual([], self.errors)


class TestRuntimeMissing(ModelSettingsTestCase):
    """A VALID path, but onnxruntime cannot be imported -- must yield the
    runtime-specific message, never the "bad path" one. Simulated via
    sys.modules patching (the same technique
    tests/test_yolo_onnx.py::TestRegistryDegradation and
    tests/test_assist.py already use for "no onnxruntime here"), so this
    exercises the same branch on every machine regardless of what is
    actually installed."""

    def test_runtime_missing_yields_the_runtime_hint_not_a_bad_path_message(self):
        with tempfile.NamedTemporaryFile(suffix='.onnx', dir=self.tmp.name, delete=False) as handle:
            handle.write(b'not really onnx, but the path exists with the right extension')
            model_path = handle.name

        with mock.patch.dict(sys.modules, {'onnxruntime': None}):
            with self.assertRaises(ModelSettingsError) as ctx:
                self.win.assist.apply_model_settings('yolo_onnx', model_path)

        message = str(ctx.exception)
        self.assertIn('onnxruntime', message)
        self.assertIn('pip install -e ".[ai]"', message)
        self.assertNotIn('not found', message.lower())
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        self.assertNotIn(SETTING_MODEL_BACKEND, self.win.settings.data)

    def test_runtime_missing_hint_names_every_missing_module(self):
        # RUNTIME_MISSING_HINT is a %-template; a sanity check that it takes
        # exactly the two placeholders apply_model_settings feeds it (module
        # list, is/are) without raising.
        self.assertIn('%s', RUNTIME_MISSING_HINT)


@unittest.skipUnless(HAS_NUMPY and HAS_ONNXRUNTIME,
                     'needs numpy + onnxruntime (pip install -e ".[ai]")')
class TestApplySucceedsWithARealModel(ModelSettingsTestCase):
    """The full success path: a real (tiny, hand-serialised) ONNX model --
    reusing tests/test_yolo_onnx.py's build_onnx_model/v8_tensor helpers, the
    same tiny fixture TestRealOnnxModel uses -- applied through
    apply_model_settings builds a REAL backend, enables the AI actions with
    NO restart, and persists both settings keys immediately."""

    # A 64x64 network, 2 classes, 20 anchors -> output (1, 6, 20), v8 layout
    # -- identical fixture shape to tests/test_yolo_onnx.py::TestRealOnnxModel.
    INPUT_SHAPE = (1, 3, 64, 64)
    OUTPUT_DIMS = (1, 6, 20)

    def write_model(self):
        rows = v8_tensor({0: (32, 32, 16, 8, (0.9, 0.1))}, 2, 20)[0]
        flat = [value for row in rows for value in row]
        path = os.path.join(self.tmp.name, 'tiny_yolo.onnx')
        with open(path, 'wb') as handle:
            handle.write(build_onnx_model(
                self.INPUT_SHAPE, self.OUTPUT_DIMS, flat, {'names': "{0: 'cat', 1: 'dog'}"}))
        return path

    def test_apply_builds_a_real_backend_and_enables_ai_actions_live(self):
        model_path = self.write_model()
        self.assertFalse(self.win.assist.is_available())
        self.assertFalse(self.win.assist.action_auto.isEnabled())

        self.win.assist.apply_model_settings('yolo_onnx', model_path)

        self.assertEqual('yolo_onnx', self.win.assist.backend_name)
        self.assertEqual(model_path, self.win.assist.model_path)
        self.assertTrue(self.win.assist.is_available())
        # Live rebuild, no restart: the SAME MainWindow/AssistController
        # instance now has the AI actions enabled.
        self.assertTrue(self.win.assist.action_auto.isEnabled())
        self.assertTrue(self.win.assist.action_accept.isEnabled() is False)  # no suggestions yet
        backend = self.win.inference_service.backend()
        self.assertIsNotNone(backend)
        self.assertEqual('yolo_onnx', backend.name)
        self.assertEqual(['cat', 'dog'], backend.class_names)

    def test_apply_persists_both_keys_immediately(self):
        model_path = self.write_model()
        self.win.assist.apply_model_settings('yolo_onnx', model_path)

        self.assertEqual('yolo_onnx', self.win.settings.data.get(SETTING_MODEL_BACKEND))
        self.assertEqual(model_path, self.win.settings.data.get(SETTING_MODEL_PATH))
        # On DISK -- not deferred to closeEvent -- so the choice survives a
        # crash. Read back with a plain pickle.load, independent of Settings.
        on_disk = self.persisted()
        self.assertEqual('yolo_onnx', on_disk.get(SETTING_MODEL_BACKEND))
        self.assertEqual(model_path, on_disk.get(SETTING_MODEL_PATH))

    def test_a_corrupt_onnx_file_is_rejected_as_construction_failure(self):
        bad_path = os.path.join(self.tmp.name, 'corrupt.onnx')
        with open(bad_path, 'wb') as handle:
            handle.write(b'\x00\x01\x02 this is not a valid onnx protobuf')

        with self.assertRaises(ModelSettingsError) as ctx:
            self.win.assist.apply_model_settings('yolo_onnx', bad_path)

        self.assertNotIn('onnxruntime', str(ctx.exception))  # not the runtime-missing message
        self.assertIsNone(self.win.assist.backend_name)
        self.assertFalse(self.win.assist.is_available())
        self.assertNotIn(SETTING_MODEL_BACKEND, self.win.settings.data)


class TestDialogIsAThinShell(ModelSettingsTestCase):
    """Construction-only checks on ModelSettingsDialog -- NEVER exec()'d or
    shown, so no modal event loop is ever entered here. Validates the
    "thin shell" contract: it collects input and reflects controller state,
    it does not itself decide anything."""

    def test_dialog_choices_exclude_stub_and_offer_exactly_two_options(self):
        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            labels = [label for label, _name in dialog.BACKEND_CHOICES]
            names = [name for _label, name in dialog.BACKEND_CHOICES]
            self.assertEqual(2, len(dialog.BACKEND_CHOICES))
            self.assertIn(None, names)
            self.assertIn('yolo_onnx', names)
            self.assertNotIn('stub', names)
            self.assertIn('사용 안 함', labels)
            self.assertIn('YOLO (ONNX)', labels)
        finally:
            dialog.deleteLater()

    def test_dialog_prefills_from_current_controller_state(self):
        self.win.assist.backend_name = 'yolo_onnx'
        self.win.assist.model_path = '/some/model.onnx'

        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            _label, selected = dialog.BACKEND_CHOICES[dialog.backend_combo.currentIndex()]
            self.assertEqual('yolo_onnx', selected)
            self.assertEqual('/some/model.onnx', dialog.path_edit.text())
        finally:
            dialog.deleteLater()

    def test_dialog_prefills_disabled_state_as_no_backend(self):
        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            _label, selected = dialog.BACKEND_CHOICES[dialog.backend_combo.currentIndex()]
            self.assertIsNone(selected)
            self.assertEqual('', dialog.path_edit.text())
        finally:
            dialog.deleteLater()


class TestBrowseHandlesThePyQt4BareStringReturn(ModelSettingsTestCase):
    """Cheap-but-confirmed fix: ModelSettingsDialog._browse's
    isinstance(path, (tuple, list)) guard must run BEFORE any tuple
    unpacking of QFileDialog.getOpenFileName's return value -- the previous
    shape (``path, _filter = QFileDialog.getOpenFileName(...)``) unpacked
    first, so the PyQt4 fallback (a bare string, not a (name, filter) tuple)
    would raise trying to unpack that string before the guard ever ran.

    QFileDialog.getOpenFileName is a MODAL file picker -- it is mocked here,
    never actually invoked, so this stays dialog-safe under offscreen."""

    def test_a_bare_string_return_does_not_raise_and_sets_the_path(self):
        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            with mock.patch('libs.assist.settings_dialog.QFileDialog.getOpenFileName',
                            return_value='/picked/model.onnx'):
                dialog._browse()  # must not raise
            self.assertEqual('/picked/model.onnx', dialog.path_edit.text())
        finally:
            dialog.deleteLater()

    def test_a_tuple_return_still_works(self):
        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            with mock.patch('libs.assist.settings_dialog.QFileDialog.getOpenFileName',
                            return_value=('/picked/model.onnx', 'ONNX Model (*.onnx)')):
                dialog._browse()
            self.assertEqual('/picked/model.onnx', dialog.path_edit.text())
        finally:
            dialog.deleteLater()

    def test_an_empty_bare_string_return_leaves_the_path_untouched(self):
        # Cancelling the (mocked) dialog: PyQt4 returns '' rather than a
        # (name, filter) tuple with an empty first element.
        dialog = ModelSettingsDialog(self.win.assist, parent=self.win)
        try:
            dialog.path_edit.setText('/already/there.onnx')
            with mock.patch('libs.assist.settings_dialog.QFileDialog.getOpenFileName',
                            return_value=''):
                dialog._browse()
            self.assertEqual('/already/there.onnx', dialog.path_edit.text())
        finally:
            dialog.deleteLater()


class TestBackendSwapClearsSuggestions(ModelSettingsTestCase):
    """P1a [BLOCKING]: swapping the backend must not leave a previous
    model's suggestions live on the canvas. AssistController.set_backend
    used to only call service.set_backend()+refresh_actions() -- never
    clear_suggestions() -- so model A's provisional boxes (and Accept/Reject
    All, which read has_suggestions straight off the canvas) survived a
    swap to model B: Accept All would then commit model A's boxes as if
    they were model B's output. See AssistController.set_backend."""

    def setUp(self):
        super(TestBackendSwapClearsSuggestions, self).setUp()
        self.win.inference_service.set_executor(SynchronousExecutor())
        self.win.assist.set_backend(StubBackend())
        self.win.assist.auto_label_image()
        self.assertEqual(2, len(self.win.canvas.shapes))  # StubBackend: 2 detections
        self.assertTrue(self.win.assist.action_accept.isEnabled())
        self.assertTrue(self.win.assist.action_reject.isEnabled())

    def test_swapping_backend_clears_stale_suggestions_from_the_canvas(self):
        self.win.assist.set_backend(StubBackend())  # model B

        self.assertEqual([], self.win.canvas.shapes)
        self.assertEqual(0, self.win.label_list.count())
        self.assertEqual([], self.win.assist._detections)

    def test_swapping_backend_disables_accept_and_reject(self):
        self.win.assist.set_backend(StubBackend())

        self.assertFalse(self.win.assist.action_accept.isEnabled())
        self.assertFalse(self.win.assist.action_reject.isEnabled())

    def test_swapping_backend_drops_dismissed_tracking_too(self):
        # Delete one suggestion by hand first (records it as dismissed) --
        # that bookkeeping is keyed by DETECTION INDEX, which means nothing
        # under a different model and must not survive the swap either, or
        # a later threshold move under the new model could silently drop a
        # box at an index that happened to collide with the old dismissal.
        shape = self.win.canvas.shapes[0]
        self.win.canvas.select_shape(shape)
        self.win.delete_selected_shape()
        self.win.canvas.de_select_shape()
        self.assertEqual({0}, self.win.assist._dismissed)

        self.win.assist.set_backend(StubBackend())

        self.assertEqual(set(), self.win.assist._dismissed)

    def test_accept_all_after_a_swap_has_nothing_stale_to_commit(self):
        self.win.assist.set_backend(StubBackend())

        accepted = self.win.assist.accept_all()

        self.assertEqual(0, accepted, 'nothing from the OLD model should have '
                         'been left for Accept All to commit')


class TestInteractiveResultDroppedAfterBackendSwap(ModelSettingsTestCase):
    """P1b [BLOCKING]: an INTERACTIVE (Ctrl+I) prediction dispatched under
    one backend must be DROPPED if it only resolves after the backend has
    since been swapped -- otherwise a slow model-A result would be silently
    accepted as model B's. _is_current alone cannot catch this: the image
    can easily still be the CURRENT one (the user did not navigate, they
    just reopened Model Settings). Mirrors the generation-tag discipline
    _batch_generation already applies to two batch runs -- see
    AssistController._interactive_generation."""

    def setUp(self):
        super(TestInteractiveResultDroppedAfterBackendSwap, self).setUp()
        self.executor, self.jobs = _deferred_executor()
        self.win.inference_service.set_executor(self.executor)
        self.win.assist.set_backend(StubBackend())

    def test_late_interactive_result_from_the_old_backend_is_dropped(self):
        self.win.assist.auto_label_image()
        self.assertEqual(1, len(self.jobs))

        # The backend swaps (exactly what apply_model_settings does) WHILE
        # that request is still outstanding.
        self.win.assist.set_backend(StubBackend())

        # The stale job (model A's answer) finally resolves.
        self.jobs.pop(0)()

        self.assertEqual([], self.win.canvas.shapes,
                         'a late result from a superseded backend generation '
                         'must not inject boxes onto the canvas')
        self.assertEqual([], self.win.assist._detections)
        self.assertFalse(self.win.assist.action_accept.isEnabled())

    def test_late_interactive_failure_from_the_old_backend_is_also_dropped(self):
        # Same generation guard, exercised through on_prediction_failed
        # instead of on_prediction_ready: model A's outstanding request
        # resolves as a FAILURE (predict() raises) only after the swap, and
        # must not be surfaced as if it were current.
        self.win.assist.set_backend(_RaisingBackend())
        self.win.assist.auto_label_image()
        self.assertEqual(1, len(self.jobs))

        self.win.assist.set_backend(StubBackend())  # swap away from model A

        statuses = []
        self.win.status = lambda message, delay=5000: statuses.append(message)

        self.jobs.pop(0)()  # model A's job finally runs; backend.predict() raises

        self.assertFalse(
            any('Model failed' in message for message in statuses),
            'a late FAILURE from a superseded backend generation must not be '
            'surfaced as if it were current: %r' % (statuses,))
        self.assertEqual([], self.errors)

    def test_a_fresh_request_dispatched_after_the_swap_still_resolves_normally(self):
        # The generation-tagging fix must not break the ordinary case: a
        # request dispatched AFTER the swap (nothing older outstanding)
        # still resolves.
        self.win.assist.set_backend(StubBackend())  # model B
        self.win.assist.auto_label_image()
        self.assertEqual(1, len(self.jobs))

        self.jobs.pop(0)()

        self.assertEqual(2, len(self.win.canvas.shapes))
        self.assertTrue(self.win.assist.action_accept.isEnabled())


class TestCancelStaysReachableWhenDisabledMidBatch(ModelSettingsTestCase):
    """P2 [BLOCKING], half (a): refresh_actions ANDed `available` over the
    WHOLE Score Folder condition, so choosing "사용 안 함" (disabling AI)
    while a batch is running made `available` False and disabled the ONLY
    control that can cancel a running batch. See refresh_actions' own
    comment: `batch_running` must be ORed over the whole expression, not
    just the has_folder half."""

    def setUp(self):
        super(TestCancelStaysReachableWhenDisabledMidBatch, self).setUp()
        self.executor, self.jobs = _deferred_executor()
        self.win.inference_service.set_executor(self.executor)
        self.win.assist.set_backend(StubBackend())

        self.win.assist.score_folder()
        self.assertTrue(self.win.assist._batch_active)
        self.assertTrue(self.win.assist.action_score_folder.isEnabled())

    def test_refresh_actions_gating_keeps_cancel_reachable_even_without_the_proactive_cancel(self):
        # Isolates refresh_actions' OWN gating fix from
        # apply_model_settings' separate "cancel proactively before
        # dropping the backend" hardening (which would otherwise cancel the
        # batch before this path is ever exercised -- a backend can become
        # unavailable mid-batch by other means too, per that method's
        # docstring, so the gating fix must hold on its own).
        with mock.patch.object(self.win.assist, 'cancel_batch_scoring', return_value=False):
            self.win.assist.apply_model_settings(None, '')

        self.assertTrue(self.win.assist._batch_active,
                        'batch should still be running -- cancel_batch_scoring was stubbed out')
        self.assertTrue(
            self.win.assist.action_score_folder.isEnabled(),
            'the only control that can cancel a running batch must stay reachable '
            'even after AI is disabled mid-scan')

    def test_apply_model_settings_disable_proactively_cancels_an_active_batch(self):
        # Half (a)'s "belt and braces" partner: apply_model_settings' own
        # disable branch cancels an active batch BEFORE dropping the
        # backend, so the batch never runs backend-less at all.
        self.win.assist.apply_model_settings(None, '')

        self.assertFalse(self.win.assist._batch_active)
        self.assertEqual([], self.errors)


class TestBackendLessBatchDoesNotRecurse(ModelSettingsTestCase):
    """P2 [BLOCKING], half (b): InferenceService.predict_async's own
    `self._backend is None` early return used to emit `predictionFailed`
    SYNCHRONOUSLY, in-process (a direct call, not a queued cross-thread
    signal) -- reaching on_prediction_failed -> _advance_batch(synchronous=
    True) -> _batch_step() recursing in the SAME call stack for every
    remaining image. Identical recursion class already fixed for
    _batch_step's own unreadable-file branch (see
    tests/test_active_learning.py's TestBatchLoadFailureDoesNotRecurse);
    this new trigger (a backend that disappears mid-scan, e.g. via Model
    Settings) never got that fix.

    SAFETY: m_img_list is set directly here, never via load_file (per the
    task's dialog-safety rule) -- and a single real, on-disk image is
    reused for every position, so this stays fast and avoids creating 700
    files on disk while still feeding a genuinely READABLE image into
    _load_model_image (the point is exercising predict_async's
    backend-None branch, not the already-fixed unreadable-file branch)."""

    N = 700

    def _drain_deferred_batch(self, max_iterations):
        """Pump the Qt event loop until the batch finishes or
        `max_iterations` is exhausted -- the deferred QTimer.singleShot(0, ...)
        steps only fire when the event loop is actually pumped."""
        for _ in range(max_iterations):
            if not self.win.assist._batch_active:
                return True
            QApplication.processEvents()
        return not self.win.assist._batch_active

    def test_batch_completes_when_the_backend_disappears_mid_scan(self):
        one_image = self.win.file_path
        self.assertTrue(os.path.isfile(one_image))
        paths = [one_image] * self.N
        self.win.m_img_list = paths
        self.win.img_count = len(paths)

        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)
        self.win.assist.set_backend(StubBackend())

        self.assertTrue(self.win.assist.score_folder())
        self.assertEqual(1, len(jobs))

        # The backend disappears mid-scan by SOME OTHER means than the
        # dialog (bypassing AssistController.set_backend on purpose, so
        # apply_model_settings' proactive cancel-batch hardening cannot
        # mask the very recursion this test exists to catch -- see that
        # method's "preferred additional hardening" note: a backend can
        # become unavailable other ways too).
        self.win.inference_service.set_backend(None)

        jobs.pop(0)()  # resolves the one in-flight (model-A) request

        finished = self._drain_deferred_batch(self.N * 3 + 200)
        self.assertTrue(
            finished,
            'batch never completed -- a hang here means a broken deferral '
            '(a RecursionError would instead raise out of this test as an '
            'exception, not hang)')
        self.assertFalse(self.win.assist._batch_active)
        self.assertEqual(self.N, self.win.assist._batch_scored)
        # HARD SAFETY: never a modal, regardless of how many consecutive
        # images failed to score.
        self.assertEqual([], self.errors)


if __name__ == '__main__':
    unittest.main()
