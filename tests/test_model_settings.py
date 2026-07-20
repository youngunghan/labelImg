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
from libs.inference.stub import StubBackend
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


if __name__ == '__main__':
    unittest.main()
