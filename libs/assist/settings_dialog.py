#!/usr/bin/env python
# -*- coding: utf8 -*-
"""ModelSettingsDialog -- the AI menu's "Model Settings..." action.

This is a THIN SHELL on purpose: it only collects a backend choice and a
model path and hands them to ``AssistController.apply_model_settings``, which
owns every actual decision (validate, persist, rebuild the backend live). See
that method's docstring for the full contract. Keeping the logic there --
not here -- is what lets the test suite exercise every success/failure path
without ever calling ``exec_()`` on this dialog (a modal event loop that a
headless/offscreen test must never enter).
"""

try:
    from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                                 QFileDialog, QFormLayout, QHBoxLayout,
                                 QLabel, QLineEdit, QPushButton, QVBoxLayout,
                                 QWidget)
except ImportError:  # pragma: no cover - the app's PyQt4 fallback path
    from PyQt4.QtGui import (QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

from libs.assist.controller import ModelSettingsError

MODEL_WEIGHTS_HINT = (
    'No model weights ship with this app (AGPL licensing -- see '
    'data/models/README.md). Point this at your own .onnx file.')


class ModelSettingsDialog(QDialog):
    """Lets the user pick a backend + model path and apply it immediately.

    ``(label, backend_name)`` pairs shown in the dropdown -- ``backend_name``
    is exactly what ``apply_model_settings``/the registry expect.  'stub' is
    deliberately NOT offered here: see
    ``AssistController.AVAILABLE_UI_BACKENDS``'s docstring -- a persisted
    'stub' is always read back as unset on the next launch, so exposing it as
    a real UI choice would let a user pick a setting that silently stops
    applying the moment they restart the app.
    """

    BACKEND_CHOICES = (
        ('사용 안 함', None),
        ('YOLO (ONNX)', 'yolo_onnx'),
    )

    def __init__(self, controller, parent=None):
        super(ModelSettingsDialog, self).__init__(parent)
        self.controller = controller
        self.setWindowTitle('AI Model Settings')

        layout = QVBoxLayout(self)

        hint = QLabel(MODEL_WEIGHTS_HINT)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()

        self.backend_combo = QComboBox()
        for label, _backend_name in self.BACKEND_CHOICES:
            self.backend_combo.addItem(label)
        form.addRow('Backend', self.backend_combo)

        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit()
        self.path_edit.setMinimumWidth(280)
        browse_button = QPushButton('Browse...')
        browse_button.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_button)
        form.addRow('Model path (.onnx)', path_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._prefill()

    # -- setup ---------------------------------------------------------

    def _prefill(self):
        """Show whatever this controller currently holds, so reopening the
        dialog reflects the app's actual state rather than always starting
        from "사용 안 함"."""
        current_backend = self.controller.backend_name
        index = 0
        for i, (_label, backend_name) in enumerate(self.BACKEND_CHOICES):
            if backend_name == current_backend:
                index = i
                break
        self.backend_combo.setCurrentIndex(index)
        self.path_edit.setText(self.controller.model_path or '')

    # -- slots -----------------------------------------------------------

    def _browse(self):
        result = QFileDialog.getOpenFileName(
            self, 'Select ONNX model', self.path_edit.text(), 'ONNX Model (*.onnx)')
        # PyQt4's getOpenFileName returns a bare string rather than a
        # (name, filter) tuple; mirrors labelImg.py's own open_file handling
        # of the same quirk. The isinstance check MUST run before any tuple
        # unpacking of `result` -- unpacking first (the previous shape of
        # this code, `path, _filter = QFileDialog.getOpenFileName(...)`)
        # raises trying to unpack a bare string under the PyQt4 fallback,
        # before this guard ever gets a chance to run.
        if isinstance(result, (tuple, list)):
            path = result[0] if result else ''
        else:  # pragma: no cover - defensive: PyQt4's bare-string return
            path = result
        if path:
            self.path_edit.setText(path)

    def _on_accept(self):
        _label, backend_name = self.BACKEND_CHOICES[self.backend_combo.currentIndex()]
        model_path = self.path_edit.text()
        try:
            self.controller.apply_model_settings(backend_name, model_path)
        except ModelSettingsError as exc:
            self.controller.app.error_message('AI Model Settings', str(exc))
            return
        self.accept()
