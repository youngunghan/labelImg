#!/usr/bin/env python
# -*- coding: utf8 -*-
"""AssistController -- the one object MainWindow delegates "AI" to.

MainWindow is already a god object; the point of this class is that adding the
ML-assist feature costs it *wiring only* (construct, splice the actions into a
menu and into ``onLoadActive``, one filter in ``save_labels``).  All behaviour --
which actions exist, what the confidence threshold means, what accept/reject do
-- lives here.

Two invariants worth stating out loud, because breaking either produces a silent
data bug rather than a crash:

**A suggestion is not data until the user accepts it.**  Provisional shapes live
on the canvas and in the label list, but ``MainWindow.save_labels`` filters them
out, so they cannot reach an annotation file.  Accepting one is exactly
"clear ``provisional``"; rejecting one is exactly "remove it".

**A result may arrive for an image the user has already left.**  Inference is
slow, navigation is not.  Every result is checked against the app's *current*
file path and dropped if it does not match -- otherwise a slow prediction for
image A would silently deposit boxes on image B, which the next save would then
write into B's annotation file as ground truth.
"""

import logging

try:
    from PyQt5.QtCore import QObject, Qt
    from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSlider, QWidget,
                                 QWidgetAction)
except ImportError:  # pragma: no cover - the app's PyQt4 fallback path
    from PyQt4.QtCore import QObject, Qt
    from PyQt4.QtGui import (QHBoxLayout, QLabel, QSlider, QWidget,
                             QWidgetAction)

from libs.assist.suggestion import detection_to_shape, style_as_committed
from libs.constants import (SETTING_CONF_THRESHOLD, SETTING_MODEL_BACKEND,
                            SETTING_MODEL_PATH)
from libs.inference.registry import DEFAULT_BACKEND, build_backend
from libs.inference.service import to_model_image
from libs.utils import generate_color_by_text, new_action

logger = logging.getLogger(__name__)

DEFAULT_CONF_THRESHOLD = 0.5

# Checked against every shortcut the app actually binds at runtime (dump of
# QAction.shortcut() over MainWindow.findChildren(QAction)), not against a
# guess: this fork has already shipped one bug caused by a double-bound key
# (Single Class Mode had to move off Ctrl+Shift+S). Qt does not warn about an
# ambiguous shortcut -- it silently disables BOTH actions bound to it.
#   Taken and deliberately avoided: Delete ("Delete RectBox"), Ctrl+D
#   ("Duplicate RectBox"), Ctrl+A/Ctrl+H (show/hide boxes), Ctrl+S/Ctrl+Shift+S,
#   Ctrl+E/Ctrl+Shift+E, a/d/w/g/b/space, Ctrl+{O,Q,R,U,V,W,Y,Z,J,L,F},
#   Ctrl+Shift+{A,C,D,F,L,O,P,R}, Ctrl+{+,-,=} and Ctrl+Shift+{+,-,=}.
SHORTCUT_AUTO_LABEL = 'Ctrl+I'
SHORTCUT_ACCEPT_ALL = 'Ctrl+Return'
SHORTCUT_REJECT_ALL = 'Ctrl+Backspace'

# Two distinct "AI disabled" causes need two distinct hints (see
# _unavailable_hint / refresh_actions): nothing was configured at all (fresh
# install, SETTING_MODEL_BACKEND unset -- DEFAULT_BACKEND is None precisely so
# this is the out-of-the-box state), versus a backend WAS named but its
# construction failed (missing extras, or extras present but SETTING_MODEL_PATH
# missing/invalid). Telling a fresh-install user to just `pip install
# labelImg[ai]` would be accurate but incomplete -- the extras alone do nothing
# without also choosing a backend and a model path -- so that case gets its own
# message rather than reusing the "something failed" one.
NO_BACKEND_CONFIGURED_HINT = (
    "No model backend configured — set a backend (e.g. 'yolo_onnx') and a model "
    "path in Settings; installing the extras alone is not enough: "
    "pip install labelImg[ai]")
BACKEND_UNAVAILABLE_HINT = (
    "Model backend %r is unavailable — install the optional extras "
    "(pip install labelImg[ai]) and check that the configured model path is valid")


class AssistController(QObject):

    def __init__(self, app, service, parent=None):
        super(AssistController, self).__init__(parent if parent is not None else app)
        self.app = app
        self.service = service

        settings = app.settings
        self.backend_name = settings.get(SETTING_MODEL_BACKEND, DEFAULT_BACKEND) or DEFAULT_BACKEND
        self.model_path = settings.get(SETTING_MODEL_PATH, None)
        self.threshold = self._sanitize_threshold(
            settings.get(SETTING_CONF_THRESHOLD, DEFAULT_CONF_THRESHOLD))

        # Detections for the current image, kept whole (unfiltered) so that
        # moving the threshold re-filters what is on screen WITHOUT re-running
        # the model. `_shapes` maps detection index -> the Shape currently
        # showing it (absent = filtered out / never shown / dismissed).
        self._detections = []
        self._shapes = {}
        # Detection indices the user removed by hand ("Delete RectBox" on a
        # suggestion). Without this, the next _sync_suggestions would see "index
        # above threshold, no shape" and put the box the user just deleted
        # straight back on the canvas.
        self._dismissed = set()
        # The shape this controller is currently removing. remove_label reports
        # EVERY removal back to us (discard_shape); this is how the ones we asked
        # for are told apart from the user's own, which must not be mistaken for
        # a dismissal.
        self._removing = None

        self.action_auto = None
        self.action_accept = None
        self.action_reject = None
        self.action_threshold = None
        self._threshold_slider = None
        self._threshold_value_label = None
        self._actions = []
        # The tooltip/statusTip each action carries while available (set by
        # new_action's `tip` argument, or '' for the threshold widget action).
        # refresh_actions overwrites both with the disabled-state hint while the
        # backend is unavailable, and must restore exactly this -- not whatever
        # is left sitting in statusTip() -- once it becomes available again.
        self._base_tips = {}

        self.service.predictionReady.connect(self.on_prediction_ready)
        self.service.predictionFailed.connect(self.on_prediction_failed)
        self.service.set_backend(self._build_backend())

    # -- backend -----------------------------------------------------------

    def _build_backend(self):
        """Ask the registry for the configured backend; None is a normal answer.

        ``conf_threshold`` is deliberately 0.0: the *backend* must not pre-filter
        by confidence, because then a user lowering the UI threshold would need a
        second inference run to see the boxes it had already discarded. The model
        returns everything it found; this controller decides what is shown.
        """
        return build_backend({
            'backend': self.backend_name,
            'model_path': self.model_path,
            'conf_threshold': 0.0,
        })

    def set_backend(self, backend):
        """Inject a backend directly (tests, and the future model picker)."""
        self.service.set_backend(backend)
        self.refresh_actions()

    def is_available(self):
        return self.service.is_available()

    def _unavailable_hint(self):
        """Which "AI disabled" message applies right now.

        ``self.backend_name`` is ``None`` on a fresh install (DEFAULT_BACKEND is
        None; see libs/inference/registry.py) -- that is "nothing configured
        yet", not "something configured is broken", and the two need different
        advice: installing the extras alone does not help someone who has not
        also picked a backend and a model path.
        """
        if not self.backend_name:
            return NO_BACKEND_CONFIGURED_HINT
        return BACKEND_UNAVAILABLE_HINT % self.backend_name

    # -- actions -----------------------------------------------------------

    def create_actions(self):
        """Build the AI actions at runtime, like MainWindow.create_classify_actions.

        MainWindow only has to splice the returned list into a menu and (for the
        image-dependent ones, see load_active_actions) into ``onLoadActive``.
        """
        app = self.app
        self.action_auto = new_action(
            app, 'Auto-label Image', self.auto_label_image, SHORTCUT_AUTO_LABEL, 'new',
            'Run the model on this image and show its boxes as suggestions',
            enabled=False)
        self.action_accept = new_action(
            app, 'Accept All Suggestions', self.accept_all, SHORTCUT_ACCEPT_ALL, 'done',
            'Turn every suggestion on this image into a real box',
            enabled=False)
        self.action_reject = new_action(
            app, 'Reject All Suggestions', self.reject_all, SHORTCUT_REJECT_ALL, 'delete',
            'Discard every suggestion on this image',
            enabled=False)
        self.action_threshold = self._create_threshold_action()

        self._actions = [self.action_auto, self.action_accept, self.action_reject,
                         self.action_threshold]
        # Captured before refresh_actions ever runs, so the very first refresh
        # (which may find no backend and stamp the hint over everything) still
        # has the real base tip to restore once a backend becomes available.
        self._base_tips = {action: action.statusTip() for action in self._actions}
        self.refresh_actions()
        return list(self._actions)

    def _create_threshold_action(self):
        """A slider embedded in the menu (QWidgetAction), like the zoom widget.

        A slider is the right control because the threshold is explored, not
        entered: the user drags it and watches suggestions appear/disappear.
        Integer percent, because a QSlider is integer-only and 1% granularity is
        far finer than a confidence number is meaningful.
        """
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 8, 2)
        layout.addWidget(QLabel('Confidence'))

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int(round(self.threshold * 100)))
        slider.setMinimumWidth(120)
        slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(slider)

        value_label = QLabel(self._threshold_text())
        value_label.setMinimumWidth(36)
        layout.addWidget(value_label)
        widget.setLayout(layout)

        action = QWidgetAction(self.app)
        action.setDefaultWidget(widget)
        action.setText('Confidence Threshold')

        self._threshold_slider = slider
        self._threshold_value_label = value_label
        return action

    def load_active_actions(self):
        """The actions that only make sense with an image open.

        The threshold is not among them: it is a preference the user may set
        before opening anything.
        """
        return (self.action_auto, self.action_accept, self.action_reject)

    def refresh_actions(self):
        """(Re-)apply availability. MainWindow calls this from toggle_actions.

        This must run *after* toggle_actions' own loop: that loop enables every
        onLoadActive action when an image loads, which would happily re-enable
        the AI actions on a machine with no model backend.
        """
        if not self._actions:
            return

        available = self.is_available()
        has_image = bool(getattr(self.app, 'file_path', None))
        has_suggestions = bool(self.provisional_shapes())

        self.action_auto.setEnabled(available and has_image)
        self.action_accept.setEnabled(available and has_image and has_suggestions)
        self.action_reject.setEnabled(available and has_image and has_suggestions)
        if self.action_threshold is not None:
            self.action_threshold.setEnabled(available)

        # A greyed-out menu with no explanation reads as a bug; say why.
        #
        # The unavailable branch below OVERWRITES both toolTip and statusTip with
        # the hint. If the backend later becomes available, `hint` is '' here --
        # falling back to action.statusTip() would read back the hint this same
        # method wrote last time (never cleared), so an ENABLED action would keep
        # showing "No model backend configured". The available branch must
        # restore the action's own base tip explicitly, not read whatever is
        # still sitting in statusTip().
        hint = self._unavailable_hint() if not available else ''
        for action in self._actions:
            if hint:
                action.setStatusTip(hint)
                action.setToolTip(hint)
            else:
                base_tip = self._base_tips.get(action, '')
                action.setStatusTip(base_tip)
                action.setToolTip(base_tip or action.text())

    # -- threshold ---------------------------------------------------------

    @staticmethod
    def _sanitize_threshold(value):
        # Unpickled from settings: a corrupted/hand-edited value must not be able
        # to hide every suggestion forever (or crash the comparison below).
        try:
            value = float(value)
        except (TypeError, ValueError):
            return DEFAULT_CONF_THRESHOLD
        return min(1.0, max(0.0, value))

    def _threshold_text(self):
        return '%d%%' % int(round(self.threshold * 100))

    def _on_slider_changed(self, percent):
        self.set_threshold(percent / 100.0)

    def set_threshold(self, value):
        """Re-filter the suggestions already on screen. Never re-runs the model."""
        value = self._sanitize_threshold(value)
        if value == self.threshold:
            return
        self.threshold = value

        if self._threshold_slider is not None:
            percent = int(round(value * 100))
            if self._threshold_slider.value() != percent:
                self._threshold_slider.blockSignals(True)
                self._threshold_slider.setValue(percent)
                self._threshold_slider.blockSignals(False)
        if self._threshold_value_label is not None:
            self._threshold_value_label.setText(self._threshold_text())

        self._sync_suggestions()
        self.refresh_actions()

    # -- inference ---------------------------------------------------------

    def auto_label_image(self, _value=False):
        if not self.is_available():
            self.app.status(self._unavailable_hint())
            return False
        file_path = self.app.file_path
        if not file_path:
            self.app.status('Open an image before running the model.')
            return False

        # Re-running replaces the previous round rather than stacking a second
        # set of boxes on top of it.
        self.clear_suggestions()

        # UI thread: the QImage is converted to plain data here, so the worker
        # thread never touches a Qt object the UI may repaint or free under it.
        image = to_model_image(self.app.image)
        if image is None:
            self.app.status('No image data to run the model on.')
            return False

        self.app.status('Running the model on %s...' % file_path)
        return self.service.predict_async(file_path, image)

    def on_prediction_ready(self, image_path, detections):
        """UI thread (queued signal): safe to build Shapes and touch the canvas."""
        if not self._is_current(image_path):
            return

        self._detections = list(detections)
        self._shapes = {}
        self._sync_suggestions()

        shown = len(self._shapes)
        self.app.status('%d suggestion(s) at >=%s (%d found)'
                        % (shown, self._threshold_text(), len(self._detections)))
        self.refresh_actions()

    def on_prediction_failed(self, image_path, reason):
        if not self._is_current(image_path):
            return
        self.app.status('Model failed: %s' % reason)
        self.refresh_actions()

    def _is_current(self, image_path):
        """STALE-RESULT DROP. See the module docstring: a result for an image the
        user has already navigated away from must never be injected into the one
        now on screen."""
        current = self.app.file_path
        if current and image_path and image_path == current:
            return True
        logger.info('Dropping stale prediction for %r (current image is %r)',
                    image_path, current)
        return False

    # -- suggestions -------------------------------------------------------

    def provisional_shapes(self):
        """Every suggestion currently ON THE CANVAS, in canvas order.

        Read from the canvas, not from `_shapes`, because `_shapes` is not the
        truth about what is on screen and cannot be:

        * a suggestion can LEAVE the canvas without this controller (the ordinary
          "Delete RectBox" action), which used to leave a stale Shape in
          `_shapes` -- and the next threshold change would then hand that dead
          shape to canvas.delete_selected(), i.e. list.remove(x) with x not in
          the list: ValueError, mid-drag, on the slider;
        * a suggestion can ARRIVE on the canvas without this controller (Ctrl+D:
          Shape.copy() faithfully clones `provisional`), and such a copy has no
          detection index, so tracking-based bulk actions could never see it --
          it stayed a dashed, unsaveable phantom no Accept/Reject All could reach.

        Reading the canvas makes Accept All / Reject All act on exactly what the
        user sees. `_shapes` keeps its one real job: mapping a detection to the
        shape showing it, so the threshold can re-filter without re-inferring.
        """
        return [shape for shape in self.app.canvas.shapes if shape.provisional]

    def discard_shape(self, shape):
        """A shape was removed from the canvas -- drop any tracking of it.

        MainWindow.remove_label calls this for EVERY shape it removes, so it must
        be a no-op for shapes we never tracked. Removals we did not ask for are
        the user deleting a suggestion by hand: that detection is recorded as
        dismissed, so a later threshold move re-filters the others without
        resurrecting the box the user just deleted.
        """
        ours = shape is self._removing
        for index in [i for i, tracked in list(self._shapes.items()) if tracked is shape]:
            del self._shapes[index]
            if not ours and shape.provisional:
                self._dismissed.add(index)
        if not ours:
            # The canvas has already dropped it, so Accept/Reject All may have
            # just become meaningless.
            self.refresh_actions()

    def _sync_suggestions(self):
        """Make what is on the canvas match (detections, threshold).

        Adds the shapes that are now above the threshold, removes the ones that
        have fallen below it. Shapes the user already accepted are left alone --
        they stopped being suggestions the moment `provisional` was cleared.
        """
        new_shapes = []
        for index, detection in enumerate(self._detections):
            if index in self._dismissed:
                continue  # the user deleted this one by hand; do not bring it back
            above = float(detection.score) >= self.threshold
            shape = self._shapes.get(index)

            if above and shape is None:
                shape = detection_to_shape(detection)
                self._shapes[index] = shape
                new_shapes.append(shape)
            elif not above and shape is not None:
                if not shape.provisional:
                    continue  # accepted: it is the user's box now, not ours
                self._remove_shape(shape, mark_dirty=False)
                # pop, not del: _remove_shape goes through remove_label, which
                # reports back into discard_shape and may already have dropped it.
                self._shapes.pop(index, None)

        if new_shapes:
            canvas = self.app.canvas
            # One repaint for the batch, not one per box.
            canvas.load_shapes(canvas.shapes + new_shapes)
            for shape in new_shapes:
                # MUST go through add_label: canvas selection looks the shape up
                # in shapes_to_items, and a canvas-only shape makes remove_label
                # (which does not guard the lookup) raise KeyError.
                self.app.add_label(shape)
            canvas.update()

    def accept_all(self, _value=False):
        """Promote every suggestion to a real box. This is what makes them saveable."""
        shapes = self.provisional_shapes()
        if not shapes:
            self.app.status('No suggestions to accept.')
            return 0

        for shape in shapes:
            style_as_committed(shape, generate_color_by_text(shape.label))
            # Mirrors new_shape(): a label that is not in label_hist has no class
            # id, which the YOLO writer needs.
            if shape.label and shape.label not in self.app.label_hist:
                self.app.label_hist.append(shape.label)

        self._forget()
        self.app.set_dirty()  # they are part of the document now
        self.app.canvas.update()
        self.app.status('Accepted %d suggestion(s).' % len(shapes))
        self.refresh_actions()
        return len(shapes)

    def reject_all(self, _value=False):
        shapes = self.provisional_shapes()
        if not shapes:
            self.app.status('No suggestions to reject.')
            return 0

        for shape in shapes:
            self._remove_shape(shape, mark_dirty=True)
        self._forget()
        self.app.status('Rejected %d suggestion(s).' % len(shapes))
        self.refresh_actions()
        return len(shapes)

    def clear_suggestions(self):
        """Drop the suggestions from the canvas (used before a re-run)."""
        for shape in self.provisional_shapes():
            self._remove_shape(shape, mark_dirty=False)
        self._forget()
        self.refresh_actions()

    def forget_suggestions(self):
        """Drop the tracking only.

        For MainWindow.reset_state, which has already cleared the canvas and the
        label list: touching them again here would be a double removal.
        """
        self._forget()

    def _forget(self):
        self._detections = []
        self._shapes = {}
        self._dismissed = set()

    def _remove_shape(self, shape, mark_dirty):
        """Remove a shape through the app's own removal path.

        Going through delete_selected_shape (rather than poking canvas.shapes)
        is what keeps items_to_shapes / shapes_to_items consistent -- remove_label
        does not guard its lookup, so a shape that skipped add_label, or a label
        item that outlives its shape, is a KeyError later.

        `mark_dirty=False` is for removals that are not document edits (a
        threshold change is a *view filter*, and a provisional shape was never
        part of the saved document): they must not make an untouched file look
        unsaved.
        """
        app = self.app
        if shape not in app.canvas.shapes:
            # Already gone (removed out of band, e.g. by load_labels replacing the
            # canvas wholesale). canvas.delete_selected() would reach
            # list.remove(x) with x not in the list -> ValueError. Drop the
            # tracking instead; there is nothing left to remove.
            self.discard_shape(shape)
            return

        was_dirty = app.dirty

        self._removing = shape
        try:
            app.canvas.select_shape(shape)
            app.delete_selected_shape()  # remove_label + set_dirty + onShapesPresent
            app.canvas.de_select_shape()
        finally:
            self._removing = None

        if not mark_dirty and not was_dirty:
            app.dirty = False
            app.actions.save.setEnabled(False)
