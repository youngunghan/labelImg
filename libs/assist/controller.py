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
    from PyQt5.QtCore import QObject, Qt, QTimer
    from PyQt5.QtGui import QColor, QImageReader
    from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QListWidgetItem, QSlider,
                                 QWidget, QWidgetAction)
except ImportError:  # pragma: no cover - the app's PyQt4 fallback path
    from PyQt4.QtCore import QObject, Qt, QTimer
    from PyQt4.QtGui import (QColor, QHBoxLayout, QImageReader, QLabel,
                             QListWidgetItem, QSlider, QWidget, QWidgetAction)

from libs.assist.suggestion import detection_to_shape, style_as_committed
from libs.constants import (SETTING_CONF_THRESHOLD, SETTING_MODEL_BACKEND,
                            SETTING_MODEL_PATH)
from libs.inference.registry import DEFAULT_BACKEND, build_backend
from libs.inference.service import to_model_image
from libs.inference.types import least_confidence
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
# Phase 4 (active-learning triage). Ctrl+Shift+U is free against the same
# checked-at-runtime dump as the block above ('U' is not in either the
# Ctrl+{...} or Ctrl+Shift+{...} taken sets listed there) -- 'U' for
# "Uncertainty". Score Folder and Restore Order deliberately get NO shortcut:
# they are one-shot/rare menu actions (a batch scan; an order reset), and
# every remaining free key is also at the mercy of File > Edit Classify
# Categories, which lets a user rebind classify actions to arbitrary single
# keys at runtime -- stacking more static shortcuts into that space buys
# little for actions nobody needs to reach without the mouse.
SHORTCUT_SORT_BY_UNCERTAINTY = 'Ctrl+Shift+U'

# Two distinct "AI disabled" causes need two distinct hints (see
# _unavailable_hint / refresh_actions): nothing was configured at all (fresh
# install, SETTING_MODEL_BACKEND unset -- DEFAULT_BACKEND is None precisely so
# this is the out-of-the-box state), versus a backend WAS named but its
# construction failed (missing extras, or extras present but SETTING_MODEL_PATH
# missing/invalid). Telling a fresh-install user to just install the extras
# would be accurate but incomplete -- they alone do nothing without also
# choosing a backend and a model path -- so that case gets its own message
# rather than reusing the "something failed" one. Both hints point at
# `pip install -e ".[ai]"` from this checkout: this fork is not published to
# PyPI, so the plain package-name form would silently fetch an unrelated
# upstream project instead.
NO_BACKEND_CONFIGURED_HINT = (
    "No model backend configured — set a backend (e.g. 'yolo_onnx') and a model "
    "path in Settings; installing the extras alone is not enough. This fork "
    "isn't on PyPI: install from your checkout with pip install -e \".[ai]\"")
BACKEND_UNAVAILABLE_HINT = (
    "Model backend %r is unavailable — install from your checkout with "
    "pip install -e \".[ai]\" (this fork isn't on PyPI) and check that the "
    "configured model path is valid")

# Before DEFAULT_BACKEND was fixed to None, it was 'stub'. Anyone who ran
# labelImg during that window and closed it even once now has 'stub' sitting
# in their settings pickle as an EXPLICIT SETTING_MODEL_BACKEND value -- the
# old, unconditional closeEvent write (see labelImg.py MainWindow.closeEvent)
# turned that implicit default into a sticky one. There is no settings-picker
# UI (see NO_BACKEND_CONFIGURED_HINT above) and no doc ever tells a user to
# set 'stub' by hand, so a persisted 'stub' can only be that old leak -- never
# a deliberate opt-in -- and reading it back must not resurrect StubBackend's
# fabricated (image-dimension-derived) detections. This is checked here, at
# the single choke point every AssistController construction reads settings
# through, so it holds no matter how many times the tainted pickle is loaded
# and re-saved. `stub` remains fully usable when selected EXPLICITLY
# in-process (AssistController.set_backend, or build_backend({'backend':
# 'stub', ...}) directly, as the tests do) -- only this settings-read path
# treats the persisted name as unset.
_LEGACY_IMPLICIT_DEFAULT_BACKEND = 'stub'


class AssistController(QObject):

    def __init__(self, app, service, parent=None):
        super(AssistController, self).__init__(parent if parent is not None else app)
        self.app = app
        self.service = service

        settings = app.settings
        raw_backend_name = settings.get(SETTING_MODEL_BACKEND, DEFAULT_BACKEND)
        if raw_backend_name == _LEGACY_IMPLICIT_DEFAULT_BACKEND:
            logger.info(
                "Ignoring persisted %s=%r: this is the pre-fix implicit "
                "default (DEFAULT_BACKEND used to be 'stub'), never a "
                "deliberate choice -- there is no settings UI that writes "
                "it. Treating it as unset instead of building StubBackend "
                "and showing fabricated detections as if a real model ran.",
                SETTING_MODEL_BACKEND, raw_backend_name)
            raw_backend_name = None
        self.backend_name = raw_backend_name or DEFAULT_BACKEND
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

        # -- active-learning triage (Phase 4) -------------------------------
        # image_path -> uncertainty (0..1, least_confidence of that image's
        # detections). Populated by a batch scan (score_folder), consumed by
        # sort_by_uncertainty; belongs to the CURRENTLY SCANNED folder, so a
        # fresh directory scan clears it (on_directory_scanned) but a
        # same-directory refresh after a classify/delete/undo move does not
        # (see import_dir_images' reset_active_learning parameter) -- that
        # refresh must not throw away the triage order mid-session.
        self._uncertainty = {}
        self._ranks_cache = None  # invalidated whenever _uncertainty changes
        # Filesystem-order snapshot taken at the same moment _uncertainty is
        # cleared, so "Restore Original Order" has something to restore even
        # after several sorts.
        self._original_order = None
        # True from a successful sort_by_uncertainty() until
        # restore_original_order() or a fresh directory scan. Lets
        # reapply_sort_if_active() keep the triage queue in uncertainty order
        # across the same-directory rescans classify/delete/undo trigger --
        # see that method's docstring for why this is not optional.
        self._sort_active = False

        # One batch run at a time; state for the currently running scan (or
        # all-default/empty when none is running). _batch_current_path is
        # the one image the batch is currently waiting a result for (used by
        # _load_model_image / status text / _advance_batch), but which FLOW
        # a result belongs to is decided by _dispatch_request /
        # _pop_request_kind's FIFO tag, not by matching this path -- see
        # _dispatch_request's docstring for why bare path equality is wrong.
        self._batch_active = False
        self._batch_snapshot = []
        self._batch_index = 0
        self._batch_total = 0
        self._batch_scored = 0
        self._batch_current_path = None

        # FIFO of 'interactive'/'batch' tags, one appended per predict_async
        # dispatch (_dispatch_request) and popped by on_prediction_ready /
        # on_prediction_failed (_pop_request_kind). InferenceService's single
        # worker resolves requests strictly in submission order, so this is
        # what tells an interactive result and a batch result apart CORRECTLY
        # even when a Ctrl+I request is still outstanding when score_folder()
        # starts and both happen to target the same image path -- routing by
        # bare path equality (the pre-fix approach) misattributes exactly
        # that race. See _dispatch_request for the full reasoning.
        self._pending_requests = []

        self.action_score_folder = None
        self.action_sort = None
        self.action_restore_order = None
        self._score_label = None
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

        self.action_score_folder = new_action(
            app, 'Score Folder for Active Learning', self.score_folder, None, 'new',
            'Run the model on every image in this folder and rank them by '
            'uncertainty (triggering again cancels a run in progress)',
            enabled=False)
        self.action_sort = new_action(
            app, 'Sort by Uncertainty', self.sort_by_uncertainty,
            SHORTCUT_SORT_BY_UNCERTAINTY, 'labels',
            'Reorder the file list so the most uncertain (highest-value) '
            'images come first -- triage them with g/b in that order',
            enabled=False)
        self.action_restore_order = new_action(
            app, 'Restore Filesystem Order', self.restore_original_order, None, 'undo',
            'Undo Sort by Uncertainty and put the file list back in scan order',
            enabled=False)

        self._actions = [self.action_auto, self.action_accept, self.action_reject,
                         self.action_threshold, self.action_score_folder,
                         self.action_sort, self.action_restore_order]
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
        """The actions that only make sense with an image (or a scanned
        folder) open.

        The threshold is not among them: it is a preference the user may set
        before opening anything. Score/Sort/Restore key off the FOLDER
        (m_img_list), not the single open image, but toggle_actions' on/off
        cycle only fires from load_file/close_file -- exactly the folder-open
        / folder-closed transitions this app has -- so it is still the right
        base gate; refresh_actions narrows each one further.
        """
        return (self.action_auto, self.action_accept, self.action_reject,
                self.action_score_folder, self.action_sort, self.action_restore_order)

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
        has_folder = bool(getattr(self.app, 'm_img_list', None))
        batch_running = self._batch_active

        # Auto-label/accept/reject are suspended for the duration of a batch
        # run: auto_label_image would otherwise queue a SECOND request behind
        # the batch's single-worker pool that _batch_step's bookkeeping does
        # not expect. This is UX hygiene, not the correctness fix for the
        # interleaving race -- that is _dispatch_request/_pop_request_kind's
        # FIFO tagging, which stays correct even for a request dispatched
        # before this disablement took effect. Sort/Restore
        # are suspended too -- reordering m_img_list out from under a batch
        # walking its own frozen snapshot would not corrupt anything (the
        # snapshot is independent), but the result would silently apply to
        # whatever order existed when scoring finishes, which is confusing.
        self.action_auto.setEnabled(available and has_image and not batch_running)
        self.action_accept.setEnabled(available and has_image and has_suggestions and not batch_running)
        self.action_reject.setEnabled(available and has_image and has_suggestions and not batch_running)
        if self.action_threshold is not None:
            self.action_threshold.setEnabled(available)

        # Score Folder stays enabled WHILE a batch is running (see above) --
        # triggering it again is exactly how the run is cancelled. Gating
        # purely on has_folder would strand a running batch: if m_img_list
        # empties (or shrinks) out from under the batch WHILE it runs (e.g.
        # every remaining image gets classified away), has_folder goes
        # False and this action -- the ONLY control that can cancel a
        # running batch -- would grey itself out with the batch still
        # active, leaving the user with no way to stop it. batch_running is
        # ORed in so cancellation stays reachable for the batch's whole
        # duration regardless of what happens to m_img_list.
        if self.action_score_folder is not None:
            self.action_score_folder.setEnabled(available and (has_folder or batch_running))
        if self.action_sort is not None:
            self.action_sort.setEnabled(available and bool(self._uncertainty) and not batch_running)
        if self.action_restore_order is not None:
            self.action_restore_order.setEnabled(
                available and self._original_order is not None and not batch_running)

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

        self._update_batch_action_text()
        self._update_score_label()

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
        return self._dispatch_request('interactive', file_path, image)

    def _dispatch_request(self, kind, path, image):
        """Submit one predict_async request, tagged with which flow started
        it ('interactive' or 'batch') -- the ONLY call sites are here and
        _batch_step, and every dispatch must go through one of them.

        ROOT-CAUSE FIX (was DESIGN POINT B): routing a result by bare path
        equality against ``_batch_current_path`` is wrong -- InferenceService
        queues rather than rejects concurrent requests (single-worker pool),
        and nothing ever stopped an interactive request from still being
        outstanding when score_folder() starts. If that request and the
        batch's own first request both target the SAME path (e.g. the
        image already open when the batch begins), path-based routing
        misattributes: the interactive result gets consumed as the batch's
        progress, and the batch's own later result for that path gets
        injected into the interactive suggestion flow instead.

        The single worker resolves requests in the exact order they were
        submitted (QThreadPool runs same-priority QRunnables FIFO; the tests'
        SynchronousExecutor resolves inline, i.e. immediately, which is
        trivially FIFO too), so tagging every dispatch by ORDER rather than
        by path is unambiguous regardless of what either request's path
        happens to be. The tag is appended here, BEFORE calling
        predict_async, because predict_async can resolve synchronously
        (SynchronousExecutor, or its own no-backend/no-image early exits) --
        appending first guarantees the tag is already there for
        on_prediction_ready/on_prediction_failed to pop, however this call
        resolves.
        """
        self._pending_requests.append(kind)
        return self.service.predict_async(path, image)

    def _pop_request_kind(self):
        """Which flow ('interactive' or 'batch') the OLDEST outstanding
        request belongs to -- see _dispatch_request. Every predict_async
        dispatch in this controller goes through _dispatch_request, which
        appends before submitting, so a result reaching on_prediction_ready/
        on_prediction_failed always has a matching tag queued ahead of it;
        the empty-queue case is defensive only (never observed) and falls
        back to 'interactive' rather than raising, so a bug here degrades to
        the ordinary stale-result guard instead of crashing the app.
        """
        if self._pending_requests:
            return self._pending_requests.pop(0)
        return 'interactive'

    def on_prediction_ready(self, image_path, detections):
        """UI thread (queued signal): safe to build Shapes and touch the canvas."""
        if self._pop_request_kind() == 'batch':
            if self._batch_active:
                self._advance_batch(image_path, least_confidence(detections))
            else:
                logger.info('Dropping late batch result for %r (batch already '
                            'finished/was cancelled)', image_path)
            return
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
        if self._pop_request_kind() == 'batch':
            if self._batch_active:
                logger.info('Batch scoring: %r failed (%s); recording max uncertainty '
                            '("could not be scored" is exactly the case a human most '
                            'needs to look at -- same reasoning as least_confidence\'s '
                            'no-detection case).', image_path, reason)
                self._advance_batch(image_path, 1.0)
            else:
                logger.info('Dropping late failed batch result for %r (batch '
                            'already finished/was cancelled)', image_path)
            return
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

    # -- active-learning triage (Phase 4) -----------------------------------
    #
    # THE IDEA: run the configured detector over every image in the folder,
    # score each image's uncertainty with libs.inference.types.least_confidence
    # (unchanged -- this module only calls it), and reorder m_img_list so the
    # most uncertain (highest-value) images sort first. The user then triages
    # with the EXISTING g/b shortcuts (MainWindow.classify_current_image) on
    # the reordered stream. Nothing here replaces or wraps those shortcuts --
    # they already index m_img_list by position, so reordering it is the
    # entire mechanism.

    def score_folder(self, _value=False):
        """Start a batch uncertainty scan over every image currently in
        ``self.app.m_img_list``, or cancel one already running -- this one
        action does both (see the tooltip built in create_actions and
        _update_batch_action_text's label swap), which is what makes
        "trigger it again" a correct description of how to cancel.

        Runs ONE image at a time (see _batch_step): load image i's QImage on
        this (UI) thread, hand it to predict_async, and only step to i+1 once
        that request's result comes back through on_prediction_ready /
        on_prediction_failed. The single-worker pool would serialise a flood
        of requests anyway; doing it this way instead of dispatching all N up
        front means at most one image's pixel data is ever resident, and --
        the actual point -- this method returns to the Qt event loop after
        every single step, so a folder of hundreds of images never blocks the
        UI thread inside a loop.
        """
        if self._batch_active:
            return self.cancel_batch_scoring()
        if not self.is_available():
            self.app.status(self._unavailable_hint())
            return False

        # A snapshot, not a live view of self.app.m_img_list: the run must
        # finish scoring exactly the images that existed when it started,
        # even if a sort/restore got through (refresh_actions disables those
        # while a batch is active, but this still makes the invariant
        # explicit rather than relying only on that gate) or a classify move
        # slips one out from under it (handled below -- a missing file scores
        # as maximally uncertain rather than wedging the run).
        images = list(self.app.m_img_list)
        if not images:
            self.app.status('Open a directory with images before scoring.')
            return False

        self._batch_snapshot = images
        self._batch_index = 0
        self._batch_total = len(images)
        self._batch_scored = 0
        self._batch_current_path = None
        self._batch_active = True
        self.refresh_actions()  # disables auto-label; flips this action's label
        self._batch_step()
        return True

    def cancel_batch_scoring(self, _value=False):
        """Stop a running batch scan cleanly. Scores already recorded for
        images processed so far are KEPT (a partial scan is still useful:
        Sort by Uncertainty happily sorts scored images first and pushes the
        rest -- which includes every image the cancelled run never reached --
        to the end, exactly like it does for a never-scored folder)."""
        if not self._batch_active:
            return False
        scored, total = self._batch_scored, self._batch_total
        self._batch_active = False
        self._batch_current_path = None
        self._batch_snapshot = []
        self._batch_index = 0
        self.app.status('Uncertainty scoring cancelled (%d/%d image(s) scored).'
                        % (scored, total))
        self.refresh_file_list()
        self.refresh_actions()
        return True

    def _batch_step(self):
        """Dispatch exactly one request: the batch's own progress -- not the
        event loop, not a callback queue -- is what drives the next step,
        via _advance_batch being the only caller of this method besides
        score_folder's initial kick."""
        if not self._batch_active:
            return
        if self._batch_index >= len(self._batch_snapshot):
            self._finish_batch()
            return

        path = self._batch_snapshot[self._batch_index]
        self._batch_current_path = path
        self.app.status('Scoring %d/%d...' % (self._batch_index + 1, self._batch_total))

        image = self._load_model_image(path)
        if image is None:
            # Unreadable/vanished file (e.g. removed by a classify move that
            # ran concurrently with the scan): record it as maximally
            # uncertain rather than silently skipping it or stalling the run
            # waiting on a request that was never dispatched -- consistent
            # with least_confidence's own "could not tell = worth a look"
            # stance on empty detections.
            #
            # CRASH FIX: this branch reaches _advance_batch DIRECTLY, in the
            # SAME stack frame as this very call to _batch_step -- unlike
            # on_prediction_ready/on_prediction_failed below, which only ever
            # run from a (queued) Qt signal, i.e. a FRESH stack frame per
            # result, there is no such boundary here. _advance_batch's own
            # tail call is what steps to the next image, so a run of many
            # CONSECUTIVE unreadable/missing files would otherwise chain
            # _batch_step -> _advance_batch -> _batch_step -> ... in one
            # unbroken synchronous call stack -- never returning to the
            # event loop the way the async predict path always does --
            # until Python's recursion limit raises RecursionError and takes
            # the whole batch (and potentially the app) down with it.
            # synchronous=False defers only the NEXT _batch_step() call via
            # QTimer.singleShot(0, ...), which posts it as a fresh event-loop
            # iteration instead of a nested call, breaking the chain while
            # still processing images strictly in order on the UI thread.
            self._advance_batch(path, 1.0, synchronous=False)
            return

        # predict_async may resolve SYNCHRONOUSLY -- the tests' executor runs
        # inline, and predict_async itself fails synchronously for a None
        # image/backend -- or ASYNCHRONOUSLY (the production thread pool,
        # via a queued signal). Either way _advance_batch is only ever called
        # from on_prediction_ready/on_prediction_failed, never from here, so
        # a synchronous resolution cannot double-advance the queue.
        self._dispatch_request('batch', path, image)

    def _advance_batch(self, path, score, synchronous=True):
        """Record one image's score and step to the next.

        ``synchronous=False`` (used only by _batch_step's load-failure
        branch -- see its comment) defers the next ``_batch_step()`` call to
        a fresh Qt event-loop iteration via ``QTimer.singleShot(0, ...)``
        instead of calling it inline, which is what stops a run of
        consecutive failures from recursing the call stack without bound.
        The default (``True``, used by on_prediction_ready/on_prediction_failed,
        which only ever run from a queued Qt signal -- already a fresh stack
        frame per result) is unchanged: those call sites cannot recurse this
        way, and posting them through the event loop too would only add a
        pointless extra round trip per image.
        """
        self._uncertainty[path] = score
        self._invalidate_ranks()
        self._batch_scored += 1
        self._batch_index += 1
        self._batch_current_path = None
        self._update_batch_action_text()
        if synchronous:
            self._batch_step()
        else:
            QTimer.singleShot(0, self._batch_step)

    def _finish_batch(self):
        scored = self._batch_scored
        self._batch_active = False
        self._batch_current_path = None
        self._batch_snapshot = []
        self._batch_index = 0
        self.app.status('Scored %d image(s) for uncertainty -- Sort by Uncertainty '
                        'is ready.' % scored)
        self.refresh_file_list()
        self.refresh_actions()

    def _load_model_image(self, path):
        """Load one image from disk and convert it exactly like the
        interactive path does (to_model_image), ON THE UI THREAD -- QImage
        construction is a Qt operation and must never happen on the worker
        (see libs/inference/service.py's module docstring).

        Reimplements labelImg.read() (QImageReader + setAutoTransform, NOT a
        bare QImage(path)) rather than importing it: labelImg.py imports this
        module at load time, so `from labelImg import read` here would be a
        circular import. Using the same loader (auto EXIF transform included)
        matters for more than cosmetics -- a real detector's confidence can
        depend on orientation, so a batch score must be produced from the
        same pixels the interactive path would have scored.
        """
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        qimage = reader.read()
        if qimage is None or qimage.isNull():
            return None
        return to_model_image(qimage)

    def _update_batch_action_text(self):
        if self.action_score_folder is None:
            return
        if self._batch_active:
            self.action_score_folder.setText(
                'Cancel Scoring (%d/%d)' % (self._batch_scored, self._batch_total))
        else:
            self.action_score_folder.setText('Score Folder for Active Learning')

    # -- reordering ----------------------------------------------------------

    def sort_by_uncertainty(self, _value=False):
        """Reorder ``self.app.m_img_list`` DESCENDING by stored uncertainty:
        most uncertain (highest score) first. Unscored images (never reached
        by a scan, or added since) sort to the END, in their existing
        relative order -- they are not dropped, just deprioritised, since a
        partial/cancelled scan is still a useful ordering for the images it
        did reach.
        """
        if self._batch_active:
            self.app.status('Wait for uncertainty scoring to finish (or cancel it) '
                            'before sorting.')
            return False
        if not self._uncertainty:
            self.app.status('Score the folder first (AI > Score Folder for '
                            'Active Learning).')
            return False

        self._reorder(self._sort_key)
        self._sort_active = True
        unscored = sum(1 for p in self.app.m_img_list if p not in self._uncertainty)
        self.app.status('Sorted %d image(s) by uncertainty, most uncertain first '
                        '(%d unscored at the end).'
                        % (len(self.app.m_img_list), unscored))
        return True

    def _sort_key(self, path):
        """DESCENDING uncertainty (most uncertain / highest score first);
        unscored images sort after every scored one. Shared by
        sort_by_uncertainty and reapply_sort_if_active so a same-directory
        rescan re-derives exactly the ordering a fresh sort would produce.
        """
        score = self._uncertainty.get(path)
        if score is None:
            return (1, 0.0)
        return (0, -score)

    def restore_original_order(self, _value=False):
        """Undo Sort by Uncertainty: put m_img_list back in the order
        scan_all_images produced it in (natural filesystem order), captured
        by on_directory_scanned at the same moment _uncertainty was cleared.
        """
        if self._batch_active:
            self.app.status('Wait for uncertainty scoring to finish (or cancel it) '
                            'before restoring order.')
            return False
        if self._original_order is None:
            self.app.status('Nothing to restore.')
            return False

        order_index = {path: i for i, path in enumerate(self._original_order)}
        # A path absent from the snapshot should not happen (the snapshot is
        # taken from the very scan that produced the current file set), but
        # sorting it to the end rather than raising keeps this from crashing
        # on a state this controller does not fully control (m_img_list is
        # MainWindow's).
        self._reorder(lambda path: order_index.get(path, len(order_index)))
        self._sort_active = False
        self.app.status('Restored filesystem order.')
        return True

    def reapply_sort_if_active(self):
        """Called by MainWindow.import_dir_images right after a SAME-directory
        refresh (``reset_active_learning=False``) has just replaced
        m_img_list with a fresh natural-order scan (scan_all_images always
        runs; only the score/order RESET is conditional -- see that method).

        That rescan is unavoidable -- classify_current_image/delete_image/
        undo_classify call it because a file genuinely left or returned to
        the folder -- but if the user had sorted by uncertainty, letting it
        stand as plain filesystem order would silently flip the queue back
        after the very first g/b press, which defeats the entire feature:
        the point is to keep triaging a SORTED queue with g/b, not to sort it
        once and lose the order on the first move. Re-running the same key
        sort_by_uncertainty uses restores exactly that order for whatever
        images remain (a newly-appeared file, if any, is unscored and sorts
        to the end, same as any other unscored image).

        A no-op when no sort is active (the common case): plain filesystem
        order from the scan is left standing, which is what
        on_directory_scanned's reset already expects for a fresh folder.

        Invalidates the rank cache UNCONDITIONALLY, before the no-op check
        above: scan_all_images has just replaced m_img_list's MEMBERSHIP
        (an image may have left via classify/delete, or returned via undo),
        and _ranks()/refresh_file_list's scored-count now compute over the
        INTERSECTION of _uncertainty and m_img_list -- so a stale cache
        built from the old membership would keep showing a rank/total that
        includes an image no longer in the folder, or omits one that just
        came back, even though _uncertainty itself did not change (which is
        the only thing _invalidate_ranks is normally called for).
        """
        self._invalidate_ranks()
        if not self._sort_active:
            return
        self.app.m_img_list.sort(key=self._sort_key)

    def _reorder(self, sort_key):
        """Reorder ``self.app.m_img_list`` IN PLACE by ``sort_key``, then
        repair every piece of state that indexes into it by POSITION.

        cur_img_idx is the one that matters: it is stored as a position, but
        what it must keep meaning is "the image currently on screen" -- an
        IDENTITY, which moves to a different position by definition whenever
        the order changes. Re-deriving it by looking the open file's path
        back up in the new order (rather than leaving the old integer
        sitting there, now pointing at whatever image happens to occupy that
        slot next) is what stops a reorder from silently swapping the canvas
        to a different image than the one the user is looking at. Every
        caller of this method (sort_by_uncertainty, restore_original_order)
        goes through it for exactly this reason -- there is no other path
        that reorders m_img_list.

        file_list_widget is rebuilt from the new order immediately after, so
        widget row i keeps mirroring m_img_list[i] -- the invariant
        file_item_double_clicked, load_file's selection highlight, and
        classify/save/navigation's index lookups all rely on.
        """
        current_path = self.app.file_path
        self.app.m_img_list.sort(key=sort_key)
        if current_path and current_path in self.app.m_img_list:
            self.app.cur_img_idx = self.app.m_img_list.index(current_path)
        self.refresh_file_list()
        self.refresh_actions()

    def on_directory_scanned(self):
        """Called by MainWindow.import_dir_images for a GENUINE (re)scan of a
        directory's contents (``reset_active_learning=True``, the default) --
        NOT for the same-directory refreshes that follow a classify move /
        delete / undo (those pass ``reset_active_learning=False``). Those
        refreshes reuse import_dir_images purely to re-derive m_img_list
        after a file left or returned to the folder; resetting the triage
        order on every one of them would defeat the feature's entire point,
        since g/b (classify_current_image) IS that refresh's caller and is
        also the normal way a user works through a sorted queue.

        A genuinely new folder's images share nothing with the old scores
        (different files, quite possibly different content at the same
        name), so both the score map and the "what is filesystem order"
        snapshot are invalidated here. Any batch run in flight was walking
        the OLD self.app.m_img_list snapshot, which import_dir_images has
        just replaced out from under it -- cancel it rather than let it
        silently keep scoring paths that may no longer be in this folder at
        all.
        """
        self.cancel_batch_scoring()
        self._uncertainty = {}
        self._invalidate_ranks()
        self._original_order = list(self.app.m_img_list)
        self._sort_active = False
        self.refresh_file_list()
        self.refresh_actions()

    # -- file list display -----------------------------------------------

    def create_status_widget(self):
        """A permanent status-bar label (MainWindow adds it via
        ``statusBar().addPermanentWidget``, next to the existing coordinate
        label) showing the CURRENT image's uncertainty rank/score.
        Permanent, not ``app.status()``'s transient message area: a
        transient message would be stomped by the very next status() call
        (e.g. load_file's "Loaded ..."), but this needs to survive
        navigation -- refresh_actions keeps it in sync on every image load
        (see _update_score_label).
        """
        self._score_label = QLabel('')
        return self._score_label

    def refresh_file_list(self):
        """(Re)build ``self.app.file_list_widget`` from ``self.app.m_img_list``,
        in order, annotating each row with its uncertainty rank/score (once
        scored) as both a text suffix and a heat-scale background colour.

        Always clears and repopulates the WHOLE widget rather than patching
        individual rows in place: row i must keep mirroring m_img_list[i] for
        every other piece of navigation, and a full rebuild is the only way
        that is guaranteed after m_img_list has been reordered.

        Repopulating clears the widget's selection, so the currently open
        image's row is reselected afterwards -- reusing the exact mechanism
        load_file's own highlight uses (``item.setSelected(True)`` on the
        row at ``m_img_list.index(file_path)``). Without this, a row
        selection set by load_file (or file_item_double_clicked) BEFORE this
        rebuild runs is simply thrown away by ``widget.clear()``, and
        nothing sets it again -- exactly what happened on a genuine Open
        Directory: import_dir_images clears the widget, then open_next_image
        (-> load_file) sets the highlight while the widget is still EMPTY
        (refresh_file_list has not repopulated it yet), so the highlight is
        set on nothing and lost the moment this method runs.
        """
        widget = getattr(self.app, 'file_list_widget', None)
        if widget is None:  # pragma: no cover - defensive; always set by MainWindow
            return
        widget.clear()
        ranks = self._ranks()
        # Total among SCORED images still present in the folder, not raw
        # len(_uncertainty): an entry for a path that left m_img_list
        # (classify-out) is deliberately kept in _uncertainty so undo can
        # restore its score (see _uncertainty's own comment), but it must
        # not keep inflating the "scored N" count or leave a rank number
        # referencing an image no longer in the list -- _ranks() already
        # excludes it, so len(ranks) is exactly the present-and-scored count.
        total_scored = len(ranks)
        for image_path in self.app.m_img_list:
            item = QListWidgetItem(self._list_item_text(image_path, ranks, total_scored))
            color = self._heat_color(image_path)
            if color is not None:
                item.setBackground(color)
            widget.addItem(item)

        current_path = getattr(self.app, 'file_path', None)
        if current_path and current_path in self.app.m_img_list:
            index = self.app.m_img_list.index(current_path)
            current_item = widget.item(index)
            if current_item is not None:
                current_item.setSelected(True)

    def _list_item_text(self, image_path, ranks, total_scored):
        """Item text stays the bare path when unscored (this is what
        file_item_double_clicked's row lookup expects to display, and what a
        user would search the list by); once scored it gets a rank/score
        suffix so the priority order is legible without opening every image.
        """
        score = self._uncertainty.get(image_path)
        if score is None:
            return image_path
        return '%s   [#%d/%d, uncertainty %.2f]' % (
            image_path, ranks[image_path], total_scored, score)

    def _heat_color(self, image_path):
        """Pale green (confident / low uncertainty) -> pale red (uncertain /
        high), so the review queue's priority is visible at a glance without
        reading numbers. ``None`` for an unscored image: no tint -- it just
        sorts to the end, it is not "confident".

        Both ends of the scale are kept pale (every channel >= 135) so the
        default (black) item text stays legible regardless of the desktop's
        light/dark widget style -- this is a native Qt list, not a themed web
        page, so there is no dark-mode stylesheet to coordinate with.
        """
        score = self._uncertainty.get(image_path)
        if score is None:
            return None
        score = min(1.0, max(0.0, float(score)))
        red = int(135 + 120 * score)
        green = int(255 - 120 * score)
        blue = 135
        return QColor(red, green, blue)

    def _ranks(self):
        """path -> 1-based rank among SCORED images STILL PRESENT in
        ``self.app.m_img_list``, most uncertain first.

        Deliberately excludes any ``_uncertainty`` entry whose path has left
        m_img_list (classify-out, delete): those entries are kept in
        ``_uncertainty`` on purpose, so undo_classify can restore the exact
        old score for an image that comes back (see ``_uncertainty``'s own
        comment) -- but until an image is actually back in the folder, it
        must not count toward the rank/total shown to the user, or the
        displayed numbers drift from what is really in the file list.

        One sort for the whole file-list refresh / status-label update
        (cached until invalidated). Invalidated both when _uncertainty
        itself changes (_invalidate_ranks, e.g. a new score recorded) AND
        whenever m_img_list's MEMBERSHIP may have changed without
        _uncertainty changing (reapply_sort_if_active invalidates
        unconditionally for exactly this reason) -- either one can change
        which entries belong in this intersection.
        """
        if self._ranks_cache is None:
            present = set(self.app.m_img_list)
            ordered = sorted(
                ((path, score) for path, score in self._uncertainty.items()
                 if path in present),
                key=lambda kv: -kv[1])
            self._ranks_cache = {path: i for i, (path, _score) in enumerate(ordered, start=1)}
        return self._ranks_cache

    def _invalidate_ranks(self):
        self._ranks_cache = None

    def _update_score_label(self):
        """Keep the permanent status-bar label (create_status_widget) in
        sync with whatever image is open right now. Called from
        refresh_actions, which already runs on every navigation
        (MainWindow.toggle_actions -> refresh_actions) and every scoring
        state change.
        """
        if self._score_label is None:
            return
        path = getattr(self.app, 'file_path', None)
        score = self._uncertainty.get(path) if path else None
        if score is None:
            self._score_label.setText('')
            self._score_label.setToolTip('')
            return
        ranks = self._ranks()
        rank = ranks.get(path)
        total = len(ranks)
        self._score_label.setText('Uncertainty %.2f (rank %d/%d)' % (score, rank, total))
        self._score_label.setToolTip(
            'Active-learning uncertainty score for the open image -- '
            'higher means the model is less sure, i.e. more worth reviewing.')
