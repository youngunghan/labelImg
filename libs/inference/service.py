#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Off-the-UI-thread model execution, and the plain-data boundary around it.

This is the ONE Qt-aware module in ``libs/inference``.  The package
``__init__`` does not import it (see its ``_LAZY_EXPORTS``), so ``import
libs.inference`` still costs nothing and the zero-dependency guarantee of the
core -- types / backend ABC / registry -- is untouched.  numpy and onnxruntime
remain absent from this module as well: numpy is used **only if it happens to be
importable**, never required (see ``to_model_image``).

Three constraints shape the design:

**Single worker.**  ``QThreadPool.maxThreadCount == 1``.  An ONNX
``Session.Run`` is not reliably safe to call concurrently on one session, and
CPU inference is already internally parallel -- oversubscribing the box with a
second inference would slow both down while adding a data race.  Requests
therefore queue; they do not race.

**The worker touches no Qt object.**  A ``QImage`` may not be read from a worker
thread while the UI thread is free to repaint/replace it, so the conversion to
model input happens on the UI thread (``to_model_image``) and the worker only
ever sees plain data it owns.  Results come back through signals emitted from
the worker thread to this QObject, which lives in the UI thread: Qt's
``AutoConnection`` therefore delivers them **queued**, so slots run on the UI
thread and may safely create ``Shape`` objects and touch the canvas.

**Every result is tagged with the image it was computed for.**  Inference is
slow enough that the user will navigate away mid-run; the consumer compares the
tag against the app's current file and drops anything stale (a detection dropped
onto the wrong image would be silently written to that image's annotation file).
"""

from __future__ import annotations

import logging

try:
    from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
    from PyQt5.QtGui import QImage
except ImportError:  # pragma: no cover - the app's PyQt4 fallback path
    from PyQt4.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
    from PyQt4.QtGui import QImage

__all__ = [
    'InferenceService',
    'RawImage',
    'to_model_image',
    'ThreadPoolExecutor',
    'SynchronousExecutor',
]

logger = logging.getLogger(__name__)


class RawImage(object):
    """Dependency-free stand-in for an ``HxWx3`` uint8 array.

    The base install has neither numpy nor onnxruntime, but the app must still
    be able to run the stub backend (and any future pure-python backend).  Every
    consumer in the core duck-types on ``.shape`` first (see
    ``libs.inference.stub.image_size``), so exposing ``.shape`` plus the raw RGB
    bytes is enough to satisfy them without dragging in a dependency.

    ``data`` is tightly packed RGB (no row padding) and is a *copy*: the QImage
    it came from stays on the UI thread and may be freed or repainted the moment
    the worker starts.
    """

    __slots__ = ('shape', 'data')

    def __init__(self, height, width, data=b''):
        self.shape = (int(height), int(width), 3)
        self.data = data

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def __repr__(self):  # pragma: no cover - debugging aid
        return '<RawImage %dx%d>' % (self.shape[1], self.shape[0])


def _rgb888_bytes(qimage):
    """(height, width, tightly-packed RGB bytes) from a QImage, on this thread.

    Qt pads each row of an RGB888 image up to a 4-byte boundary, so the raw
    buffer is *not* h*w*3 bytes; the padding is stripped here rather than left
    for every consumer to rediscover.
    """
    image = qimage.convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()

    bits = image.constBits()
    # sizeInBytes() is the modern spelling; byteCount() is the one PyQt5 shipped
    # for years and the one PyQt4 has.
    size = image.sizeInBytes() if hasattr(image, 'sizeInBytes') else image.byteCount()
    bits.setsize(size)
    buffer = bytes(bits)  # copy — the worker must not alias Qt-owned memory

    row_bytes = width * 3
    if stride == row_bytes:
        return height, width, buffer
    packed = b''.join(buffer[y * stride:y * stride + row_bytes] for y in range(height))
    return height, width, packed


def to_model_image(qimage):
    """Convert a QImage to model input **on the calling (UI) thread**.

    Returns a real ``numpy`` HxWx3 uint8 RGB array when numpy is importable (the
    ``[ai]`` extra ships it, and the ONNX backends of Phase 2 need it), and a
    ``RawImage`` carrier otherwise.  Either way the result is plain data the
    worker thread owns outright -- no QImage crosses the thread boundary.

    numpy is imported lazily, inside the function: a top-level import would make
    this module -- and therefore the whole AI menu -- unusable on the base
    install, which is exactly the failure mode the inference package exists to
    avoid.
    """
    if qimage is None or qimage.isNull():
        return None

    height, width, packed = _rgb888_bytes(qimage)

    try:
        import numpy as np
    except ImportError:
        return RawImage(height, width, packed)

    array = np.frombuffer(packed, dtype=np.uint8).reshape(height, width, 3)
    # frombuffer is read-only and aliases `packed`; backends expect to own (and
    # some will write to) their input.
    return np.ascontiguousarray(array)


class ThreadPoolExecutor(object):
    """Default executor: one worker thread, requests queued behind each other."""

    def __init__(self, parent=None):
        self._pool = QThreadPool(parent) if parent is not None else QThreadPool()
        # See module docstring: one session, one worker. Not a tuning knob.
        self._pool.setMaxThreadCount(1)

    def submit(self, job):
        self._pool.start(_Job(job))

    def wait_for_done(self, msecs=30000):
        return self._pool.waitForDone(msecs)


class SynchronousExecutor(object):
    """Runs the job inline, on the calling thread.

    Injected by the tests: it makes inference deterministic and removes any
    dependence on the Qt event loop being pumped, so an assertion can follow the
    call immediately instead of racing a worker thread.  Signals emitted from
    here are delivered directly (same thread), which is exactly what a test
    wants -- and is safe, because the "results must be queued" rule exists to
    protect the UI thread, and here there is no other thread.
    """

    def submit(self, job):
        job()

    def wait_for_done(self, msecs=0):  # pragma: no cover - nothing to wait for
        return True


class _Job(QRunnable):
    """Adapts a plain callable to QRunnable (QThreadPool takes nothing else)."""

    def __init__(self, fn):
        super(_Job, self).__init__()
        self._fn = fn

    def run(self):
        self._fn()


class InferenceService(QObject):
    """Runs a ``ModelBackend`` off the UI thread and reports back by signal.

    Signals carry the image path the request was made for; the consumer is
    responsible for dropping results whose path is no longer the current image
    (it is the only layer that knows what "current" means).
    """

    # (image_path, list[Detection])
    predictionReady = pyqtSignal(str, object)
    # (image_path, human-readable reason)
    predictionFailed = pyqtSignal(str, str)

    def __init__(self, parent=None, backend=None, executor=None):
        super(InferenceService, self).__init__(parent)
        self._backend = backend
        self._executor = executor if executor is not None else ThreadPoolExecutor(self)
        # Per-image encoder output, keyed by image path. Interactive segmenters
        # (SAM-like) split into an expensive per-image encode and a cheap
        # per-click decode; the slot exists now so the click path has somewhere
        # to look, but nothing populates it until that backend lands.
        self._embeddings = {}

    # -- backend -----------------------------------------------------------

    def backend(self):
        return self._backend

    def set_backend(self, backend):
        """Swap the model. Invalidates the embedding cache (it belonged to the
        old model's encoder and means nothing to a new one)."""
        if backend is self._backend:
            return
        self._backend = backend
        self._embeddings.clear()

    def is_available(self):
        return self._backend is not None

    # -- executor ----------------------------------------------------------

    def set_executor(self, executor):
        """Swap how jobs are run. The tests inject SynchronousExecutor so that
        inference is deterministic and does not depend on the event loop being
        pumped; production keeps the single-worker thread pool."""
        self._executor = executor

    # -- embedding cache ---------------------------------------------------

    def embedding(self, image_path):
        return self._embeddings.get(image_path)

    def set_embedding(self, image_path, embedding):
        self._embeddings[image_path] = embedding

    def clear_embeddings(self):
        self._embeddings.clear()

    # -- inference ---------------------------------------------------------

    def predict_async(self, image_path, image):
        """Queue a detection run for ``image``, tagged with ``image_path``.

        ``image`` must already be plain data (``to_model_image``): this call is
        made from the UI thread, and whatever it is handed is what the worker
        thread will read.  Returns False when there is nothing to run.
        """
        if self._backend is None:
            self.predictionFailed.emit(image_path or '', 'No model backend is loaded.')
            return False
        if image is None:
            self.predictionFailed.emit(image_path or '', 'No image to run the model on.')
            return False

        backend = self._backend

        def job():
            # Worker thread from here on: no Qt objects, no self.* mutation.
            # Both emits cross back to the UI thread as queued signals because
            # this QObject lives there.
            try:
                detections = backend.predict(image)
            except Exception as exc:  # noqa: BLE001 - a bad model must not kill the app
                logger.warning('Inference failed for %s: %s', image_path, exc)
                self.predictionFailed.emit(image_path or '', '%s' % exc)
                return
            self.predictionReady.emit(image_path or '', list(detections))

        self._executor.submit(job)
        return True

    def wait_for_done(self, msecs=30000):
        """Block until queued work drains (shutdown / tests). No-op for the
        synchronous executor, which has already finished by the time it returns."""
        return self._executor.wait_for_done(msecs)
