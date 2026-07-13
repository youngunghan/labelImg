#!/usr/bin/env python
# -*- coding: utf8 -*-
"""The UI half of the ML-assist spine: where model output becomes editable boxes.

``libs/inference`` knows nothing about Qt or ``Shape``; ``libs/shape`` knows
nothing about ``Detection``.  This package is the only place the two meet, which
is what keeps the model layer testable with no GUI and the GUI free of any
model-specific branching:

    libs.inference.Detection --(suggestion.py)--> libs.shape.Shape (provisional)
    MainWindow --(controller.py)--> InferenceService --> ModelBackend

``controller.py`` holds the behaviour (actions, threshold, accept/reject) so that
``labelImg.py`` only has to wire it up -- the god-object does not grow an AI
feature, it grows one attribute.

Nothing here is imported by ``libs.inference``; the dependency runs one way
only (assist -> inference), so the zero-dependency core stays importable on a
machine with no PyQt at all.
"""
