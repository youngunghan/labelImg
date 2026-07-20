# Fork changes vs upstream

- **Upstream**: [HumanSignal/labelImg](https://github.com/HumanSignal/labelImg) @ `b33f965` — archived February 2024 (read-only), so these changes cannot be merged back.
- **This fork**: [youngunghan/labelImg](https://github.com/youngunghan/labelImg) — maintained independently. Latest snapshot: [`v1.8.6-fork.3`](https://github.com/youngunghan/labelImg/releases/tag/v1.8.6-fork.3) — the first release to include the AI-assist work (Phase 1–4 + COCO).

## At a glance

| Metric | Value |
|---|---|
| Diff vs upstream | 63 files, **+12,641 / −114** |
| Core app (`labelImg.py`) | +754 / −34 |
| New documentation | `docs/` tree: 21 files, ~2,050 lines (Diátaxis: tutorials / how-to / reference / explanation) |
| Packaging | reproducible PyInstaller `labelImg.spec` (SPECPATH-anchored, bundles `data/`); optional `ai` extra (`pip install -e ".[ai]"` from this checkout — not published to PyPI under the `labelImg` name) |
| Tests | **280/280 passing**, 14 files (up from 30/8) — dependency-requiring tests SKIP on the base install rather than erroring |
| Upstream bugs fixed | 6 crash / silent-failure / data-integrity defects (see table) |
| CI | GitHub Actions: 3 jobs — core test matrix (Linux/Windows × Py3.9/3.12, headless Qt, base install only), `test-ai` (base + `[ai]` extra, exercises the ONNX-dependent tests), ruff critical-rules lint |

## Features added

### AI-assisted auto-labeling (`Ctrl+I` / `Ctrl+Return` / `Ctrl+Backspace`)

Upstream labelImg has no model-in-the-loop assist at all — every box is drawn by hand.
This fork adds one, behind a seam designed so the UI never has to know which model is
underneath (`libs/inference/backend.py`'s `ModelBackend` ABC — `predict`/`segment`/`embed`
plus capability flags):

- **`libs/inference/`** is a zero-dependency core: `types.py` (`Detection`, `Mask`,
  `SegPrompt`, `Prediction`, `least_confidence` for future active learning),
  `backend.py`, `stub.py` (a deterministic `StubBackend` that drives most of the test
  suite with no model file and no `onnxruntime`), `registry.py` (`build_backend`).
  `import libs.inference` pulls in no numpy, no onnxruntime, no Qt.
- **`libs/inference/yolo_onnx.py`** — the first real backend: ONNX YOLOv5/v8 detection
  via `onnxruntime`. Aspect-preserving letterbox preprocessing with whole-pixel pads (a
  single `_letterbox_geometry()` feeds both the forward paste and the inverse mapping,
  so boxes land back on original image pixels exactly — see the bug table below for what
  happens when forward/inverse compute this twice). Autodetects the YOLOv5
  `(1,N,5+nc)` and YOLOv8 `(1,4+nc,N)` export layouts from the output shape and **fails
  loudly** rather than guessing when a shape is ambiguous. Class-aware NMS (a "person"
  and the backpack they're wearing don't suppress each other). Pipeline order is
  decode → confidence filter → NMS → inverse-letterbox, with NMS run in model-input
  space on purpose.
- **`libs/inference/service.py`** — `InferenceService` runs the model on a single-worker
  `QThreadPool` so a slow prediction never blocks the UI thread; workers see only plain
  data (no `QImage` crosses the thread boundary), results come back as queued Qt
  signals, and every result is tagged with the image path it was computed for so
  navigating away mid-inference can't deposit boxes on the wrong image.
- **`libs/assist/`** — `AssistController`, the one object `MainWindow` delegates AI
  behaviour to (construction, menu wiring, and one filter at the save choke point is
  all `MainWindow` itself does). A new **AI menu** adds *Auto-label Image* (`Ctrl+I`),
  *Accept All* (`Ctrl+Return`), *Reject All* (`Ctrl+Backspace`), and a confidence
  threshold slider that **re-filters on-screen suggestions without re-running the
  model** (the backend is asked for everything at `conf_threshold=0.0`; the slider is a
  pure display filter).
- **`libs/shape.py`** gained `provisional` / `confidence` / `shape_type`. A
  model's guess is drawn dashed and translucent and is **never written to disk** — one
  filter at the single save choke point (`save_labels`, `labelImg.py:1054`) drops every
  provisional shape, so accepting one is exactly "clear the flag" and every existing
  VOC/YOLO/CreateML/COCO writer stays byte-for-byte the same for real annotations.
- **No model weights ship with this fork.** Ultralytics YOLOv5/v8 weights are
  AGPL-3.0 and would conflict with this MIT app (see `data/models/README.md` for the
  full reasoning and permissive alternatives such as YOLOX/Apache-2.0). The user points
  the model-path setting at their own `.onnx` file. `pip install -e ".[ai]"` (run from
  this checkout — this fork is not on PyPI, so `pip install labelImg[ai]` would fetch
  the unrelated upstream package) pulls in
  `onnxruntime>=1.15` + `numpy`; the base install stays `pyqt5`+`lxml` and ships with
  no backend configured at all, so the AI menu stays disabled (with an install/configure
  hint) until both the extra is installed and a model backend/path are set — installing
  the extra by itself is not enough.

### COCO import/export (dataset-level lane)

The other most-requested gap: upstream ships VOC, YOLO and CreateML but no COCO. COCO
doesn't fit the per-image-sidecar model the other three formats use — **a COCO dataset
is one JSON covering many images** — so it's wired in as its own lane rather than a
fourth `<stem>.json` writer:

- `libs/coco_io.py` — `COCOReader`/`COCOWriter`. The default dataset target is
  `<save dir>/annotations.json`; explicit *Import COCO...* / *Export COCO...*
  File-menu commands let you point at a different file. Saving does a
  read-modify-write merge: the current image's `images[]`/`annotations[]` entries are
  replaced in place, every other image's entries are left untouched.
- Images are keyed by their path **relative to the dataset JSON**, not by bare
  basename — `train/0001.jpg` and `val/0001.jpg` are recognised as different images
  even though labelImg's recursive directory scan can put both in view at once (see
  the bug table below for what basename-keying used to do).
- `bbox` is `[x, y, w, h]`, `area = w*h`, `iscrowd = 0`; `category_id`↔name mapping is
  stable across merges (existing ids are never renumbered, so re-saving doesn't
  relabel annotations already in the file).
- Both COCO and CreateML use the `.json` extension, so every load site content-sniffs
  before picking a reader: a top-level dict with `images`/`annotations`/`categories`
  is COCO, a top-level list is CreateML.
- COCO has no schema slot for `difficult` or `verified` — both are dropped on write
  and read back `False`; the app posts a status-bar note when this happens rather than
  losing the flag silently.
- COCO is **not** wired into per-image autosave in the sense of writing a
  `<stem>.json` sidecar — it never does that — but `Ctrl+S`/Save As/next-prev
  autosave/verify all merge into the same dataset file, so "I saved but the box isn't
  there" can't happen either.

### Good/Bad image triage (`g` / `b` / `Ctrl+Z`) — ~200 LOC

The core addition. Press `g` or `b` to move the current image **and its label file**
(`.xml`/`.txt`/`.json`; save-dir first, image folder as fallback) into a
`<folder>_good` / `<folder>_bad` sibling directory and advance to the next image.

- **Atomic**: if any label move fails, everything already moved is rolled back and
  freshly-created empty target folders are cleaned up — an image is never separated
  from its label.
- **Undoable**: `Ctrl+Z` pops a move-set from a history stack and restores the files;
  a partially failed undo re-queues itself so retrying self-heals.
- **Safe**: name collisions are resolved by `stem_N` renaming; a standalone image
  opened via *Open File* is guarded from being mis-moved relative to a stale directory.

### In-app class editing (`Ctrl+Shift+E`) — ~60 LOC

*File → Edit Default Classes* opens a multiline editor for the predefined class list and
persists it. The label dialog (`w`) and the default-label combo refresh immediately —
no restart. For the packaged exe (where the bundled file lives in a temp dir and is
lost on exit) the list is bootstrapped to a writable file **next to the executable**.

### Command-line save dir respected at startup

`labelImg.py <image_dir> [class_file] [save_dir]` now behaves as documented: the
directory passed on the command line wins over the remembered `lastOpenDir`, the CLI
save dir survives startup, a cancelled directory dialog returns early, and the first
image is no longer imported (and its boxes loaded) twice.

### User-defined triage categories (2026-07-08)

good/bad are just the defaults: *File > Edit Classify Categories* accepts any
number of `<shortcut> <folder-name>` pairs (e.g. ``k keep`` / ``x trash``),
rebuilds the menu and shortcuts live, and persists them in the settings
pickle. Atomicity and undo are category-agnostic.

### Shortcut conflict fix

Single Class Mode moved from `Ctrl+Shift+S` — which upstream double-bound to
*Save As* — to `Ctrl+Shift+C`.

## Bugs fixed (upstream defects)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `tools/label_to_csv.py` crashes with `NotADirectoryError` when a stray file sits in a set folder | inner loop's `isdir` check re-tested the **parent** dir (always true) instead of the child | test the child path; stray files are skipped |
| 2 | *Copy Previous Bounding Boxes* crashes with `ValueError` on a standalone (Open File) image | unguarded `m_img_list.index(file_path)` | membership guard, quiet no-op |
| 3 | Saving with an unsupported format silently did nothing | `else` branch called the commented-out `LabelFile.save` → would `AttributeError` | raise `LabelFileError`, caught by the existing handler → clear error dialog |
| 4 | Opening a YOLO folder crashed on a missing `classes.txt`, a malformed line, or an out-of-range class index | `YoloReader` had no error handling (`try/except` left commented out) | clear error dialog (`YoloParseError`), malformed lines skipped and counted, status-bar notice |
| 5 | Multi-image CreateML files showed another image's *verified* badge | reader always took `verified` from the first JSON entry | read it from the entry matching the current image |
| 6 | Non-ASCII labels could mangle or crash on locale-encoded systems (e.g. cp949) | `classes.txt` and CreateML JSON were read/written with the OS default encoding | all six I/O call sites fixed to UTF-8 |

## Defects caught by cross-engine review before merge

The AI-assist work (COCO IO + `libs/inference/` + `libs/assist/`) went through a
second-engine (Codex/GPT) review pass before each phase merged. Listed here — not
because these are upstream bugs, but because they're a fair sample of what that review
process is actually worth catching:

| # | Symptom | Root cause | Fix | Where |
|---|---|---|---|---|
| 1 | A plain "Delete RectBox" on an AI suggestion, followed by a threshold-slider move, aborts the app with an unhandled `ValueError` inside a Qt slot | `AssistController` tracked which shapes were suggestions in its own dict, treating it as ground truth for "what's on the canvas"; an ordinary delete left a stale reference that the next re-filter tried to remove a second time | `provisional_shapes()` now reads the canvas directly instead of the tracking dict; every shape removal (not just ones the controller itself triggered) is reported back via `discard_shape` | `libs/assist/controller.py`, `a32acd3` |
| 2 | Two images sharing a basename in different subfolders (`train/0001.jpg`, `val/0001.jpg`) silently cross-contaminate each other's boxes in the shared COCO dataset json | `images[]` entries were keyed on bare basename; labelImg scans directories recursively, so both images collapsed onto one dataset entry and their annotations were unioned on read | images are keyed on the path relative to the dataset json (`dataset_relative_name`), with a guarded one-candidate-only migration for datasets written by other tools | `libs/coco_io.py`, `a32acd3` |
| 3 | Every detected box lands ~0.6–0.8px off in the original image on any image whose letterbox padding is an odd number of pixels | the forward preprocessing rounded the pad to a whole pixel for the actual array paste, while the inverse mapping subtracted the unrounded float pad — a half-model-pixel bias that a self-consistency round-trip test couldn't catch (it only checked the inverse against itself, never against the numpy paste) | forward paste and inverse mapping now derive from one `_letterbox_geometry()` call, so they cannot disagree; new tests read the paste offset off the actual tensor | `libs/inference/yolo_onnx.py`, `c2ecf8e` |
| 4 | `DEFAULT_BACKEND` was fixed from `'stub'` to `None` (fresh installs no longer light up the AI menu with `StubBackend`'s fabricated, image-dimension-derived boxes) — but anyone who had run an earlier build of this branch and closed it even once now has `'stub'` sitting explicitly in `~/.labelImgSettings.pkl`, because `closeEvent` wrote `self.assist.backend_name` unconditionally on every close. `AssistController.__init__` read that persisted value back and rebuilt `StubBackend`, completely bypassing the `DEFAULT_BACKEND=None` fix for exactly the users it was meant to protect | (a) the fix addressed the default, not the settings file it had already written; (b) `closeEvent` never distinguished "nothing configured" from "the constructor's own fallback value" when persisting | two-part fix: `AssistController.__init__` now treats a persisted `'stub'` as unset at read time, permanently (`_LEGACY_IMPLICIT_DEFAULT_BACKEND`, robust across repeated save/load cycles since it re-checks on every construction); `closeEvent` now only writes `SETTING_MODEL_BACKEND` when a backend was actually configured, and drops a stale key instead of leaving it, so this class of bug cannot recur | `libs/assist/controller.py`, `labelImg.py` |
| 5 | Scoring a folder while a slow (Ctrl+I) interactive request for the same, currently-open image was still outstanding could swap the two results: the interactive prediction got consumed as the batch's own progress, and the batch's real result for that image leaked into the interactive suggestion flow instead | routing a result to "interactive" vs. "batch" by comparing its image path against `_batch_current_path` breaks the instant both flows target the same path at once — `InferenceService`'s single-worker pool queues rather than rejects the second request | every `predict_async` dispatch is now tagged 'interactive'/'batch' in a FIFO queue at submit time (`_dispatch_request`/`_pop_request_kind`), so results are attributed by **submission order**, not by path | `libs/assist/controller.py:504-555` |
| 6 | A folder of many consecutive unreadable/vanished images could crash the whole app with `RecursionError` during Score Folder | the load-failure branch of `_batch_step` recorded the score and called the next `_batch_step()` **inline**, in the same stack frame — unlike the async predict path (always a fresh Qt signal frame), a long run of back-to-back failures chained unboundedly on the call stack | the next step is deferred via `QTimer.singleShot(0, ...)` (`_advance_batch(..., synchronous=False)`), posting it as a new event-loop iteration instead of a nested call | `libs/assist/controller.py:868-942`, regression test `TestBatchLoadFailureDoesNotRecurse` |
| 7 | If every remaining image in the folder got classified away *while* a batch scan was still running, Score Folder — the only control that can cancel a running batch — could grey itself out, leaving the run uncancellable | the action's enabled state was gated on `has_folder` alone; `m_img_list` emptying mid-batch made that `False` even though the batch (walking its own frozen snapshot) was still active | gate widened to `available and (has_folder or batch_running)` | `libs/assist/controller.py:413`, regression test `TestScoreFolderStaysEnabledWhenFolderEmptiesMidBatch` |
| 8 | After a genuine Open Directory, the newly-opened image's row in the file list was not highlighted | `import_dir_images` clears `file_list_widget` before loading the first image, so `load_file`'s own highlight attempt ran against an still-empty widget; nothing re-selected the row once `refresh_file_list` repopulated it afterwards | `refresh_file_list` re-selects the current image's row itself, after repopulating | `libs/assist/controller.py:1203-1208`, regression test `TestFileListSelectionAfterOpenDirectory` |
| 9 | The displayed uncertainty rank / "scored N" total kept counting an image after it left the folder (classified out, deleted) | `_ranks()` counted every entry in `_uncertainty`, which deliberately keeps an out-of-folder image's score around so undo can restore it | `_ranks()`/the displayed total now count only the intersection of `_uncertainty` and the current `m_img_list` (the score itself is still retained, not deleted, for undo) | `libs/assist/controller.py:1242-1269`, regression tests `TestRankAndTotalExcludeAbsentImages` |

Two more come from the same review passes and are handled the same way (`load_labels`
appending instead of clearing on `Import COCO...`, and the `setup.py` support floor
disagreeing with the `dataclass`/`from __future__ import annotations` code it declares
support for) — see `a32acd3`'s commit message for the full list.

## Documentation added

Upstream shipped no developer docs. This fork adds a `docs/` tree (Korean, Diátaxis):
1 tutorial, 10 how-to guides (including new COCO-annotation and auto-labeling guides
added alongside this work), 4 reference pages (byte-level format specs, module
inventory, full shortcut table, settings keys), 4 explanations (architecture,
canvas interaction model, annotation-format design, and the ML-assist design doc
added alongside the AI work, kept current through both merged phases), plus an
`llms.txt` routing map. Every factual claim was audited against the code
(multi-agent audit, 2026-07; 34 confirmed corrections applied in the initial pass).

## Roadmap — candidate improvements

Verified against the code (each item was checked at the cited location before listing).
Effort: **S** small / **M** medium / **L** large.

### Robustness

1. ~~**[M] YOLO reader crash-safety**~~ — **done (2026-07-07)**: missing `classes.txt`
   raises `YoloParseError` → error dialog; malformed lines are skipped and counted
   (`libs/yolo_io.py:104-168`, `labelImg.py:2244-2265`), with regression tests.
2. ~~**[S] Consistent text encodings**~~ — **done (2026-07-08)**: all YOLO and
   CreateML I/O call sites now read/write UTF-8; non-ASCII labels round-trip
   (regression-tested with Korean labels).
3. ~~**[S] CreateML `verified` read from the wrong entry**~~ — **done (2026-07-08)**:
   read from the matching entry (`create_ml_io.py:121`), with multi-image tests.
4. **[S] Surface CreateML read errors** — decode failures are swallowed (`ValueError`
   catch) and the image loads as if it had no annotations.
5. **[S] Working *Reset All* restart** — `QProcess.startDetached(__file__)` can't work on
   Windows or in the frozen exe; use `sys.executable`.
6. **[S] Revisit the 1-px minimum-coordinate clamp** — boxes drawn at the image edge are
   silently shifted from 0 to 1 on save (VOC/YOLO path, `libs/labelFile.py:196-203`), a
   2015-era faster-rcnn workaround that degrades edge annotations.

### Triage workflow

7. ~~**[M] User-defined categories**~~ — **done (2026-07-08)**: `(shortcut, name)`
   pairs in settings, edited live via *File > Edit Classify Categories*
   (menu/shortcut rebuild without restart, duplicate-shortcut validation).
8. **[M] Incremental list update on classify** — currently every `g`/`b` rescans the whole
   directory and decodes the image twice; pop from the list instead (also applies to
   undo and delete-image paths).
9. **[L] Batch classify** — multi-select in the file list, classify as one atomic,
   one-`Ctrl+Z` unit.
10. **[S] Honor auto-save on classify** — same save policy as next/prev navigation, so
    edits aren't dropped when triaging with auto-save on.
11. **[S] Status-bar triage stats** — remaining / per-category counts / undo depth.
12. **[M] Persist classify history across sessions** — `Ctrl+Z` after a restart.
13. **[S] i18n for the fork's new actions** — register the four new menu items in
    `stringBundle` so ja/zh locales aren't mixed-language.

### Engineering

14. ~~**[M] CI**~~ — **done (2026-07-07)**: `ci.yml` runs the test suite on
    Linux/Windows × Python 3.9/3.12 (offscreen Qt) plus a ruff critical-rules lint;
    the inherited packaging workflow is bumped to `upload-artifact@v4` and moved to
    manual dispatch.
15. ~~**[M] Tests for the fork features**~~ — **done (2026-07-07)**: `tests/test_classify.py`
    drives a real `MainWindow` (CLI dir import, move+advance, collision rename,
    fault-injected rollback, undo) and `tests/test_yolo_reader.py` covers the YOLO
    hardening; plain `unittest`, no new test dependency.
16. ~~**[M] Release automation**~~ — **partially done (2026-07-08)**: `release.yml`
    builds the Windows exe from `labelImg.spec` on `v*` tag push and attaches
    exe + SHA256 to the GitHub Release. Remaining: `pyproject.toml` packaging
    under a distinct distribution name.
17. **[M] Remove Python 2 / PyQt4 remnants** — dead import fallbacks in 15 files, no-op
    `ustr()` wrappers, `qt4py2` build paths; unblocks ruff/type-checking.
18. ~~**[S] Demo GIF in README**~~ — **done (2026-07-08)**: an app-driven recording
    (g/b classify + Ctrl+Z undo, captured programmatically frame-by-frame) now
    tops the fork section. Badge fixed 2026-07-07. Upstream-hot-linked demo
    images remain to be re-pointed.
19. **[L] Qt6 migration (PySide6)** — after tests and dead-code removal land.

### AI-assisted labeling

20. ~~**[L] COCO import/export**~~ — **done (2026-07-13)**: dataset-level Import/Export
    COCO... lane, content-sniffed against CreateML at every `.json` load site
    (`libs/coco_io.py`, `6b48a38`).
21. ~~**[L] Model-in-the-loop assist**~~ — **done (2026-07-13/14)**: `ModelBackend` seam +
    `InferenceService` + `AssistController` + provisional shapes shipped Phase 1
    (`6b48a38`, hardened in `a32acd3`); the real `YoloOnnxBackend` (ONNX YOLOv5/v8 via
    `onnxruntime`) and the optional `[ai]` extra shipped Phase 2 (`d324e41`, letterbox
    fix `c2ecf8e`).
22. ~~**[M] Auto-label Folder**~~ — **done (2026-07-15, Phase 4)**: `AssistController.score_folder`
    batch-runs the model across every image in `m_img_list`, one at a time (so a large
    folder never blocks the UI thread), and is cancellable mid-run by triggering the
    same action again (`libs/assist/controller.py:792-866`).
23. ~~**[S/M] Active learning (uncertainty-sorted review queue)**~~ — **done (2026-07-15,
    Phase 4)**: `sort_by_uncertainty` (`Ctrl+Shift+U`) reorders `m_img_list` by
    `least_confidence`, most-uncertain-first, so the existing `g`/`b` triage walks the
    reordered stream (`libs/assist/controller.py:994-1017`); `Restore Filesystem Order`
    undoes it. File-list rows show a live rank/score suffix and heat-map tint
    (`refresh_file_list`, `:1161-1208`).
24. **[L] Polygon / keypoint annotation modes** — `Shape.shape_type` exists as a label
    but the rectangle assumption is still load-bearing throughout `Canvas`
    (4-point vertex arithmetic, bbox-reducing writers); this is the most expensive
    remaining rework, not a one-field change.
25. **[L] MobileSAM prompt-based segmentation** — `ModelBackend.segment()`/`embed()` are
    already declared on the ABC and `InferenceService` already has a per-image embedding
    cache slot; no segmentation backend implements them yet.
