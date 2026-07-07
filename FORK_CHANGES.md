# Fork changes vs upstream

- **Upstream**: [HumanSignal/labelImg](https://github.com/HumanSignal/labelImg) @ `b33f965` — archived February 2024 (read-only), so these changes cannot be merged back.
- **This fork**: [youngunghan/labelImg](https://github.com/youngunghan/labelImg) — maintained independently. Latest snapshot: [`v1.8.6-fork.1`](https://github.com/youngunghan/labelImg/releases/tag/v1.8.6-fork.1).

## At a glance

| Metric | Value |
|---|---|
| Diff vs upstream | 32 files, **+2,098 / −71** |
| Core app (`labelImg.py`) | +306 / −19 |
| New documentation | `docs/` tree: 18 files, ~1,200 lines (Diátaxis: tutorials / how-to / reference / explanation) |
| Packaging | reproducible PyInstaller `labelImg.spec` (SPECPATH-anchored, bundles `data/`) |
| Tests | **18/18 passing** — 8 fork tests added (atomic-move rollback via fault injection, undo, YOLO robustness); suite made environment-independent |
| Upstream bugs fixed | 4 crash / silent-failure defects (see table) |
| CI | GitHub Actions: test matrix (Linux/Windows × Py3.9/3.12, headless Qt) + ruff critical-rules lint |

## Features added

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

## Documentation added

Upstream shipped no developer docs. This fork adds a `docs/` tree (Korean, Diátaxis):
1 tutorial, 8 how-to guides, 4 reference pages (byte-level format specs, module
inventory, full shortcut table, settings keys), 3 explanations (architecture,
canvas interaction model, annotation-format design), plus an `llms.txt` routing map.
Every factual claim was audited against the code (multi-agent audit, 2026-07;
34 confirmed corrections applied).

## Roadmap — candidate improvements

Verified against the code (each item was checked at the cited location before listing).
Effort: **S** small / **M** medium / **L** large.

### Robustness

1. ~~**[M] YOLO reader crash-safety**~~ — **done (2026-07-07)**: missing `classes.txt`
   raises `YoloParseError` → error dialog; malformed lines are skipped and counted
   (`libs/yolo_io.py:104-168`, `labelImg.py:1908-1920`), with regression tests.
2. **[S] Consistent text encodings** — annotation `.txt` is written UTF-8 but
   `classes.txt` and CreateML JSON are read/written with the OS locale encoding
   (`yolo_io.py`, `create_ml_io.py`) → mojibake/crashes for non-ASCII labels on Windows (cp949).
3. **[S] CreateML `verified` read from the wrong entry** — always taken from the first
   JSON array entry (`create_ml_io.py:115`), so multi-image files show another image's
   verified badge.
4. **[S] Surface CreateML read errors** — decode failures are swallowed (`ValueError`
   catch) and the image loads as if it had no annotations.
5. **[S] Working *Reset All* restart** — `QProcess.startDetached(__file__)` can't work on
   Windows or in the frozen exe; use `sys.executable`.
6. **[S] Revisit the 1-px minimum-coordinate clamp** — boxes drawn at the image edge are
   silently shifted from 0 to 1 on save (VOC/YOLO path, `labelFile.py:168-172`), a
   2015-era faster-rcnn workaround that degrades edge annotations.

### Triage workflow

7. **[M] User-defined categories** — generalize hard-coded good/bad to configurable
   N categories (the move/rollback/undo core is already generic).
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
16. **[M] Release automation** — build the Windows exe from `labelImg.spec` on tag push
    and attach it to the GitHub Release; move packaging to `pyproject.toml` under a
    distinct distribution name.
17. **[M] Remove Python 2 / PyQt4 remnants** — dead import fallbacks in 15 files, no-op
    `ustr()` wrappers, `qt4py2` build paths; unblocks ruff/type-checking.
18. **[S] Demo GIF in README** — the selling point is interaction; demo images still
    hot-link the upstream repo. (Badge part done 2026-07-07: the dead upstream workflow
    badge now points at this fork's CI.)
19. **[L] Qt6 migration (PySide6)** — after tests and dead-code removal land.
