# Models

**No model weights ship with labelImg.** This directory is a convention -- a
place to keep the `.onnx` file you point the tool at -- not a download cache.
Nothing here is loaded automatically; the model is whatever the model-path
setting says it is.

## Why no bundled weights: the licence

The obvious default model would be an Ultralytics YOLOv5 / YOLOv8 checkpoint.
We cannot bundle one:

| | Licence |
|---|---|
| labelImg (this app) | **MIT** |
| Ultralytics YOLOv5 / YOLOv8 (code *and* pretrained weights) | **AGPL-3.0** |

AGPL-3.0 is copyleft and its network clause reaches further than the GPL's.
Shipping AGPL weights inside an MIT application -- or making the app depend on
them -- would put the combined work under obligations MIT does not carry, and
would misrepresent the licence users think they are getting. So labelImg ships
the *backend* (MIT, written here, weight-agnostic) and you supply the *weights*.

This is a distribution boundary, not a technical one. The backend loads an
Ultralytics export perfectly well -- see below -- and doing that **for yourself**
is fine. What you may not do is redistribute those weights (or a labelImg build
containing them) as if they were MIT. If you build a product on top of this,
AGPL-3.0 obligations attach to *your* use of *those* weights; read the licence,
or buy Ultralytics' commercial licence, or use one of the permissive
alternatives below.

## Using your own model

1. Put an ONNX detector anywhere on disk (this directory is a reasonable spot).
2. Set the model path in labelImg's settings (`model/path`) and the backend to
   `yolo_onnx` (`model/backend`).
3. Class names are resolved for you, in this order:
   1. the `names` entry in the ONNX file's own metadata (Ultralytics exports
      write one),
   2. a `classes.txt` **next to the model file** -- one class name per line, in
      class-id order, same format as `data/predefined_classes.txt`,
   3. `class_0`, `class_1`, ... as a last resort.

If the AI actions stay greyed out, the extras are missing. This fork is not
published to PyPI, so install them from this checkout (from the repository
root, not from `data/models/`):
`pip install -e ".[ai]"` (onnxruntime + numpy).

If you are instead running the prebuilt Windows exe from a GitHub Release,
`labelImg.spec` already bundles onnxruntime + numpy into it (see
[`docs/how-to/install-and-build.md`](../../docs/how-to/install-and-build.md#onnxruntime-onnx-런타임-번들)),
so you do **not** need `pip install -e ".[ai]"` for that exe -- the runtime is
already inside it. The exe still ships **no model weights**, though: the AI
menu stays greyed out until you point `model/path` at your own `.onnx` file
(see "Using your own model" above); there is still no in-app file picker for
this.

### Exporting from Ultralytics (AGPL-3.0 -- applies to you)

```bash
pip install ultralytics          # AGPL-3.0
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
# -> yolov8n.onnx, with class names in the model metadata
```

Same for YOLOv5 (`python export.py --weights yolov5s.pt --include onnx`). Both
produce a file this backend decodes. Do **not** commit the result to a public MIT
repository.

Export **without** the end-to-end / NMS option (`nms=False`, the default): this
backend does its own decoding and NMS, and an export that already applied NMS has
a completely different output shape, which the layout autodetection will refuse
rather than mis-decode.

### Permissively-licensed alternatives

If AGPL is a problem for you (it usually is, in a commercial pipeline), these
detectors have the same box-detector shape and permissive licences:

| Model | Licence | Notes |
|---|---|---|
| **YOLOX** (Megvii) | Apache-2.0 | ONNX export in-repo; decodes as the v5-style layout after its standard export. |
| **YOLOv6** (Meituan) | GPL-3.0 | Copyleft, but not AGPL -- still not bundleable here. |
| **RT-DETR** (PaddlePaddle) | Apache-2.0 | Different output layout -- **not supported by this backend** as of Phase 2. |
| Your own model | yours | Anything that exports to one of the two layouts below. |

Weights trained *by you* on *your* data with an AGPL toolchain are still subject
to that toolchain's licence terms -- the training code, not just the checkpoint,
is what carries them. When in doubt, train with an Apache-2.0 codebase.

## What the `yolo_onnx` backend expects

**Input:** one tensor, `float32` (or `float16`), NCHW, RGB, `0..1`, letterboxed
(aspect-preserving resize, centred, grey `114` padding). The input size is read
from the model's own input shape; if that axis is dynamic, it falls back to the
`input_size` config key, else 640.

**Output:** one tensor, in one of the two Ultralytics layouts, autodetected from
its shape:

| Layout | Shape | Contents |
|---|---|---|
| YOLOv8 | `(1, 4+nc, N)` | `cx, cy, w, h` then `nc` class scores. No objectness. |
| YOLOv5 | `(1, N, 5+nc)` | `cx, cy, w, h, objectness`, then `nc` class scores; final score = objectness x class score. |

`cx, cy, w, h` are in **model-input pixels** (not normalised). The backend undoes
the letterbox itself and reports boxes in **original image pixels**.

Detection is by shape, on two pieces of evidence: the class count implied by each
reading is checked against the model's own declared class names (the strong
signal), and failing that, the anchor axis is never shorter than the channel axis
(8400 vs 84 -- a "YOLOv5 output" with 4 anchors and 8395 classes is not a
detector). An output shape that fits neither reading -- or a square one, which
fits both with nothing to choose between them -- is **refused with an error**
rather than guessed at, because a wrong guess silently produces plausible boxes
in the wrong place. Set the `layout` config key (`v8` / `v5`) to force one.

Segmentation outputs (the `(1, 32, 160, 160)` mask-prototype head of a `-seg`
model) are ignored: this backend is a detector. Masks are a later phase.
