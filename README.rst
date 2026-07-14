.. image:: /readme/images/labelimg.png
        :target: https://github.com/heartexlabs/label-studio

Label Studio is a modern, multi-modal data annotation tool
=======

LabelImg, the popular image annotation tool created by Tzutalin with the help of dozens contributors, is no longer actively being developed and has become part of the Label Studio community. Check out `Label Studio <https://github.com/heartexlabs/label-studio>`__, the most flexible open source data labeling tool for images, text, hypertext, audio, video and time-series data. `Install <https://labelstud.io/guide/install.html>`__ Label Studio and join the `slack community <https://label-studio.slack.com/>`__ to get started.

.. image:: /readme/images/label-studio-1-6-player-screenshot.png
        :target: https://github.com/heartexlabs/label-studio

About LabelImg
========

.. image:: https://img.shields.io/pypi/v/labelimg.svg
        :target: https://pypi.python.org/pypi/labelimg

.. image:: https://github.com/youngunghan/labelImg/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/youngunghan/labelImg/actions/workflows/ci.yml
        :alt: CI status

.. image:: https://img.shields.io/badge/lang-en-blue.svg
        :target: https://github.com/tzutalin/labelImg

.. image:: https://img.shields.io/badge/lang-zh-green.svg
        :target: https://github.com/tzutalin/labelImg/blob/master/readme/README.zh.rst

.. image:: https://img.shields.io/badge/lang-jp-green.svg
        :target: https://github.com/tzutalin/labelImg/blob/master/readme/README.jp.rst

LabelImg is a graphical image annotation tool.

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
by `ImageNet <http://www.image-net.org/>`__.  Besides, it also supports YOLO and CreateML formats.

.. image:: https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg
     :alt: Demo Image

.. image:: https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo.jpg
     :alt: Demo Image

`Watch a demo video <https://youtu.be/p0nR2YsCY_U>`__

About this fork
------------------

This repository is a fork of `HumanSignal/labelImg <https://github.com/HumanSignal/labelImg>`__,
which was archived in February 2024 (read-only) when LabelImg joined the Label
Studio community. Changes can no longer be merged upstream, so this fork
carries them independently.

.. image:: /readme/images/demo-triage.gif
        :alt: Keyboard triage demo — g/b classify an image with its label (atomic move), Ctrl+Z undo

What this fork adds on top of upstream ``b33f965``:

- **AI-assisted auto-labeling** (``Ctrl+I`` / ``Ctrl+Return`` / ``Ctrl+Backspace``,
  new **AI** menu) — run an ONNX YOLOv5/v8 detector on the current image and get
  provisional (dashed) suggestion boxes back; reject any suggestion individually
  (``Delete``), or accept or reject all of them at once (per-box accept is not yet
  implemented — see the roadmap), with a confidence-threshold slider that re-filters
  suggestions on screen without re-running the model. Inference runs on a
  background thread, so it never blocks the UI, and a suggestion is never written
  to disk until you accept it. Requires the optional ``ai`` extra, installed from
  this checkout (``pip install -e ".[ai]"`` — see *Build from source* below; this
  fork is not published to PyPI, so ``pip install labelImg[ai]`` would fetch the
  unrelated upstream package instead; adds ``onnxruntime`` + ``numpy``) and a model file
  you supply yourself — see `data/models/README.md <data/models/README.md>`__ for
  why no weights ship with this MIT-licensed app (Ultralytics YOLOv5/v8 weights are
  AGPL-3.0) and for permissively-licensed alternatives. Without the extra and a
  configured model, the AI menu stays greyed out and the rest of the app is
  unaffected.
- **COCO import/export** (*File > Import COCO...* / *Export COCO...*) — a fourth
  annotation format alongside PASCAL VOC/YOLO/CreateML. Unlike those three, COCO
  is dataset-level (one JSON describes many images), so it is not a per-image
  save format you switch to — it is an explicit Import/Export lane that merges the
  current image into a shared dataset ``.json`` (``annotations.json`` in the save
  directory by default) without disturbing any other image's entries.
- **Good/Bad image triage** — press ``g``/``b`` to move the current image *and*
  its label file into a ``<folder>_good`` / ``<folder>_bad`` sibling folder and
  advance to the next image. Moves are atomic (rolled back if a label move
  fails) and undoable with ``Ctrl+Z``.
- **Edit Default Classes in-app** (``Ctrl+Shift+E``) — edit the predefined
  class list from the File menu; changes persist (written back to the class
  file, or stored next to the executable for the packaged exe).
- **Command-line save dir respected at startup** —
  ``labelImg.py <image_dir> [class_file] [save_dir]`` now prefers the folder
  given on the command line over the remembered ``lastOpenDir``, so existing
  labels are found where you expect.
- **Single Class Mode moved to** ``Ctrl+Shift+C`` (upstream's ``Ctrl+Shift+S``
  collided with Save As).
- **Robustness fixes** — ``tools/label_to_csv.py`` no longer crashes when a
  stray file sits in a set folder; Copy Previous Bounding Boxes no longer
  crashes on a standalone (Open File) image; saving with an unsupported format
  raises a clear error instead of an ``AttributeError``.
- **Developer documentation** — a `docs/ <docs/README.md>`__ tree (Korean)
  covering architecture, annotation formats, the ML-assist design, shortcuts
  and the fork features, plus a PyInstaller ``labelImg.spec``.

The PyPI package (``pip3 install labelImg``) is the upstream project and does
not include these changes — build from source to use them.

See `FORK_CHANGES.md <FORK_CHANGES.md>`__ for a change-by-change comparison
against upstream (with numbers) and the improvement roadmap, and the
`releases page <https://github.com/youngunghan/labelImg/releases>`__ for
working snapshots.

Installation
------------------

Get from PyPI but only python3.0 or above
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the simplest (one-command) install method on modern Linux distributions such as Ubuntu and Fedora.

Note: the PyPI package is the upstream labelImg and does NOT include this
fork's additions (AI-assisted auto-labeling, COCO import/export, g/b classify,
Ctrl+Z undo classify, Ctrl+Shift+E class editing, Ctrl+Shift+C
single-class-mode shortcut). To use this fork, build from source in this
repository instead.

.. code:: shell

    pip3 install labelImg
    labelImg
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]


Build from source
~~~~~~~~~~~~~~~~~

Linux/Ubuntu/Mac requires at least `Python
2.6 <https://www.python.org/getit/>`__ and has been tested with `PyQt
4.8 <https://www.riverbankcomputing.com/software/pyqt/intro>`__. However, `Python
3 or above <https://www.python.org/getit/>`__ and  `PyQt5 <https://pypi.org/project/PyQt5/>`__ are strongly recommended.

This fork requires **Python 3.7+** (``labelImg.py`` imports the AI-assist core at
module load time, and that core uses dataclasses and postponed annotation
evaluation, both 3.7+ features — this applies even if you never use the AI menu).
The AI-assisted auto-labeling *model backend* itself is an additional, optional
extra on top of the ``pyqt5``/``lxml`` base install below:

.. code:: shell

    pip install -e ".[ai]"   # from the repository root; adds onnxruntime>=1.15 and numpy

Note: this fork is not published to PyPI under the ``labelImg`` name, so
``pip install labelImg[ai]`` would install the unrelated upstream package
(which has none of this fork's AI code) instead of these extras — always run
the command above from a checkout of *this* repository.

No model weights ship with labelImg — point the ``model/path`` setting at your
own ``.onnx`` file (see `data/models/README.md <data/models/README.md>`__).
Without this extra and a configured model, labelImg still runs as a normal
annotation tool; the AI menu is simply greyed out.


Ubuntu Linux
^^^^^^^^^^^^

Python 3 + Qt5

.. code:: shell

    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]

macOS
^^^^^

Python 3 + Qt5

.. code:: shell

    brew install qt  # Install qt-5.x.x by Homebrew
    brew install libxml2

    or using pip

    pip3 install pyqt5 lxml # Install qt and lxml by pip

    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]


Python 3 Virtualenv (Recommended)

Virtualenv can avoid a lot of the QT / Python version issues

.. code:: shell

    brew install python3
    pip3 install pipenv
    pipenv run pip install pyqt5==5.15.2 lxml
    pipenv run make qt5py3
    pipenv run python3 labelImg.py
    [Optional] rm -rf build dist; pipenv run python setup.py py2app -A;mv "dist/labelImg.app" /Applications

Note: The Last command gives you a nice .app file with a new SVG Icon in your /Applications folder. You can consider using the script: build-tools/build-for-macos.sh


Windows
^^^^^^^

Install `Python <https://www.python.org/downloads/windows/>`__,
`PyQt5 <https://www.riverbankcomputing.com/software/pyqt/download5>`__
and `install lxml <http://lxml.de/installation.html>`__.

Open cmd and go to the `labelImg <#labelimg>`__ directory

.. code:: shell

    pyrcc4 -o libs/resources.py resources.qrc
    For pyqt5, pyrcc5 -o libs/resources.py resources.qrc

    python labelImg.py
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]

If you want to package it into a separate EXE file

.. code:: shell

    Install pyinstaller and execute:

    pip install pyinstaller
    pyinstaller --hidden-import=pyqt5 --hidden-import=lxml -F -n "labelImg" -c labelImg.py -p ./libs -p ./

Windows + Anaconda
^^^^^^^^^^^^^^^^^^

Download and install `Anaconda <https://www.anaconda.com/download/#download>`__ (Python 3+)

Open the Anaconda Prompt and go to the `labelImg <#labelimg>`__ directory

.. code:: shell

    conda install pyqt=5
    conda install -c anaconda lxml
    pyrcc5 -o libs/resources.py resources.qrc
    python labelImg.py
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]

Use Docker
~~~~~~~~~~~~~~~~~
.. code:: shell

    docker run -it \
    --user $(id -u) \
    -e DISPLAY=unix$DISPLAY \
    --workdir=$(pwd) \
    --volume="/home/$USER:/home/$USER" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    tzutalin/py2qt4

    make qt4py2;./labelImg.py

You can pull the image which has all of the installed and required dependencies. `Watch a demo video <https://youtu.be/nw1GexJzbCI>`__

Note: the ``py2qt4`` image runs Python 2 + Qt4; this fork's classify (g/b)
feature requires Python 3 and is not supported there.


Usage
-----

Steps (PascalVOC)
~~~~~~~~~~~~~~~~~

1. Build and launch using the instructions above.
2. Click 'Open Dir'. The annotation save folder is automatically set to
   the opened image folder.
3. (Optional) Click 'Change default saved annotation folder' in Menu/File
   to save annotations elsewhere. Do this *after* 'Open Dir' — opening a
   directory always resets the save folder to that directory. Exception:
   a save dir passed on the command line (3rd argument) is kept when the
   image dir is opened at startup.
4. Click 'Create RectBox'
5. Click and release left mouse to select a region to annotate the rect
   box
6. You can use right mouse to drag the rect box to copy or move it

The annotation will be saved to the folder you specify (by default, next
to the images).

You can refer to the below hotkeys to speed up your workflow.

Steps (YOLO)
~~~~~~~~~~~~

1. In ``data/predefined_classes.txt`` define the list of classes that will be used for your training.

2. Build and launch using the instructions above.

3. Right below "Save" button in the toolbar, click "PascalVOC" button to switch to YOLO format.

4. You may use Open/OpenDIR to process single or multiple images. When finished with a single image, click save.

A txt file of YOLO format will be saved in the same folder as your image with same name. A file named "classes.txt" is saved to that folder too. "classes.txt" defines the list of class names that your YOLO label refers to.

Note:

- Your label list shall not change in the middle of processing a list of images. When you save an image, classes.txt will also get updated, while previous annotations will not be updated.

- You shouldn't use "default class" function when saving to YOLO format, it will not be referred.

- When saving as YOLO format, "difficult" flag is discarded.

Create pre-defined classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can edit the
`data/predefined\_classes.txt <https://github.com/tzutalin/labelImg/blob/master/data/predefined_classes.txt>`__
to load pre-defined classes

In this fork you can also edit the class list in-app via File > Edit Default
Classes (Ctrl+Shift+E); changes are saved permanently. When running from
source they are written back to ``data/predefined_classes.txt`` (or to the
class file passed as the 2nd command-line argument, if given); when running
the packaged exe, the list is stored in ``predefined_classes.txt`` next to the
executable.

Annotation visualization
~~~~~~~~~~~~~~~~~~~~~~~~

1. Copy the existing lables file to same folder with the images. The labels file name must be same with image file name.

2. Click File and choose 'Open Dir' then Open the image folder.

3. Select image in File List, it will appear the bounding box and label for all objects in that image.

(Choose Display Labels mode in View to show/hide lablels)


Hotkeys
~~~~~~~

+--------------------+--------------------------------------------+
| Ctrl + u           | Load all of the images from a directory    |
+--------------------+--------------------------------------------+
| Ctrl + r           | Change the default annotation target dir   |
+--------------------+--------------------------------------------+
| Ctrl + s           | Save                                       |
+--------------------+--------------------------------------------+
| Ctrl + Shift + s   | Save As                                    |
+--------------------+--------------------------------------------+
| Ctrl + Shift + c   | Toggle single class mode                   |
|                    | (changed from Ctrl+Shift+S in this fork)   |
+--------------------+--------------------------------------------+
| Ctrl + d           | Copy the current label and rect box        |
+--------------------+--------------------------------------------+
| Ctrl + Shift + d   | Delete the current image                   |
+--------------------+--------------------------------------------+
| Space              | Flag the current image as verified         |
+--------------------+--------------------------------------------+
| w                  | Create a rect box                          |
+--------------------+--------------------------------------------+
| d                  | Next image                                 |
+--------------------+--------------------------------------------+
| a                  | Previous image                             |
+--------------------+--------------------------------------------+
| del                | Delete the selected rect box               |
+--------------------+--------------------------------------------+
| Ctrl++             | Zoom in                                    |
+--------------------+--------------------------------------------+
| Ctrl--             | Zoom out                                   |
+--------------------+--------------------------------------------+
| ↑→↓←               | Keyboard arrows to move selected rect box  |
+--------------------+--------------------------------------------+
| g                  | Move image + label to <folder>_good (fork) |
+--------------------+--------------------------------------------+
| b                  | Move image + label to <folder>_bad (fork)  |
+--------------------+--------------------------------------------+
| Ctrl + z           | Undo the last good/bad classify (fork)     |
+--------------------+--------------------------------------------+
| Ctrl + Shift + e   | Edit default classes in-app (fork)         |
+--------------------+--------------------------------------------+
| Ctrl + I           | Auto-label current image (fork)            |
+--------------------+--------------------------------------------+
| Ctrl + Return      | Accept all AI suggestions (fork)           |
+--------------------+--------------------------------------------+
| Ctrl + Backspace   | Reject all AI suggestions (fork)           |
+--------------------+--------------------------------------------+

g / b / Ctrl+Z are fork-specific classify hotkeys: the current image and its
label file (.xml / .txt / .json) are moved to a sibling ``<folder>_good`` /
``<folder>_bad`` directory, then the next image is loaded; Ctrl+Z undoes the
last move.

Ctrl+I / Ctrl+Return / Ctrl+Backspace are fork-specific AI-assist hotkeys: run
the model on the current image, accept every suggestion it made, or reject
every suggestion it made. There is also a confidence-threshold slider in the
new **AI** menu that re-filters suggestions on screen without re-running the
model.

**Verify Image:**

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when creating a dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

**Difficult:**

The difficult field is set to 1 indicates that the object has been annotated as "difficult", for example, an object which is clearly visible but difficult to recognize without substantial use of context.
According to your deep neural network implementation, you can include or exclude difficult objects during training.

How to reset the settings
~~~~~~~~~~~~~~~~~~~~~~~~~

In case there are issues with loading the classes, you can either:

1. From the top menu of the labelimg click on Menu/File/Reset All
2. Remove the `.labelImgSettings.pkl` from your home directory. In Linux and Mac you can do:
    `rm ~/.labelImgSettings.pkl`


How to contribute
~~~~~~~~~~~~~~~~~

Send a pull request

License
~~~~~~~
`Free software: MIT license <https://github.com/tzutalin/labelImg/blob/master/LICENSE>`_

Citation: Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg

Related and additional tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. `Label Studio <https://github.com/heartexlabs/label-studio>`__ to label images, text, audio, video and time-series data for machine learning and AI
2. `ImageNet Utils <https://github.com/tzutalin/ImageNet_Utils>`__ to
   download image, create a label text for machine learning, etc
3. `Use Docker to run labelImg <https://hub.docker.com/r/tzutalin/py2qt4>`__
4. `Generating the PASCAL VOC TFRecord files <https://github.com/tensorflow/models/blob/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/g3doc/preparing_inputs.md#generating-the-pascal-voc-tfrecord-files>`__
5. `App Icon based on Icon by Nick Roach (GPL) <https://www.elegantthemes.com/>`__
6. `Setup python development in vscode <https://tzutalin.blogspot.com/2019/04/set-up-visual-studio-code-for-python-in.html>`__
7. `The link of this project on iHub platform <https://code.ihub.org.cn/projects/260/repository/labelImg>`__
8. `Convert annotation files to CSV format or format for Google Cloud AutoML <https://github.com/tzutalin/labelImg/tree/master/tools>`__



Stargazers over time
~~~~~~~~~~~~~~~~~~~~

.. image:: https://starchart.cc/tzutalin/labelImg.svg

