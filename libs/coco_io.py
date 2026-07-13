#!/usr/bin/env python
# -*- coding: utf8 -*-
import json
import os

from libs.constants import DEFAULT_ENCODING

JSON_EXT = '.json'
ENCODE_METHOD = DEFAULT_ENCODING

# COCO is a *dataset-level* format: a single json holds images[] + annotations[]
# + categories[] for many images at once. The other formats labelImg supports
# (VOC/YOLO/CreateML) are per-image sidecars whose path is derived from the
# image stem, so COCO cannot be driven that way — every save merges the current
# image into one fixed dataset file. This is the file name used for that target
# when the user has not picked one explicitly (Import/Export COCO...).
COCO_DEFAULT_DATASET_NAME = 'annotations.json'


class COCOParseError(ValueError):
    """Raised when a .json path is not a readable COCO dataset.

    Subclasses ValueError on purpose: save_labels() in labelImg.py already
    treats ValueError from a merge-read as a recoverable "bad existing json"
    and shows an error dialog, so a mis-targeted save (e.g. at a CreateML
    json) reports cleanly instead of crashing or silently clobbering the file.
    """


def _empty_dataset():
    return {'images': [], 'annotations': [], 'categories': []}


def is_coco_dict(data):
    """Sniff already-parsed json content: COCO dataset vs CreateML.

    CreateML writes a top-level list, COCO a top-level dict with the dataset
    sections. Both use the .json extension, so the extension alone cannot pick
    a reader.
    """
    if not isinstance(data, dict):
        return False
    return any(key in data for key in ('images', 'annotations', 'categories'))


def is_coco_json(json_path):
    """Content-sniff a .json file on disk. False for anything unreadable, so
    callers can fall back to the CreateML reader instead of failing."""
    try:
        with open(json_path, 'r', encoding=ENCODE_METHOD) as file:
            data = json.load(file)
    except (ValueError, OSError, UnicodeError):
        return False
    return is_coco_dict(data)


def load_coco_dataset(json_path):
    """Read a COCO dataset json, normalising the three sections we touch."""
    with open(json_path, 'r', encoding=ENCODE_METHOD) as file:
        data = json.load(file)

    if not is_coco_dict(data):
        raise COCOParseError('"%s" is not a COCO dataset json (expected a dict '
                             'with images/annotations/categories).' % json_path)

    for key in ('images', 'annotations', 'categories'):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


class COCOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        # Caveat: COCO has no per-annotation difficult flag, so this is
        # discarded when saved as coco format (same as yolo format) and reads
        # back as False.
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def sync_categories(self, dataset, class_list):
        """Map class name -> category_id, reusing the ids already in the file.

        Renumbering existing categories on a merge would silently relabel every
        annotation the dataset already holds, so existing ids are kept as-is and
        only unseen names get a fresh id.
        """
        name_to_id = {}
        for category in dataset['categories']:
            name = category.get('name')
            if name is not None and category.get('id') is not None:
                name_to_id[name] = category['id']

        next_id = max(list(name_to_id.values()) or [0]) + 1
        # The predefined class list defines the taxonomy (like classes.txt for
        # YOLO); box names are added too so an ad-hoc label still gets an id.
        for name in list(class_list or []) + [box['name'] for box in self.box_list]:
            if name not in name_to_id:
                name_to_id[name] = next_id
                dataset['categories'].append({'id': next_id, 'name': name, 'supercategory': ''})
                next_id += 1
        return name_to_id

    def sync_image(self, dataset):
        """Return the image id for this image, adding an images[] entry if new."""
        height, width = self.img_size[0], self.img_size[1]

        for image in dataset['images']:
            if image.get('file_name') == self.filename and image.get('id') is not None:
                image['width'] = width
                image['height'] = height
                return image['id']

        image_id = max([image.get('id', 0) for image in dataset['images']] or [0]) + 1
        dataset['images'].append({
            'id': image_id,
            'file_name': self.filename,
            'width': width,
            'height': height,
        })
        return image_id

    def save(self, class_list=None, target_file=None):
        # Read-modify-write merge: the dataset file describes many images, so it
        # is loaded first and only this image's entry is replaced.
        if target_file is None:
            raise COCOParseError('COCO needs an explicit dataset json target.')

        if os.path.isfile(target_file):
            dataset = load_coco_dataset(target_file)
        else:
            dataset = _empty_dataset()

        name_to_id = self.sync_categories(dataset, class_list)
        image_id = self.sync_image(dataset)

        # Drop only this image's annotations; every other image keeps its own.
        dataset['annotations'] = [annotation for annotation in dataset['annotations']
                                  if annotation.get('image_id') != image_id]
        next_id = max([annotation.get('id', 0) for annotation in dataset['annotations']] or [0]) + 1

        for box in self.box_list:
            x_min = box['xmin']
            y_min = box['ymin']
            width = box['xmax'] - x_min
            height = box['ymax'] - y_min
            dataset['annotations'].append({
                'id': next_id,
                'image_id': image_id,
                'category_id': name_to_id[box['name']],
                # COCO bbox is [x, y, width, height] measured from the top-left
                # corner — not the [xmin, ymin, xmax, ymax] the rest of labelImg
                # passes around.
                'bbox': [x_min, y_min, width, height],
                'area': width * height,
                'iscrowd': 0,
            })
            next_id += 1

        with open(target_file, 'w', encoding=ENCODE_METHOD) as file:
            # ensure_ascii=False keeps non-ASCII class names readable in the
            # file; it round-trips because the file is UTF-8 on both ends.
            json.dump(dataset, file, ensure_ascii=False, indent=2)


class COCOReader:

    def __init__(self, json_path, file_path):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.json_path = json_path
        self.filename = os.path.basename(file_path)
        # COCO has no verified flag (CreateML does); report False rather than
        # letting the canvas keep another file's verified state.
        self.verified = False
        # A dataset covers many images and may not contain this one at all —
        # the caller needs to tell "no entry" from "entry with no boxes".
        self.found_image = False
        try:
            self.parse_json()
        except COCOParseError as e:
            print(e)
        except ValueError:
            print("JSON decoding failed")

    def parse_json(self):
        dataset = load_coco_dataset(self.json_path)

        id_to_name = {}
        for category in dataset['categories']:
            if category.get('id') is not None:
                id_to_name[category['id']] = category.get('name', '')

        image_ids = set()
        for image in dataset['images']:
            # file_name may carry a directory prefix in datasets exported by
            # other tools — match on the basename, like CreateMLReader does.
            file_name = os.path.basename(str(image.get('file_name', '')))
            if file_name == self.filename and image.get('id') is not None:
                image_ids.add(image['id'])

        if not image_ids:
            return
        self.found_image = True

        for annotation in dataset['annotations']:
            if annotation.get('image_id') not in image_ids:
                continue
            bbox = annotation.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            label = id_to_name.get(annotation.get('category_id'))
            if label is None:
                continue
            self.add_shape(label, bbox)

    def add_shape(self, label, bbox):
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        # difficult is not representable in COCO, so it always reads back False.
        self.shapes.append((label, points, None, None, False))

    def get_shapes(self):
        return self.shapes
