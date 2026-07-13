import json
import os
import sys
import tempfile
import unittest

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from libs.coco_io import (COCOReader, COCOWriter, COCOParseError,
                          dataset_relative_name, is_coco_json)


class TestCOCORW(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.dataset = os.path.join(self.dir, 'annotations.json')

    def tearDown(self):
        self.tmp.cleanup()

    def write_image(self, filename, boxes, class_list=None, img_size=(512, 512, 3)):
        """Merge one image (list of (x_min, y_min, x_max, y_max, name)) into the dataset."""
        writer = COCOWriter('tests', filename, img_size,
                            local_img_path=os.path.join('tests', filename))
        for x_min, y_min, x_max, y_max, name in boxes:
            writer.add_bnd_box(x_min, y_min, x_max, y_max, name, 0)
        writer.save(class_list=class_list, target_file=self.dataset)
        return writer

    def read_dataset(self):
        with open(self.dataset, 'r', encoding='utf-8') as file:
            return json.load(file)

    def test_round_trip(self):
        self.write_image('test.512.512.bmp',
                         [(60, 40, 430, 504, 'person'),
                          (113, 40, 450, 403, 'face')],
                         class_list=['person', 'face'])

        shapes = COCOReader(self.dataset, 'tests/test.512.512.bmp').get_shapes()
        self.assertEqual(2, len(shapes), 'shape count is wrong')

        person, face = shapes
        self.assertEqual('person', person[0])
        self.assertEqual([(60, 40), (430, 40), (430, 504), (60, 504)], person[1])
        self.assertEqual('face', face[0])
        self.assertEqual([(113, 40), (450, 40), (450, 403), (113, 403)], face[1])

    def test_bbox_is_xywh_in_raw_json(self):
        # COCO stores [x, y, width, height] from the top-left corner, not the
        # [xmin, ymin, xmax, ymax] the rest of labelImg passes around.
        self.write_image('test.512.512.bmp', [(60, 40, 430, 504, 'person')],
                         class_list=['person'])

        data = self.read_dataset()
        annotation = data['annotations'][0]
        self.assertEqual([60, 40, 370, 464], annotation['bbox'])
        self.assertEqual(370 * 464, annotation['area'])
        self.assertEqual(0, annotation['iscrowd'])

        image = data['images'][0]
        self.assertEqual('test.512.512.bmp', image['file_name'])
        self.assertEqual(512, image['width'])
        self.assertEqual(512, image['height'])
        self.assertEqual(image['id'], annotation['image_id'])

    def test_merge_preserves_other_images(self):
        self.write_image('a.bmp', [(10, 10, 20, 20, 'person')], class_list=['person'])
        before = self.read_dataset()
        a_image = before['images'][0]
        a_annotations = [a for a in before['annotations'] if a['image_id'] == a_image['id']]
        a_category_ids = {c['name']: c['id'] for c in before['categories']}

        self.write_image('b.bmp', [(30, 30, 60, 60, 'face')], class_list=['person', 'face'])
        after = self.read_dataset()

        # image A survives untouched...
        self.assertEqual(['a.bmp', 'b.bmp'], sorted(i['file_name'] for i in after['images']))
        self.assertIn(a_image, after['images'])
        for annotation in a_annotations:
            self.assertIn(annotation, after['annotations'])
        # ...and so do its category ids.
        for name, category_id in a_category_ids.items():
            self.assertEqual(category_id, {c['name']: c['id'] for c in after['categories']}[name])

        # both images still read back their own boxes
        self.assertEqual([(10, 10), (20, 10), (20, 20), (10, 20)],
                         COCOReader(self.dataset, 'a.bmp').get_shapes()[0][1])
        self.assertEqual([(30, 30), (60, 30), (60, 60), (30, 60)],
                         COCOReader(self.dataset, 'b.bmp').get_shapes()[0][1])

    def test_category_ids_stable_across_merge(self):
        self.write_image('a.bmp', [(10, 10, 20, 20, 'person')], class_list=['person', 'face'])
        first = {c['name']: c['id'] for c in self.read_dataset()['categories']}

        # a merge that introduces a new class must not renumber the old ones —
        # every annotation already in the file points at those ids.
        self.write_image('b.bmp', [(30, 30, 60, 60, 'cat')], class_list=['person', 'face', 'cat'])
        second = {c['name']: c['id'] for c in self.read_dataset()['categories']}

        self.assertEqual(first['person'], second['person'])
        self.assertEqual(first['face'], second['face'])
        self.assertNotIn(second['cat'], (first['person'], first['face']))
        self.assertEqual(len(set(second.values())), len(second), 'category ids must be unique')

    def test_rewriting_same_image_replaces_only_its_annotations(self):
        self.write_image('a.bmp', [(10, 10, 20, 20, 'person')], class_list=['person'])
        self.write_image('b.bmp', [(30, 30, 60, 60, 'person')], class_list=['person'])
        # re-save image A with a different box
        self.write_image('a.bmp', [(11, 11, 25, 25, 'person')], class_list=['person'])

        data = self.read_dataset()
        self.assertEqual(2, len(data['images']), 'image A must not be duplicated')
        self.assertEqual(2, len(data['annotations']))

        a_shapes = COCOReader(self.dataset, 'a.bmp').get_shapes()
        self.assertEqual(1, len(a_shapes))
        self.assertEqual([(11, 11), (25, 11), (25, 25), (11, 25)], a_shapes[0][1])
        # B is untouched
        self.assertEqual([(30, 30), (60, 30), (60, 60), (30, 60)],
                         COCOReader(self.dataset, 'b.bmp').get_shapes()[0][1])

    def test_non_ascii_label_roundtrips_as_utf8(self):
        self.write_image('a.bmp', [(10, 10, 20, 20, '사람')], class_list=['사람'])

        with open(self.dataset, 'r', encoding='utf-8') as file:
            raw = file.read()
        self.assertIn('사람', raw, 'class name must be stored as utf-8, not escaped bytes')

        shapes = COCOReader(self.dataset, 'a.bmp').get_shapes()
        self.assertEqual('사람', shapes[0][0])

    def test_difficult_is_dropped_and_reads_back_false(self):
        writer = COCOWriter('tests', 'a.bmp', (512, 512, 3))
        writer.add_bnd_box(10, 10, 20, 20, 'person', 1)
        writer.save(class_list=['person'], target_file=self.dataset)

        # COCO has no difficult field, so it cannot survive the write...
        annotation = self.read_dataset()['annotations'][0]
        self.assertNotIn('difficult', annotation)

        # ...and the 5-tuple the canvas consumes reports False.
        shapes = COCOReader(self.dataset, 'a.bmp').get_shapes()
        self.assertEqual(False, shapes[0][4])

    def test_absent_image_reads_no_shapes(self):
        self.write_image('a.bmp', [(10, 10, 20, 20, 'person')], class_list=['person'])

        reader = COCOReader(self.dataset, 'absent.bmp')
        self.assertFalse(reader.found_image)
        self.assertEqual([], reader.get_shapes())

    def test_sniff_distinguishes_coco_from_create_ml(self):
        # both formats use .json — the dispatch in labelImg.py relies on this.
        self.write_image('a.bmp', [(10, 10, 20, 20, 'person')], class_list=['person'])
        self.assertTrue(is_coco_json(self.dataset))

        create_ml = os.path.join(self.dir, 'create_ml.json')
        with open(create_ml, 'w', encoding='utf-8') as file:
            json.dump([{'image': 'a.bmp', 'verified': False, 'annotations': []}], file)
        self.assertFalse(is_coco_json(create_ml))

        self.assertFalse(is_coco_json(os.path.join(self.dir, 'missing.json')))

    def test_save_refuses_to_clobber_a_non_coco_json(self):
        # the dataset target is user-pickable, so it can be aimed at a CreateML
        # file by mistake — that must fail loudly, not silently overwrite it.
        with open(self.dataset, 'w', encoding='utf-8') as file:
            json.dump([{'image': 'a.bmp', 'verified': False, 'annotations': []}], file)

        writer = COCOWriter('tests', 'a.bmp', (512, 512, 3))
        writer.add_bnd_box(10, 10, 20, 20, 'person', 0)
        with self.assertRaises(COCOParseError):
            writer.save(class_list=['person'], target_file=self.dataset)


class TestSameBasenameInDifferentSubfolders(unittest.TestCase):
    """MainWindow.scan_all_images walks the directory tree RECURSIVELY, so
    `train/0001.jpg` and `val/0001.jpg` are two different images sharing one
    basename — and, COCO being a dataset-level format, one shared json.

    Keying images[] on the basename (as the writer and reader used to) makes them
    overwrite each other's entry and union each other's annotations on read:
    silent data corruption in exactly the workflow this format is built for.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.dataset = os.path.join(self.dir, 'annotations.json')
        for sub in ('train', 'val', 'other'):
            os.makedirs(os.path.join(self.dir, sub))

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, *parts):
        # Absolute, like MainWindow.file_path always is.
        return os.path.join(self.dir, *parts)

    def write(self, image_path, boxes, class_list=None, img_size=(512, 512, 3)):
        writer = COCOWriter(os.path.basename(os.path.dirname(image_path)),
                            os.path.basename(image_path), img_size,
                            local_img_path=image_path)
        for x_min, y_min, x_max, y_max, name in boxes:
            writer.add_bnd_box(x_min, y_min, x_max, y_max, name, 0)
        writer.save(class_list=class_list, target_file=self.dataset)

    def read_dataset(self):
        with open(self.dataset, 'r', encoding='utf-8') as file:
            return json.load(file)

    def write_raw(self, dataset):
        with open(self.dataset, 'w', encoding='utf-8') as file:
            json.dump(dataset, file)

    def test_two_images_one_basename_get_one_entry_each(self):
        self.write(self.image('train', '0001.jpg'), [(10, 10, 20, 20, 'person')],
                   class_list=['person'])
        self.write(self.image('val', '0001.jpg'), [(30, 30, 60, 60, 'face')],
                   class_list=['person', 'face'])

        data = self.read_dataset()
        self.assertEqual(['train/0001.jpg', 'val/0001.jpg'],
                         sorted(image['file_name'] for image in data['images']),
                         'the two images must not collapse into one images[] entry')
        self.assertEqual(2, len(data['images']))
        self.assertEqual(2, len(data['annotations']))
        # file_name is dataset-relative and slash-normalised even on Windows.
        for image in data['images']:
            self.assertNotIn('\\', image['file_name'])

    def test_each_image_reads_back_only_its_own_boxes(self):
        self.write(self.image('train', '0001.jpg'), [(10, 10, 20, 20, 'person')],
                   class_list=['person'])
        self.write(self.image('val', '0001.jpg'), [(30, 30, 60, 60, 'face')],
                   class_list=['person', 'face'])

        train = COCOReader(self.dataset, self.image('train', '0001.jpg'))
        self.assertTrue(train.found_image)
        self.assertEqual(1, len(train.get_shapes()), "val's boxes leaked into train")
        self.assertEqual('person', train.get_shapes()[0][0])
        self.assertEqual([(10, 10), (20, 10), (20, 20), (10, 20)], train.get_shapes()[0][1])

        val = COCOReader(self.dataset, self.image('val', '0001.jpg'))
        self.assertEqual(1, len(val.get_shapes()), "train's boxes leaked into val")
        self.assertEqual('face', val.get_shapes()[0][0])
        self.assertEqual([(30, 30), (60, 30), (60, 60), (30, 60)], val.get_shapes()[0][1])

    def test_rewriting_one_does_not_clobber_the_other(self):
        self.write(self.image('train', '0001.jpg'), [(10, 10, 20, 20, 'person')],
                   class_list=['person'], img_size=(512, 512, 3))
        self.write(self.image('val', '0001.jpg'), [(30, 30, 60, 60, 'face')],
                   class_list=['person', 'face'], img_size=(256, 128, 3))
        # re-save train with a different box; val must be untouched, including
        # its width/height (the basename writer used to overwrite those too).
        self.write(self.image('train', '0001.jpg'), [(11, 11, 25, 25, 'person')],
                   class_list=['person'], img_size=(512, 512, 3))

        data = self.read_dataset()
        self.assertEqual(2, len(data['images']))
        val_entry = next(i for i in data['images'] if i['file_name'] == 'val/0001.jpg')
        self.assertEqual((128, 256), (val_entry['width'], val_entry['height']))

        self.assertEqual([(11, 11), (25, 11), (25, 25), (11, 25)],
                         COCOReader(self.dataset, self.image('train', '0001.jpg')).get_shapes()[0][1])
        val_shapes = COCOReader(self.dataset, self.image('val', '0001.jpg')).get_shapes()
        self.assertEqual(1, len(val_shapes))
        self.assertEqual([(30, 30), (60, 30), (60, 60), (30, 60)], val_shapes[0][1])

    def test_legacy_basename_entry_is_adopted_and_migrated(self):
        # A dataset written by the previous (basename-keyed) writer, or by a tool
        # that stores bare names: its entry is reused — id and annotations kept —
        # and rewritten to the relative key, so the NEXT same-basename image
        # cannot latch onto it.
        self.write_raw({
            'images': [{'id': 4, 'file_name': '0001.jpg', 'width': 512, 'height': 512}],
            'annotations': [{'id': 9, 'image_id': 4, 'category_id': 1,
                             'bbox': [10, 10, 10, 10], 'area': 100, 'iscrowd': 0}],
            'categories': [{'id': 1, 'name': 'person', 'supercategory': ''}],
        })

        self.write(self.image('train', '0001.jpg'), [(11, 11, 25, 25, 'person')],
                   class_list=['person'])
        data = self.read_dataset()
        self.assertEqual(1, len(data['images']), 'the legacy entry must be reused, not doubled')
        self.assertEqual(4, data['images'][0]['id'], 'existing ids must not be renumbered')
        self.assertEqual('train/0001.jpg', data['images'][0]['file_name'])
        self.assertEqual(1, data['categories'][0]['id'])

        # ...and now the second image with that basename gets its own entry.
        self.write(self.image('val', '0001.jpg'), [(30, 30, 60, 60, 'person')],
                   class_list=['person'])
        data = self.read_dataset()
        self.assertEqual(2, len(data['images']))
        self.assertEqual(['train/0001.jpg', 'val/0001.jpg'],
                         sorted(image['file_name'] for image in data['images']))
        self.assertEqual([(11, 11), (25, 11), (25, 25), (11, 25)],
                         COCOReader(self.dataset, self.image('train', '0001.jpg')).get_shapes()[0][1])

    def test_reader_falls_back_to_a_unique_basename(self):
        # Datasets authored elsewhere key file_name to THEIR root, not to this
        # json's directory. A unique basename is unambiguous, so it still resolves.
        self.write_raw({
            'images': [{'id': 7, 'file_name': 'images/train/0001.jpg',
                        'width': 512, 'height': 512}],
            'annotations': [{'id': 1, 'image_id': 7, 'category_id': 1,
                             'bbox': [5, 5, 10, 10], 'area': 100, 'iscrowd': 0}],
            'categories': [{'id': 1, 'name': 'person', 'supercategory': ''}],
        })

        reader = COCOReader(self.dataset, self.image('train', '0001.jpg'))
        self.assertTrue(reader.found_image)
        self.assertEqual([(5, 5), (15, 5), (15, 15), (5, 15)], reader.get_shapes()[0][1])

    def test_reader_refuses_to_guess_between_two_same_basename_entries(self):
        # THE corruption path: with two entries sharing a basename and no exact
        # match, the old reader unioned BOTH images' annotations onto whichever
        # image was open. Refusing to guess is the only safe answer.
        self.write(self.image('train', '0001.jpg'), [(10, 10, 20, 20, 'person')],
                   class_list=['person'])
        self.write(self.image('val', '0001.jpg'), [(30, 30, 60, 60, 'face')],
                   class_list=['person', 'face'])

        reader = COCOReader(self.dataset, self.image('other', '0001.jpg'))
        self.assertFalse(reader.found_image)
        self.assertEqual([], reader.get_shapes(), 'annotations were cross-contaminated')


class TestDatasetRelativeName(unittest.TestCase):

    def test_relative_to_the_dataset_directory_with_forward_slashes(self):
        root = os.path.abspath(os.path.join(os.sep, 'data'))
        dataset = os.path.join(root, 'annotations.json')
        image = os.path.join(root, 'train', '0001.jpg')
        self.assertEqual('train/0001.jpg', dataset_relative_name(image, dataset))

    def test_image_outside_the_dataset_directory_still_gets_a_unique_key(self):
        # A separate save dir is legal in labelImg; the key just walks up.
        dataset = os.path.abspath(os.path.join(os.sep, 'labels', 'annotations.json'))
        image = os.path.abspath(os.path.join(os.sep, 'photos', 'train', '0001.jpg'))
        name = dataset_relative_name(image, dataset)
        self.assertTrue(name.endswith('photos/train/0001.jpg'), name)
        self.assertNotIn('\\', name)

    def test_unrelatable_paths_degrade_to_the_basename(self):
        # Relative input (no anchor to compute from) -> basename, as before.
        self.assertEqual('0001.jpg',
                         dataset_relative_name(os.path.join('train', '0001.jpg'),
                                               os.path.abspath('annotations.json')))
        self.assertEqual('', dataset_relative_name(None, 'annotations.json'))


if __name__ == '__main__':
    unittest.main()
