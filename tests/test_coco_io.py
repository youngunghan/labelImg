import json
import os
import sys
import tempfile
import unittest

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from libs.coco_io import COCOReader, COCOWriter, COCOParseError, is_coco_json


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


if __name__ == '__main__':
    unittest.main()
