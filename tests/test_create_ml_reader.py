import json
import os
import sys
import tempfile
import unittest

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from libs.create_ml_io import CreateMLReader


class TestCreateMLReader(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, entries):
        path = os.path.join(self.dir, 'ann.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False)
        return path

    @staticmethod
    def entry(image, verified, label='person'):
        return {'image': image, 'verified': verified,
                'annotations': [{'label': label,
                                 'coordinates': {'x': 50, 'y': 50, 'width': 20, 'height': 20}}]}

    def test_verified_read_from_matching_entry_not_first(self):
        # first entry verified=True, ours=False — the badge must be ours
        path = self.write_json([self.entry('other.png', True),
                                self.entry('mine.png', False)])
        self.assertFalse(CreateMLReader(path, 'mine.png').verified)
        # and the other way around
        path = self.write_json([self.entry('other.png', False),
                                self.entry('mine.png', True)])
        self.assertTrue(CreateMLReader(path, 'mine.png').verified)
        # image not in the file at all -> stays False
        path = self.write_json([self.entry('other.png', True)])
        self.assertFalse(CreateMLReader(path, 'absent.png').verified)

    def test_non_ascii_label_roundtrips_as_utf8(self):
        # utf-8 JSON with a Korean label must load regardless of the OS
        # locale encoding (used to be opened with the platform default).
        path = self.write_json([self.entry('mine.png', False, label='사람')])
        reader = CreateMLReader(path, 'mine.png')
        shapes = reader.get_shapes()
        self.assertEqual(1, len(shapes))
        self.assertEqual('사람', shapes[0][0])


if __name__ == '__main__':
    unittest.main()
