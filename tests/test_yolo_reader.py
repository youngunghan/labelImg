import os
import sys
import tempfile
import unittest

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from PyQt5.QtGui import QImage

from libs.yolo_io import YoloParseError, YoloReader


class TestYoloReader(unittest.TestCase):
    """Crash-safety of YoloReader: a missing classes.txt or malformed lines
    must not abort loading (upstream crashed on both)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        # width=200, height=100
        self.image = QImage(200, 100, QImage.Format_RGB32)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_valid_file_loads_all_shapes(self):
        self.write('classes.txt', 'person\ncar\n')
        txt = self.write('img.txt',
                         '0 0.5 0.5 0.5 0.5\n'
                         '1 0.25 0.25 0.1 0.1\n')
        reader = YoloReader(txt, self.image)
        shapes = reader.get_shapes()
        self.assertEqual(2, len(shapes))
        label, points, _, _, difficult = shapes[0]
        self.assertEqual('person', label)
        # center (0.5, 0.5), size (0.5, 0.5) on 200x100 -> x 50..150, y 25..75
        self.assertEqual([(50, 25), (150, 25), (150, 75), (50, 75)], points)
        self.assertFalse(difficult)
        self.assertEqual('car', shapes[1][0])
        self.assertEqual(0, reader.skipped_lines)

    def test_missing_classes_txt_raises_clear_error(self):
        txt = self.write('img.txt', '0 0.5 0.5 0.5 0.5\n')
        with self.assertRaises(YoloParseError) as ctx:
            YoloReader(txt, self.image)
        self.assertIn('classes.txt', str(ctx.exception))

    def test_malformed_lines_are_skipped_not_fatal(self):
        self.write('classes.txt', 'person\n')
        txt = self.write('img.txt', '\n'.join([
            '0 0.5 0.5 0.5 0.5',        # valid
            '',                          # blank -> ignored silently
            '0 0.1 0.1',                 # too few fields
            'x 0.5 0.5 0.5 0.5',         # non-numeric class index
            '0 a 0.5 0.5 0.5',           # non-numeric coordinate
            '5 0.5 0.5 0.5 0.5',         # class index out of range
            '-1 0.5 0.5 0.5 0.5',        # negative class index
            '0 0.2 0.2 0.1 0.1 extra',   # too many fields
            '0 nan 0.5 0.1 0.1',         # NaN coordinate (round() ValueError)
            '0 1e308 0.5 0.1 0.1',       # overflows to inf (OverflowError)
        ]) + '\n')
        reader = YoloReader(txt, self.image)
        self.assertEqual(1, len(reader.get_shapes()))
        self.assertEqual('person', reader.get_shapes()[0][0])
        self.assertEqual(8, reader.skipped_lines)

    def test_bom_in_classes_txt_is_stripped(self):
        # Windows Notepad writes a BOM; it must not leak into the first name.
        with open(os.path.join(self.dir, 'classes.txt'), 'w', encoding='utf-8-sig') as f:
            f.write('person\ncar\n')
        txt = self.write('img.txt', '0 0.5 0.5 0.2 0.2\n')
        reader = YoloReader(txt, self.image)
        self.assertEqual('person', reader.get_shapes()[0][0])

    def test_non_ascii_classes_read_as_utf8(self):
        # classes.txt is written utf-8 by YOLOWriter; the reader must decode
        # it as utf-8 too, independent of the OS locale (cp949 on Korean
        # Windows used to mangle or reject these).
        with open(os.path.join(self.dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            f.write('사람\n자동차\n')
        txt = self.write('img.txt', '1 0.5 0.5 0.2 0.2\n')
        reader = YoloReader(txt, self.image)
        self.assertEqual('자동차', reader.get_shapes()[0][0])
        self.assertEqual(0, reader.skipped_lines)


if __name__ == '__main__':
    unittest.main()
