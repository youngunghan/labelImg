import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dir_name, '..'))

from PyQt5.QtWidgets import QApplication

# QApplication is a process-wide singleton; create it once at import so every
# test module (and get_main_app, which reuses an existing instance) shares it.
APP = QApplication.instance() or QApplication([])

from PyQt5.QtGui import QImage

from labelImg import get_main_app


class TestClassifyWorkflow(unittest.TestCase):
    """End-to-end tests for the fork's g/b triage feature: atomic image+label
    moves, rollback on failure, and Ctrl+Z undo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        self.good_dir = self.dir + '_good'
        os.makedirs(self.dir)
        for stem in ('a', 'b', 'c'):
            img = QImage(16, 16, QImage.Format_RGB32)
            img.fill(0xffffffff)
            img.save(os.path.join(self.dir, stem + '.png'))
            with open(os.path.join(self.dir, stem + '.xml'), 'w') as f:
                f.write('<annotation></annotation>')

        # Open the directory the same way the CLI does (fork keeps the CLI
        # dir ahead of any remembered lastOpenDir). Patch expanduser for the
        # construction window so Settings() reads a fresh pickle from tmp —
        # the developer's real ~/.labelImgSettings.pkl must steer neither
        # construction (read) nor teardown (write).
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        # Error dialogs would block the test run — record them instead.
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def path(self, base, name):
        return os.path.join(base, name)

    def test_startup_imports_cli_dir(self):
        self.assertEqual(3, len(self.win.m_img_list))
        self.assertEqual(self.path(self.dir, 'a.png'), self.win.file_path)

    def test_classify_moves_image_with_label_and_advances(self):
        self.win.classify_current_image('good')

        self.assertTrue(os.path.isfile(self.path(self.good_dir, 'a.png')))
        self.assertTrue(os.path.isfile(self.path(self.good_dir, 'a.xml')))
        self.assertFalse(os.path.exists(self.path(self.dir, 'a.png')))
        self.assertFalse(os.path.exists(self.path(self.dir, 'a.xml')))
        # advanced to the next image, list rescanned
        self.assertEqual(self.path(self.dir, 'b.png'), self.win.file_path)
        self.assertEqual(2, len(self.win.m_img_list))
        # one history entry holding both moves (image first)
        self.assertEqual(1, len(self.win.classify_history))
        self.assertEqual(2, len(self.win.classify_history[0]))
        self.assertEqual([], self.errors)

    def test_classify_renames_on_collision(self):
        os.makedirs(self.good_dir)
        with open(self.path(self.good_dir, 'a.png'), 'w') as f:
            f.write('occupied')

        self.win.classify_current_image('good')

        self.assertTrue(os.path.isfile(self.path(self.good_dir, 'a_1.png')))
        self.assertTrue(os.path.isfile(self.path(self.good_dir, 'a_1.xml')))
        # the pre-existing file is untouched
        with open(self.path(self.good_dir, 'a.png')) as f:
            self.assertEqual('occupied', f.read())
        self.assertEqual([], self.errors)

    def test_classify_rolls_back_atomically_when_label_move_fails(self):
        real_move = shutil.move

        def failing_move(src, dst, *args, **kwargs):
            # Fail only the outbound label move; rollback moves (back into
            # the source dir) must still succeed.
            if src.endswith('a.xml') and os.path.dirname(dst) == self.good_dir:
                raise OSError('disk full')
            return real_move(src, dst, *args, **kwargs)

        with mock.patch('labelImg.shutil.move', side_effect=failing_move):
            self.win.classify_current_image('good')

        # image and label are both back (or never left) — no half-moved pair
        self.assertTrue(os.path.isfile(self.path(self.dir, 'a.png')))
        self.assertTrue(os.path.isfile(self.path(self.dir, 'a.xml')))
        # the freshly created, now-empty target dir was cleaned up
        self.assertFalse(os.path.exists(self.good_dir))
        # nothing recorded as undoable, user was told, current image unchanged
        self.assertEqual([], self.win.classify_history)
        self.assertEqual(1, len(self.errors))
        self.assertIn('Reverted', self.errors[0][1])
        self.assertEqual(self.path(self.dir, 'a.png'), self.win.file_path)

    def test_undo_classify_restores_pair(self):
        self.win.classify_current_image('good')
        self.assertEqual(self.path(self.dir, 'b.png'), self.win.file_path)

        self.win.undo_classify()

        self.assertTrue(os.path.isfile(self.path(self.dir, 'a.png')))
        self.assertTrue(os.path.isfile(self.path(self.dir, 'a.xml')))
        self.assertFalse(os.path.exists(self.path(self.good_dir, 'a.png')))
        self.assertEqual([], self.win.classify_history)
        # the restored image becomes current again
        self.assertEqual(self.path(self.dir, 'a.png'), self.win.file_path)
        self.assertEqual(3, len(self.win.m_img_list))


if __name__ == '__main__':
    unittest.main()
