#!/usr/bin/env python
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

__author__ = 'TzuTaLin'

dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, '..', 'libs')
sys.path.insert(0, libs_path)
from settings import Settings

class TestSettings(unittest.TestCase):

    def test_basic(self):
        # Settings.__init__ resolves its path from os.path.expanduser("~"):
        # without mocking that, this test wrote test0/test1/test2 into --
        # and then reset() DELETED -- the real developer's
        # ~/.labelImgSettings.pkl (found during a labelImg.py/controller.py
        # ML-assist persistence audit, when a full `python -m unittest
        # discover tests` run silently wiped a real settings file). Route it
        # at a throwaway temp dir instead, the same way every other test
        # module in this suite that touches Settings already does (see
        # tests/test_assist.py's AssistTestCase.launch).
        with tempfile.TemporaryDirectory() as tmp_home:
            with mock.patch('os.path.expanduser', return_value=tmp_home):
                settings = Settings()
                settings['test0'] = 'hello'
                settings['test1'] = 10
                settings['test2'] = [0, 2, 3]
                self.assertEqual(settings.get('test3', 3), 3)
                self.assertEqual(settings.save(), True)

                settings.load()
                self.assertEqual(settings.get('test0'), 'hello')
                self.assertEqual(settings.get('test1'), 10)

                settings.reset()


if __name__ == '__main__':
    unittest.main()
