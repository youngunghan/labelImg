import os
import sys
import unittest
import resources
from stringBundle import StringBundle

class TestStringBundle(unittest.TestCase):

    def test_loadDefaultBundle_withoutError(self):
        str_bundle = StringBundle.get_bundle('en')
        self.assertEqual(str_bundle.get_string("openDir"), 'Open Dir', 'Fail to load the default bundle')

    def test_fallback_withoutError(self):
        str_bundle = StringBundle.get_bundle('zh-TW')
        self.assertEqual(str_bundle.get_string("openDir"), u'\u958B\u555F\u76EE\u9304', 'Fail to load the zh-TW bundle')

    def test_setInvaleLocaleToEnv_printErrorMsg(self):
        # 환경에 LC_ALL/LANG이 없어도 동작하도록 get()으로 안전 조회 후 원복.
        prev_lc = os.environ.get('LC_ALL')
        prev_lang = os.environ.get('LANG')
        os.environ['LC_ALL'] = 'UTF-8'
        os.environ['LANG'] = 'UTF-8'
        try:
            str_bundle = StringBundle.get_bundle()
            self.assertEqual(str_bundle.get_string("openDir"), 'Open Dir', 'Fail to load the default bundle')
        finally:
            if prev_lc is None:
                os.environ.pop('LC_ALL', None)
            else:
                os.environ['LC_ALL'] = prev_lc
            if prev_lang is None:
                os.environ.pop('LANG', None)
            else:
                os.environ['LANG'] = prev_lang


if __name__ == '__main__':
    unittest.main()
