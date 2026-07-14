
import tempfile
from unittest import TestCase, mock

from labelImg import get_main_app


class TestMainWindow(TestCase):

    app = None
    win = None

    def setUp(self):
        # Settings.__init__ resolves its path from os.path.expanduser("~"):
        # without mocking that, tearDown's close() (which runs closeEvent ->
        # settings.save()) wrote real window/session state into -- and
        # `test_settings.py`'s unmocked test_basic even DELETED -- the real
        # developer's ~/.labelImgSettings.pkl (found during a labelImg.py/
        # controller.py ML-assist persistence audit, when a full
        # `python -m unittest discover tests` run silently clobbered a real
        # settings file). Route it at a throwaway temp dir instead, the same
        # way every other test module that touches Settings already does
        # (see tests/test_assist.py's AssistTestCase.launch, tests/test_classify.py).
        self._tmp_home = tempfile.TemporaryDirectory()
        self._expanduser_patcher = mock.patch(
            'os.path.expanduser', return_value=self._tmp_home.name)
        self._expanduser_patcher.start()
        self.app, self.win = get_main_app()

    def tearDown(self):
        self.win.close()
        self.app.quit()
        self._expanduser_patcher.stop()
        self._tmp_home.cleanup()

    def test_noop(self):
        pass
