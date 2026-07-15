import os
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
from libs.inference.backend import ModelBackend
from libs.inference.service import SynchronousExecutor
from libs.inference.stub import image_size
from libs.inference.types import Detection, least_confidence

# StubBackend's score sequence (0.9, 0.8, ...) depends only on detection
# INDEX/COUNT, not on image content -- every image gets the same uncertainty
# under a fixed StubBackend instance. Sort/order tests need genuinely
# different per-image scores, so this backend keys its response off image
# WIDTH instead (the fixtures below are square images sized to match).
WIDTH_BLANK = 32       # no detections -> uncertainty 1.0 (least_confidence's
                        # documented max-uncertainty case)
WIDTH_MID = 96          # one detection at 0.5 -> uncertainty 0.5
WIDTH_CONFIDENT = 64    # one detection at 0.95 -> uncertainty 0.05
WIDTH_OTHER = 128       # one detection at 0.8 -> uncertainty 0.2 (used, then
                        # its score is deliberately dropped to simulate an
                        # image the batch never reached)


class VaryingUncertaintyStub(ModelBackend):
    """Deterministic per-image detections keyed off image width.

    Unlike StubBackend (whose scores are a pure function of detection COUNT,
    identical for every image), this backend gives genuinely different
    uncertainty to differently-sized fixtures, which is what the ordering/
    sort tests need to be meaningful rather than checking a tie.
    """

    name = 'varying'
    supports_detection = True
    supports_segmentation = False

    def predict(self, image):
        height, width = image_size(image)
        if width == WIDTH_BLANK:
            return []
        if width == WIDTH_MID:
            return [Detection(label='x', box=(0, 0, 1, 1), score=0.5)]
        if width == WIDTH_CONFIDENT:
            return [Detection(label='x', box=(0, 0, 1, 1), score=0.95)]
        return [Detection(label='x', box=(0, 0, 1, 1), score=0.8)]


def _deferred_executor():
    """An executor that records jobs instead of running them, so a test can
    control exactly when each batch step "completes" -- mirrors the idiom in
    tests/test_assist.py's TestStaleResults.test_navigating_away_mid_inference_drops_the_result.
    """
    jobs = []
    executor = type('Deferred', (), {
        'submit': lambda _self, job: jobs.append(job),
        'wait_for_done': lambda _self, msecs=0: True,
    })()
    return executor, jobs


class ActiveLearningTestCase(unittest.TestCase):
    """Same idiom as tests/test_assist.py's AssistTestCase: a real MainWindow
    headless, a deterministic backend + SynchronousExecutor injected so
    inference is immediate and assertions never race a worker thread."""

    # (stem, width) -- alphabetical stem order is also the natural scan
    # order scan_all_images produces (natural_sort on lowercase filename).
    FILES = (('a', WIDTH_BLANK), ('b', WIDTH_MID), ('c', WIDTH_CONFIDENT),
             ('d', WIDTH_OTHER))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        for stem, width in self.FILES:
            img = QImage(width, width, QImage.Format_RGB32)
            img.fill(0xffffffff)
            img.save(os.path.join(self.dir, stem + '.png'))
        self.win = None
        self.launch()

    def launch(self, backend=None):
        if self.win is not None:
            self.win.close()
        with mock.patch('os.path.expanduser', return_value=self.tmp.name):
            self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

        # Inference runs inline on this thread by default; individual tests
        # swap in a deferred executor where they need to control timing.
        self.win.inference_service.set_executor(SynchronousExecutor())
        self.backend = backend if backend is not None else VaryingUncertaintyStub()
        self.win.assist.set_backend(self.backend)
        self.win.default_save_dir = self.dir

    def tearDown(self):
        self.win.dirty = False
        self.win.close()
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name)

    def score_folder_sync(self):
        """Run a full batch scan to completion (SynchronousExecutor is
        already installed, so score_folder() drives every step inline)."""
        return self.win.assist.score_folder()


class TestLeastConfidenceOrdering(unittest.TestCase):
    """Spec-conformance check on the scoring function itself (types.py:118),
    independent of the app: higher-confidence detections must yield lower
    uncertainty, and "no detections" must be the maximum."""

    def test_higher_confidence_detections_yield_lower_uncertainty(self):
        confident = [Detection(label='x', box=(0, 0, 1, 1), score=0.95),
                    Detection(label='x', box=(0, 0, 1, 1), score=0.90)]
        unsure = [Detection(label='x', box=(0, 0, 1, 1), score=0.40),
                 Detection(label='x', box=(0, 0, 1, 1), score=0.30)]

        self.assertLess(least_confidence(confident), least_confidence(unsure))

    def test_no_detections_is_more_uncertain_than_any_real_prediction(self):
        unsure = [Detection(label='x', box=(0, 0, 1, 1), score=0.05)]

        self.assertEqual(1.0, least_confidence([]))
        self.assertGreater(least_confidence([]), least_confidence(unsure))


class TestBatchScoring(ActiveLearningTestCase):

    def test_scoring_populates_a_score_for_every_image_in_m_img_list(self):
        self.assertEqual(4, len(self.win.m_img_list))

        self.score_folder_sync()

        scores = self.win.assist._uncertainty
        self.assertEqual(4, len(scores))
        for path in self.win.m_img_list:
            self.assertIn(path, scores)
        self.assertAlmostEqual(1.0, scores[self.path('a.png')])   # no detections
        self.assertAlmostEqual(0.5, scores[self.path('b.png')])
        self.assertAlmostEqual(0.05, scores[self.path('c.png')])
        self.assertAlmostEqual(0.2, scores[self.path('d.png')])

    def test_batch_result_for_a_non_current_image_is_recorded_not_dropped(self):
        # DESIGN POINT B: the folder is opened on a.png (alphabetically
        # first), so b.png/c.png/d.png are NOT the current image while the
        # batch scores them -- exactly the case the interactive stale-result
        # guard (_is_current) would otherwise drop.
        self.assertEqual(self.path('a.png'), self.win.file_path)

        self.score_folder_sync()

        for name in ('b.png', 'c.png', 'd.png'):
            self.assertIn(self.path(name), self.win.assist._uncertainty)
        # And it did NOT get funnelled into the interactive suggestion flow
        # for the current image either -- no suggestions were requested.
        self.assertEqual([], self.win.canvas.shapes)

    def test_batch_result_for_the_current_image_is_claimed_by_the_batch_not_the_interactive_flow(self):
        # A step-by-step drive with a deferred executor: even when the image
        # being scored IS the currently open one, the result must be
        # consumed by the batch collector (recorded in _uncertainty), not by
        # on_prediction_ready's interactive branch (which would otherwise
        # start populating provisional suggestions no one asked for).
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)

        self.win.assist.score_folder()
        self.assertEqual(1, len(jobs))
        self.assertEqual(self.path('a.png'), self.win.assist._batch_current_path)
        self.assertEqual(self.path('a.png'), self.win.file_path)

        jobs.pop(0)()  # a.png's result arrives -- current image AND batch target

        self.assertIn(self.path('a.png'), self.win.assist._uncertainty)
        self.assertEqual([], self.win.canvas.shapes, 'batch result leaked into suggestions')
        # Batch advanced to the next image.
        self.assertEqual(self.path('b.png'), self.win.assist._batch_current_path)

    def test_interactive_auto_label_is_disabled_while_a_batch_is_running(self):
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)

        self.win.assist.score_folder()

        self.assertTrue(self.win.assist._batch_active)
        self.assertFalse(self.win.assist.action_auto.isEnabled())
        # Draining the run re-enables it.
        while jobs:
            jobs.pop(0)()
        self.assertFalse(self.win.assist._batch_active)
        self.assertTrue(self.win.assist.action_auto.isEnabled())

    def test_status_bar_shows_progress_while_scoring(self):
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)
        self.win.assist.score_folder()

        self.assertIn('Scoring', self.win.statusBar().currentMessage())


class TestSortByUncertainty(ActiveLearningTestCase):

    def test_sort_reorders_descending_with_no_detection_image_first(self):
        self.score_folder_sync()

        self.assertTrue(self.win.assist.sort_by_uncertainty())

        # a=1.0 (no detections) > b=0.5 > d=0.2 > c=0.05
        expected = [self.path(n) for n in ('a.png', 'b.png', 'd.png', 'c.png')]
        self.assertEqual(expected, self.win.m_img_list)

    def test_unscored_images_sort_to_the_end_preserving_relative_order(self):
        self.score_folder_sync()
        # Simulate an image the batch never reached (e.g. a cancelled run,
        # or a file that appeared after scoring): drop its recorded score.
        del self.win.assist._uncertainty[self.path('d.png')]
        self.win.assist._invalidate_ranks()

        self.win.assist.sort_by_uncertainty()

        expected = [self.path(n) for n in ('a.png', 'b.png', 'c.png', 'd.png')]
        self.assertEqual(expected, self.win.m_img_list)

    def test_sort_disabled_and_no_op_before_any_scoring(self):
        self.assertFalse(self.win.assist.action_sort.isEnabled())
        self.assertFalse(self.win.assist.sort_by_uncertainty())
        self.assertEqual([self.path(n + '.png') for n, _ in self.FILES], self.win.m_img_list)

    def test_currently_open_image_stays_selected_after_sort(self):
        self.score_folder_sync()
        self.win.open_next_image()  # a -> b
        self.win.open_next_image()  # b -> c
        self.assertEqual(self.path('c.png'), self.win.file_path)

        self.win.assist.sort_by_uncertainty()

        self.assertEqual(self.path('c.png'), self.win.file_path,
                         'sorting must not silently change the open image')
        self.assertEqual(self.win.file_path, self.win.m_img_list[self.win.cur_img_idx],
                         'cur_img_idx must be re-derived for the new position')

    def test_restore_original_order(self):
        self.score_folder_sync()
        self.win.assist.sort_by_uncertainty()
        self.assertNotEqual([self.path(n + '.png') for n, _ in self.FILES], self.win.m_img_list)

        self.assertTrue(self.win.assist.restore_original_order())

        self.assertEqual([self.path(n + '.png') for n, _ in self.FILES], self.win.m_img_list)

    def test_restore_is_available_and_a_no_op_before_any_sort(self):
        # A fresh scan always seeds _original_order (on_directory_scanned),
        # so restore is available as soon as a folder is open, even with no
        # sort yet -- but it is a no-op (nothing to restore FROM a change).
        self.assertTrue(self.win.assist.action_restore_order.isEnabled())
        self.assertTrue(self.win.assist.restore_original_order())
        self.assertEqual([self.path(n + '.png') for n, _ in self.FILES], self.win.m_img_list)


class TestReorderedNavigationAndClassify(ActiveLearningTestCase):
    """After sort_by_uncertainty reorders m_img_list, the EXISTING navigation
    and g/b triage machinery (which only ever indexes m_img_list by
    position) must keep working unchanged."""

    def setUp(self):
        super(TestReorderedNavigationAndClassify, self).setUp()
        self.score_folder_sync()
        self.win.assist.sort_by_uncertainty()
        # New order: a(1.0), b(0.5), d(0.2), c(0.05) -- see test_sort_reorders...
        self.assertEqual([self.path(n) for n in ('a.png', 'b.png', 'd.png', 'c.png')],
                         self.win.m_img_list)

    def test_navigation_follows_the_new_order(self):
        self.assertEqual(self.path('a.png'), self.win.file_path)

        self.win.open_next_image()
        self.assertEqual(self.path('b.png'), self.win.file_path)

        self.win.open_next_image()
        self.assertEqual(self.path('d.png'), self.win.file_path)

        self.win.open_prev_image()
        self.assertEqual(self.path('b.png'), self.win.file_path)

    def test_classify_move_advances_correctly_on_the_reordered_list(self):
        # Current image (a.png, rank 1) is moved out; the app must advance to
        # what is now the front of the queue -- b.png -- not to whatever
        # would be alphabetically next.
        self.win.classify_current_image('good')

        self.assertTrue(os.path.isfile(os.path.join(self.dir + '_good', 'a.png')))
        self.assertFalse(os.path.exists(self.path('a.png')))
        self.assertEqual(self.path('b.png'), self.win.file_path)

    def test_classify_move_preserves_the_uncertainty_order_for_remaining_images(self):
        # THE key coherence property: g/b is the normal way a user works
        # through a sorted queue, so the order must survive a move, not
        # silently revert to filesystem order (see
        # AssistController.reapply_sort_if_active).
        self.win.classify_current_image('good')  # removes a.png

        self.assertEqual([self.path(n) for n in ('b.png', 'd.png', 'c.png')],
                         self.win.m_img_list)

    def test_save_file_path_still_derives_from_the_open_image_after_reorder(self):
        self.win.open_next_image()  # -> b.png
        self.win.save_file()

        self.assertTrue(os.path.isfile(os.path.splitext(self.path('b.png'))[0] + '.xml'))


class TestCancellation(ActiveLearningTestCase):

    def test_cancel_stops_the_run_keeping_partial_scores(self):
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)

        self.win.assist.score_folder()
        jobs.pop(0)()  # a.png scored
        self.assertEqual(1, len(self.win.assist._uncertainty))
        self.assertTrue(self.win.assist._batch_active)

        self.assertTrue(self.win.assist.cancel_batch_scoring())

        self.assertFalse(self.win.assist._batch_active)
        self.assertEqual(1, len(self.win.assist._uncertainty))  # a.png's score kept

    def test_a_late_result_after_cancellation_does_not_crash_or_reactivate_the_batch(self):
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)

        self.win.assist.score_folder()
        self.assertEqual(1, len(jobs))
        self.win.assist.cancel_batch_scoring()

        # The in-flight request from before cancellation cannot be
        # interrupted mid-job in a real thread pool -- it still resolves.
        jobs.pop(0)()

        self.assertFalse(self.win.assist._batch_active)
        self.assertEqual([], self.errors)
        # The app is still fully usable afterwards.
        self.assertTrue(self.win.assist.action_score_folder.isEnabled())
        self.assertTrue(self.win.assist.score_folder())

    def test_second_trigger_while_running_cancels(self):
        executor, jobs = _deferred_executor()
        self.win.inference_service.set_executor(executor)

        self.win.assist.score_folder()
        self.assertTrue(self.win.assist._batch_active)

        result = self.win.assist.score_folder()  # second trigger == cancel

        self.assertTrue(result, 'score_folder() delegates to cancel_batch_scoring(), which succeeded')
        self.assertFalse(self.win.assist._batch_active)


class TestNoBackendActiveLearning(unittest.TestCase):
    """Mirrors tests/test_assist.py's TestNoBackend: build_backend() returning
    None is a normal outcome and the new Score/Sort/Restore actions must
    degrade the same way the existing AI actions do -- disabled, no crash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'photos')
        os.makedirs(self.dir)
        img = QImage(32, 32, QImage.Format_RGB32)
        img.fill(0xffffffff)
        img.save(os.path.join(self.dir, 'a.png'))
        self.win = None
        with mock.patch('libs.assist.controller.build_backend', return_value=None):
            with mock.patch('os.path.expanduser', return_value=self.tmp.name):
                self.app, self.win = get_main_app([sys.argv[0], self.dir])
        self.win.settings.path = os.path.join(self.tmp.name, 'settings.pkl')
        self.errors = []
        self.win.error_message = lambda title, msg: self.errors.append((title, msg))

    def tearDown(self):
        self.win.dirty = False
        self.win.close()
        self.tmp.cleanup()

    def test_actions_disabled(self):
        self.assertFalse(self.win.assist.is_available())
        self.assertFalse(self.win.assist.action_score_folder.isEnabled())
        self.assertFalse(self.win.assist.action_sort.isEnabled())
        self.assertFalse(self.win.assist.action_restore_order.isEnabled())

    def test_score_folder_does_not_crash_without_a_backend(self):
        self.assertFalse(self.win.assist.score_folder())
        self.assertEqual({}, self.win.assist._uncertainty)
        self.assertEqual([], self.errors)

    def test_sort_does_not_crash_without_a_backend(self):
        self.assertFalse(self.win.assist.sort_by_uncertainty())
        self.assertEqual([], self.errors)


if __name__ == '__main__':
    unittest.main()
