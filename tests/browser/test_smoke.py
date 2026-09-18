"""独立 Chromium 冒烟测试；tests/browser/ 不作为基础测试包导入。"""
import os
import unittest
from pathlib import Path

import paths  # noqa: F401 先启用项目 libs/，再导入 Playwright
from playwright.sync_api import expect, sync_playwright

from flask_server import local_flask_server

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(ROOT / 'data' / 'playwright-browsers'))


class BrowserSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_context = local_flask_server()
        cls.base_url = cls.server_context.__enter__()
        cls.addClassCleanup(cls.server_context.__exit__, None, None, None)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(headless=True)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context()
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()

    def test_home_opens_first_lesson_without_mnist(self):
        self.page.goto(self.base_url)
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '00')
        expect(self.page.locator('.lesson-step:visible')).to_have_count(1)

    def test_navigation_opens_convolution_lesson(self):
        self.page.goto(self.base_url)
        self.page.locator('#courseNav a[href="#/learn/02"]').click()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '02')

    def test_convolution_next_step_calculates_first_cell(self):
        self.page.goto(self.base_url + '/#/learn/02')
        slot = self.page.locator('[data-activity="convolution"]')
        slot.get_by_role('button', name='下一格').click()
        expect(slot.locator('.matrix-cell').first).to_have_text('3')

    def test_missing_mnist_shows_download_instruction_only_for_training(self):
        self.page.goto(self.base_url)
        expect(self.page.locator('.lesson')).to_be_visible()
        self.page.locator('#courseNav a[href="#/training"]').click()
        expect(self.page.locator('#notice')).to_contain_text('download_data.py')

    def test_unknown_course_link_returns_to_start(self):
        self.page.goto(self.base_url + '/#/unknown')
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '00')
        self.assertTrue(self.page.url.endswith('#/learn/00'))

    def test_activity_mount_receives_scrollable_host(self):
        self.page.goto(self.base_url + '/#/learn/01')
        slot = self.page.locator('[data-activity="pixels"]')
        expect(slot).to_contain_text('点击小格')
        self.assertIn('activity-host', slot.get_attribute('class') or '')

    def test_training_route_mounts_new_page_without_data(self):
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#trainingTitle')).to_be_visible()
        expect(self.page.locator('#notice')).to_contain_text('download_data.py')

    def test_prediction_route_handles_empty_input(self):
        self.page.goto(self.base_url + '/#/prediction')
        expect(self.page.locator('#predictionTitle')).to_be_visible()
        self.page.locator('#predictButton').click()
        expect(self.page.locator('#predictResult')).to_contain_text('输入为空')

    def test_lab_deep_link_focuses_settings(self):
        self.page.goto(self.base_url + '/#/lab')
        expect(self.page.locator('#settings')).to_be_visible()
        expect(self.page.locator('#routeStatus')).to_be_hidden()
        self.assertEqual(self.page.evaluate('document.activeElement.id'), 'settings')

    def test_old_hashes_still_reach_corresponding_pages(self):
        targets = {'intro': ('data-lesson', '01'), 'theory': ('data-lesson', '02'),
                   'model': ('data-lesson', '03'), 'gloss': ('data-lesson', '04'),
                   'live': ('id', 'trainingTitle'), 'predict': ('id', 'predictionTitle'),
                   'lab': ('id', 'settings')}
        for old, (attribute, value) in targets.items():
            with self.subTest(old=old):
                self.page.goto(self.base_url + '/#' + old)
                if attribute == 'data-lesson':
                    expect(self.page.locator('.lesson')).to_have_attribute(attribute, value)
                else:
                    expect(self.page.locator('#' + value)).to_be_visible()
    def test_skip_link_focuses_current_page_without_changing_route(self):
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#routeStatus')).to_be_hidden()
        self.page.locator('.skip-link').focus()
        self.page.keyboard.press('Enter')
        expect(self.page.locator('#trainingTitle')).to_be_visible()
        self.assertTrue(self.page.url.endswith('#/training'))
        self.assertEqual(self.page.evaluate('document.activeElement.id'), 'courseView')


if __name__ == '__main__':
    unittest.main()
