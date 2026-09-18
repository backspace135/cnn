"""逐课检查浏览器里的学习、反馈和窄屏操作。"""
import unittest

import paths  # noqa: F401 项目依赖仍位于 libs/
from playwright.sync_api import expect, sync_playwright
from flask_server import local_flask_server


class CourseBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = local_flask_server()
        cls.base_url = cls.server.__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(headless=True)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context()
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()

    def test_all_seven_lessons_load_without_mnist(self):
        for number in range(7):
            with self.subTest(lesson=number):
                self.page.goto(self.base_url + f'/#/learn/{number:02d}')
                expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', f'{number:02d}')
                expect(self.page.locator('.lesson-step:visible')).to_have_count(1)
                expect(self.page.locator('#routeStatus')).to_be_hidden()

    def test_stepper_quiz_and_refresh_resume(self):
        self.page.goto(self.base_url + '/#/learn/01')
        self.page.locator('.step-controls button.primary').click()
        expect(self.page.locator('.lesson-step:visible')).to_have_attribute('data-step', '01-2')
        self.page.locator('[data-quiz] button[data-correct="false"]').click()
        feedback = self.page.locator('.answer-feedback')
        expect(feedback).to_contain_text('51 ÷ 255')
        self.assertIn('is-incorrect', feedback.get_attribute('class') or '')
        self.page.locator('[data-quiz] button[data-correct="true"]').click()
        expect(feedback).to_contain_text('答对了')
        self.assertIn('is-correct', feedback.get_attribute('class') or '')
        self.assertEqual(feedback.inner_text().count('答对了'), 1)
        self.page.reload()
        expect(self.page.locator('.lesson-step:visible')).to_have_attribute('data-step', '01-2')
        self.page.locator('.step-controls button').first.click()
        expect(self.page.locator('.lesson-step:visible')).to_have_attribute('data-step', '01-1')

    def test_convolution_first_cell_matches_lesson_number(self):
        self.page.goto(self.base_url + '/#/learn/02')
        slot = self.page.locator('[data-activity="convolution"]')
        expect(slot).to_contain_text('第一格')
        slot.get_by_role('button', name='下一格').click()
        expect(slot).to_contain_text('= 3')
        self.page.locator('.step-controls button.primary').click()
        self.page.locator('[data-quiz] button[data-correct="true"]').click()
        expect(self.page.locator('.answer-feedback')).to_contain_text('权重')

    def test_storage_unavailable_does_not_block_navigation(self):
        self.page.add_init_script("Object.defineProperty(window, 'localStorage', {get(){throw Error('blocked')}})")
        self.page.goto(self.base_url + '/#/learn/01')
        self.page.locator('.step-controls button.primary').click()
        expect(self.page.locator('.lesson-step:visible')).to_have_attribute('data-step', '01-2')
        self.page.locator('a[href="#/learn/02"]').first.click()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '02')

    def test_narrow_screen_has_no_page_level_horizontal_scroll(self):
        for width in (375, 400):
            with self.subTest(width=width):
                self.page.set_viewport_size({'width': width, 'height': 800})
                self.page.goto(self.base_url + '/#/learn/02')
                expect(self.page.locator('.lesson')).to_be_visible()
                overflow = self.page.evaluate('document.documentElement.scrollWidth - window.innerWidth')
                self.assertLessEqual(overflow, 1)
                self.page.goto(self.base_url + '/#/training')
                expect(self.page.locator('#trainingTitle')).to_be_visible()
                overflow = self.page.evaluate('document.documentElement.scrollWidth - window.innerWidth')
                self.assertLessEqual(overflow, 1)
                self.page.goto(self.base_url + '/#/prediction')
                expect(self.page.locator('#predictionTitle')).to_be_visible()
                overflow = self.page.evaluate('document.documentElement.scrollWidth - window.innerWidth')
                self.assertLessEqual(overflow, 1)
    def test_browser_back_and_forward_restore_chapter(self):
        self.page.goto(self.base_url + '/#/learn/00')
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '00')
        self.page.locator('#courseNav a[href="#/learn/01"]').click()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '01')
        self.page.locator('#courseNav a[href="#/learn/02"]').click()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '02')
        self.page.go_back()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '01')
        self.page.go_forward()
        expect(self.page.locator('.lesson')).to_have_attribute('data-lesson', '02')


if __name__ == '__main__':
    unittest.main()
