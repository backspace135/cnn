"""新课程壳的静态契约；浏览器行为由独立浏览器测试覆盖。"""
import unittest
from html.parser import HTMLParser
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.modules = set()
        self.inline_handlers = []
        self.articles = []
        self.steps = []
        self.quizzes = 0
        self.forbidden = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "article" and "data-lesson" in values:
            self.articles.append(values["data-lesson"])
        if tag == "section" and "data-step" in values:
            self.steps.append(values["data-step"])
        if "data-quiz" in values:
            self.quizzes += 1
        if tag in {"script", "iframe"}:
            self.forbidden.append(tag)
        if tag == "script" and values.get("type") == "module":
            self.modules.add(values.get("src"))
        self.inline_handlers.extend(name for name in values if name.startswith("on"))


class CourseShellTests(unittest.TestCase):
    def test_home_shell_has_single_module_and_content_region(self):
        page = STATIC / "index.html"
        self.assertTrue(page.is_file())
        elements = Elements()
        elements.feed(page.read_text(encoding="utf-8"))
        self.assertIn("courseView", elements.ids)
        self.assertEqual({"/static/js/main.js"}, elements.modules)
        self.assertFalse(elements.inline_handlers)

    def test_registry_contains_every_lesson_and_feature_route(self):
        source = (STATIC / "js" / "core" / "lesson-registry.js").read_text(encoding="utf-8")
        for number in range(7):
            self.assertIn(f"{number:02d}-", source)
        for route in ("training", "prediction", "lab"):
            self.assertRegex(source, rf"\b{route}\b")

    def test_core_modules_have_no_inline_global_handlers(self):
        for name in ("main.js", "core/router.js", "core/api.js", "core/lesson-loader.js"):
            source = (STATIC / "js" / name).read_text(encoding="utf-8")
            self.assertNotRegex(source, r"window\.(?:control|modelAction|runLab)\s*=")
            self.assertNotRegex(source, r"\bonclick\s*=")
    def test_seven_lessons_have_unique_steps_and_self_checks(self):
        names = ["00-start", "01-pixels", "02-patterns", "03-network",
                 "04-learning", "05-evaluation", "06-try-it"]
        all_steps = []
        for number, name in enumerate(names):
            with self.subTest(lesson=name):
                page = STATIC / "lessons" / f"{name}.html"
                self.assertTrue(page.is_file(), f"缺少课程 {name}")
                elements = Elements()
                elements.feed(page.read_text(encoding="utf-8"))
                self.assertEqual([f"{number:02d}"], elements.articles)
                self.assertGreaterEqual(len(elements.steps), 2)
                self.assertGreaterEqual(elements.quizzes, 1)
                self.assertFalse(elements.inline_handlers)
                self.assertFalse(elements.forbidden)
                all_steps.extend(elements.steps)
        self.assertEqual(len(all_steps), len(set(all_steps)), "课程步骤 ID 必须唯一")
    def test_pixel_activity_appears_after_zero_to_one_explanation(self):
        page = (STATIC / "lessons" / "01-pixels.html").read_text(encoding="utf-8")
        first = page.split('data-step="01-1"', 1)[1].split('</section>', 1)[0]
        second = page.split('data-step="01-2"', 1)[1].split('</section>', 1)[0]
        self.assertNotIn('data-activity="pixels"', first)
        self.assertIn('data-activity="pixels"', second)


if __name__ == "__main__":
    unittest.main()
