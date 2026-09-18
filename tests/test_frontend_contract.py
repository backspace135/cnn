"""正式首页与课程片段的资源契约；交互由独立浏览器测试覆盖。"""
import unittest
from html.parser import HTMLParser
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / 'static'


class PageAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.scripts = []
        self.styles = []
        self.anchors = []
        self.inline_handlers = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'script':
            self.scripts.append((attrs.get('type'), attrs.get('src')))
        if tag == 'link' and attrs.get('rel') == 'stylesheet':
            self.styles.append(attrs.get('href'))
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.anchors.append(attrs['href'])
        self.inline_handlers.extend(key for key in attrs if key.startswith('on'))


class FrontendContractTests(unittest.TestCase):
    def test_home_is_one_module_shell_with_local_assets(self):
        page = PageAssets()
        page.feed((STATIC / 'index.html').read_text(encoding='utf-8'))
        self.assertTrue({'courseView', 'courseNav', 'routeStatus'} <= page.ids)
        self.assertEqual([('module', '/static/js/main.js')], page.scripts)
        self.assertFalse(page.inline_handlers)
        self.assertEqual({'#courseView', '#/learn/00'}, set(page.anchors))
        for resource in [source for _, source in page.scripts] + page.styles:
            with self.subTest(resource=resource):
                self.assertTrue(resource.startswith('/static/'))
                self.assertTrue((STATIC / resource.removeprefix('/static/')).is_file())

    def test_old_course_hashes_are_routed_not_dead_links(self):
        registry = (STATIC / 'js/core/lesson-registry.js').read_text(encoding='utf-8')
        for old in ('intro', 'model', 'gloss', 'theory', 'live', 'predict', 'lab'):
            with self.subTest(old=old):
                self.assertIn(f'{old}:', registry)
        for feature in ('training', 'prediction'):
            self.assertTrue((STATIC / f'pages/{feature}.html').is_file())
            self.assertTrue((STATIC / f'js/features/{feature}/index.js').is_file())


if __name__ == '__main__':
    unittest.main()
