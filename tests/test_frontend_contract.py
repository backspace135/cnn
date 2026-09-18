"""无需浏览器的静态页面/API 契约检查；交互仍需手工浏览器验收。"""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'static'


class PageIds(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.scripts = set()
        self.anchors = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'script' and 'src' in attrs:
            self.scripts.add(attrs['src'])
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.anchors.add(attrs['href'][1:])


class FrontendContractTests(unittest.TestCase):
    def test_every_static_js_target_and_navigation_anchor_exists(self):
        parser = PageIds()
        parser.feed((STATIC / 'index.html').read_text(encoding='utf-8'))
        for name in ('app', 'lessons', 'predict'):
            self.assertIn(f'/static/{name}.js', parser.scripts)
            source = (STATIC / f'{name}.js').read_text(encoding='utf-8')
            for target in re.findall(r"\$\('([^']+)'\)|byId\('([^']+)'\)", source):
                referenced = target[0] or target[1]
                self.assertIn(referenced, parser.ids, f'{name}.js refers to missing #{referenced}')
        self.assertTrue(parser.anchors.issubset(parser.ids))

    def test_learning_sections_are_present(self):
        parser = PageIds()
        parser.feed((STATIC / 'index.html').read_text(encoding='utf-8'))
        self.assertTrue({'intro', 'theory', 'model', 'live', 'predict', 'lab',
                         'confusionMatrix', 'modelLayers', 'experimentHistory'} <= parser.ids)


if __name__ == '__main__':
    unittest.main()
