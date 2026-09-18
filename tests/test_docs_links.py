"""检查入门文档的本地链接，防止整理目录后留下失效路径。"""
import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(?<!!)\[[^\]]+\]\(([^)]+)\)')


def heading_anchors(markdown):
    anchors = set()
    for line in markdown.splitlines():
        if not line.startswith('#'):
            continue
        title = line.lstrip('#').strip().lower()
        title = re.sub(r'[^\w\s-]', '', title)
        anchors.add(re.sub(r'\s+', '-', title))
    return anchors


class DocumentationLinksTests(unittest.TestCase):
    def test_all_local_document_links_exist(self):
        documents = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
        for document in documents:
            source = document.read_text(encoding='utf-8')
            for link in LINK.findall(source):
                if link.startswith(('http:', 'https:', 'mailto:', '#')):
                    continue
                path, _, fragment = unquote(link).partition('#')
                target = (document.parent / path).resolve()
                with self.subTest(document=document.relative_to(ROOT), link=link):
                    self.assertTrue(target.is_file(), f'失效的本地链接：{link}')
                    if fragment and target.suffix == '.md':
                        anchors = heading_anchors(target.read_text(encoding='utf-8'))
                        self.assertIn(fragment.lower(), anchors, f'找不到标题锚点：{link}')


if __name__ == '__main__':
    unittest.main()
