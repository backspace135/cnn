"""训练新页的静态契约：旧首页继续可用，新页独立挂载。"""
import re
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / 'static'
FEATURE = STATIC / 'js' / 'features' / 'training'


class TrainingPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.actions = set()
        self.inline_handlers = []
        self.styles = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if 'data-action' in attrs:
            self.actions.add(attrs['data-action'])
        self.inline_handlers.extend(key for key in attrs if key.startswith('on'))
        if tag == 'link':
            self.styles.add(attrs.get('href'))


class TrainingFrontendTests(unittest.TestCase):
    def setUp(self):
        page_path = STATIC / 'pages' / 'training.html'
        self.assertTrue(page_path.is_file(), 'training.html 尚未创建')
        self.html = page_path.read_text(encoding='utf-8')
        self.page = TrainingPage()
        self.page.feed(self.html)
        self.scripts = {file.name: file.read_text(encoding='utf-8') for file in FEATURE.glob('*.js')}
        self.source = '\n'.join(self.scripts.values())

    def test_view_covers_training_observations_and_settings(self):
        self.assertTrue({'status', 'notice', 'lossChart', 'accChart', 'accBars',
                         'chartTable', 'fmaps', 'confusionMatrix', 'samples', 'smax', 'testResult',
                         'log', 'modelLayers', 'experimentHistory', 'settings',
                         'epochs', 'lr', 'batch', 'dropout', 'device'} <= self.page.ids)
        self.assertTrue({'start', 'pause', 'resume', 'stop', 'reset', 'apply',
                         'save', 'load', 'refresh-history'} <= self.page.actions)
        self.assertIn('/static/css/training.css', self.page.styles)
        self.assertFalse(self.page.inline_handlers)

    def test_module_entry_and_api_contract(self):
        entry = self.scripts['index.js']
        self.assertRegex(entry, r'export\s+function\s+mount\(root,\s*\{\s*focus\s*\}\s*=\s*\{\s*\}\)')
        self.assertIn('return () =>', entry)
        self.assertIn("../../core/api.js", entry)
        for endpoint in ('/api/state', '/api/control', '/api/model-info',
                         '/api/experiments', '/api/model'):
            self.assertIn(endpoint, self.source)
        self.assertIn('700', entry)
        self.assertIn('AbortController', entry)
        self.assertIn('clearTimeout(timer)', entry)
        self.assertIn('abort.abort()', entry)
        self.assertIn("request('/api/state', {signal})", entry)
        self.assertIn('setTimeout(poll, 700)', entry)
        self.assertIn('pending || button.disabled', entry)
        self.assertIn('fmaps_version', self.scripts['results.js'])
        self.assertNotIn('window.', self.source)
        self.assertNotIn('innerHTML', self.source)
        self.assertTrue({'index.js', 'controls.js', 'charts.js', 'results.js',
                         'details.js'} <= self.scripts.keys())

    def test_settings_and_resume_controls_execute_without_browser(self):
        if not shutil.which('node'):
            self.skipTest('Node.js 未安装，无法执行模块行为测试')
        script = r"""
import assert from 'node:assert/strict';
import {readSettings, resetSettings, renderControls} from './js/features/training/controls.js';
const fields = new Map(Object.entries({
  epochs: {value: '6'}, lr: {value: '0.001'}, batch: {value: '64'},
  dropout: {value: '0.3'}, device: {value: 'auto', querySelector: () => ({disabled: false})},
  deviceHint: {textContent: ''},
}));
for (const id of ['epochs', 'lr', 'batch', 'dropout']) fields.get(id).reportValidity = () => true;
const actions = new Map(['start', 'resume', 'pause', 'stop', 'reset', 'apply', 'save', 'load']
  .map(name => [name, {disabled: false, hidden: false}]));
const root = {querySelector: selector => selector[0] === '#'
  ? fields.get(selector.slice(1)) : actions.get(selector.match(/data-action="([^"]+)"/)[1])};
assert.deepEqual(readSettings(root), {epochs: 6, lr: 0.001, batch_size: 64,
  dropout: 0.3, device_mode: 'auto'});
fields.get('epochs').reportValidity = () => false;
assert.equal(readSettings(root), null);
fields.get('epochs').reportValidity = () => true;
assert.deepEqual(resetSettings(root, 'cpu'), {epochs: 6, lr: 0.001,
  batch_size: 64, dropout: 0.3, device_mode: 'cpu'});
renderControls(root, {status: 'paused', hardware: {cuda_available: false}}, false, false);
assert.equal(actions.get('start').hidden, true);
assert.equal(actions.get('resume').hidden, false);
assert.equal(actions.get('resume').disabled, false);
assert.equal(actions.get('pause').disabled, true);
renderControls(root, {status: 'running', hardware: {}}, true, false);
assert.equal(actions.get('pause').disabled, true);
assert.equal(actions.get('reset').disabled, true);
console.log('training controls: passed');
"""
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                cwd=STATIC, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('training controls: passed', result.stdout)

    def test_every_scoped_dom_lookup_has_a_target(self):
        for filename, source in self.scripts.items():
            for target in re.findall(r"\$\('([^']+)'\)", source):
                self.assertIn(target, self.page.ids, f'{filename}: #{target}')


if __name__ == '__main__':
    unittest.main()
