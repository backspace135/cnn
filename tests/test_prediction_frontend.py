"""预测页的静态契约和无依赖 Canvas 预处理回归测试。"""
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'static/pages/prediction.html'
FEATURE = ROOT / 'static/js/features/prediction'
CSS = ROOT / 'static/css/prediction.css'


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.labels = set()
        self.inputs = {}
        self.links = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'label':
            self.labels.add(attrs.get('for'))
        if tag == 'input':
            self.inputs[attrs.get('id')] = attrs
        if tag == 'link':
            self.links.append(attrs.get('href'))


class PredictionFrontendTests(unittest.TestCase):
    def test_fragment_is_self_contained_and_upload_has_keyboard_access(self):
        parser = Elements()
        parser.feed(PAGE.read_text(encoding='utf-8'))
        self.assertIn('/static/css/prediction.css', parser.links)
        self.assertTrue(CSS.is_file())
        self.assertTrue({'drawCanvas', 'clearDraw', 'brushSize', 'uploadDigit',
                         'invertImage', 'inputPreview', 'predictButton',
                         'predictResult', 'predictProbs'} <= parser.ids)
        self.assertEqual(parser.inputs['uploadDigit']['type'], 'file')
        self.assertIn('uploadDigit', parser.labels)
        self.assertNotIn('tabindex', parser.inputs['uploadDigit'])
        self.assertIn('image/png,image/jpeg', parser.inputs['uploadDigit']['accept'])

    def test_modules_are_scoped_and_only_send_matrix(self):
        modules = {name: (FEATURE / f'{name}.js').read_text(encoding='utf-8')
                   for name in ('index', 'board', 'preprocess', 'results')}
        self.assertIn('export function mount(root)', modules['index'])
        self.assertIn("from '../../core/api.js'", modules['index'])
        self.assertIn("request('/api/predict'", modules['index'])
        self.assertIn('body: {pixels}', modules['index'])
        self.assertIn('AbortController', modules['index'])
        self.assertIn('URL.revokeObjectURL', modules['index'])
        self.assertIn('5 * 1024 * 1024', modules['index'])
        for source in modules.values():
            self.assertNotIn('window.modelAction', source)
            self.assertNotIn('fetch(', source)
            self.assertNotIn('document.getElementById', source)
        self.assertNotIn('FormData', modules['index'])
        self.assertNotIn('readAsDataURL', modules['index'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js 仅用于开发测试，学员无需安装')
    def test_javascript_syntax(self):
        for path in sorted(FEATURE.glob('*.js')):
            with self.subTest(path=path.name):
                result = subprocess.run(['node', '--check', str(path)],
                                        capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js 仅用于开发测试，学员无需安装')
    def test_preprocessing_crops_inverts_and_centers(self):
        script = r"""
import assert from 'node:assert/strict';
import {preprocess} from '__MODULE__';
function canvas(width, height, data) {
  const surface = {width, height, pixels: data || new Uint8ClampedArray(width * height * 4)};
  const context = {
    getImageData: () => ({data: surface.pixels}),
    createImageData: (w, h) => ({data: new Uint8ClampedArray(w * h * 4)}),
    putImageData: image => { surface.pixels = image.data; },
    fillRect: () => { surface.pixels = new Uint8ClampedArray(surface.width * surface.height * 4); },
    drawImage: (src, dx, dy, dw, dh) => {
      for (let y = 0; y < surface.height; y++) for (let x = 0; x < surface.width; x++) {
        if (x < dx || y < dy || x >= dx + dw || y >= dy + dh) continue;
        const sx = Math.floor((x - dx) * src.width / dw);
        const sy = Math.floor((y - dy) * src.height / dh);
        const from = (sy * src.width + sx) * 4;
        const to = (y * surface.width + x) * 4;
        surface.pixels.set(src.pixels.subarray(from, from + 4), to);
      }
    },
  };
  surface.getContext = () => context;
  return surface;
}
globalThis.document = {createElement: () => canvas(0, 0)};
const raw = new Uint8ClampedArray(3 * 3 * 4);
for (let i = 0; i < 9; i++) raw.set([255, 255, 255, 255], i * 4);
raw.set([0, 0, 0, 255], 4 * 4);
const matrix = preprocess(canvas(3, 3, raw), true);
assert.equal(matrix.length, 28);
assert(matrix.every(row => row.length === 28 && row.every(v => v >= 0 && v <= 1)));
assert.equal(matrix[0][0], 0);
assert.equal(matrix[14][14], 1);
const blank = new Uint8ClampedArray(3 * 3 * 4);
assert.throws(() => preprocess(canvas(3, 3, blank), false), /输入为空/);
""".replace('__MODULE__', (FEATURE / 'preprocess.js').as_uri())
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
    @unittest.skipUnless(shutil.which('node'), 'Node.js 仅用于开发测试，学员无需安装')
    def test_cleanup_aborts_request_and_upload_stays_local(self):
        script = r"""
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const dir = '__DIR__';
let code = readFileSync(dir + '/index.js', 'utf8');
code = code.replace("import {request} from '../../core/api.js';", 'const request = (...args) => globalThis.mockRequest(...args);');
for (const name of ['board', 'preprocess', 'results']) {
  code = code.replace(`'./${name}.js'`, `'${pathToFileURL(dir + '/' + name + '.js')}'`);
}
const {mount} = await import('data:text/javascript,' + encodeURIComponent(code));
class Element extends EventTarget {
  constructor() { super(); this.value = ''; this.disabled = false; this.textContent = ''; this.children = []; this.style = {}; }
  replaceChildren() { this.children = []; }
  appendChild(child) { this.children.push(child); }
  append(...children) { this.children.push(...children); }
}
const doc = {createElement(tag) {
  if (tag === 'canvas') return makeCanvas(0, 0);
  const item = new Element(); item.ownerDocument = doc; return item;
}};
function makeCanvas(width, height) {
  const item = new Element();
  item.width = width; item.height = height;
  item.pixels = new Uint8ClampedArray(width * height * 4);
  item.ownerDocument = doc;
  item.getBoundingClientRect = () => ({left: 0, top: 0, width: item.width, height: item.height});
  item.setPointerCapture = () => {};
  const context = {
    getImageData: () => ({data: item.pixels}),
    createImageData: (w, h) => ({data: new Uint8ClampedArray(w * h * 4)}),
    putImageData: image => { item.pixels = image.data; },
    fillRect: () => { item.pixels = new Uint8ClampedArray(item.width * item.height * 4); },
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    drawImage(src, dx, dy, dw, dh) {
      if (dw === undefined) { item.pixels = src.pixels.slice(); return; }
      for (let y = 0; y < item.height; y++) for (let x = 0; x < item.width; x++) {
        if (x < dx || x >= dx + dw || y < dy || y >= dy + dh) continue;
        const sx = Math.floor((x - dx) * src.width / dw);
        const sy = Math.floor((y - dy) * src.height / dh);
        item.pixels.set(src.pixels.subarray((sy * src.width + sx) * 4, (sy * src.width + sx) * 4 + 4),
          (y * item.width + x) * 4);
      }
    },
  };
  item.getContext = () => context;
  return item;
}
const ids = Object.fromEntries(['brushSize', 'uploadDigit', 'invertImage', 'inputPreview',
  'predictButton', 'predictResult', 'predictProbs', 'clearDraw'].map(id => [id, new Element()]));
for (const item of Object.values(ids)) item.ownerDocument = doc;
ids.drawCanvas = makeCanvas(280, 280);
ids.inputPreview = makeCanvas(28, 28);
ids.brushSize.value = '20';
const root = {ownerDocument: doc, querySelector: selector => ids[selector.slice(1)]};
globalThis.document = doc;
const revoked = [];
URL.createObjectURL = () => 'blob:local';
URL.revokeObjectURL = url => revoked.push(url);
globalThis.Image = class {
  width = 3; height = 3;
  pixels = new Uint8ClampedArray(3 * 3 * 4);
  constructor() { this.pixels.set([255, 255, 255, 255], 4 * 4); }
  decode() { return Promise.resolve(); }
};
let answer;
let calls = [];
globalThis.mockRequest = (path, options) => {
  calls.push({path, options});
  return new Promise(resolve => { answer = resolve; });
};
const cleanup = mount(root);
ids.uploadDigit.files = [{type: 'image/png', size: 100}];
ids.uploadDigit.dispatchEvent(new Event('change'));
await new Promise(resolve => setImmediate(resolve));
assert.match(ids.predictResult.textContent, /本地读取/);
assert.deepEqual(revoked, ['blob:local']);
ids.predictButton.dispatchEvent(new Event('click'));
assert.equal(calls.length, 1);
assert.equal(calls[0].path, '/api/predict');
assert.deepEqual(Object.keys(calls[0].options.body), ['pixels']);
assert.equal(calls[0].options.body.pixels.length, 28);
answer({ok: true, pred: 1, prob: .9, probs: Array(10).fill(.1), trained: false});
await new Promise(resolve => setImmediate(resolve));
assert.match(ids.predictResult.textContent, /尚未训练/);
assert.equal(ids.predictProbs.children.length, 10);
ids.predictButton.dispatchEvent(new Event('click'));
assert.equal(calls.length, 2);
answer({ok: false, error: {message: '模型暂不可用'}});
await new Promise(resolve => setImmediate(resolve));
assert.equal(ids.predictResult.textContent, '模型暂不可用');
ids.predictButton.dispatchEvent(new Event('click'));
assert.equal(calls.length, 3);
const previous = ids.predictResult.textContent;
cleanup();
assert(calls[2].options.signal.aborted);
answer({ok: true, pred: 1, prob: .9, probs: Array(10).fill(.1), trained: true});
await new Promise(resolve => setImmediate(resolve));
assert.equal(ids.predictResult.textContent, previous);
assert.equal(ids.predictProbs.children.length, 10);
ids.clearDraw.dispatchEvent(new Event('click'));
assert.equal(ids.predictResult.textContent, previous);
""".replace('__DIR__', FEATURE.as_posix())
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
