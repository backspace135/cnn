"""用 Node 运行浏览器模块的纯计算接口；无需浏览器或第三方 JS 包。"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / 'static' / 'js' / 'activities' / 'convolution.js'
PIXELS_MODULE = MODULE.with_name('pixels.js')
NODE_SCRIPT = r"""
import {readFileSync} from 'node:fs';
try {
  const source = readFileSync(process.argv[1], 'utf8');
  const {parseMatrix, computeConvolution, nextPixelValue} = await import('data:text/javascript,' + encodeURIComponent(source));
  const request = JSON.parse(process.argv[2]);
  const value = request.op === 'parse' ? parseMatrix(request.text, '输入')
    : request.op === 'pixel' ? nextPixelValue(request.value)
    : computeConvolution(request.input, request.kernel, request.stride, request.padding);
  console.log(JSON.stringify({value}));
} catch (error) {
  console.log(JSON.stringify({error: error.message}));
}
"""


@unittest.skipUnless(shutil.which('node'), '需要 Node.js 执行 ES module 数学测试')
class ActivityMathTests(unittest.TestCase):
    def evaluate(self, module=MODULE, **request):
        process = subprocess.run(
            ['node', '--input-type=module', '-e', NODE_SCRIPT, str(module), json.dumps(request)],
            capture_output=True, text=True, encoding='utf-8', check=True,
        )
        return json.loads(process.stdout)

    def test_fixed_first_cell_uses_kernel_without_flipping(self):
        result = self.evaluate(
            input=[[1, 2, 0], [0, 1, 0], [1, 0, 1]],
            kernel=[[1, 0, -1], [0, 1, 0], [1, 0, 0]], stride=1, padding=0,
        )['value']
        self.assertEqual((result['height'], result['width']), (1, 1))
        self.assertEqual(result['steps'][0]['products'],
                         [[1, 1], [2, 0], [0, -1], [0, 0], [1, 1], [0, 0],
                          [1, 1], [0, 0], [1, 0]])
        self.assertEqual(result['steps'][0]['sum'], 3)

    def test_sliding_produces_row_major_outputs(self):
        result = self.evaluate(
            input=[[1, 2, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
            kernel=[[1, 0, -1], [0, 1, 0], [1, 0, 0]], stride=1, padding=0,
        )['value']
        self.assertEqual((result['height'], result['width']), (2, 2))
        self.assertEqual([(step['r'], step['c'], step['sum']) for step in result['steps']],
                         [(0, 0, 3), (0, 1, 2), (1, 0, 0), (1, 1, 2)])

    def test_padding_uses_zeros_and_stride_skips_positions(self):
        result = self.evaluate(input=[[1, 2], [3, 4]],
                               kernel=[[1] * 3] * 3, stride=2, padding=1)['value']
        self.assertEqual((result['height'], result['width']), (1, 1))
        self.assertEqual(result['steps'][0]['products'],
                         [[0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [2, 1],
                          [0, 1], [3, 1], [4, 1]])
        self.assertEqual(result['steps'][0]['sum'], 10)

    def test_parse_rejects_empty_ragged_non_numeric_and_oversized(self):
        for text in ('', '1 2\n3', '1 x\n2 3', '1 2 3 4 5 6 7 8 9', '1\n\n2'):
            with self.subTest(text=text):
                self.assertIn('矩阵', self.evaluate(op='parse', text=text)['error'])
        self.assertEqual(self.evaluate(op='parse', text='1 0\n-2 0.5')['value'],
                         [[1, 0], [-2, 0.5]])

    def test_invalid_kernel_stride_padding_and_output_size_have_chinese_errors(self):
        base = dict(input=[[1, 2], [3, 4]], kernel=[[1] * 3] * 3, stride=1, padding=1)
        for change, keyword in (({'kernel': [[1]]}, '卷积核'),
                                ({'stride': 0}, '步长'),
                                ({'stride': 1.5}, '步长'),
                                ({'padding': -1}, '填充'),
                                ({'padding': 3}, '填充'),
                                ({'padding': 0}, '卷积核')):
            with self.subTest(change=change):
                self.assertIn(keyword, self.evaluate(**(base | change))['error'])
    def test_pixel_cycle_visits_three_brightness_levels(self):
        for value, expected in ((0, 0.5), (0.5, 1), (1, 0)):
            with self.subTest(value=value):
                self.assertEqual(self.evaluate(module=PIXELS_MODULE, op='pixel', value=value)['value'],
                                 expected)


if __name__ == '__main__':
    unittest.main()
