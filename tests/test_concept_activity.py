"""在无浏览器依赖的 Node 小型 DOM 中检查概念互动的可见行为。"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVITIES = ROOT / 'static' / 'js' / 'activities'

# 只模拟 DOM 的容器、事件和文本；被测模块逻辑始终是实际 JS 代码。
HARNESS = r"""
import {pathToFileURL} from 'node:url';
class Element {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.listeners = {};
    this._text = ''; this.value = ''; this.required = false;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...children) {
    if (this.tagName === 'select' && !this.children.length && children.length) {
      this.value = children[0].value;
    }
    this.children.push(...children);
  }
  replaceChildren(...children) { this.children = [...children]; this._text = ''; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  removeEventListener(type, listener) {
    this.listeners[type] = (this.listeners[type] ?? []).filter(fn => fn !== listener);
  }
  dispatch(type) {
    for (const listener of this.listeners[type] ?? []) listener({preventDefault() {}});
  }
}
globalThis.document = {createElement: tag => new Element(tag)};
globalThis.localStorage = {getItem() { throw Error('disabled'); }, setItem() { throw Error('disabled'); }};
const walk = (root, tag) => [root, ...root.children.flatMap(child => walk(child, tag))]
  .filter(el => el.tagName === tag);
const mount = async name => {
  const module = await import(pathToFileURL(`${process.argv[1]}/${name}.js`).href);
  const slot = new Element('div');
  const cleanup = module.mount(slot);
  return {slot, cleanup};
};
const network = await mount('network');
const networkBefore = network.slot.textContent;
walk(network.slot, 'button')[0].dispatch('click');
const networkAfter = network.slot.textContent;
network.cleanup();
const weight = await mount('weight-step');
const weightBefore = weight.slot.textContent;
walk(weight.slot, 'button').find(b => b.textContent.includes('+0.5')).dispatch('click');
const weightAfter = weight.slot.textContent;
weight.cleanup();
const evaluation = await mount('evaluation');
const evaluationBefore = evaluation.slot.textContent;
walk(evaluation.slot, 'button')[0].dispatch('click');
const evaluationAfter = evaluation.slot.textContent;
evaluation.cleanup();
const experiment = await mount('experiment');
const [before, after] = walk(experiment.slot, 'input');
const textarea = walk(experiment.slot, 'textarea')[0];
before.value = '0.1'; after.value = '0.01'; textarea.value = '验证表现提高';
walk(experiment.slot, 'form')[0].dispatch('submit');
const experimentAfter = experiment.slot.textContent;
const experimentNote = walk(experiment.slot, 'li')[0].textContent;
experiment.cleanup();
const stored = new Map();
globalThis.localStorage = {
  getItem(key) { return stored.get(key) ?? null; },
  setItem(key, value) { stored.set(key, value); },
};
const saved = await mount('experiment');
const [oldValue, newValue] = walk(saved.slot, 'input');
const observation = walk(saved.slot, 'textarea')[0];
oldValue.value = '4'; newValue.value = '4'; observation.value = '还没改变';
walk(saved.slot, 'form')[0].dispatch('submit');
const unchangedCount = walk(saved.slot, 'li').length;
newValue.value = '8'; observation.value = '观察到差异';
walk(saved.slot, 'form')[0].dispatch('submit');
saved.cleanup();
const reopened = await mount('experiment');
const restoredNote = walk(reopened.slot, 'li')[0].textContent;
reopened.cleanup();
console.log(JSON.stringify({
  networkBefore, networkAfter, networkClean: network.slot.children.length,
  weightBefore, weightAfter, weightClean: weight.slot.children.length,
  evaluationBefore, evaluationAfter, evaluationClean: evaluation.slot.children.length,
  experimentAfter, experimentNote, experimentClean: experiment.slot.children.length,
  unchangedCount, restoredNote,
}));
"""


@unittest.skipUnless(shutil.which('node'), '需要 Node.js 才能执行模块行为测试')
class ConceptActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['node', '--input-type=module', '-e', HARNESS, str(ACTIVITIES)],
            capture_output=True, text=True, encoding='utf-8',
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.view = json.loads(result.stdout)

    def test_network_shows_sizes_and_changes_ten_example_scores(self):
        before, after = self.view['networkBefore'], self.view['networkAfter']
        for size in ('28×28', '14×14', '7×7'):
            self.assertIn(size, before)
        for digit in range(10):
            self.assertIn(f'{digit}：', before)
            self.assertIn(f'{digit}：', after)
        self.assertIn('最高是数字 3', before)
        self.assertIn('最高是数字 8', after)
        self.assertIn('教学示意', before)
        self.assertNotEqual(before, after)
        self.assertEqual(self.view['networkClean'], 0)

    def test_weight_step_shows_arithmetic_and_smaller_error_after_trial(self):
        self.assertIn('x = 2', self.view['weightBefore'])
        self.assertIn('目标 = 4', self.view['weightBefore'])
        self.assertIn('平方误差 = 4', self.view['weightBefore'])
        self.assertIn('w = 1.5', self.view['weightAfter'])
        self.assertIn('平方误差 = 1', self.view['weightAfter'])
        self.assertEqual(self.view['weightClean'], 0)

    def test_evaluation_separates_teaching_samples_and_accuracy(self):
        before, after = self.view['evaluationBefore'], self.view['evaluationAfter']
        self.assertIn('教学示意', before)
        self.assertIn('训练', before)
        self.assertIn('验证', before)
        self.assertIn('4/4 = 100%', before)
        self.assertIn('1/4 = 25%', before)
        self.assertIn('3/4 = 75%', after)
        self.assertEqual(self.view['evaluationClean'], 0)

    def test_experiment_works_when_storage_is_unavailable(self):
        self.assertEqual(self.view['experimentNote'], '学习率：0.1 → 0.01；观察：验证表现提高')
        self.assertIn('当前页面', self.view['experimentAfter'])
        self.assertEqual(self.view['experimentClean'], 0)
    def test_experiment_rejects_unchanged_value_and_restores_saved_note(self):
        self.assertEqual(self.view['unchangedCount'], 0)
        self.assertEqual(self.view['restoredNote'], '学习率：4 → 8；观察：观察到差异')


if __name__ == '__main__':
    unittest.main()
