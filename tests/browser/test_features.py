"""真实浏览器下检查训练与识别页的原 API 连接和卸载行为。"""
import base64
import json
import unittest

import paths  # noqa: F401 先启用项目 libs/
from playwright.sync_api import expect, sync_playwright
from flask_server import local_flask_server


IDLE = {
    'status': 'idle', 'epoch_current': 0, 'epoch_total': 6, 'step': 0,
    'steps_per_epoch': 10, 'train_loss_cur': None, 'elapsed_sec': 0,
    'eta_sec': None, 'loss_history': [], 'loss_steps': [], 'smooth_loss': [],
    'epochs_train_acc': [], 'epochs_val_acc': [], 'class_acc': [],
    'fmaps': [], 'samples': [], 'confusion': [], 'log': [],
    'dataset_sizes': {'train': 54000, 'validation': 6000, 'test': 10000},
    'config': {'epochs': 6, 'lr': 0.001, 'batch_size': 64,
               'dropout': 0.3, 'device': 'cpu', 'device_mode': 'cpu'},
    'hardware': {'cuda_available': False, 'cuda_reason': '可使用 CPU'},
}


class FeatureBrowserTests(unittest.TestCase):
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

    def test_training_settings_model_and_history_use_existing_api(self):
        control = []
        models = []
        self.page.route('**/api/state', lambda route: route.fulfill(json=IDLE))
        self.page.route('**/api/model-info', lambda route: route.fulfill(json={
            'ok': True, 'layers': [{'layer': '卷积', 'shape': [1, 32, 28, 28], 'parameters': 320}],
            'summary': '测试用模型结构'}))
        self.page.route('**/api/experiments', lambda route: route.fulfill(json={
            'ok': True, 'experiments': [{'id': '例一', 'config': IDLE['config'],
                                        'validation_acc': 88.5, 'validation_loss': 0.5}]}))

        def respond_control(route):
            control.append(route.request.post_data_json)
            route.fulfill(json={'ok': True, 'status': 'running'})

        def respond_model(route):
            models.append(route.request.post_data_json)
            route.fulfill(json={'ok': True, 'model': 'latest.pt', 'status': 'idle'})

        self.page.route('**/api/control', respond_control)
        self.page.route('**/api/model', respond_model)
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#modelLayers')).to_contain_text('测试用模型结构')
        expect(self.page.locator('#experimentHistory')).to_contain_text('88.50%')
        self.page.locator('[data-action="save"]').click()
        expect(self.page.locator('#notice')).to_contain_text('模型已保存')
        self.page.locator('[data-action="load"]').click()
        expect(self.page.locator('#notice')).to_contain_text('已加载本地存档')
        self.page.locator('#lr').fill('0.0001')
        self.page.locator('[data-action="apply"]').click()
        expect(self.page.locator('#settingsMessage')).to_contain_text('已应用设置')
        self.assertEqual(models, [{'action': 'save'}, {'action': 'load'}])
        self.assertEqual(control[-1], {'action': 'reset', 'epochs': 6,
            'lr': 0.0001, 'batch_size': 64, 'dropout': 0.3, 'device_mode': 'cpu'})

    def test_experiment_history_recovers_when_record_is_written_after_done(self):
        completed = {**IDLE, 'status': 'done', 'epochs_val_acc': [88.5]}
        requests = []
        self.page.route('**/api/state', lambda route: route.fulfill(json=completed))

        def experiments(route):
            requests.append(route.request.url)
            entries = [] if len(requests) == 1 else [{
                'id': '本次完成', 'config': IDLE['config'],
                'validation_acc': 88.5, 'validation_loss': 0.5}]
            route.fulfill(json={'ok': True, 'experiments': entries})

        self.page.route('**/api/experiments', experiments)
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#status')).to_have_text('训练完成')
        expect(self.page.locator('#experimentHistory')).to_contain_text('本次完成', timeout=4000)
        self.assertGreaterEqual(len(requests), 2)

    def test_history_can_be_refreshed_manually_after_delayed_write(self):
        self.page.route('**/api/state', lambda route: route.fulfill(json=IDLE))
        calls = []

        def experiments(route):
            calls.append(1)
            entries = [] if len(calls) == 1 else [{
                'id': '稍后写入', 'config': IDLE['config'],
                'validation_acc': 88.5, 'validation_loss': 0.5}]
            route.fulfill(json={'ok': True, 'experiments': entries})

        self.page.route('**/api/experiments', experiments)
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#experimentHistory')).to_contain_text('暂无完整运行')
        self.page.locator('[data-action="refresh-history"]').click()
        expect(self.page.locator('#experimentHistory')).to_contain_text('稍后写入')

    def test_leaving_training_page_stops_state_polling(self):
        calls = []

        def state(route):
            calls.append(route.request.url)
            route.fulfill(json=IDLE)

        self.page.route('**/api/state', state)
        self.page.goto(self.base_url + '/#/training')
        expect(self.page.locator('#status')).to_have_text('尚未开始')
        self.page.locator('a[href="#/prediction"]').click()
        expect(self.page.locator('#predictionTitle')).to_be_visible()
        count = len(calls)
        self.page.wait_for_timeout(1000)
        self.assertEqual(len(calls), count)

    def test_handwriting_sends_only_28_by_28_numbers(self):
        requests = []

        def predict(route):
            requests.append(route.request.post_data_json)
            route.fulfill(json={'ok': True, 'pred': 3, 'prob': 0.82,
                                'probs': [0.82 if digit == 3 else 0.02 for digit in range(10)],
                                'trained': False})

        self.page.route('**/api/predict', predict)
        self.page.goto(self.base_url + '/#/prediction')
        canvas = self.page.locator('#drawCanvas')
        rect = canvas.bounding_box()
        x, y = rect['x'] + rect['width'] / 2, rect['y'] + rect['height'] / 2
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        self.page.mouse.move(x + 20, y + 30, steps=4)
        self.page.mouse.up()
        preview_has_ink = self.page.locator('#inputPreview').evaluate(
            "canvas => { const pixels = canvas.getContext('2d').getImageData(0, 0, 28, 28).data; "
            "return [...pixels].some((value, index) => index % 4 !== 3 && value > 0); }")
        self.assertTrue(preview_has_ink, '按识别前应能看见模型输入预览')
        self.page.locator('#predictButton').click()
        expect(self.page.locator('#predictResult')).to_contain_text('尚未训练')
        self.assertEqual(len(requests), 1)
        self.assertEqual(list(requests[0]), ['pixels'])
        matrix = requests[0]['pixels']
        self.assertEqual(len(matrix), 28)
        self.assertTrue(all(len(row) == 28 and all(0 <= n <= 1 for n in row) for row in matrix))
        self.assertTrue(any(n > 0 for row in matrix for n in row))
    def test_upload_decode_survives_invert_change(self):
        self.page.add_init_script("""
            const decode = Image.prototype.decode;
            Image.prototype.decode = function() {
              return new Promise((resolve, reject) => {
                window.__releaseDecode = () => decode.call(this).then(resolve, reject);
              });
            };
        """)
        sent = []

        def predict(route):
            sent.append(route.request.post_data_json['pixels'])
            route.fulfill(json={'ok': True, 'pred': 1, 'prob': 0.9,
                                'probs': [0.01] * 9 + [0.91], 'trained': True})

        self.page.route('**/api/predict', predict)
        self.page.goto(self.base_url + '/#/prediction')
        expect(self.page.locator('#routeStatus')).to_be_hidden()
        data_url = self.page.evaluate("""() => {
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = 3;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = 'white'; ctx.fillRect(0, 0, 3, 3);
          ctx.fillStyle = 'black'; ctx.fillRect(1, 1, 1, 1);
          return canvas.toDataURL('image/png');
        }""")
        png = base64.b64decode(data_url.split(',', 1)[1])
        self.page.locator('#uploadDigit').set_input_files(
            {'name': 'digit.png', 'mimeType': 'image/png', 'buffer': png})
        self.page.wait_for_function('typeof window.__releaseDecode === "function"')
        self.page.locator('#invertImage').check()
        expect(self.page.locator('#predictButton')).to_be_disabled()
        self.page.evaluate('window.__releaseDecode()')
        expect(self.page.locator('#predictResult')).to_contain_text('图片已在本地读取')
        preview_has_ink = self.page.locator('#inputPreview').evaluate(
            "canvas => [...canvas.getContext('2d').getImageData(0, 0, 28, 28).data]"
            ".some((value, index) => index % 4 !== 3 && value > 0)")
        self.assertTrue(preview_has_ink, '上传并反色后应先显示模型输入预览')
        self.page.locator('#predictButton').click()
        expect(self.page.locator('#predictResult')).to_contain_text('预测')
        self.assertEqual(len(sent), 1)
        self.assertTrue(any(value > 0 for row in sent[0] for value in row))


if __name__ == '__main__':
    unittest.main()
