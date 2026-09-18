"""不用下载 MNIST 的数据、模型、推理与 API 回归测试。"""
import gzip
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import paths  # noqa: F401  项目内依赖优先
import numpy as np
import torch

import data
import download_data
import server
from model import HandwritingCNN
from test_training_devices import TinyTrainer


class DataTests(unittest.TestCase):
    def test_split_is_stratified_reproducible_and_disjoint(self):
        images = np.zeros((100, 28, 28), dtype=np.float32)
        labels = np.repeat(np.arange(10, dtype=np.int64), 10)
        for i in range(100):
            images[i, 0, 0] = i / 100  # 唯一像素，检测集合交叉
        train_x, train_y, val_x, val_y, sizes = data.split_train_validation(images, labels)
        self.assertEqual(sizes, {"train": 90, "validation": 10})
        self.assertTrue(np.array_equal(np.bincount(val_y), np.ones(10)))
        self.assertFalse(set(train_x[:, 0, 0]) & set(val_x[:, 0, 0]))
        again = data.split_train_validation(images, labels)
        self.assertTrue(np.array_equal(val_x, again[2]))

    def test_npz_rejects_missing_or_invalid_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.npz'
            with self.assertRaises(FileNotFoundError):
                data.load_npz(path)
            images = np.zeros((4, 28, 28), dtype=np.float32)
            np.savez(path, x_train=images)
            with self.assertRaisesRegex(ValueError, '缺少'):
                data.load_npz(path)
            np.savez(path, x_train=images, y_train=np.array([0.3] * 4),
                     x_test=images, y_test=np.zeros(4, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, '整数'):
                data.load_npz(path)

    def test_idx_rejects_bad_magic_length_and_label(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'image.gz'
            path.write_bytes(gzip.compress(struct.pack('>IIII', 2051, 1, 28, 28) + bytes(784)))
            self.assertEqual(download_data.read_idx_gz(path).shape, (1, 28, 28))
            path.write_bytes(gzip.compress(struct.pack('>IIII', 2051, 1, 28, 28) + bytes(12)))
            with self.assertRaisesRegex(ValueError, '长度'):
                download_data.read_idx_gz(path)
            path.write_bytes(gzip.compress(struct.pack('>II', 2049, 1) + bytes([12])))
            with self.assertRaisesRegex(ValueError, '标签'):
                download_data.read_idx_gz(path)
            path.write_bytes(gzip.compress(struct.pack('>II', 999, 1)))
            with self.assertRaisesRegex(ValueError, '魔数'):
                download_data.read_idx_gz(path)
    def test_download_failure_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(download_data, 'DATA_DIR', Path(temp)), \
                    patch.object(download_data, 'BASE_URLS', ('https://example.test/',)), \
                    patch.object(download_data.urllib.request, 'urlopen', side_effect=TimeoutError('timeout')) as fetch, \
                    patch.object(download_data.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'train-images'):
                    download_data.download_file('train-images-idx3-ubyte.gz')
                self.assertEqual(fetch.call_count, 3)
            self.assertFalse((Path(temp) / 'train-images-idx3-ubyte.gz.part').exists())


class LearningApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        with patch('torch.cuda.is_available', return_value=False):
            self.trainer = TinyTrainer(model_file=Path(self.temp.name) / 'latest.pt')
        self.client = server.create_app(self.trainer).test_client()

    def tearDown(self):
        self.trainer.stop()
        self.temp.cleanup()

    def test_model_shapes_and_initial_examples(self):
        model = HandwritingCNN()
        x = torch.zeros(2, 1, 28, 28)
        self.assertEqual(tuple(model(x).shape), (2, 10))
        self.assertEqual(tuple(model.conv1_activations(x[:1]).shape), (1, 32, 28, 28))
        layers = model.describe_layers()
        self.assertEqual(layers[-1]['shape'], [1, 10])
        self.assertEqual(layers[3]['shape'], [1, 32, 14, 14])
        self.assertEqual(self.client.get('/api/model-info').status_code, 200)
        self.assertEqual(self.trainer.snapshot()['dataset_sizes'],
                         {'train': 29, 'validation': 3, 'test': 8})
        self.assertIsNone(self.trainer.snapshot()['samples'][0]['pred'])

    def test_predict_validation_and_eval_mode(self):
        blank = [[0.0] * 28 for _ in range(28)]
        for pixels in (blank, [[0] * 3], [[True] * 28] * 28, [[float('nan')] * 28] * 28):
            self.assertEqual(self.client.post('/api/predict', json={'pixels': pixels}).status_code, 400)
        blank[14][14] = 1.0
        before = self.trainer.model.features[1].running_mean.clone()
        self.assertTrue(self.trainer.model.training)
        response = self.client.post('/api/predict', json={'pixels': blank})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()['probs']), 10)
        self.assertFalse(response.get_json()['trained'])
        self.assertTrue(self.trainer.model.training)
        self.assertTrue(torch.equal(before, self.trainer.model.features[1].running_mean))

    def test_predict_preserves_weights_and_original_mode(self):
        pixels = [[0.0] * 28 for _ in range(28)]
        pixels[14][14] = 1.0
        model = self.trainer.model
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                result = self.trainer.predict(pixels)
                self.assertEqual(len(result['probs']), 10)
                self.assertEqual(model.training, training)
                for key, value in before.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value), key)
        model.train()
        with patch.object(model, 'forward', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                self.trainer.predict(pixels)
        self.assertTrue(model.training)

    def test_predict_rejects_bad_pixels_without_changing_model(self):
        pixels = [[0.0] * 28 for _ in range(28)]
        model = self.trainer.model
        for bad in (pixels, pixels[:27], [[0] * 3], [[True] * 28] * 28,
                    [[float('nan')] * 28] * 28, [[float('inf')] * 28] * 28,
                    [[1.1] * 28] * 28, 'not pixels'):
            with self.subTest(bad=repr(bad)[:40]), self.assertRaises(ValueError):
                self.trainer.predict(bad)
        self.assertIs(self.trainer.model, model)
        self.assertTrue(model.training)
        self.assertEqual(self.trainer.snapshot()['generation'], 0)

    def test_training_final_test_and_save_load_roundtrip(self):
        self.trainer.reset(epochs=1, device_mode='cpu')
        self.trainer.start()
        self.trainer._thread.join(timeout=30)
        state = self.trainer.snapshot()
        self.assertEqual(state['status'], 'done')
        self.assertEqual(sum(map(sum, state['confusion'])), 3)
        self.assertEqual(sum(map(sum, state['test_confusion'])), 8)
        self.assertIsNotNone(state['test_acc'])
        entries = self.client.get('/api/experiments').get_json()['experiments']
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['dataset_sizes']['validation'], 3)
        self.assertNotIn('test_acc', entries[0])
        self.assertEqual(state['samples'][0]['correct'],
                         state['samples'][0]['pred'] == state['samples'][0]['true'])
        with self.assertRaises(RuntimeError):
            self.trainer.start()
        self.assertEqual(self.client.post('/api/model', json={'action': 'save'}).status_code, 200)
        pixels = [[0.0] * 28 for _ in range(28)]; pixels[0][0] = 1.0
        before = self.trainer.predict(pixels)['probs']
        self.trainer.reset(epochs=1)
        self.assertEqual(self.client.post('/api/model', json={'action': 'load'}).status_code, 200)
        self.assertEqual(before, self.trainer.predict(pixels)['probs'])
        self.assertTrue(self.trainer.predict(pixels)['trained'])

    def test_done_is_published_before_experiment_history_is_saved(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.trainer.record_experiment

        def delayed_record():
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError('实验记录等待超时')
            original()

        with patch.object(self.trainer, 'record_experiment', side_effect=delayed_record):
            self.trainer.reset(epochs=1, device_mode='cpu')
            self.trainer.start()
            self.assertTrue(entered.wait(timeout=20), '训练应在完成时开始记录实验')
            try:
                self.assertEqual(self.trainer.snapshot()['status'], 'done')
                self.assertEqual(self.trainer.experiments(), [])
            finally:
                release.set()
            self.trainer._thread.join(timeout=30)
        self.assertEqual(self.trainer.snapshot()['status'], 'done')
        self.assertEqual(len(self.trainer.experiments()), 1)

    def test_corrupt_model_does_not_replace_current_model(self):
        self.trainer.model_file.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.model_file.write_bytes(b'not a checkpoint')
        old_model = self.trainer.model
        response = self.client.post('/api/model', json={'action': 'load'})
        self.assertEqual(response.status_code, 400)
        self.assertIs(self.trainer.model, old_model)

    def test_invalid_checkpoint_payload_keeps_model_and_state(self):
        self.trainer.configure(epochs=2)
        model = self.trainer.model
        optimizer = self.trainer.optimizer
        state = self.trainer.snapshot()
        pending = self.trainer._pending_config.copy()
        config = state['config']
        for payload in ({'version': 1, 'config': config},
                        {'version': 1, 'config': config, 'state_dict': {}}):
            with self.subTest(payload=payload):
                self.trainer.model_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, self.trainer.model_file)
                with self.assertRaises(ValueError):
                    self.trainer.load_model()
                self.assertIs(self.trainer.model, model)
                self.assertIs(self.trainer.optimizer, optimizer)
                self.assertEqual(self.trainer.snapshot(), state)
                self.assertEqual(self.trainer._pending_config, pending)

    def test_app_without_data_serves_course_and_reports_missing_data(self):
        with patch.object(server, 'trainer', None), \
                patch.object(server, 'Trainer', side_effect=FileNotFoundError('请先下载 MNIST')):
            client = server.create_app().test_client()
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            response.close()
            self.assertEqual(client.get('/api/health').status_code, 503)
            state = client.get('/api/state')
            self.assertEqual(state.status_code, 503)
            self.assertEqual(state.get_json()['error']['code'], 'data_unavailable')

    def test_app_factory_and_bad_requests(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.post('/api/control', json={'action': 'reset', 'epochs': True}).status_code, 400)
        self.assertEqual(self.client.post('/api/control', json={'action': 'reset', 'epochs': None}).status_code, 400)
        self.assertEqual(self.client.post('/api/control', json=['start']).status_code, 400)
        self.assertEqual(self.client.post('/api/model', json={'action': 'load'}).status_code, 404)
        self.assertEqual(self.client.post('/api/control', json={'action': 'stop'}).status_code, 200)
        snapshot = self.client.get('/api/state').get_json()
        snapshot['config']['batch_size'] = 999
        self.assertEqual(self.client.get('/api/state').get_json()['config']['batch_size'], 64)


if __name__ == '__main__':
    unittest.main()
