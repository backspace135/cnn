import unittest
from unittest.mock import patch

import server
from test_training_devices import TinyTrainer


class DeviceApiTests(unittest.TestCase):
    def setUp(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.trainer = TinyTrainer()
        self.replacement = patch.object(server, 'trainer', self.trainer)
        self.replacement.start()
        self.client = server.app.test_client()

    def tearDown(self):
        self.trainer.stop()
        self.replacement.stop()

    def test_cpu_selection_reaches_model_and_state(self):
        response = self.client.post('/api/control', json={
            'action': 'reset', 'device_mode': 'cpu', 'epochs': 1,
        })
        self.assertEqual(response.status_code, 200)
        self.trainer._thread.join(timeout=15)
        state = self.client.get('/api/state').get_json()
        self.assertEqual(state['config']['device_mode'], 'cpu')
        self.assertEqual(state['config']['device'], 'cpu')
        self.assertEqual(state['status'], 'done')
        self.assertEqual(state['epoch_current'], 1)
        self.assertEqual(len(state['epochs_val_acc']), 1)

    def test_unavailable_cuda_is_actionable_client_error(self):
        with patch('torch.cuda.is_available', return_value=False):
            response = self.client.post('/api/control', json={
                'action': 'reset', 'device_mode': 'cuda',
            })
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['ok'])
        self.assertIn('CUDA', response.get_json()['msg'])
        self.assertEqual(self.trainer.snapshot()['status'], 'idle')

    def test_bad_mode_and_nonobject_body_return_json_error(self):
        for body in ({'action': 'reset', 'device_mode': 'tpu'}, ['reset']):
            with self.subTest(body=body):
                response = self.client.post('/api/control', json=body)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.get_json()['ok'])


if __name__ == '__main__':
    unittest.main()
