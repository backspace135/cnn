import threading
import unittest
from unittest.mock import patch

import train
import numpy as np
import torch


class TinyTrainer(train.Trainer):
    """Use small in-memory data; exercise the real model and training loop."""

    def _load_data(self):
        self.x_train = np.zeros((32, 28, 28), dtype=np.float32)
        self.y_train = np.zeros(32, dtype=np.int64)
        self.x_test = self.x_train[:8]
        self.y_test = self.y_train[:8]
        self._data_shape = (self.x_train.shape, self.x_test.shape)


class DeviceTests(unittest.TestCase):
    def make_trainer(self):
        with patch('torch.cuda.is_available', return_value=False):
            return TinyTrainer()

    def test_initial_state_reports_cpu_and_cuda_unavailable(self):
        trainer = self.make_trainer()
        state = trainer.snapshot()
        self.assertEqual(state['config']['device'], 'cpu')
        self.assertEqual(state['config']['device_mode'], 'auto')
        self.assertFalse(state['hardware']['cuda_available'])
        self.assertTrue(state['hardware']['cuda_reason'])

    def test_explicit_cpu_selection_and_training(self):
        trainer = self.make_trainer()
        trainer.reset(epochs=1, device_mode='cpu')
        trainer.start()
        trainer._thread.join(timeout=15)
        self.assertFalse(trainer._thread.is_alive())
        self.assertEqual(trainer.snapshot()['status'], 'done')
        self.assertEqual(trainer.snapshot()['epoch_current'], 1)
        self.assertEqual(next(trainer.model.parameters()).device.type, 'cpu')

    def test_cuda_unavailable_preserves_existing_model_and_config(self):
        trainer = self.make_trainer()
        trainer.configure(epochs=2)
        before = trainer.snapshot()['config'].copy()
        pending = trainer._pending_config.copy()
        model = trainer.model
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                trainer.reset(epochs=1, device_mode='cuda')
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                trainer.configure(device_mode='cuda')
        self.assertIs(trainer.model, model)
        self.assertEqual(trainer.snapshot()['config'], before)
        self.assertEqual(trainer._pending_config, pending)

    def test_unknown_mode_is_rejected(self):
        trainer = self.make_trainer()
        with self.assertRaises(ValueError):
            trainer.reset(device_mode='bogus')

    def test_reset_stops_old_worker_before_replacing_state(self):
        trainer = self.make_trainer()
        old_stop = trainer._stop_iter
        started = threading.Event()
        exited = threading.Event()

        def old_worker():
            started.set()
            old_stop.wait(timeout=3)
            exited.set()

        worker = threading.Thread(target=old_worker)
        trainer._thread = worker
        trainer.pause_event.set()
        worker.start()
        self.assertTrue(started.wait(timeout=1))
        try:
            trainer.reset(epochs=1)
            self.assertTrue(exited.is_set(), 'Reset must wait for the old worker')
            self.assertFalse(worker.is_alive())
            self.assertFalse(trainer.pause_event.is_set())
            self.assertEqual(trainer.snapshot()['epoch_total'], 1)
        finally:
            old_stop.set()
            worker.join(timeout=3)

    def test_invalid_parameters_leave_active_configuration_untouched(self):
        trainer = self.make_trainer()
        before = trainer.snapshot()['config'].copy()
        for cfg in ({'epochs': 0}, {'lr': float('nan')}, {'batch_size': 0}, {'dropout': 2}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                trainer.reset(**cfg)
        self.assertEqual(trainer.snapshot()['config'], before)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware/runtime unavailable')
    def test_real_cuda_training_and_switch_back_to_cpu(self):
        trainer = TinyTrainer()
        trainer.reset(epochs=1, device_mode='cuda')
        trainer.start()
        trainer._thread.join(timeout=30)
        self.assertFalse(trainer._thread.is_alive())
        self.assertEqual(trainer.snapshot()['status'], 'done')
        self.assertEqual(len(trainer.snapshot()['epochs_val_acc']), 1)
        self.assertEqual(next(trainer.model.parameters()).device.type, 'cuda')
        trainer.reset(epochs=1, device_mode='cpu')
        self.assertEqual(next(trainer.model.parameters()).device.type, 'cpu')
        self.assertEqual(trainer.snapshot()['config']['device_mode'], 'cpu')


if __name__ == '__main__':
    unittest.main()
