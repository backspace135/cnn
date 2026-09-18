# -*- coding: utf-8 -*-
"""
手写数字识别训练器（带强学习注释）
============================================================
这份文件是"训练循环"的核心，也是机器学习里最值得理解的代码之一。
注释会解释清楚三个训练中的核心概念：

  1. Epoch / Batch / Step 的区别
  2. 前向传播(forward) / 反向传播(backward) / 更新参数(step)
  3. 训练集 vs 验证集：为什么训练准确率高不代表模型好

【实时可视化】训练跑在独立后台线程 _run_loop 里，每走一步就把
loss、准确率、预测样例、特征图等写进 self.state（一个普通字典）。
web 服务器 (server.py) 读取这个 state 供浏览器展示 —— 这就是
"看到训练过程"的原理。

所有对外字段都经过 JSON 序列化保证能传给浏览器：
  status                idle | running | paused | done | cancelled | error
  epoch_current/total   当前轮次 / 总轮次
  step / steps_per_epoch
  train_loss_cur        最近一个 batch 的 loss
  loss_steps            每个记录点的步数（横轴）
  loss_history          每个记录点的训练 loss（原始，未平滑）
  smooth_loss           指数滑动平均后的 loss（曲线更易读）
  epochs_train_acc      每个 epoch 后的训练准确率(0~100)
  epochs_val_acc        每个 epoch 后的验证准确率(0~100)
  class_acc            10 个类别的验证准确率(0~100)
  confusion             10x10 混淆矩阵
  samples               验证集样例：像素 + 真实标签 + 预测 + 各类概率
  fmaps / fmaps_version 第 1 层卷积特征图（学习可视化用）
  elapsed_sec / eta_sec 已用时间 / 预计剩余
  config                当前超参数（可在网页上改后重置）
  log                   最近若干条滚动日志
"""
import copy
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import paths  # noqa: F401  先把项目 libs/ 加入 sys.path，再导入数据模块和 torch
import data as mnist_data

import numpy as np
import torch
import torch.nn as nn

from model import HandwritingCNN
from training.checkpoint import MODEL_VERSION, load_checkpoint, save_checkpoint
from training.device import hardware_info, resolve_device
from training.inference import predict_probs, validate_pixels

DATA_FILE = Path(__file__).resolve().parent / "data" / "mnist.npz"
MODEL_FILE = Path(__file__).resolve().parent / "data" / "checkpoints" / "latest.pt"

# 默认超参数（可在网页"调参实验室"里改，再点重置生效）
DEFAULT_EPOCHS = 6
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 64
DEFAULT_DROPOUT = 0.3

SEED = 42            # 随机种子：固定后每次结果可复现（机器学习要讲究可复现）
MAX_LOG = 40         # 滚动日志最多保留条数
MAX_LOSS_POINTS = 6000  # loss 曲线最多保留的数据点个数

# 从验证集固定抽取的几张图，用于在网页上实时展示"模型现在的预测效果"
SAMPLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  11, 12, 14, 20, 33, 55, 88, 100,
                  150, 300, 500, 800, 1200, 2000, 3500, 5000]


class Trainer:
    """训练器：负责数据 / 模型 / 训练循环 / 对外状态，全部封装在这里。"""

    def __init__(self, device_mode="auto", data_file=None, model_file=None):
        # 固定随机种子：让"重新跑一次"的结果可复现，
        # 是研究机器学习时非常重要的好习惯。
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # 优先用 GPU（如果电脑有 NVIDIA 显卡且装了 CUDA），否则用 CPU
        self.device = resolve_device(device_mode)
        self.hardware = hardware_info()

        # ---- 注意初始化顺序：必须先建好锁和事件 ----
        # 因为后面的 _load_data / log 都需要 self.lock 和 self.state 已存在。
        # 多线程里所有对 self.state 的读写都要加 self.lock，避免数据竞争。
        self.lock = threading.Lock()
        self.control_lock = threading.RLock()
        # 模型模式切换（train/eval）、参数更新和单张预测不能同时发生。
        self.model_lock = threading.RLock()
        self.data_file = Path(data_file) if data_file is not None else DATA_FILE
        self.model_file = Path(model_file) if model_file is not None else MODEL_FILE
        self.experiment_file = self.model_file.parent / "experiments.json"
        self.generation = 0
        self._pending_config = None
        self.pause_event = threading.Event()   # 置位=True 表示暂停训练
        self._stop_iter = threading.Event()
        self._thread = None                    # 训练线程
        self.start_ts = None                   # 本轮训练开始时间
        self._paused_total = 0.0
        self._pause_started = None

        # ---- 载入数据（见 _load_data）----
        self._load_data()
        # TinyTrainer 等自定义教学数据源沿用 _load_data 覆盖点。
        if not hasattr(self, "x_val"):
            self.x_train, self.y_train, self.x_val, self.y_val, _ = (
                mnist_data.split_train_validation(self.x_train, self.y_train, seed=SEED))
        self.dataset_sizes = {"train": len(self.x_train), "validation": len(self.x_val),
                              "test": len(self.x_test)}
        # steps_per_epoch：一个 epoch 里有多少个 batch（步）
        # = len(训练集) / batch_size 向上取整
        self.steps_per_epoch = int(np.ceil(len(self.x_train) / DEFAULT_BATCH))

        # ---- 搭建模型 / 优化器 / 损失函数 ----
        self.model = HandwritingCNN(dropout_p=DEFAULT_DROPOUT).to(self.device)
        # 优化器：Adam。它负责"根据梯度更新每个权重"。lr 学习率=每次走多大步。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=DEFAULT_LR)
        # 损失函数：交叉熵 CrossEntropyLoss。
        # 它衡量"模型给出的预测分布"和"真实答案"差多少，越小越好。
        self.criterion = nn.CrossEntropyLoss()

        # ---- 初始化对外状态 ----
        self.state = self._fresh_state("idle")
        self.state["steps_per_epoch"] = self.steps_per_epoch
        self.state["config"].update(device=str(self.device), device_mode=device_mode,
                                    steps_per_epoch=self.steps_per_epoch)
        self.state["hardware"] = self.hardware
        self.state["dataset_sizes"] = self.dataset_sizes
        self.state["samples"] = self._sample_pixels()
        self.log(f"载入数据: 训练 {len(self.x_train):,} 张, 验证 {len(self.x_val):,} 张, "
                 f"测试 {len(self.x_test):,} 张")
        self.log(f"设备: {self.device}")
        self.log(self.model.summary())

    # ======================================================================
    # 数据加载
    # ======================================================================
    def _load_data(self):
        """从已下载的 npz 读取数据。验证集从原训练集划分，测试集留到最后。"""
        arrays = mnist_data.load_npz(self.data_file)
        (self.x_train, self.y_train, self.x_val, self.y_val, _) = (
            mnist_data.split_train_validation(arrays["x_train"], arrays["y_train"], seed=SEED))
        self.x_test, self.y_test = arrays["x_test"], arrays["y_test"]
        self._data_shape = (self.x_train.shape, self.x_test.shape)

    def _sample_pixels(self):
        """未训练前也展示真实图片；预测字段等首次评估后才有值。"""
        return [{"pixels": (image * 255).round().astype(int).tolist(),
                 "true": int(label), "pred": None, "correct": None,
                 "prob": None, "probs": []}
                for image, label in zip(self.x_val[:10], self.y_val[:10])]

    def _fresh_state(self, status: str) -> dict:
        """构造一个全新的、结构固定的 state 字典（所有字段保持默认）。"""
        return {
            "status": status, "generation": self.generation,
            "epoch_current": 0, "epoch_total": DEFAULT_EPOCHS,
            "step": 0, "steps_per_epoch": 0,
            "train_loss_cur": None,
            "loss_steps": [], "loss_history": [], "smooth_loss": [],
            "epochs_train_acc": [], "epochs_val_acc": [],
            "val_loss": None, "test_loss": None, "test_acc": None,
            "final_acc": None, "test_confusion": [], "error": None,
            "dataset_sizes": getattr(self, "dataset_sizes", {}),
            "checkpoint_available": self.model_file.is_file(),
            "class_acc": [],
            "confusion": [],
            "samples": [],
            "fmaps": [], "fmaps_version": 0,
            "elapsed_sec": 0.0, "eta_sec": None,
            "config": {"batch_size": DEFAULT_BATCH, "lr": DEFAULT_LR,
                       "epochs": DEFAULT_EPOCHS, "dropout": DEFAULT_DROPOUT,
                       "device": "", "device_mode": "auto", "steps_per_epoch": 0},
            "log": [],
        }

    def _elapsed_seconds(self):
        """训练有效耗时不包括暂停时间；调用者应持有 state 锁。"""
        if self.start_ts is None:
            return self.state["elapsed_sec"]
        now = time.time()
        current_pause = now - self._pause_started if self._pause_started is not None else 0
        return round(max(0, now - self.start_ts - self._paused_total - current_pause), 1)

    # ======================================================================
    # 日志（写进滚动列表，网页实时显示）
    # ======================================================================
    def log(self, msg: str):
        with self.lock:
            self.state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.state["log"] = self.state["log"][-MAX_LOG:]

    # ======================================================================
    # 控制接口（由网页 /api/control 调用）
    # ======================================================================
    def start(self):
        """只有 idle 或 paused 可以启动；完成后必须明确重置。"""
        with self.control_lock:
            with self.lock:
                status = self.state["status"]
                loaded = self.state.get("model_loaded", False)
            if loaded:
                raise RuntimeError("已加载模型仅用于识别；如需重新训练，请先重置。")
            if status == "paused" and self._thread is not None and self._thread.is_alive():
                self.resume()
                return
            if status != "idle":
                raise RuntimeError("当前状态不能开始训练；如需再训练，请先重置。")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("旧训练线程尚未退出，请稍后重试。")
            self._stop_iter = threading.Event()
            self.pause_event.clear()
            self.start_ts = time.time()
            self._paused_total = 0.0
            self._pause_started = None
            with self.lock:
                self.state["status"] = "running"
            stop_event = self._stop_iter
            self._thread = threading.Thread(target=self._run_loop, args=(stop_event,), daemon=True)
            self._thread.start()

    def pause(self):
        with self.control_lock:
            with self.lock:
                status = self.state["status"]
            if status != "running" or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("只有正在训练时才能暂停。")
            self.pause_event.set()
            with self.lock:
                self._pause_started = time.time()
                self.state["status"] = "paused"

    def resume(self):
        with self.control_lock:
            with self.lock:
                status = self.state["status"]
            if status != "paused" or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("只有暂停中的训练才能继续。")
            self.pause_event.clear()
            with self.lock:
                if self._pause_started is not None:
                    self._paused_total += time.time() - self._pause_started
                    self._pause_started = None
                self.state["status"] = "running"

    def stop(self):
        """停止当前 worker，join 时不能持有 state/model 锁。"""
        with self.control_lock:
            self._stop_iter.set()
            self.pause_event.clear()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=30)
                if self._thread.is_alive():
                    raise RuntimeError("训练仍在停止中，请稍后重试。")
            self._thread = None
            with self.lock:
                if self.state["status"] in ("running", "paused"):
                    self.state["status"] = "cancelled"
                self.start_ts = None

    def _validated_config(self, epochs=None, lr=None, batch_size=None,
                          dropout=None, device_mode=None):
        with self.lock:
            cfg = dict(self._pending_config or self.state["config"])
        overrides = {"epochs": epochs, "lr": lr, "batch_size": batch_size,
                     "dropout": dropout, "device_mode": device_mode}
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
        try:
            for key, low, high in (("epochs", 1, 30), ("batch_size", 16, 512)):
                raw = cfg[key]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"{key} 必须是 {low}–{high} 之间的整数。")
                value = float(raw)
                if not math.isfinite(value) or not value.is_integer() or not low <= value <= high:
                    raise ValueError(f"{key} 必须是 {low}–{high} 之间的整数。")
                cfg[key] = int(value)
            for key, low, high in (("lr", .0001, .1), ("dropout", 0, .7)):
                raw = cfg[key]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"{key} 必须在 {low}–{high} 之间。")
                value = float(raw)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{key} 必须在 {low}–{high} 之间。")
                cfg[key] = value
        except (TypeError, OverflowError, KeyError) as exc:
            raise ValueError("训练参数必须为有效数字。") from exc
        resolve_device(cfg["device_mode"])
        return cfg

    def configure(self, epochs=None, lr=None, batch_size=None, dropout=None,
                  device_mode=None):
        """Stage parameters for the next reset, without changing a live batch."""
        with self.control_lock:
            cfg = self._validated_config(epochs, lr, batch_size, dropout, device_mode)
            with self.lock:
                self._pending_config = cfg

    def reset(self, epochs=None, lr=None, batch_size=None, dropout=None,
              device_mode=None):
        """Validate first, stop the old worker, then replace model and state."""
        with self.control_lock:
            cfg = self._validated_config(epochs, lr, batch_size, dropout, device_mode)
            device = resolve_device(cfg["device_mode"])
            self.stop()
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            # Keep the old model intact if allocation fails.
            model = HandwritingCNN(dropout_p=cfg["dropout"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            with self.model_lock:
                self.model, self.optimizer, self.device = model, optimizer, device
            self.steps_per_epoch = int(np.ceil(len(self.x_train) / cfg["batch_size"]))
            self.start_ts = None
            self.generation += 1
            cfg.update(device=str(device), steps_per_epoch=self.steps_per_epoch)
            with self.lock:
                self.state = self._fresh_state("idle")
                self.state["epoch_total"] = cfg["epochs"]
                self.state["steps_per_epoch"] = self.steps_per_epoch
                self.state["config"] = cfg
                self.state["hardware"] = self.hardware
                self.state["samples"] = self._sample_pixels()
                self._pending_config = None
            self.log(f"已重置模型（lr={cfg['lr']}, batch={cfg['batch_size']}, "
                     f"dropout={cfg['dropout']}），共 {cfg['epochs']} 轮，设备 {device}")

    # ======================================================================
    # 训练主循环（在后台线程里跑）
    # ======================================================================
    def _run_loop(self, stop_event):
        """一个训练线程只持有属于自己的停止事件。"""
        with self.lock:
            epochs = self.state["epoch_total"]
            if self.state["status"] != "paused":
                self.state["status"] = "running"
        try:
            for epoch in range(1, epochs + 1):
                if stop_event.is_set():
                    break
                self._run_epoch(epoch, epochs)
            if stop_event.is_set():
                return
            # test 集只在完整训练结束后评估一次，绝不用来选学习率。
            conf, test_acc, _, test_loss = self._evaluate(self.x_test, self.y_test)
            if stop_event.is_set():
                return
            self.log(f"最终测试: acc={test_acc:.2f}% loss={test_loss:.4f}")
            with self.lock:
                self.state["test_acc"] = test_acc
                self.state["test_loss"] = test_loss
                self.state["test_confusion"] = conf.tolist()
                self.state["final_acc"] = self.state["epochs_val_acc"][-1] \
                    if self.state["epochs_val_acc"] else None
                # 先结束训练状态，避免写盘期间仍允许用户发起取消。
                self.state["status"] = "done"
            self.log("训练完成！")
            try:
                self.record_experiment()
            except (OSError, ValueError) as exc:
                logging.warning("实验记录写入失败: %s", exc)
        except Exception as exc:
            logging.exception("训练失败")
            self.log(f"训练出错: {exc}")
            with self.lock:
                self.state["status"] = "error"
                self.state["error"] = str(exc)
        finally:
            with self.lock:
                if stop_event.is_set() and self.state["status"] in ("running", "paused"):
                    self.state["status"] = "cancelled"
                if self.start_ts is not None:
                    self.state["elapsed_sec"] = self._elapsed_seconds()
                    self.start_ts = None
                    self._pause_started = None

    def _run_epoch(self, epoch: int, total: int):
        """
        一个 epoch = 把训练集完整看一遍。
        里面是一步一步（step/batch）地做"前向->算loss->反向->更新"。
        """
        with self.model_lock:
            self.model.train()   # Dropout/BatchNorm 按训练模式
        # 打乱训练集顺序：避免模型学到"顺序"这种假规律，也提升泛化
        permutation = np.random.permutation(len(self.x_train))
        running_loss = 0.0   # 累计本 epoch 的 loss，最后求平均
        n_batches = 0
        correct = 0          # 统计训练时预测对的个数（不算 loss，只算准确率）
        seen = 0

        for b in range(self.steps_per_epoch):
            # ---- 响应暂停 / 重置 ----
            while self.pause_event.is_set() and not self._stop_iter.is_set():
                time.sleep(0.2)
            if self._stop_iter.is_set():
                return

            # ---- 取一个 batch ----
            idx = permutation[b * self.config_bs:(b + 1) * self.config_bs]
            xb = torch.from_numpy(self.x_train[idx]).unsqueeze(1).to(self.device)
            yb = torch.from_numpy(self.y_train[idx]).to(self.device)

            # ---- 核心三连：前向 -> 反向 -> 更新 ----
            with self.model_lock:
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                pred = out.argmax(1)
            running_loss += loss.item()
            n_batches += 1
            correct += (pred == yb).sum().item()
            seen += len(yb)

            # ---- 记录到 state（网页实时刷）----
            with self.lock:
                step = self.state["step"] + 1
                self.state["step"] = step
                self.state["epoch_current"] = epoch
                self.state["train_loss_cur"] = loss.item()
                if len(self.state["loss_steps"]) < MAX_LOSS_POINTS:
                    self.state["loss_steps"].append(step)
                    self.state["loss_history"].append(loss.item())
                    previous = self.state["smooth_loss"][-1] if self.state["smooth_loss"] else loss.item()
                    self.state["smooth_loss"].append(.98 * previous + .02 * loss.item())

        # ---- 一个 epoch 结束：算训练平均 loss 和训练准确率 ----
        train_loss_avg = running_loss / max(n_batches, 1)
        train_acc = 100.0 * correct / max(seen, 1)
        self.log(f"Epoch {epoch}/{total} 训练: loss={train_loss_avg:.4f} "
                 f"acc={train_acc:.2f}%")

        # ---- 只在验证集上观察每一轮；原始 test 集等训练全部完成再使用 ----
        conf, val_acc, class_acc, val_loss = self._evaluate(self.x_val, self.y_val)
        if self._stop_iter.is_set():
            return

        # 一次性更新多个展示字段（在锁内，避免读到一半的状态）
        with self.lock:
            self.state["epochs_train_acc"].append(float(train_acc))
            self.state["epochs_val_acc"].append(float(val_acc))
            self.state["val_loss"] = float(val_loss)
            self.state["class_acc"] = [float(x) for x in class_acc]
        # _update_samples 内部获取 state lock，所以不能在上方 lock 中调用。
        self._update_samples(conf)
        self.log(f"Epoch {epoch}/{total} 验证: acc={val_acc:.2f}% "
                 f"loss={val_loss:.4f}")

    @property
    def config_bs(self):
        return self.state["config"].get("batch_size", DEFAULT_BATCH)

    # ======================================================================
    # 验证集评估
    # ======================================================================
    def _evaluate(self, images=None, labels=None):
        """不更新权重，返回指定集合的混淆矩阵、准确率和平均 loss。"""
        if images is None:
            images, labels = self.x_val, self.y_val
        correct, total = 0, 0
        loss_sum = 0.0
        class_correct = np.zeros(10, dtype=np.int64)
        class_total = np.zeros(10, dtype=np.int64)
        conf = np.zeros((10, 10), dtype=np.int64)
        with self.model_lock:
            previous_mode = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    for b in range(int(np.ceil(len(images) / 256))):
                        if self._stop_iter.is_set():
                            break
                        idx = slice(b * 256, (b + 1) * 256)
                        xb = torch.from_numpy(images[idx]).unsqueeze(1).to(self.device)
                        yb = torch.from_numpy(labels[idx]).to(self.device)
                        out = self.model(xb)
                        loss_sum += self.criterion(out, yb).item() * len(yb)
                        pred = out.argmax(1).cpu().numpy()
                        true = labels[idx]
                        correct += (pred == true).sum()
                        total += len(yb)
                        for t, p in zip(true, pred):
                            conf[t, p] += 1
                            class_total[t] += 1
                            if t == p:
                                class_correct[t] += 1
            finally:
                self.model.train(previous_mode)
        acc = 100.0 * correct / max(total, 1)
        class_acc = 100.0 * class_correct / np.maximum(class_total, 1)
        return conf, float(acc), class_acc, float(loss_sum / max(total, 1))

    # ======================================================================
    # 展示数据生成（网页上"看模型学到了什么"）
    # ======================================================================
    def _update_samples(self, conf):
        """
        对验证集里固定几张图做预测，生成：
          - samples：每张图的像素、真实标签、预测、10 个类的概率
          - fmaps： 第 1 层卷积的特征图（把"CNN 看到的"可视化）
          - confusion：混淆矩阵
        这些都只在每个 epoch 结束更新一次，网页据此刷新。
        """
        idx = np.array([i for i in SAMPLE_INDICES if i < len(self.x_val)], dtype=np.int64)
        if not len(idx):
            with self.lock:
                self.state["confusion"] = conf.tolist()
            return
        xb = torch.from_numpy(self.x_val[idx]).unsqueeze(1).to(self.device)
        with self.model_lock:
            previous_mode = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    probs = torch.softmax(self.model(xb), dim=1).cpu().numpy()
                    act = self.model.conv1_activations(xb[0:1])
            finally:
                self.model.train(previous_mode)
        pred = probs.argmax(1)
        true = self.y_val[idx]

        samples = []
        for i, (p, t) in enumerate(zip(pred, true)):
            samples.append({
                "pixels": (self.x_val[idx[i]] * 255.0).round().astype(int).tolist(),
                "true": int(t),
                "pred": int(p), "correct": bool(p == t),
                "prob": float(probs[i, p]),
                # 10 个类的概率，网页上画成"Softmax 概率条"
                "probs": [float(v) for v in probs[i]],
            })

        # ---- 特征图可视化：拿第一张样例图喂给第一层卷积 ----
        fmaps = []
        # act 来自上方同一次 eval 前向，不必再次切换模型模式。
        act = act[0].cpu().numpy()                          # (32,28,28)
        # 用 2x2 全局均值把每张特征图缩到 14x14，体积更小更好展示
        pooled = act.reshape(32, 14, 2, 14, 2).mean(4).mean(2)
        # 归一化到 0~255 便于网页当灰度图画
        lo, hi = pooled.min(), pooled.max()
        pooled = (pooled - lo) / (hi - lo + 1e-6) * 255.0
        for k in range(pooled.shape[0]):
            fmaps.append(pooled[k].round().astype(int).tolist())

        with self.lock:
            self.state["samples"] = samples
            self.state["fmaps"] = fmaps
            self.state["fmaps_version"] = self.state.get("fmaps_version", 0) + 1
            self.state["confusion"] = conf.tolist()

    def model_info(self):
        with self.model_lock:
            layers = self.model.describe_layers()
            summary = self.model.summary()
        return {"layers": layers, "summary": summary}

    def experiments(self):
        """仅返回本地已完整结束的真实运行；没有模拟成绩。"""
        if not self.experiment_file.is_file():
            return []
        try:
            with open(self.experiment_file, encoding="utf-8") as source:
                entries = json.load(source)
            return entries if isinstance(entries, list) else []
        except (OSError, ValueError):
            logging.warning("无法读取实验历史: %s", self.experiment_file)
            return []

    def record_experiment(self):
        """记录配置与验证结果供学习者对照；不按最终测试分数排序。"""
        with self.lock:
            entry = {"id": f"{int(time.time())}-{self.generation}",
                     "seed": SEED, "config": dict(self.state["config"]),
                     "dataset_sizes": dict(self.dataset_sizes),
                     "validation_acc": self.state["final_acc"],
                     "validation_loss": self.state["val_loss"],
                     "elapsed_sec": self._elapsed_seconds(),
                     "status": "done"}
        entries = self.experiments()[-19:] + [entry]
        self.experiment_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.experiment_file.with_suffix(".part")
        try:
            with open(temporary, "w", encoding="utf-8") as output:
                json.dump(entries, output, ensure_ascii=False, indent=2)
            os.replace(temporary, self.experiment_file)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def validate_pixels(pixels):
        return validate_pixels(pixels)

    def predict(self, pixels):
        """推理不会修改模型权重、BatchNorm 统计或原来的 train/eval 模式。"""
        image = self.validate_pixels(pixels)
        if not image.any():
            raise ValueError("画布为空，请先写一个数字。")
        with self.model_lock:
            probs = predict_probs(self.model, image, self.device)
        label = int(np.argmax(probs))
        with self.lock:
            trained = bool(self.state["epochs_val_acc"]) or self.state.get("model_loaded", False)
            generation = self.state["generation"]
        return {"pred": label, "prob": probs[label], "probs": probs,
                "trained": trained, "generation": generation}

    def save_model(self):
        """只在训练停止后保存本地模型；临时文件原子替换防止半截存档。"""
        with self.control_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("请先停止或等待训练完成再保存模型。")
            with self.lock:
                cfg = dict(self.state["config"])
                epoch = self.state["epoch_current"]
                trained = bool(self.state["epochs_val_acc"]) or self.state.get("model_loaded", False)
            if not trained:
                raise RuntimeError("模型尚未训练；请先完成至少一轮训练。")
            with self.model_lock:
                save_checkpoint(self.model_file, self.model, cfg, epoch, SEED)
            with self.lock:
                self.state["checkpoint_available"] = True
            return str(self.model_file)

    def load_model(self):
        """加载受固定路径约束、仅含张量与基本类型的存档；不会恢复优化器。"""
        with self.control_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("请先停止训练再加载模型。")
            payload, config = load_checkpoint(self.model_file)
            validated = self._validated_config(**config, device_mode=str(self.device))
            candidate = HandwritingCNN(dropout_p=validated["dropout"]).to(self.device)
            try:
                candidate.load_state_dict(payload["state_dict"], strict=True)
            except (RuntimeError, TypeError, KeyError) as exc:
                raise ValueError("存档权重与当前 CNN 结构不兼容。") from exc
            with self.model_lock:
                self.model = candidate
                self.optimizer = torch.optim.Adam(candidate.parameters(), lr=validated["lr"])
            self.steps_per_epoch = int(np.ceil(len(self.x_train) / validated["batch_size"]))
            with self.lock:
                self.generation += 1
                self._pending_config = None
                self.state = self._fresh_state("idle")
                self.state["generation"] = self.generation
                self.state["model_loaded"] = True
                self.state["checkpoint_available"] = True
                self.state["config"].update(validated)
                self.state["config"].update(device=str(self.device), device_mode=str(self.device),
                                            steps_per_epoch=self.steps_per_epoch)
                self.state["hardware"] = self.hardware
                self.state["steps_per_epoch"] = self.steps_per_epoch
                self.state["samples"] = self._sample_pixels()
            return str(self.model_file)

    # ======================================================================
    # 对外读取接口
    # ======================================================================
    def snapshot(self):
        """返回 state 的一份拷贝（线程安全），给 /api/state 用。"""
        with self.lock:
            if self.start_ts:
                self.state["elapsed_sec"] = self._elapsed_seconds()
            if self.state["steps_per_epoch"] and self.state["status"] == "running":
                # 用"每轮耗时"估算剩余时间(ETA)，给网页进度显示用
                ep = self.state["elapsed_sec"] / max(self.state["epoch_current"], 1)
                remain = (self.state["epoch_total"] - self.state["epoch_current"]) * ep
                self.state["eta_sec"] = round(max(remain, 0), 1)
            else:
                self.state["eta_sec"] = None
            return copy.deepcopy(self.state)


if __name__ == "__main__":
    # 命令行独立运行（不启动 web），方便快速验证训练流程是否正常
    t = Trainer()
    print(f"设备: {t.device}")
    t.start()
    while t.snapshot()["status"] not in ("done", "error"):
        time.sleep(1)
    if t.snapshot()["status"] == "error":
        raise SystemExit(t.snapshot()["error"])
    print("最终验证准确率: %.2f%%" % t.snapshot()["final_acc"])
