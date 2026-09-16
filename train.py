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
  status                idle | running | paused | done
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
import threading
import time
from pathlib import Path

import paths  # noqa: F401  把项目 libs/ 里的 torch 注入 sys.path

import numpy as np
import torch
import torch.nn as nn

from model import HandwritingCNN

DATA_FILE = Path(__file__).resolve().parent / "data" / "mnist.npz"

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

    def __init__(self):
        # 固定随机种子：让"重新跑一次"的结果可复现，
        # 是研究机器学习时非常重要的好习惯。
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # 优先用 GPU（如果电脑有 NVIDIA 显卡且装了 CUDA），否则用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 注意初始化顺序：必须先建好锁和事件 ----
        # 因为后面的 _load_data / log 都需要 self.lock 和 self.state 已存在。
        # 多线程里所有对 self.state 的读写都要加 self.lock，避免数据竞争。
        self.lock = threading.Lock()
        self.pause_event = threading.Event()   # 置位=True 表示暂停训练
        self._stop_iter = threading.Event()
        self._thread = None                    # 训练线程
        self.start_ts = None                   # 本轮训练开始时间

        # ---- 载入数据（见 _load_data）----
        self._load_data()
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
        tr, te = self._data_shape
        self.log(f"载入数据: 训练 {tr[0]:,} 张, 测试 {te[0]:,} 张")
        self.log(f"设备: {self.device}")
        self.log(self.model.summary())

    # ======================================================================
    # 数据加载
    # ======================================================================
    def _load_data(self):
        """从 data/mnist.npz 读取 MNIST。npz 是 numpy 的压缩存档格式。"""
        data = np.load(DATA_FILE)
        self.x_train = data["x_train"]   # (60000, 28, 28) float32 [0,1] 训练图
        self.y_train = data["y_train"]   # (60000,)               int64  训练标签
        self.x_test = data["x_test"]     # (10000, 28, 28)               测试图
        self.y_test = data["y_test"]     # (10000,)
        self._data_shape = (self.x_train.shape, self.x_test.shape)

    @staticmethod
    def _fresh_state(status: str) -> dict:
        """构造一个全新的、结构固定的 state 字典（所有字段保持默认）。"""
        return {
            "status": status,
            "epoch_current": 0, "epoch_total": DEFAULT_EPOCHS,
            "step": 0, "steps_per_epoch": 0,
            "train_loss_cur": None,
            "loss_steps": [], "loss_history": [], "smooth_loss": [],
            "epochs_train_acc": [], "epochs_val_acc": [],
            "val_loss": None,
            "class_acc": [],
            "confusion": [],
            "samples": [],
            "fmaps": [], "fmaps_version": 0,
            "elapsed_sec": 0.0, "eta_sec": None,
            "config": {"batch_size": DEFAULT_BATCH, "lr": DEFAULT_LR,
                       "epochs": DEFAULT_EPOCHS, "dropout": DEFAULT_DROPOUT,
                       "device": "", "steps_per_epoch": 0},
            "log": [],
        }

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
        """启动或继续训练。若线程还没创建，则创建并启动后台线程。"""
        if self._thread is not None and self._thread.is_alive():
            self.resume()
            return
        self._stop_iter.clear()
        self.pause_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        """暂停训练（训练线程会在下一个 batch 前停下来等）。"""
        self.pause_event.set()
        with self.lock:
            self.state["status"] = "paused"

    def resume(self):
        """继续暂停的训练。"""
        self.pause_event.clear()
        with self.lock:
            self.state["status"] = "running"

    def configure(self, epochs=None, lr=None, batch_size=None, dropout=None):
        """【网页调参】把超参存进 state，等 reset() 时按新参数重建模型。"""
        with self.lock:
            if epochs is not None:
                self.state["config"]["epochs"] = int(epochs)
            if lr is not None:
                self.state["config"]["lr"] = float(lr)
            if batch_size is not None:
                self.state["config"]["batch_size"] = int(batch_size)
            if dropout is not None:
                self.state["config"]["dropout"] = float(dropout)

    def reset(self, epochs=None, lr=None, batch_size=None, dropout=None):
        """
        按（可选的）新超参重建模型、优化器和 state，然后自动开始训练。
        注意：此方法应在线程结束后调用（网页重置时会先停旧线程）。
        """
        if lr is not None or epochs is not None or batch_size is not None or dropout is not None:
            self.configure(epochs, lr, batch_size, dropout)

        # 读取配置
        epochs = self.state["config"]["epochs"]
        lr = self.state["config"]["lr"]
        batch_size = self.state["config"]["batch_size"]
        dropout = self.state["config"]["dropout"]

        # 重置随机种子 -> 重建模型/优化器（每次重置都从"没学过"开始）
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.model = HandwritingCNN(dropout_p=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._stop_iter = threading.Event()
        self.steps_per_epoch = int(np.ceil(len(self.x_train) / batch_size))

        with self.lock:
            self.state = self._fresh_state("idle")
            self.state["epoch_total"] = epochs
            self.state["steps_per_epoch"] = self.steps_per_epoch
            self.state["config"] = {"batch_size": batch_size, "lr": lr,
                                    "epochs": epochs, "dropout": dropout,
                                    "device": str(self.device),
                                    "steps_per_epoch": self.steps_per_epoch}
        self.log(f"已重置模型（lr={lr}, batch={batch_size}, "
                 f"dropout={dropout}），共 {epochs} 轮，设备 {self.device}")

    # ======================================================================
    # 训练主循环（在后台线程里跑）
    # ======================================================================
    def _run_loop(self):
        """整个训练的主流程：一轮一轮地跑，直到跑完或被打断。"""
        self.start_ts = time.time()
        epochs = self.state["epoch_total"]
        with self.lock:
            self.state["status"] = "running"

        try:
            for epoch in range(1, epochs + 1):
                if self._stop_iter.is_set():
                    break
                self._run_epoch(epoch, epochs)
            # 训练正常结束，记录最终准确率
            self.log("训练完成！")
            with self.lock:
                self.state["final_acc"] = self.state["epochs_val_acc"][-1] \
                    if self.state["epochs_val_acc"] else None
                self.state["status"] = "done"
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"训练出错: {e}")
            with self.lock:
                self.state["status"] = "done"

    def _run_epoch(self, epoch: int, total: int):
        """
        一个 epoch = 把训练集完整看一遍。
        里面是一步一步（step/batch）地做"前向->算loss->反向->更新"。
        """
        self.model.train()   # 切换到训练模式（Dropout/BatchNorm 按训练行为）
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
            self.optimizer.zero_grad()      # 清空上一次的梯度
            out = self.model(xb)            # 前向：图片 -> 10 个分数
            loss = self.criterion(out, yb)  # 算 loss：和正确答案差多少
            loss.backward()                 # 反向：算出每个权重的梯度
            self.optimizer.step()           # 更新：权重沿梯度走一小步

            running_loss += loss.item()
            n_batches += 1
            pred = out.argmax(1)            # 取分数最大的类作为预测
            correct += (pred == yb).sum().item()
            seen += len(yb)

            # ---- 记录到 state（网页实时刷）----
            step = self.state["step"] + 1
            with self.lock:
                self.state["step"] = step
                self.state["epoch_current"] = epoch
                self.state["train_loss_cur"] = loss.item()
                if len(self.state["loss_steps"]) < MAX_LOSS_POINTS:
                    self.state["loss_steps"].append(step)
                    self.state["loss_history"].append(loss.item())

        # ---- 一个 epoch 结束：算训练平均 loss 和训练准确率 ----
        train_loss_avg = running_loss / max(n_batches, 1)
        train_acc = 100.0 * correct / max(seen, 1)
        self.log(f"Epoch {epoch}/{total} 训练: loss={train_loss_avg:.4f} "
                 f"acc={train_acc:.2f}%")

        # ---- 在验证集上评估（模型没见过测试集，测真实水平）----
        conf, val_acc, class_acc, val_loss = self._evaluate()

        # 一次性更新多个展示字段（在锁内，避免读到一半的状态）
        with self.lock:
            self.state["epochs_train_acc"].append(float(train_acc))
            self.state["epochs_val_acc"].append(float(val_acc))
            self.state["val_loss"] = float(val_loss)
            self.state["class_acc"] = [float(x) for x in class_acc]
            self._write_smooth_loss()
        # 注意：_update_samples 内部自身会获取 self.lock，所以在锁外调用，
        # 否则 (非可重入锁) 会造成死锁 —— 这是多线程常见的坑。
        self._update_samples(conf)
        self.log(f"Epoch {epoch}/{total} 验证: acc={val_acc:.2f}% "
                 f"loss={val_loss:.4f}")

    @property
    def config_bs(self):
        return self.state["config"].get("batch_size", DEFAULT_BATCH)

    # ======================================================================
    # 验证集评估
    # ======================================================================
    def _evaluate(self):
        """
        用测试集 (x_test) 评估模型：算准确率、每类准确率、混淆矩阵。
        这里不更新参数 (torch.no_grad)，只用来看模型学得怎么样。
        为什么需要验证集？因为模型"背"训练集可能背得很好（overfit），
        只有用没见过的数据测试，才是真实水平。
        """
        self.model.eval()   # 切换到评估模式（Dropout 关闭）
        correct, total = 0, 0
        loss_sum = 0.0
        class_correct = np.zeros(10, dtype=np.int64)
        class_total = np.zeros(10, dtype=np.int64)
        conf = np.zeros((10, 10), dtype=np.int64)
        with torch.no_grad():
            for b in range(int(np.ceil(len(self.x_test) / 256))):
                idx = slice(b * 256, (b + 1) * 256)
                xb = torch.from_numpy(self.x_test[idx]).unsqueeze(1).to(self.device)
                yb = torch.from_numpy(self.y_test[idx]).to(self.device)
                out = self.model(xb)
                loss_sum += self.criterion(out, yb).item() * len(yb)
                pred = out.argmax(1).cpu().numpy()
                true = self.y_test[idx]
                correct += (pred == true).sum()
                total += len(yb)
                for t, p in zip(true, pred):   # 累加混淆矩阵
                    conf[t, p] += 1
                    class_total[t] += 1
                    if t == p:
                        class_correct[t] += 1
        acc = 100.0 * correct / max(total, 1)
        class_acc = 100.0 * class_correct / np.maximum(class_total, 1)
        val_loss = loss_sum / max(total, 1)
        return conf, float(acc), class_acc, float(val_loss)

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
        self.model.eval()
        idx = np.array([i for i in SAMPLE_INDICES if i < len(self.x_test)], dtype=np.int64)
        xb = torch.from_numpy(self.x_test[idx]).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        true = self.y_test[idx]

        samples = []
        for i, (p, t) in enumerate(zip(pred, true)):
            samples.append({
                "pixels": (self.x_test[idx[i]] * 255.0).round().astype(int).tolist(),
                "true": int(t),
                "pred": int(p),
                "prob": float(probs[i, p]),
                # 10 个类的概率，网页上画成"Softmax 概率条"
                "probs": [float(v) for v in probs[i]],
            })

        # ---- 特征图可视化：拿第一张样例图喂给第一层卷积 ----
        fmaps = []
        with torch.no_grad():
            act = self.model.conv1_activations(xb[0:1])   # (1,32,28,28)
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

    def _write_smooth_loss(self, smooth: float = 0.98):
        """指数滑动平均：把上上下下的原始 loss 抹平，曲线更好读。"""
        h = self.state["loss_history"]
        if not h:
            return
        s = h[0]
        out = []
        for v in h:
            s = smooth * s + (1 - smooth) * v
            out.append(s)
        self.state["smooth_loss"] = out

    # ======================================================================
    # 对外读取接口
    # ======================================================================
    def snapshot(self):
        """返回 state 的一份拷贝（线程安全），给 /api/state 用。"""
        with self.lock:
            if self.start_ts:
                self.state["elapsed_sec"] = round(time.time() - self.start_ts, 1)
            if self.state["steps_per_epoch"] and self.state["status"] == "running":
                # 用"每轮耗时"估算剩余时间(ETA)，给网页进度显示用
                ep = self.state["elapsed_sec"] / max(self.state["epoch_current"], 1)
                remain = (self.state["epoch_total"] - self.state["epoch_current"]) * ep
                self.state["eta_sec"] = round(max(remain, 0), 1)
            else:
                self.state["eta_sec"] = None
            return {**self.state}


if __name__ == "__main__":
    # 命令行独立运行（不启动 web），方便快速验证训练流程是否正常
    t = Trainer()
    print(f"设备: {t.device}")
    t.start()
    while t.snapshot()["status"] != "done":
        time.sleep(1)
    print("最终验证准确率: %.2f%%" % t.snapshot()["final_acc"])
