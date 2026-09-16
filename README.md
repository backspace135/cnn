# 手写数字识别 · 训练可视化 Demo

一个**用 CNN 训练 MNIST 手写数字识别**的小项目，重点在于**可视化完整训练过程**——
打开网页即可实时看到 loss 曲线、准确率变化、模型的实时预测效果、训练日志。

> 纯 CPU 即可运行（6 轮 MNIST 约 1.5 分钟跑完），无需 GPU。

---

## 一、这是什么 / 能学到什么

| 你关心的 | 对应到本项目的哪里 |
|---|---|
| 数据怎么来的 | `download_data.py` 从官方镜像下载 MNIST 并解析（含详细注释） |
| 卷积网络怎么搭 | `model.py` 一个 32→64 通道的小 CNN |
| 训练循环怎么写 | `train.py` 的 `_run_epoch`，逐 batch 前向→反向→更新 |
| 训练过程怎么「看」 | `train.py` 把每一步 loss/准确率写进 `self.state`，`server.py` 用 Flask 暴露，浏览器轮询渲染 |
| 模型预测对不对 | 面板上的「预测效果」区显示验证集真实手写图 + 模型判断 |

MNIST = 6 万张手写数字（0-9）灰度图，每张 28×28。这是深度学习的「Hello World」。

---

## 二、目录结构

```
cnn/
├─ download_data.py   下载并解析 MNIST -> data/mnist.npz
├─ model.py           CNN 模型定义（带逐层注释）
├─ train.py           训练器：后台线程训练 + 维护实时状态 state
├─ server.py          Flask 服务器：/api/state 提供状态，/ 返回面板
├─ paths.py           把项目 libs/ 里的 torch 注入 Python 路径
├─ static/index.html  可视化训练面板（原生 SVG/Canvas，无外部依赖）
├─ data/              MNIST 数据集（已下载）+ mnist.npz
└─ libs/              用 pip 下载到项目内的第三方库（torch 等）
```

---

## 三、怎么运行

环境：Windows + Python 3.9+（本项目在 3.13 验证）。依赖已装进项目 `libs/`。

**方式 A（双击一键启动）**：运行根目录的 `start.bat`。会自动打开浏览器并启动训练。

**方式 B（命令行）**：

```bash
# 首次：下载依赖到项目 libs/（torch CPU 版）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --target libs

# 首次：下载 MNIST 数据集到 data/
python download_data.py

# 启动训练 + 可视化面板
python server.py
```

然后在浏览器打开 **http://127.0.0.1:5000** 即可看到实时训练过程。

面板上可以点 **开始/继续 / 暂停 / 重置** 控制训练。默认 6 轮，约 1.5 分钟完成，验证准确率 ≈ 99%。

> 只跑训练、不开网页：`python train.py`

---

## 四、训练过程怎么「可视化」的（核心设计）

三块配合，构成「看训练过程」的闭环：

1. **train.py（生产者）**：训练在独立后台线程 `_run_loop` 里跑。每一步把
   `loss`、每个 epoch 后的训练/验证准确率、预测样例、日志写入 `self.state`（一个 dict）。
   用 `self.lock` 保证多线程安全，`snapshot()` 返回一份拷贝供外部读取。

2. **server.py（传输层）**：Flask 提供
   - `GET /api/state` → 返回 `snapshot()`
   - `POST /api/control` → `{action: start|pause|resume|reset}` 控制训练线程

3. **static/index.html（消费者）**：用 `setInterval` 每 0.7 秒轮询 `/api/state`，
   把所有指标画成 SVG 曲线 / Canvas 手写图 / 柱状图。完全原生实现，**不依赖任何 CDN**，离线也能跑。

这样训练和数据展示彻底解耦：以后换数据集、换模型、甚至换 GPU 训练，面板都不用改。

---

## 五、模型结构速览（想进一步学习）

```
输入 28×28×1
 └─ Conv2d(1→32, 3×3, padding=1) → BatchNorm → ReLU → MaxPool2d(2)    14×14
    └─ Conv2d(32→64, 3×3, padding=1) → BatchNorm → ReLU → MaxPool2d(2)  7×7
       └─ Flatten → Linear(64·7·7 → 128) → ReLU → Dropout(0.3) → Linear(128→10)
```

- **卷积层**提取局部特征（笔画、边、圈），**池化**逐步压缩尺寸、保留主要特征。
- **Dropout** 随机丢弃部分神经元，防止过拟合。
- 最后一层输出 10 个得分（对应数字 0-9），取最大即为预测类。
- 参数总量约 **42 万**，非常轻量，适合讲解与演示。

---

## 六、调参玩一玩（都改在 `train.py` 顶部）

| 参数 | 作用 | 试试改它看曲线变化 |
|---|---|---|
| `EPOCHS` | 训练轮数 | 调大看准确率是否继续涨/停滞 |
| `LR`    | 学习率 | 调大易震荡，调小收敛变慢 |
| `BATCH_SIZE` | 每次更新用的样本数 | 调大波动更平滑但更慢 |
| `SEED`  | 随机种子 | 固定后结果可复现 |

---

## 常见问题

- **端口被占**：改运行 `$env:PORT=5001; python server.py`
- **想用 GPU**：改成安装 `--index-url https://download.pytorch.org/whl/cu124` 的 torch，
  程序会自动检测并使用 `cuda`（需已装 CUDA 驱动）。
