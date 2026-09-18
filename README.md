# 从像素到预测：CNN 手写数字入门课

一个能**边读、边算、边运行**的中文卷积神经网络入门项目。你不需要先学微积分：先看像素和一个神经元，再亲手计算卷积，最后训练真实 MNIST 模型、分析错误并识别自己写的数字。

- **从零学习**：四个按顺序运行的小脚本 + [分课讲义](docs/learning-guide.md) + [实验记录模板](docs/experiments.md)。
- **看见学习过程**：浏览器显示实时 loss、训练/验证准确率、卷积特征图、预测概率和混淆矩阵。
- **动手验证**：可编辑的卷积矩阵逐格计算；画布/本地图片识别；学习率、批大小、Dropout 实验。
- **本地优先**：Flask + NumPy + PyTorch + 原生 HTML/JS，无账号、数据库或 CDN。准备依赖和首次下载数据需联网，之后可离线学习。

## 0. 环境与安装（Windows）

需要 Python 3.11 或更新版本。本项目曾在 Python 3.13 上验证 CUDA 路径；其他版本、硬件和安装时间取决于你的环境。只需要 CPU 就能完成全部课程。

```powershell
# 在项目目录运行；首次需要联网
.\install.bat
python download_data.py
.\start.bat
```

双击同名 bat 文件也可以。网页地址是 <http://127.0.0.1:5000>。**网页默认不会自动开始训练**，请阅读前几课后点击「训练实践」→「开始训练」。命令行演示想自动启动时，可先设置 `$env:AUTO_TRAIN='1'` 再运行 `python server.py`。

安装脚本把 Flask、NumPy 和 CPU 版 PyTorch 放在项目的 `libs/`（Git 不收录）。若偏好自己管理环境，也可先用 `python -m pip install -r requirements.txt` 安装 Flask/NumPy，再按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与自己系统匹配的 Torch；`paths.py` 会优先使用已有 `libs/`/`libs-cuda/`。如果切换 Python 版本，最好重新安装依赖。

有兼容的 NVIDIA GPU 且想尝试 CUDA：关闭服务后运行 `install_cuda.bat`，重启并在调参区选择 CUDA。它单独安装在 `libs-cuda/`，不覆盖 CPU 目录；CUDA 不可用时可继续选择 CPU。不需要 GPU 也能学完。

如无 MNIST，前三个入门脚本仍可运行；访问网页会看到数据准备提示，`start.bat` 则要求先完成下载。`data/`、`libs/`、`libs-cuda/` 均为本地生成，不应提交到 Git。

## 1. 学习路线（推荐按顺序）

| 步骤 | 理解什么 | 运行/操作 |
|---|---|---|
| 图片是数字 | 灰度、归一化、维度 `(N,C,H,W)` | `python examples/01_pixels_and_tensors.py` |
| 一个神经元怎么学习 | 权重、loss、梯度与更新 | `python examples/02_neuron_and_gradient.py` |
| 手算卷积和池化 | 局部窗口、共享核、stride/padding | `python examples/03_convolution_and_pooling.py`；网页「原理实验」 |
| 组装网络 | 从 `(1,28,28)` 到 10 个类别得分 | 阅读 `model.py`；网页「认识模型」 |
| 最小训练循环 | `zero_grad → forward → loss → backward → step` | `python examples/04_minimal_training.py --epochs 1 --max-samples 512 --device cpu` |
| 实时训练与评估 | 验证集曲线、混淆矩阵、独立测试 | 网页「训练实践」；阅读 `train.py` |
| 自己识别和实验 | eval/预处理、控制变量 | 网页「手写识别」「调参实验」；填写 `docs/experiments.md` |

**建议每步先猜答案再运行。** [完整课程地图、自测与答案](docs/learning-guide.md) 解释了每课要看的代码。第四个脚本与网页共享 `data.py` 的划分规则和同一个 `HandwritingCNN`；它没有线程和 Web 服务，更方便逐行阅读。小样本练习的成绩不能代表完整 MNIST 成绩。

## 2. 项目结构

```text
cnn/
├── data.py                 读取/校验 npz，固定种子按类别划分验证集
├── download_data.py        下载并检查 IDX/Gzip，生成 data/mnist.npz
├── model.py                两组卷积 + 池化 + 全连接分类器
├── train.py                后台训练、逐轮验证、最终测试、单张预测与模型存档
├── server.py               本地 Flask API，默认打开课程但不自动训练
├── paths.py                在项目内优先加载 libs/ 和可选 libs-cuda/
├── examples/               四个可独立运行的小实验（前 3 个无需 MNIST）
├── docs/                   完整课程地图与实验模板
├── static/                 原生 HTML/CSS/JS 学习页面
├── tests/                  无网络的小数据单元/API 测试
├── install.bat             准备 CPU 依赖
├── install_cuda.bat        可选 CUDA 安装
├── start.bat               启动前检查依赖和数据
└── data/                   本地 MNIST、存档（运行后生成，Git 忽略）
```

## 3. 数据、模型与评估

MNIST 原始数据包含 60,000 张训练图和 10,000 张测试图，每张是黑底白字的 28×28 灰度图。项目以固定种子按类别把原始训练集划成约 **54,000 张训练集** + **6,000 张验证集**；原始 10,000 张测试图**只在完整训练结束后评估一次**。训练集用于更新权重，验证集用于每轮观察/调参，测试集相当于最终考试。

```text
(N,1,28,28) → Conv(1→32,3×3)+BN+ReLU+Pool → (N,32,14,14)
             → Conv(32→64,3×3)+BN+ReLU+Pool → (N,64,7,7)
             → Flatten(3136)+Linear(128)+ReLU+Dropout → Linear(10 logits)
```

模型输出的 **logits 是原始类别得分，不是概率**。训练时 `CrossEntropyLoss` 直接接收 logits，不应事先 Softmax；展示概率时才使用 Softmax。每个 batch 清梯度、计算预测和交叉熵、反向传播梯度，再由 Adam 更新权重。训练中的准确率与轮末 eval 模式下的验证准确率不是同一时刻/模式的严格对照。固定种子有助于比较实验，但不同设备不一定逐位一致。

混淆矩阵的**行是真实数字、列是预测数字**；对角线越集中，越少误认。首层特征图是卷积响应可视化，不保证一个通道只代表一种能说清的笔画。准确率、耗时与机器性能及训练设置有关，没有固定必达值。

## 4. 自己识别与保存模型

在网页「手写识别」用鼠标或触屏画**单个数字**，或者选择本地 PNG/JPEG。浏览器会裁剪前景、保持比例缩放并居中到 28×28；黑字白底文件应勾选反色，然后检查输入预览。浏览器只发送处理后的 28×28 数字矩阵给本机 `/api/predict`，不会把原始图片发往外部服务。复杂背景照片、多位数字或与 MNIST 不同的笔迹可能失败；最高概率并不保证预测正确。

完整训练结束后可以在训练实践页点击「保存已训练模型」，写入本机 `data/checkpoints/latest.pt`；下次运行点击「加载本地模型用于识别」。这里只保存识别所需权重与配置，**并非完整训练断点续训**，想重新训练请重置。只加载固定本地路径，不接收用户上传的 checkpoint 文件。

## 5. 调参实验

先记录默认设置作为基线，每次**只改变一个变量**，记录自己的预测、实际曲线和验证指标：

- 学习率：较大可能震荡，较小可能学习缓慢，但具体效果要看本次数据与轮数。
- Batch Size：改变更新次数与噪声；并非越大越快。
- Dropout：与训练/验证表现的差距一起观察；关闭它不意味着一定过拟合。

填写 [实验记录模板](docs/experiments.md)。网页还会把**完整结束**的运行配置、验证成绩和耗时保存在本机 `data/checkpoints/experiments.json`，在调参页供比较；取消/失败的运行不列入完整成绩。不要反复查看最终测试集来选参数。

## 本地 API 速查

浏览器只调用本机服务；以下接口也可用 Flask 的 `test_client` 学习或测试。错误统一包含 `ok:false`、兼容旧页面的 `msg` 和 `error.code/message`，参数错误为 400，状态冲突为 409，数据尚未准备为 503。

| 接口 | 用途 |
|---|---|
| `GET /api/health` | 服务、数据与设备状态 |
| `GET /api/state` | 当前训练状态、验证曲线、最终测试与样例 |
| `POST /api/control` | `{"action":"start"}`、`pause`、`resume`、`stop`、`reset` 或 `configure`；重置可带 epochs/lr/batch_size/dropout/device_mode |
| `GET /api/model-info` | 由实际模型前向产生的各层形状和参数量 |
| `POST /api/predict` | `{"pixels":[28 行 × 每行 28 个 0–1 数字]}`；返回预测类及十类概率 |
| `POST /api/model` | `{"action":"save"}` 或 `{"action":"load"}`；只操作固定本地存档 |
| `GET /api/experiments` | 本机已完整完成的实验记录 |

示例：在 PowerShell 中运行 `Invoke-RestMethod http://127.0.0.1:5000/api/health`。本地服务无登录机制，因此仅绑定 `127.0.0.1`，不要把它暴露到公共网络。

## 6. 测试与常见问题

```powershell
python -m unittest discover -s tests -v
python examples/01_pixels_and_tensors.py
python examples/02_neuron_and_gradient.py
python examples/03_convolution_and_pooling.py
```

基础测试使用内存中的小数据，不下载完整 MNIST；有真实 GPU 时会额外测试 CUDA，否则相应测试跳过。做完整演示再运行 `python server.py`，打开 <http://127.0.0.1:5000>。

- **找不到数据**：运行 `python download_data.py`；若原始 gzip 缓存损坏，下载脚本会重新校验并下载。
- **缺少依赖**：运行 `install.bat`；检查 Python 路径/版本是否与安装时一致。
- **端口被占用**：在 PowerShell 先设置 `$env:PORT='5001'`，然后 `python server.py`。
- **CUDA 不可用**：选择 CPU；CUDA 需要兼容的 NVIDIA GPU、驱动和 CUDA 版 PyTorch，不是所有机器都支持。
- **画的数字识别错误**：先查看输入预览，保持黑底白字、居中、单个数字；MNIST 模型不擅长真实照片。
- **小样本准确率低**：例子 04 默认仅取少量数据做教学，完整训练与它的数字不可直接比较。

这是单机学习项目，不是面向互联网的生产服务；Flask 只监听 `127.0.0.1`。课程练习中提供的权重/训练例子仅供理解原理。

## 许可证

本仓库的源代码和项目文档采用 [MIT License](LICENSE)，Copyright © 2026 backspace135。运行时下载的 MNIST 数据及安装的第三方依赖不属于本仓库的 MIT 授权范围，请分别遵守其自身的许可条款。
