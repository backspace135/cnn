# Python 与源码：选修路线

[网页 00–06 主线](../learning-guide.md) 不要求写代码或看懂导数。这一页适合已经启动「从像素开始」课程、想知道程序从哪里来的读者；若还没装 Python 和依赖，先按 [Windows 零基础启动指南](../windows-start.md) 安装后直接打开课程，**MNIST 不必在打开网页之前下载**。依赖在本项目 `libs/`，安装脚本不会替你安装 Python；首次装依赖要联网，MNIST 留到 04 的真实训练或运行例子 04 前再联网下载。

## 先认识最少的 Python 符号

| 看见 | 先这样读 |
|---|---|
| `x = 3` | 给名字 `x` 保存数字 3；之后可用这个名字。 |
| `numbers[0]` | 取一组值里的第一个（从 0 开始数）。 |
| `f(x)` | 调用叫 `f` 的函数，把 `x` 交给它。 |
| `for ... in ...:` | 按顺序重复做缩进的几行；冒号和缩进界定这一段。 |
| `import torch` | 使用已经安装的库；若提示没有这个模块，先检查安装。 |
| `# ...` | 给人看的注释，Python 不执行它。 |

不需要一次读懂整份文件。先运行一个例子，看输出，再找它最前面的文字说明和打印语句；看不懂的部分可以跳过，继续用网页。

## 运行四个原有例子（不影响网页课程）

在项目文件夹（能看见 `install.bat` 的地方）的资源管理器地址栏输入 `powershell` 并回车。完成 `install.bat` 后，在打开的 PowerShell **逐条**运行下面的命令；前三个生成自己的玩具数据，不需 MNIST，但都需要 Python、NumPy 和 PyTorch：

```powershell
python examples/01_pixels_and_tensors.py
python examples/02_neuron_and_gradient.py
python examples/03_convolution_and_pooling.py
```

| 例子 | 应该看什么 | 对应网页 |
|---|---|---|
| `01` 像素与张量 | 原图、归一化后的数和 `(1, 1, 28, 28)` 输入尺寸；`0` 是黑、`255` 是白。 | **01 数字图片**；张量维度是选修扩展 |
| `02` 神经元与梯度 | 一次小规模的“预测、误差、更新”；这个平方误差玩具例子不是实际 CNN 的分类损失。 | **04 让机器学习**；导数可先跳过 |
| `03` 卷积与池化 | 小矩阵逐项乘加的结果、池化结果及与 PyTorch 的核对。 | **02 找局部线索** |

想继续看**独立于网页**的一次最小训练，先在同一个项目文件夹运行下载命令，等出现 `[完成]`；如果数据已下载，脚本会校验本地原始文件，然后再生成 `data/mnist.npz`：

```powershell
python download_data.py
python examples/04_minimal_training.py --epochs 1 --max-samples 512 --device cpu
```

第四个例子会打印训练与验证 loss、准确率；它只取小样本教学，不保存模型、不启动网页，也不报告最终测试成绩。`--max-samples 512` **同时**限制训练与验证部分，不能只把增加这个值的效果解释成“训练图片更多”。缺少 `data/mnist.npz` 时回到 [启动指南第 5 步](../windows-start.md#5-到训练课再下载手写数字数据)；缺少 `torch` 或 `numpy` 时重新运行 `install.bat`，并确认使用同一个 Python。

## 看懂脚本和网页如何连接

1. `paths.py` 把项目里的 `libs/` 加入导入路径；直接运行 `examples/` 里的文件时，脚本会先找到项目根目录再导入它。无需把依赖移到系统的其他文件夹。
2. `download_data.py` 下载并检查原始 MNIST，生成 `data/mnist.npz`；`data.py` 读取、验证图片和标签，把原训练图片分为训练集和验证集；原测试集留到最后。
3. `model.py` 定义 `HandwritingCNN`：两组卷积/池化处理图像，最后输出十个原始类别分数（logits）。从这里先找 `features`、`classifier` 和 `forward`；不必马上理解所有 PyTorch 类。
4. `examples/04_minimal_training.py` 用这个模型同步演示每批的 `zero_grad → forward → loss → backward → step`，轮末只在验证集上评估。`train.py` 将同类工作放在后台、记录进度与实验数据；`server.py` 提供本机网页调用的接口，`static/` 是页面文件。**网页训练和例子 04 是两次独立运行**。

若想理解 `examples/02` 里的数字，先读 [数学进阶](math.md) 的四则运算部分；导数是可选的。回到 [学习指南](../learning-guide.md) 可继续只用网页学习。

> 运行前请先完成 `install.bat` 的依赖安装；前三个例子不需要 MNIST，第四个例子需要 `data/mnist.npz`。
