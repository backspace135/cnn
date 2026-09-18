# -*- coding: utf-8 -*-
"""自己实现小卷积核扫描与最大池化；不需要下载 MNIST。

运行：python examples/03_convolution_and_pooling.py
这里的“卷积”和 PyTorch Conv2d 一样，不翻转卷积核，严格说是互相关。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import paths  # noqa: F401  让项目 libs/ 中的 torch 可被导入

import numpy as np
import torch
import torch.nn.functional as F


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """无填充、步长为 1：在每个位置用窗口与卷积核逐项相乘后求和。"""
    height, width = image.shape
    kh, kw = kernel.shape
    output = np.empty((height - kh + 1, width - kw + 1), dtype=np.float32)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            window = image[row:row + kh, col:col + kw]
            output[row, col] = np.sum(window * kernel)
    return output


def max_pool2x2(image: np.ndarray) -> np.ndarray:
    """每个不重叠的 2×2 区域保留最大值；奇数边长丢弃末尾一行/列。"""
    height, width = image.shape
    output = np.empty((height // 2, width // 2), dtype=image.dtype)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = np.max(image[2 * row:2 * row + 2,
                                            2 * col:2 * col + 2])
    return output


if __name__ == "__main__":
    image = np.array([[0, 0, 0, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0]], dtype=np.float32)
    kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
    feature_map = convolve2d(image, kernel)
    pooled = max_pool2x2(feature_map)

    # PyTorch 需要 (批量, 通道, 高, 宽)；手算结果可与框架核对。
    x = torch.tensor(image)[None, None]
    weight = torch.tensor(kernel)[None, None]
    torch_feature = F.conv2d(x, weight)
    print("原图（4×4）：\n", image)
    print("卷积核（2×2）：\n", kernel)
    print("扫描后的特征图（3×3）：\n", feature_map)
    print("最大池化后（1×1）：\n", pooled)
    print("手算与 PyTorch Conv2d 一致：",
          np.allclose(feature_map, torch_feature[0, 0].numpy()))
    print("HandwritingCNN 用 padding=1 的 3×3 卷积保持宽高不变，再池化减半。")
