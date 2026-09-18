# -*- coding: utf-8 -*-
"""从一个像素走到 CNN 输入张量；不需要下载 MNIST。

运行：python examples/01_pixels_and_tensors.py
这里的像素是灰度值：0 是黑色，255 是白色。模型使用 [0, 1] 的浮点数。
"""
import sys
from pathlib import Path

# 直接运行 examples/ 下的脚本时，Python 默认找不到项目根目录的 paths.py。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import paths  # noqa: F401  先把项目 libs/ 加入搜索路径，再导入第三方库

import numpy as np
import torch


def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    """将 0~255 的原始灰度值转换为新的 float32 数组，不修改原图。"""
    return np.asarray(pixels, dtype=np.float32) / 255.0


def to_model_input(image: np.ndarray) -> torch.Tensor:
    """将一张 (高, 宽) 图变成 (批量=1, 通道=1, 高, 宽) 张量。"""
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    # 不用真实数据：自己画一张 28×28 的小图，只展示左上角四个像素。
    pixels = np.zeros((28, 28), dtype=np.uint8)
    pixels[:2, :2] = [[0, 128], [255, 64]]
    image = normalize_pixels(pixels)
    batch = to_model_input(image)

    print("原始像素（左上角）：\n", pixels[:2, :2])
    print("归一化后（左上角）：\n", image[:2, :2])
    print("NumPy 图片形状:", image.shape)  # (28, 28)
    print("PyTorch 输入形状:", tuple(batch.shape))  # (1, 1, 28, 28)
    print("四个维度依次是：图片数量、颜色通道数、高、宽")
    print("原图未改变：", pixels[0, 1] == 128)
