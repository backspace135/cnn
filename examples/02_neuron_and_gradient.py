# -*- coding: utf-8 -*-
"""一个神经元、平方误差和梯度；不需要下载 MNIST。

运行：python examples/02_neuron_and_gradient.py
先用普通函数手算，再用 PyTorch 的自动求导核对同一个导数。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import paths  # noqa: F401  让项目 libs/ 中的 torch 可被导入

import numpy as np
import torch


def neuron_output(inputs: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """神经元先计算加权和，再用 ReLU 把负值变成 0。"""
    weighted_sum = float(np.dot(inputs, weights) + bias)
    return max(0.0, weighted_sum)


def squared_error(prediction: float, target: float) -> float:
    """预测值离目标越远，平方误差越大。"""
    return (prediction - target) ** 2


def single_weight_gradient(weight: float, input_value: float, target: float) -> float:
    """当预测 = weight × input_value 时，误差对 weight 的导数。"""
    prediction = weight * input_value
    return 2 * (prediction - target) * input_value


if __name__ == "__main__":
    inputs = np.array([2.0, 3.0])
    weights = np.array([0.5, -0.25])
    print("神经元输出（ReLU 后）:", neuron_output(inputs, weights, bias=0.1))

    # 为了只观察一个梯度，下面使用没有 ReLU、没有偏置的单权重线性模型。
    x, target, learning_rate = 3.0, 4.0, 0.01
    weight = 2.0
    before = squared_error(weight * x, target)
    hand_gradient = single_weight_gradient(weight, x, target)

    torch_weight = torch.tensor(weight, requires_grad=True)
    loss = (torch_weight * x - target) ** 2
    loss.backward()  # 沿计算图反向传播，得到 loss 对 weight 的导数
    updated_weight = weight - learning_rate * hand_gradient

    print(f"更新前：权重={weight:.2f}，误差={before:.4f}")
    print(f"手算梯度={hand_gradient:.4f}，自动求导={torch_weight.grad.item():.4f}")
    print(f"沿梯度反方向更新：权重={updated_weight:.2f}，"
          f"误差={squared_error(updated_weight * x, target):.4f}")
    print("CNN 也重复 前向→误差→反向→更新，只是权重更多、误差函数不同。")
