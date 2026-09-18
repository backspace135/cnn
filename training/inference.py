"""单张手写数字的输入校验与无梯度推理。"""

import math

import numpy as np
import torch


def validate_pixels(pixels):
    """预测 API 只接收一张 28×28 数值矩阵，不接受文件路径或 pickle。"""
    if not isinstance(pixels, list) or len(pixels) != 28:
        raise ValueError("pixels 必须是 28×28 数字矩阵。")
    for row in pixels:
        if not isinstance(row, list) or len(row) != 28:
            raise ValueError("pixels 必须是 28×28 数字矩阵。")
        for value in row:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("像素必须是 [0,1] 之间的有限数字。")
    return np.asarray(pixels, dtype=np.float32)


def predict_probs(model, image, device):
    """调用者持有 model_lock；推理结束后恢复原来的 train/eval 模式。"""
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            return torch.softmax(model(x), dim=1)[0].cpu().tolist()
    finally:
        model.train(previous_mode)
