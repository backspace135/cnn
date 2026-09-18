# -*- coding: utf-8 -*-
"""最小同步训练循环：读取 MNIST，训练一轮，再在留出的验证集上评估。

先运行 python download_data.py，再运行：
    python examples/04_minimal_training.py --epochs 1 --max-samples 512 --device cpu
脚本不会启动网页，也不会用测试集调参或保存权重。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import paths  # noqa: F401  先加入项目 libs/，再导入 NumPy / PyTorch

import numpy as np
import torch
import torch.nn as nn

from data import load_npz, split_train_validation
from model import HandwritingCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="同步运行一个可逐行阅读的 MNIST 训练循环")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（默认 1）")
    parser.add_argument("--max-samples", type=int, default=512,
                        help="训练集和验证集各自最多使用多少张图（默认 512）")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="计算设备（默认 cpu）")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "mnist.npz",
                        help="MNIST npz 文件路径（默认 data/mnist.npz）")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_samples < 1:
        parser.error("--epochs 和 --max-samples 都必须大于 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("未检测到 CUDA；请使用 --device cpu")

    # 固定种子：模型初始权重和每轮打乱数据的顺序可重现。
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device(args.device)
    arrays = load_npz(args.data)
    x_train, y_train, x_val, y_val, sizes = split_train_validation(
        arrays["x_train"], arrays["y_train"]
    )
    # 先切分，再限量：验证图始终不参与训练；测试集完全留到最后。
    x_train, y_train = x_train[:args.max_samples], y_train[:args.max_samples]
    x_val, y_val = x_val[:args.max_samples], y_val[:args.max_samples]
    if not len(x_val):
        parser.error("验证集为空；请提供更多训练图片")
    print(f"原始划分：训练 {sizes['train']} 张，验证 {sizes['validation']} 张；"
          f"本次使用训练 {len(x_train)} 张、验证 {len(x_val)} 张。")
    print(f"设备：{device}；测试集 {len(arrays['x_test'])} 张未用于训练/验证。")

    model = HandwritingCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 输入是 10 个原始分数（logits），无需先 softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 64

    for epoch in range(1, args.epochs + 1):
        model.train()  # BatchNorm 使用当前批次统计量；Dropout 生效
        total_loss, correct = 0.0, 0
        order = rng.permutation(len(x_train))  # 每轮打乱一次，切出不重叠的 batch
        for start in range(0, len(x_train), batch_size):
            indices = order[start:start + batch_size]
            xb = torch.from_numpy(x_train[indices]).unsqueeze(1).to(device)
            yb = torch.from_numpy(y_train[indices]).to(device)

            optimizer.zero_grad()      # 清除上一个 batch 的梯度
            logits = model(xb)         # 前向：图片 -> 十个类别的分数
            loss = criterion(logits, yb)
            loss.backward()           # 反向：求每个参数的梯度
            optimizer.step()          # 沿梯度反方向更新权重

            total_loss += loss.item() * len(yb)  # 按图片数加权，最后一批可能较小
            correct += (logits.argmax(dim=1) == yb).sum().item()

        model.eval()  # Dropout 关闭；BatchNorm 使用训练阶段累计的统计量
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():  # 验证时不求梯度，也不更新权重
            for start in range(0, len(x_val), batch_size):
                xb = torch.from_numpy(x_val[start:start + batch_size]).unsqueeze(1).to(device)
                yb = torch.from_numpy(y_val[start:start + batch_size]).to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()

        print(f"第 {epoch}/{args.epochs} 轮："
              f"训练 loss={total_loss / len(x_train):.4f}、"
              f"准确率={100 * correct / len(x_train):.2f}%；"
              f"验证 loss={val_loss / len(x_val):.4f}、"
              f"准确率={100 * val_correct / len(x_val):.2f}%")


if __name__ == "__main__":
    main()
