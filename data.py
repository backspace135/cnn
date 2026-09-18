# -*- coding: utf-8 -*-
"""MNIST 数据读取、校验和 train/validation/test 划分。

这个模块故意只使用 NumPy，不依赖 PyTorch 或 Flask，方便学习者单独阅读，
也方便测试使用很小的临时数组而不下载完整 MNIST。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import paths  # noqa: F401  允许仅在项目 libs/ 安装 NumPy
import numpy as np


IMAGE_SHAPE = (28, 28)
NUM_CLASSES = 10


def validate_arrays(images: np.ndarray, labels: np.ndarray, name: str = "数据集") -> None:
    """检查一组图片和标签是否满足训练约定。"""
    if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError(f"{name} 必须是 NumPy 数组。")
    if images.ndim != 3 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"{name} 图片形状必须是 (N, 28, 28)，实际为 {images.shape}。")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError(f"{name} 图片和标签数量不一致。")
    if len(images) == 0:
        raise ValueError(f"{name} 不能为空。")
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise ValueError(f"{name} 图片必须只包含有限数字。")
    if float(images.min()) < 0 or float(images.max()) > 1:
        raise ValueError(f"{name} 图片像素必须在 [0, 1] 范围内。")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} 标签必须是整数。")
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError(f"{name} 标签必须在 0 到 9 之间。")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """安全读取并校验项目生成的 mnist.npz。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}。请先运行 download_data.py。")
    required = {"x_train", "y_train", "x_test", "y_test"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"数据文件缺少字段：{', '.join(sorted(missing))}。")
        arrays = {key: np.asarray(archive[key]) for key in required}
    validate_arrays(arrays["x_train"], arrays["y_train"], "训练集")
    validate_arrays(arrays["x_test"], arrays["y_test"], "测试集")
    for key in ("x_train", "x_test"):
        arrays[key] = arrays[key].astype(np.float32, copy=False)
    for key in ("y_train", "y_test"):
        arrays[key] = arrays[key].astype(np.int64, copy=False)
    return arrays


def split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """按类别、固定随机种子切出验证集，返回 train/validation 四个数组。

    按类别切分能避免某个小数据集的验证集恰好没有某些数字。对于每一类，
    至少保留一个训练样本；真实 MNIST 的样本量远大于这个边界。
    """
    validate_arrays(images, labels, "训练数据")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间。")
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == label)
        if not len(indices):
            continue
        indices = indices.copy()
        rng.shuffle(indices)
        val_count = int(round(len(indices) * validation_fraction))
        if len(indices) > 1:
            val_count = max(1, min(val_count, len(indices) - 1))
        else:
            val_count = 0
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    val_idx = np.asarray(val_indices, dtype=np.int64)
    result = (images[train_idx], labels[train_idx], images[val_idx], labels[val_idx])
    sizes = {"train": len(train_idx), "validation": len(val_idx)}
    return (*result, sizes)


def describe(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """返回可直接放入 API state 的数据量摘要。"""
    return {
        "train": int(len(arrays["x_train"])),
        "test": int(len(arrays["x_test"])),
    }
