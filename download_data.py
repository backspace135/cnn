# -*- coding: utf-8 -*-
"""
下载 MNIST 手写数字数据集并解析为 numpy 数组，保存到 data/mnist.npz

MNIST: 6 万张训练图 + 1 万张测试图，每张 28x28 灰度图，标签 0-9
数据来源: PyTorch 官方镜像 ossci-datasets.s3.amazonaws.com
"""
import gzip
import os
import struct
import urllib.request
from pathlib import Path

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": 0,
    "train-labels-idx1-ubyte.gz": 0,
    "t10k-images-idx3-ubyte.gz": 0,
    "t10k-labels-idx1-ubyte.gz": 0,
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def download_file(name: str) -> Path:
    """下载 MNIST 原始 gz 文件到 data/ 目录。"""
    target = DATA_DIR / name
    if target.exists():
        print(f"[跳过] {name} 已存在")
        return target

    url = BASE_URL + name
    print(f"[下载] {name} <- {url}")
    tmp = target.parent / (name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(target)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"下载 {name} 失败: {e}")
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"[完成] {name} ({size_mb:.1f} MB)")
    return target


def read_idx_gz(path: Path):
    """解析 MNIST 的 idx 格式（gzip 压缩）。魔数 2049=标签, 2051=图片。"""
    import numpy as np
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:  # 图片: [n, 行, 列]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        elif magic == 2049:  # 标签: [n]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n)
        else:
            raise ValueError(f"未知魔数 magic={magic}")
    return data


def main():
    import numpy as np

    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("开始下载并解析 MNIST 手写数字数据集")
    print("=" * 50)

    paths = [download_file(name) for name in FILES]
    train_img = read_idx_gz(paths[0])
    train_lbl = read_idx_gz(paths[1])
    test_img = read_idx_gz(paths[2])
    test_lbl = read_idx_gz(paths[3])

    print(f"\n训练集: {train_img.shape} (图片), {train_lbl.shape} (标签)")
    print(f"测试集: {test_img.shape} (图片), {test_lbl.shape} (标签)")

    # 归一化到 [0, 1] 并转为 float32，便于训练
    out = {
        "x_train": train_img.astype(np.float32) / 255.0,
        "y_train": train_lbl.astype(np.int64),
        "x_test": test_img.astype(np.float32) / 255.0,
        "y_test": test_lbl.astype(np.int64),
    }
    dest = DATA_DIR / "mnist.npz"
    np.savez_compressed(dest, **out)
    print(f"\n[保存] 处理后的数据 -> {dest} "
          f"({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    print("完成。")


if __name__ == "__main__":
    main()
