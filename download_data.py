# -*- coding: utf-8 -*-
"""下载 MNIST 原始 IDX/Gzip，检查完整性并生成 data/mnist.npz。

首次运行需要网络；生成的 npz 留在本地，后续训练和网页不再访问网络。
"""
import gzip
import os
import struct
import time
import urllib.request
from pathlib import Path

import paths  # noqa: F401  先把项目 libs/ 加入 sys.path
import numpy as np

BASE_URLS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
FILES = (
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
)
DATA_DIR = Path(__file__).resolve().parent / "data"


def read_idx_gz(path: Path):
    """IDX 大端头包含 magic、样本数和图片尺寸；长度错误必须立即报错。"""
    try:
        with gzip.open(path, "rb") as handle:
            contents = handle.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"无法解压 {path}，请删除损坏文件后重试。") from exc
    if len(contents) < 8:
        raise ValueError(f"{path} 的 IDX 头不完整。")
    magic, count = struct.unpack_from(">II", contents)
    if not count:
        raise ValueError(f"{path} 样本数为零。")
    if magic == 2051:
        if len(contents) < 16:
            raise ValueError(f"{path} 的图片头不完整。")
        rows, cols = struct.unpack_from(">II", contents, 8)
        if (rows, cols) != (28, 28):
            raise ValueError(f"{path} 图片尺寸必须为 28×28。")
        expected = 16 + count * rows * cols
        shape, offset = (count, rows, cols), 16
    elif magic == 2049:
        expected = 8 + count
        shape, offset = (count,), 8
    else:
        raise ValueError(f"{path} 的 IDX 魔数无效：{magic}。")
    if len(contents) != expected:
        raise ValueError(f"{path} 长度不符：声明 {count} 条，实际 {len(contents)} 字节。")
    result = np.frombuffer(contents, dtype=np.uint8, offset=offset).reshape(shape).copy()
    if magic == 2049 and np.any(result > 9):
        raise ValueError(f"{path} 含有不在 0–9 内的标签。")
    return result


def download_file(name: str) -> Path:
    """下载到临时路径，通过 gzip/IDX 校验后原子替换；缓存也会重新校验。"""
    if name not in FILES:
        raise ValueError("未知 MNIST 文件名。")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    if target.exists():
        try:
            read_idx_gz(target)
            print(f"[缓存] {name} 已通过校验")
            return target
        except ValueError:
            print(f"[损坏] {name} 将重新下载")
    temp = target.with_name(target.name + ".part")
    errors = []
    for url_base in BASE_URLS:
        for attempt in range(1, 4):
            url = url_base + name
            try:
                print(f"[下载] {name} ({url}, 第 {attempt} 次)")
                with urllib.request.urlopen(url, timeout=30) as source, open(temp, "wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                read_idx_gz(temp)
                os.replace(temp, target)
                return target
            except (OSError, ValueError, EOFError) as exc:
                errors.append(f"{url}: {exc}")
                temp.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"下载 {name} 失败，请检查网络或手动放入 data/。最后错误：{errors[-1]}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_img, train_lbl, test_img, test_lbl = (
        read_idx_gz(download_file(name)) for name in FILES)
    if len(train_img) != len(train_lbl) or len(test_img) != len(test_lbl):
        raise ValueError("MNIST 图片数与标签数不一致。")
    arrays = {
        "x_train": train_img.astype(np.float32) / 255,
        "y_train": train_lbl.astype(np.int64),
        "x_test": test_img.astype(np.float32) / 255,
        "y_test": test_lbl.astype(np.int64),
    }
    target = DATA_DIR / "mnist.npz"
    temp = DATA_DIR / "mnist.npz.part"
    try:
        with open(temp, "wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[完成] 训练 {len(train_img):,}、测试 {len(test_img):,} 张 → {target}")


if __name__ == "__main__":
    main()
