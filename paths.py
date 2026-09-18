# -*- coding: utf-8 -*-
"""把下载到项目 libs/ 文件夹的第三方库注入到 sys.path。

这样项目自包含：不必把 torch 等装进全局 site-packages。
被 train.py / server.py 在导入 torch 之前 import。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIBS = ROOT / "libs"
CUDA_LIBS = ROOT / "libs-cuda"

if LIBS.is_dir() or CUDA_LIBS.is_dir():
    # CUDA builds also support CPU; prefer the optional CUDA installation.
    # Do not shadow a working CPU installation with an incomplete download.
    search_paths = [LIBS]
    if (CUDA_LIBS / "torch" / "__init__.py").is_file():
        search_paths.append(CUDA_LIBS)
    search_paths.append(ROOT)
    for p in search_paths:
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)
    # 模型权重、torch 缓存放到项目目录下，保持自包含
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch_cache"))
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
