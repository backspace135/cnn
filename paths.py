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

if LIBS.is_dir():
    for p in [LIBS, ROOT]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    # 模型权重、torch 缓存放到项目目录下，保持自包含
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch_cache"))
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
