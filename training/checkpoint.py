"""本地模型存档的原子写入与格式检查；训练状态仍由 Trainer 管理。"""

import os

import torch

MODEL_VERSION = 1


def save_checkpoint(model_file, model, config, epoch, seed):
    """仅写入已有格式的模型权重与基础配置，不保存优化器。"""
    model_file.parent.mkdir(parents=True, exist_ok=True)
    temp = model_file.with_suffix(".part")
    try:
        torch.save({"version": MODEL_VERSION, "state_dict": model.state_dict(),
                    "config": config, "epoch": epoch, "seed": seed,
                    "input": "28x28 grayscale [0,1], white digit on black"}, temp)
        os.replace(temp, model_file)
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint(model_file):
    """在替换当前模型之前先检查路径、版本与必要字段。"""
    if not model_file.is_file():
        raise FileNotFoundError("没有已保存模型，请先完成训练并保存。")
    payload = torch.load(model_file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != MODEL_VERSION:
        raise ValueError("存档版本不兼容。")
    config = payload.get("config")
    required = ("epochs", "lr", "batch_size", "dropout")
    if not isinstance(config, dict) or "state_dict" not in payload or any(k not in config for k in required):
        raise ValueError("存档缺少完整的模型配置或权重。")
    return payload, {key: config[key] for key in required}
