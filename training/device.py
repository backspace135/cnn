"""训练器的设备检测与选择。"""

import torch


def hardware_info():
    """Describe the PyTorch runtime; a CPU build cannot use a CUDA GPU."""
    available = torch.cuda.is_available()
    reason = ""
    if not available:
        reason = ("当前为 CPU 版 PyTorch，请运行 install_cuda.bat 后重启服务。"
                  if torch.version.cuda is None else
                  "CUDA 不可用，请检查 NVIDIA 显卡及驱动，或选择 CPU 模式。")
    return {
        "cuda_available": available,
        "cuda_reason": reason,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "torch_version": str(torch.__version__),
    }


def resolve_device(mode):
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError("训练模式必须为 auto、cpu 或 cuda。")
    if mode == "cuda" and not torch.cuda.is_available():
        raise ValueError("无法启用 CUDA：" + hardware_info()["cuda_reason"])
    return torch.device("cuda" if mode == "cuda" or
                        (mode == "auto" and torch.cuda.is_available()) else "cpu")
