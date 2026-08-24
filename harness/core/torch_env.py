"""Torch runtime guards applied before any model is constructed.

On some GPU/driver combinations cuDNN returns wrong convolution results for
these detection backbones without raising — the model emits degenerate
predictions that look like a weak model rather than a broken kernel. Disabling
the cuDNN backend falls back to kernels that match CPU output.

Set DLA_ALLOW_CUDNN=1 to keep cuDNN enabled.
"""
import os


def configure():
    info = {"cudnn_guard": False}
    try:
        import torch
    except Exception:
        return info
    info["torch"] = torch.__version__
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        return info
    cap = torch.cuda.get_device_capability()
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = f"sm_{cap[0]}{cap[1]}"
    info["arch_list"] = torch.cuda.get_arch_list()
    if os.environ.get("DLA_ALLOW_CUDNN", "0") == "1":
        info["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        return info
    torch.backends.cudnn.enabled = False
    info["cudnn_guard"] = True
    info["cudnn_enabled"] = False
    info["reason"] = "cudnn disabled: incorrect convolution results observed on some devices"
    return info
