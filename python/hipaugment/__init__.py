"""hipAugment: batched GPU image preprocessing on PyTorch tensors.

Every function takes tensors on the GPU and returns a new tensor on the GPU.

    src     uint8  [N, H, W, 3]   images padded to the largest H and W in the batch
    roi     int32  [N, 4]         x, y, w, h of the valid region of each image
    mirror  uint8  [N]            1 flips that image in x

Kernels are queued on the default stream, the one PyTorch uses, and return before
they finish, exactly like PyTorch's own GPU operations.
"""
import torch

from . import _core

IMAGENET_MEAN = (123.675, 116.28, 103.53)  # 0 to 255 units
IMAGENET_STD = (58.395, 57.12, 57.375)

__all__ = ["resize", "gaussian_blur", "crop_mirror_normalize", "fused_rcmn",
           "IMAGENET_MEAN", "IMAGENET_STD"]


def _prep(src, roi, mirror=None):
    src = src.contiguous()
    roi = roi.to(device=src.device, dtype=torch.int32).contiguous()
    if mirror is None:
        mirror = torch.zeros(src.shape[0], dtype=torch.uint8, device=src.device)
    mirror = mirror.to(device=src.device, dtype=torch.uint8).contiguous()
    return src, roi, mirror


def resize(src, roi, size):
    """Bilinear resize of each ROI to size = (height, width). Returns uint8 [N, h, w, 3]."""
    src, roi, _ = _prep(src, roi)
    out = torch.empty((src.shape[0], size[0], size[1], 3), dtype=torch.uint8, device=src.device)
    _core.resize(src, roi, out)
    return out


def gaussian_blur(src, roi, ksize=5, sigma=1.0):
    """Gaussian blur (ksize 3 or 5) inside each ROI. Pixels outside the ROI are copied."""
    src, roi, _ = _prep(src, roi)
    out = src.clone()
    _core.gaussian_blur(src, roi, out, ksize, sigma)
    return out


def crop_mirror_normalize(src, roi, size, mirror=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Crop each ROI (which must be `size` = (height, width)), mirror, normalize.

    Returns float32 [N, 3, h, w]."""
    src, roi, mirror = _prep(src, roi, mirror)
    out = torch.empty((src.shape[0], 3, size[0], size[1]), dtype=torch.float32, device=src.device)
    _core.crop_mirror_normalize(src, roi, mirror, mean, std, out)
    return out


def fused_rcmn(src, roi, size, mirror=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """ROI crop, bilinear resize to size = (height, width), mirror and normalize in one kernel.

    Returns float32 [N, 3, h, w]."""
    src, roi, mirror = _prep(src, roi, mirror)
    out = torch.empty((src.shape[0], 3, size[0], size[1]), dtype=torch.float32, device=src.device)
    _core.fused_rcmn(src, roi, mirror, mean, std, out)
    return out
