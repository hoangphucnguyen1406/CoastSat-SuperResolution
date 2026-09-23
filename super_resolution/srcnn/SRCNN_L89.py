"""
Inference SRCNN 5-band cho Landsat 8/9.

Mô hình được huấn luyện bằng bài toán tự giám sát:
60 m giả lập -> bilinear 30 m -> SRCNN -> target 30 m.

Trong CoastSat, mô hình được áp dụng thử nghiệm lên ảnh multispectral
đã bilinear từ 30 m lên 15 m.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN5Band(nn.Module):
    """SRCNN cổ điển cho ảnh Landsat 5 band."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=64,
                kernel_size=9,
                padding=4,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )

        self.reconstruction = nn.Conv2d(
            in_channels=32,
            out_channels=5,
            kernel_size=5,
            padding=2,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.reconstruction(x)
        return x


_MODEL = None
_DEVICE = None


def _get_model():
    """Nạp model một lần rồi tái sử dụng cho các ảnh tiếp theo."""
    global _MODEL, _DEVICE

    if _MODEL is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "best_srcnn_l89.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model SRCNN L8/L9: {model_path}"
        )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=_DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=_DEVICE,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    _MODEL = SRCNN5Band().to(_DEVICE)
    _MODEL.load_state_dict(state_dict, strict=True)
    _MODEL.eval()

    print(
        f"SRCNN L8/L9 loaded | "
        f"device={_DEVICE} | "
        f"model={model_path.name}"
    )

    return _MODEL, _DEVICE


@torch.inference_mode()
def enhance_l89(im_ms, reflect_pad=6):
    """
    Áp dụng SRCNN lên ảnh multispectral L8/L9.

    Parameters
    ----------
    im_ms : numpy.ndarray
        Ảnh CoastSat có dạng (H, W, 5).
    reflect_pad : int
        Số pixel reflect padding để tránh viền đen.

    Returns
    -------
    numpy.ndarray
        Ảnh SRCNN có cùng shape và dtype với ảnh đầu vào.
    """
    image = np.asarray(im_ms)

    if image.ndim != 3 or image.shape[2] != 5:
        raise ValueError(
            "SRCNN L8/L9 yêu cầu ảnh dạng (H, W, 5), "
            f"nhưng nhận được {image.shape}"
        )

    original_dtype = image.dtype
    image_float = image.astype(np.float32, copy=True)

    # Ghi nhớ pixel không hợp lệ
    invalid_mask = ~np.isfinite(image_float).all(axis=2)

    # Không truyền NaN/Inf vào mạng
    image_float[~np.isfinite(image_float)] = 0.0

    # HWC -> NCHW
    tensor = (
        torch.from_numpy(
            np.moveaxis(image_float, -1, 0)
        )
        .unsqueeze(0)
    )

    model, device = _get_model()
    tensor = tensor.to(device)

    height, width = image_float.shape[:2]

    # Reflect padding yêu cầu padding nhỏ hơn kích thước ảnh
    pad = min(
        int(reflect_pad),
        height - 1,
        width - 1,
    )

    if pad > 0:
        padded = F.pad(
            tensor,
            (pad, pad, pad, pad),
            mode="reflect",
        )

        output = model(padded)

        output = output[
            :,
            :,
            pad:-pad,
            pad:-pad,
        ]
    else:
        output = model(tensor)

    # NCHW -> HWC
    enhanced = (
        output.squeeze(0)
        .cpu()
        .numpy()
    )
    enhanced = np.moveaxis(enhanced, 0, -1)

    if enhanced.shape != image_float.shape:
        raise RuntimeError(
            f"Sai kích thước SRCNN: "
            f"{image_float.shape} -> {enhanced.shape}"
        )

    # Giữ nguyên vùng nodata/không hợp lệ
    enhanced[invalid_mask] = image_float[invalid_mask]

    return enhanced.astype(original_dtype, copy=False)