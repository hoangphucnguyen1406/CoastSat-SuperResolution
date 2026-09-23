"""
Inference SRCNN 5-band dành riêng cho Landsat 7.

Mô hình được huấn luyện bằng bài toán tự giám sát:
60 m giả lập -> bilinear 30 m -> SRCNN -> target L7 gốc 30 m.

Trong CoastSat, mô hình được áp dụng thử nghiệm lên ảnh multispectral L7
đã được nội suy từ 30 m lên 15 m.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN5Band(nn.Module):
    """SRCNN cổ điển dành cho ảnh Landsat 7 gồm 5 band."""

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
    """Nạp model L7 một lần rồi tái sử dụng cho các ảnh tiếp theo."""
    global _MODEL, _DEVICE

    if _MODEL is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "best_srcnn_l7.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model SRCNN L7: {model_path}"
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

    model = SRCNN5Band().to(_DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model

    checkpoint_epoch = (
        checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else None
    )

    epoch_text = (
        f" | epoch={checkpoint_epoch}"
        if checkpoint_epoch is not None
        else ""
    )

    print(
        f"SRCNN L7 loaded | "
        f"device={_DEVICE} | "
        f"model={model_path.name}"
        f"{epoch_text}"
    )

    return _MODEL, _DEVICE


@torch.inference_mode()
def enhance_l7(im_ms, reflect_pad=6):
    """
    Áp dụng SRCNN dành riêng cho ảnh multispectral Landsat 7.

    Parameters
    ----------
    im_ms : numpy.ndarray
        Ảnh CoastSat có dạng (H, W, 5).
    reflect_pad : int, default=6
        Số pixel reflect padding dùng để giảm artefact ở biên ảnh.

    Returns
    -------
    numpy.ndarray
        Ảnh SRCNN có cùng shape và dtype với ảnh đầu vào.
    """
    image = np.asarray(im_ms)

    if image.ndim != 3 or image.shape[2] != 5:
        raise ValueError(
            "SRCNN L7 yêu cầu ảnh dạng (H, W, 5), "
            f"nhưng nhận được {image.shape}"
        )

    original_dtype = image.dtype
    image_float = image.astype(np.float32, copy=True)

    # Ghi nhớ pixel chứa NaN hoặc Inf
    invalid_mask = ~np.isfinite(image_float).all(axis=2)

    # Không đưa NaN/Inf vào mạng
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

    # Reflect padding phải nhỏ hơn kích thước ảnh
    pad = min(
        max(int(reflect_pad), 0),
        height - 1,
        width - 1,
    )

    if pad > 0:
        padded_tensor = F.pad(
            tensor,
            (pad, pad, pad, pad),
            mode="reflect",
        )

        padded_output = model(padded_tensor)

        output = padded_output[
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
            "Sai kích thước đầu ra SRCNN L7: "
            f"{image_float.shape} -> {enhanced.shape}"
        )

    # Phục hồi giá trị gốc tại vùng nodata
    enhanced[invalid_mask] = image_float[invalid_mask]

    return enhanced.astype(original_dtype, copy=False)