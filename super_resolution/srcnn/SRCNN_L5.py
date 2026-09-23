from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN5Band(nn.Module):
    """SRCNN for Landsat 5 bandes."""

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
        return self.reconstruction(self.features(x))


_MODEL = None
_DEVICE = None


def load_srcnn_l5(
    checkpoint_path=None,
    device=None,
):
    """
    Load checkpoint SRCNN L5.
    """

    global _MODEL, _DEVICE

    if _MODEL is not None:
        return _MODEL, _DEVICE

    if checkpoint_path is None:
        checkpoint_path = (
            Path(__file__).resolve().parent
            / "models"
            / "best_srcnn_l5.pth"
        )
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Donot find checkpoint SRCNN L5: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = SRCNN5Band().to(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "none")
    else:
        state_dict = checkpoint
        epoch = "none"

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _DEVICE = device

    print(
        f"SRCNN L5 loaded | "
        f"device={device} | epoch={epoch}"
    )

    return _MODEL, _DEVICE


@torch.inference_mode()
def enhance_l5(
    im_ms,
    checkpoint_path=None,
    reflect_pad=6,
):

    im_ms = np.asarray(im_ms)

    if im_ms.ndim != 3:
        raise ValueError(
            f"im_ms must be  H x W x 5, "
            f"but shape={im_ms.shape}"
        )

    if im_ms.shape[2] != 5:
        raise ValueError(
            f"SRCNN L5 require 5 bandes, "
            f"but shape={im_ms.shape}"
        )

    original = im_ms.astype(np.float32, copy=True)

    invalid_mask = ~np.all(np.isfinite(original), axis=2)

    clean = np.nan_to_num(
        original,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    tensor = (
        torch.from_numpy(clean)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    model, device = load_srcnn_l5(
        checkpoint_path=checkpoint_path
    )

    tensor = tensor.to(
        device=device,
        dtype=torch.float32,
    )

    if reflect_pad > 0:
        tensor = F.pad(
            tensor,
            (
                reflect_pad,
                reflect_pad,
                reflect_pad,
                reflect_pad,
            ),
            mode="reflect",
        )

    prediction = model(tensor)

    if reflect_pad > 0:
        prediction = prediction[
            ...,
            reflect_pad:-reflect_pad,
            reflect_pad:-reflect_pad,
        ]

    # BCHW -> HWC
    output = (
        prediction
        .squeeze(0)
        .permute(1, 2, 0)
        .float()
        .cpu()
        .numpy()
    )

    output[invalid_mask] = original[invalid_mask]

    if output.shape != original.shape:
        raise RuntimeError(
            f"SRCNN changes size: "
            f"{original.shape} -> {output.shape}"
        )

    return output.astype(np.float32, copy=False)
