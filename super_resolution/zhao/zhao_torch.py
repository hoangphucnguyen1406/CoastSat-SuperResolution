"""
PyTorch implementation of Ningning Zhao et al. Algorithm 1.

Image-domain L2 regularization:

    min_x 0.5 ||y - S H x||_2^2
          + tau ||x - x_bar||_2^2

where:
    y     : original LR Landsat image
    H     : Gaussian blur operator
    S     : decimation operator
    x_bar : bicubic interpolation of y
    tau   : regularization parameter

No training, optimizer or neural-network weights are used.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal


# ============================================================
# Gaussian PSF and FFT operators
# ============================================================

def gaussian_kernel(
    size: int,
    variance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create a normalized 2D Gaussian kernel.

    The paper uses a 9x9 Gaussian kernel with variance = 3
    in its default experiment.
    """

    if size <= 0 or size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd integer."
        )

    if variance <= 0:
        raise ValueError(
            "kernel_variance must be positive."
        )

    coordinates = torch.arange(
        size,
        device=device,
        dtype=dtype,
    )

    coordinates = (
        coordinates - (size - 1) / 2
    )

    yy, xx = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    kernel = torch.exp(
        -(xx.square() + yy.square())
        / (2.0 * variance)
    )

    kernel = kernel / kernel.sum()

    return kernel


def psf_to_otf(
    psf: torch.Tensor,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Convert a spatial PSF to an optical transfer function.

    Circular boundary conditions are used, as assumed by
    Algorithm 1.
    """

    output_height, output_width = output_shape
    kernel_height, kernel_width = psf.shape

    if kernel_height > output_height:
        raise ValueError(
            "PSF height is larger than the HR image."
        )

    if kernel_width > output_width:
        raise ValueError(
            "PSF width is larger than the HR image."
        )

    padded_psf = torch.zeros(
        (output_height, output_width),
        dtype=psf.dtype,
        device=psf.device,
    )

    padded_psf[
        :kernel_height,
        :kernel_width,
    ] = psf

    padded_psf = torch.roll(
        padded_psf,
        shifts=(
            -(kernel_height // 2),
            -(kernel_width // 2),
        ),
        dims=(0, 1),
    )

    return torch.fft.fft2(padded_psf)


def apply_h(
    image: torch.Tensor,
    otf: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the blur operator H.
    """

    return torch.fft.ifft2(
        torch.fft.fft2(image, dim=(-2, -1))
        * otf
    ).real


def apply_ht(
    image: torch.Tensor,
    otf: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the adjoint blur operator H^T.
    """

    return torch.fft.ifft2(
        torch.fft.fft2(image, dim=(-2, -1))
        * otf.conj()
    ).real


# ============================================================
# Decimation operators
# ============================================================

def apply_s(
    image: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    """
    Apply decimation operator S.

    One pixel is retained every 'scale' pixels along each
    spatial direction.
    """

    return image[
        ...,
        ::scale,
        ::scale,
    ]


def apply_st(
    low_resolution: torch.Tensor,
    scale: int,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Apply S^T by inserting zeros on the HR grid.

    This is not an interpolation.
    """

    output = torch.zeros(
        (
            *low_resolution.shape[:-2],
            output_shape[0],
            output_shape[1],
        ),
        dtype=low_resolution.dtype,
        device=low_resolution.device,
    )

    output[
        ...,
        ::scale,
        ::scale,
    ] = low_resolution

    return output


# ============================================================
# Algorithm 1
# ============================================================

def zhao_algorithm1(
    y: torch.Tensor,
    scale: int,
    otf: torch.Tensor,
    tau: float,
    x_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Algorithm 1 for image-domain L2 regularization.

    Parameters
    ----------
    y
        LR observation with shape [B, C, H, W].
    scale
        Super-resolution scale factor.
    otf
        FFT of the Gaussian blur kernel on the HR grid.
    tau
        Regularization parameter.
    x_bar
        Bicubic prior with shape [B, C, scale*H, scale*W].

    Returns
    -------
    x_hat
        Zhao SR result with the same shape as x_bar.
    """

    if tau <= 0:
        raise ValueError(
            "tau must be strictly positive."
        )

    if y.ndim != 4:
        raise ValueError(
            f"Expected y in BCHW format, got {y.shape}."
        )

    if x_bar.ndim != 4:
        raise ValueError(
            "x_bar must be in BCHW format."
        )

    output_shape = x_bar.shape[-2:]

    expected_shape = (
        y.shape[-2] * scale,
        y.shape[-1] * scale,
    )

    if output_shape != expected_shape:
        raise ValueError(
            f"Incorrect x_bar shape: {output_shape}; "
            f"expected {expected_shape}."
        )

    # --------------------------------------------------------
    # r = H^T S^T y + 2 tau x_bar
    # --------------------------------------------------------

    st_y = apply_st(
        low_resolution=y,
        scale=scale,
        output_shape=output_shape,
    )

    r = (
        apply_ht(st_y, otf)
        + 2.0 * tau * x_bar
    )

    # --------------------------------------------------------
    # q = S H r
    # --------------------------------------------------------

    q = apply_s(
        apply_h(r, otf),
        scale,
    )

    # --------------------------------------------------------
    # Impulse response of A_lr = S H H^T S^T
    # --------------------------------------------------------

    impulse_lr = torch.zeros(
        (
            1,
            1,
            y.shape[-2],
            y.shape[-1],
        ),
        dtype=y.dtype,
        device=y.device,
    )

    impulse_lr[..., 0, 0] = 1.0

    impulse_hr = apply_st(
        low_resolution=impulse_lr,
        scale=scale,
        output_shape=output_shape,
    )

    a_lr = apply_s(
        apply_h(
            apply_ht(
                impulse_hr,
                otf,
            ),
            otf,
        ),
        scale,
    )

    # --------------------------------------------------------
    # z = (2 tau I + S H H^T S^T)^(-1) q
    # --------------------------------------------------------

    denominator = (
        2.0 * tau
        + torch.fft.fft2(
            a_lr,
            dim=(-2, -1),
        )
    )

    q_fft = torch.fft.fft2(
        q,
        dim=(-2, -1),
    )

    z = torch.fft.ifft2(
        q_fft / denominator,
        dim=(-2, -1),
    ).real

    # --------------------------------------------------------
    # x_hat = (r - H^T S^T z) / (2 tau)
    # --------------------------------------------------------

    st_z = apply_st(
        low_resolution=z,
        scale=scale,
        output_shape=output_shape,
    )

    correction = apply_ht(
        st_z,
        otf,
    )

    x_hat = (
        r - correction
    ) / (2.0 * tau)

    return x_hat


# ============================================================
# Zhao SR for a PyTorch tensor
# ============================================================

@torch.no_grad()
def run_zhao_tensor(
    y: torch.Tensor,
    scale: int = 2,
    tau: float = 0.01,
    kernel_size: int = 9,
    kernel_variance: float = 3.0,
):
    """
    Create the bicubic prior and apply Algorithm 1.

    Parameters
    ----------
    y
        Original LR image, shape [B, C, H, W].
    scale
        Spatial SR scale.
    tau
        Image-domain L2 regularization parameter.
    kernel_size
        Gaussian PSF size.
    kernel_variance
        Gaussian PSF variance, not standard deviation.
    """

    if y.ndim != 4:
        raise ValueError(
            f"Expected BCHW input, got {y.shape}."
        )

    if scale < 2:
        raise ValueError(
            "scale must be at least 2."
        )

    output_shape = (
        y.shape[-2] * scale,
        y.shape[-1] * scale,
    )

    # Bicubic prior x_bar from Algorithm 1
    x_bar = F.interpolate(
        y,
        size=output_shape,
        mode="bicubic",
        align_corners=False,
    )

    psf = gaussian_kernel(
        size=kernel_size,
        variance=kernel_variance,
        device=y.device,
        dtype=y.dtype,
    )

    otf = psf_to_otf(
        psf=psf,
        output_shape=output_shape,
    )

    x_hat = zhao_algorithm1(
        y=y,
        scale=scale,
        otf=otf,
        tau=tau,
        x_bar=x_bar,
    )

    if not torch.isfinite(x_hat).all():
        raise ValueError(
            "Zhao output contains NaN or Inf."
        )

    return x_hat, x_bar, otf


# ============================================================
# Diagnostic: forward-model consistency
# ============================================================

@torch.no_grad()
def calculate_forward_errors(
    y: torch.Tensor,
    x_hat: torch.Tensor,
    x_bar: torch.Tensor,
    otf: torch.Tensor,
    scale: int,
):
    """
    Compare Zhao and bicubic through the observation model.
    """

    y_from_zhao = apply_s(
        apply_h(x_hat, otf),
        scale,
    )

    y_from_bicubic = apply_s(
        apply_h(x_bar, otf),
        scale,
    )

    zhao_error = torch.mean(
        (y_from_zhao - y).square()
    ).item()

    bicubic_error = torch.mean(
        (y_from_bicubic - y).square()
    ).item()

    return zhao_error, bicubic_error


# ============================================================
# GeoTIFF input/output
# ============================================================

@torch.no_grad()
def zhao_superresolve_tif(
    fn_in,
    fn_out,
    scale: int = 2,
    tau: float = 0.01,
    kernel_size: int = 9,
    kernel_variance: float = 3.0,
    device: str | None = None,
):
    """
    Read an original Landsat GeoTIFF, apply Zhao Algorithm 1,
    and write a georeferenced SR GeoTIFF.
    """

    fn_in = str(Path(fn_in))
    fn_out = str(Path(fn_out))

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    torch_device = torch.device(device)

    source = gdal.Open(
        fn_in,
        gdal.GA_ReadOnly,
    )

    if source is None:
        raise FileNotFoundError(
            f"Cannot open input GeoTIFF: {fn_in}"
        )

    image = source.ReadAsArray()

    if image.ndim == 2:
        image = image[None, ...]

    if image.ndim != 3:
        raise ValueError(
            f"Expected CHW GeoTIFF, got {image.shape}."
        )

    image = image.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(image).all():
        raise ValueError(
            "Input image contains NaN or Inf."
        )

    y = torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0).to(torch_device)

    x_hat, x_bar, otf = run_zhao_tensor(
        y=y,
        scale=scale,
        tau=tau,
        kernel_size=kernel_size,
        kernel_variance=kernel_variance,
    )

    zhao_error, bicubic_error = (
        calculate_forward_errors(
            y=y,
            x_hat=x_hat,
            x_bar=x_bar,
            otf=otf,
            scale=scale,
        )
    )

    output = (
        x_hat
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if not np.isfinite(output).all():
        raise ValueError(
            "Output contains NaN or Inf."
        )

    band_count, output_height, output_width = (
        output.shape
    )

    expected_height = (
        source.RasterYSize * scale
    )

    expected_width = (
        source.RasterXSize * scale
    )

    if output_height != expected_height:
        raise ValueError(
            f"Incorrect output height: {output_height}; "
            f"expected {expected_height}."
        )

    if output_width != expected_width:
        raise ValueError(
            f"Incorrect output width: {output_width}; "
            f"expected {expected_width}."
        )

    output_directory = Path(fn_out).parent
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    driver = gdal.GetDriverByName("GTiff")

    destination = driver.Create(
        fn_out,
        output_width,
        output_height,
        band_count,
        gdal.GDT_Float32,
        options=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
        ],
    )

    if destination is None:
        raise RuntimeError(
            f"Cannot create output GeoTIFF: {fn_out}"
        )

    input_georef = list(
        source.GetGeoTransform()
    )

    output_georef = input_georef.copy()

    # Divide the two pixel vectors by the scale factor.
    output_georef[1] /= scale
    output_georef[2] /= scale
    output_georef[4] /= scale
    output_georef[5] /= scale

    destination.SetGeoTransform(
        output_georef
    )

    destination.SetProjection(
        source.GetProjection()
    )

    for band_index in range(band_count):
        output_band = destination.GetRasterBand(
            band_index + 1
        )

        output_band.WriteArray(
            output[band_index]
        )

        source_band = source.GetRasterBand(
            band_index + 1
        )

        description = (
            source_band.GetDescription()
        )

        if description:
            output_band.SetDescription(
                description
            )

        output_band.FlushCache()

    destination.FlushCache()

    destination = None
    source = None

    return {
        "device": str(torch_device),
        "tau": float(tau),
        "input_shape": tuple(image.shape),
        "output_shape": tuple(output.shape),
        "input_min": float(image.min()),
        "input_max": float(image.max()),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "zhao_forward_mse": float(zhao_error),
        "bicubic_forward_mse": float(
            bicubic_error
        ),
    }
