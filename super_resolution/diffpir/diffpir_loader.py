
import os
import numpy as np


def load_precomputed_diffpir(im_ms,image_filename,satname,diffpir_dir,crop_mode="resize",):
    
    if satname not in ("L5", "L7"):
        raise ValueError(
            f"Precomputed DiffPIR integration currently supports L5/L7 only, "
            f"but received {satname}."
        )

    diffpir_dir = os.path.abspath(os.fspath(diffpir_dir))
    if not os.path.isdir(diffpir_dir):
        raise NotADirectoryError(
            "DiffPIR directory not found: " + diffpir_dir
        )

    image_name = os.path.basename(os.fspath(image_filename))
    image_prefix = image_name[:19]
    diffpir_suffix = "_DiffPIR_x2_HWC.npy"

    matches = sorted([
        os.path.join(diffpir_dir, name)
        for name in os.listdir(diffpir_dir)
        if (
            name.startswith(image_prefix)
            and f"_{satname}_" in name
            and name.endswith(diffpir_suffix)
        )
    ])

    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one DiffPIR result for "
            f"{image_name}, sat={satname}, prefix={image_prefix}; "
            f"found {len(matches)}: {matches}"
        )

    diffpir_path = matches[0]
    im_diffpir = np.load(diffpir_path).astype(np.float32)

    # Allow CHW or HWC, standardise to HWC.
    if (
        im_diffpir.ndim == 3
        and im_diffpir.shape[0] == 5
        and im_diffpir.shape[-1] != 5
    ):
        im_diffpir = np.moveaxis(im_diffpir, 0, -1)

    if im_diffpir.ndim != 3 or im_diffpir.shape[2] != 5:
        raise ValueError(
            "Invalid DiffPIR shape: "
            f"{im_diffpir.shape} | file={diffpir_path}"
        )

    expected_h = im_ms.shape[0]
    expected_w = im_ms.shape[1]
    expected_shape = (expected_h, expected_w, 5)

    if im_diffpir.shape != expected_shape:
        dh = im_diffpir.shape[0] - expected_h
        dw = im_diffpir.shape[1] - expected_w

        # Preserve the behaviour of the tested DiffPIR-CoastSat pipeline:
        # only a very small (+/- 1 pixel) mismatch is auto-corrected.
        if abs(dh) <= 1 and abs(dw) <= 1:
            print(
                f"\nDIFFPIR ALIGN {satname} | "
                f"before={im_diffpir.shape} | "
                f"expected={expected_shape} | "
                f"mode={crop_mode}"
            )

            if crop_mode == "resize":
                from skimage.transform import resize as sk_resize

                resized = np.empty(expected_shape, dtype=np.float32)
                for band in range(5):
                    resized[:, :, band] = sk_resize(
                        im_diffpir[:, :, band],
                        (expected_h, expected_w),
                        order=1,
                        mode="edge",
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(np.float32)

                im_diffpir = resized

            elif crop_mode in ("drop_left", "drop_right"):
                if dh == 1:
                    im_diffpir = im_diffpir[:expected_h, :, :]
                elif dh == -1:
                    raise ValueError(
                        "DiffPIR is one row smaller than CoastSat; "
                        "crop mode cannot fix this."
                    )

                if dw == 1:
                    if crop_mode == "drop_left":
                        im_diffpir = im_diffpir[:, 1:, :]
                    else:
                        im_diffpir = im_diffpir[:, :expected_w, :]
                elif dw == -1:
                    raise ValueError(
                        "DiffPIR is one column smaller than CoastSat; "
                        "crop mode cannot fix this."
                    )
            else:
                raise ValueError(
                    "Invalid diffpir_crop_mode: "
                    f"{crop_mode}. Use 'resize', 'drop_left' or 'drop_right'."
                )

            print(
                f"DIFFPIR ALIGN DONE {satname} | "
                f"after={im_diffpir.shape}"
            )
        else:
            raise ValueError(
                "DiffPIR/CoastSat shape mismatch too large: "
                f"file={image_name}, DiffPIR={im_diffpir.shape}, "
                f"expected={expected_shape}, original im_ms={im_ms.shape}"
            )

    if im_diffpir.shape != expected_shape:
        raise ValueError(
            "DiffPIR alignment failed: "
            f"file={image_name}, got={im_diffpir.shape}, "
            f"expected={expected_shape}"
        )

    if not np.all(np.isfinite(im_diffpir)):
        raise ValueError(
            "DiffPIR result contains NaN or Inf: " + diffpir_path
        )

    return im_diffpir, diffpir_path
