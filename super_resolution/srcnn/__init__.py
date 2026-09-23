from .SRCNN_L5 import enhance_l5
from .SRCNN_L7 import enhance_l7
from .SRCNN_L89 import enhance_l89


def enhance_srcnn(im_ms, satname, reflect_pad=6):
    """
    Apply the satellite-specific SRCNN model.

    Parameters
    ----------
    im_ms : numpy.ndarray
        Multispectral image with shape (H, W, 5).

    satname : str
        Satellite mission: L5, L7, L8 or L9.

    reflect_pad : int
        Reflect padding used during SRCNN inference.
    """

    if satname == "L5":
        return enhance_l5(
            im_ms,
            reflect_pad=reflect_pad,
        )

    elif satname == "L7":
        return enhance_l7(
            im_ms,
            reflect_pad=reflect_pad,
        )

    elif satname in ["L8", "L9"]:
        return enhance_l89(
            im_ms,
            reflect_pad=reflect_pad,
        )

    else:
        return im_ms
