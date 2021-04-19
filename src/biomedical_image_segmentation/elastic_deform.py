import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from typing import Tuple, Union, List


def custom_2d_elastic_deform(
    img: np.ndarray,
    mask: np.ndarray,
    ref_ratio: float = 3.0,
    alpha_affine: Tuple[float, float] = (0.01, 0.2),
    sigma: float = 10.0,
    alpha: float = 1.0,
    adjustment_pixel_range: Tuple[int, int] = None,
    adjusted_pixel: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform elastic transformation on input image and its mask.

    Note: Results of the function is non-reproducible.

    Reference: https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations

    Parameters
    ----------

    img: numpy array of shape (H, W)
        H: height of image
        W: width of image

    mask: numpy array of shape (H, W)
        H: height of image
        W: width of image

    alpha_affine: float
        Used to compute affine transformation matrix.

    sigma: float
        Used in computation of Gaussian filter.

    alpha: float
        Used in computation of Gaussian filter.
        Used to scale Gaussian filter.

    ref_ratio: float, default=3.
        Used to pick reference points in original image space

    Returns
    -------
    transformed img: numpy array of shape (H, W)
        H: height of image
        W: width of image

    transformed mask: numpy array of shape (H, W)
        H: height of image
        W: width of image
    """

    assert img.ndim == 2
    assert img.shape == mask.shape

    height, width = img.shape
    center_coords = np.float32([height, width]) // 2
    ref_point = min([height, width]) // ref_ratio

    # fix 3 points in original image space
    pts_src = np.float32(
        [
            center_coords - ref_point,
            center_coords + np.array([1.0, -1.0]) * ref_point,
            center_coords + ref_point,
        ]
    )

    # get 3 points in affine transformed image space
    alpha_affine = min([height, width]) * np.random.uniform(*alpha_affine)
    alpha_tranform = np.random.uniform(
        low=-alpha_affine, high=alpha_affine, size=pts_src.shape
    ).astype(np.float32)
    pts_dst = pts_src + alpha_tranform

    # get affine transformation matrix to transform (x,y) points
    # from original image space to affine transformed image space
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    M = cv2.getAffineTransform(pts_src, pts_dst)

    # perform affine transformation
    # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
    img = cv2.warpAffine(
        src=img, M=M, dsize=(width, height), borderMode=cv2.BORDER_REFLECT_101
    )

    mask = cv2.warpAffine(
        src=mask,
        M=M,
        dsize=(width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.ones(1))

    dx = (
        gaussian_filter(
            (np.random.rand(*(height, width, 1)) * 2 - 1),
            sigma=sigma,
            order=0,
            mode="reflect",
        )
        * alpha
    )

    dy = (
        gaussian_filter(
            (np.random.rand(*(height, width, 1)) * 2 - 1),
            sigma=sigma,
            order=0,
            mode="reflect",
        )
        * alpha
    )
    dz = np.zeros_like(dx)

    coordinates = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z + dz, (-1, 1)),
    )

    img = (
        map_coordinates(
            np.expand_dims(img, axis=2),
            coordinates=coordinates,
            order=3,
            mode="reflect",
        )
        .squeeze(axis=1)
        .reshape(width, height)
        .T
    )

    mask = (
        map_coordinates(
            np.expand_dims(mask, axis=2),
            coordinates=coordinates,
            order=3,
            mode="reflect",
        )
        .squeeze(axis=1)
        .reshape(width, height)
        .T
    )

    if adjustment_pixel_range is not None and adjusted_pixel is not None:
        conditions = [
            mask <= 0 + adjustment_pixel_range[0],
            mask >= 255 - adjustment_pixel_range[-1],
        ]
        choices = [0, 255]
        mask = np.select(conditions, choices, adjusted_pixel)

    return img, mask


