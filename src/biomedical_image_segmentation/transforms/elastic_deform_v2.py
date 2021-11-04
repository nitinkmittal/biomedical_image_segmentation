from typing import Optional, Tuple, Union

import cv2
import numpy as np

from numpy.random._generator import Generator

from numbers import Number


def _validate_image(img: np.ndarray):
    """Validate type, shape and datatype for given image."""

    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Expected NumPy image array, got {type(img)} instead."
        )

    if img.ndim != 3:
        raise ValueError(
            f"Expected NumPy image array with 3 dimensions "
            "(height * width * number of channels), "
            f"got {img.ndim}-dimension/s instead."
        )

    if img.dtype != np.uint8:
        raise TypeError(
            "Expected datatype for values in NumPy array of type uint8, "
            f"got {img.dtype}-datatype instead."
        )


def _get_scale_shear_translate_T(
    scale_x: float, scale_y: float, shear_x: float, shear_y: float, dtype: type
) -> np.ndarray:
    """Generate transformation matrix to perform scaling, shearing and translation x and y axis.

    Args:
        scale_x: A float value used to scale along x

        scale_y: A float value used to scale along y

        shear_x: A float value used to shear along x

        shear_y: A float value used to shear along y

        dtype: value datatype for values in transformation matrix

    Returns:
        A NumPy transformation matrix
    """
    T = np.eye(2, dtype=dtype)
    T[0, 0] = scale_x
    T[1, 1] = scale_y
    T[0, 1] = shear_x
    T[1, 0] = shear_y
    return T


def _get_rotation_T(radian: float) -> np.ndarray:
    """Generate Tranformation matrix to perform rotation along x and y axis.

    Args:
        radian: A float value in radians used to perform rotation

    Returns:
        A NumPy rotation matrix.
    """
    cos, sin = np.cos(radian), np.sin(radian)
    return np.array([[cos, -sin], [sin, cos]])


class AffineTransform:
    def __init__(
        self,
        scale_xy: Union[float, Tuple[float, float]] = 1.0,
        shear_xy: Union[float, Tuple[float, float]] = 0.0,
        translate_xy: Union[float, Tuple[float, float]] = 0.0,
        angle: float = 0.0,
        rotate_first: bool = False,
        border_mode: int = 4,
        dtype: type = np.float32,
    ):
        """Affine transformations.

        Note: all tranformations are performed w.r.t to origin
            where origin is considered at i=0, j=0 for an array of shape (x, y)

        Args:
            scale_xy: A single float value or tuple of 2 float values, default=1.0
                scale along x and y axis

            shear_xy: A single float value or tuple of 2 float values, default=0.0
                shear along x and y axis

            translate_xy: A single float value or tuple of 2 float values, default=0.0
                translate along x and y axis

            angle: degree of rotation in clockwise direction, default=0.0
                Negative degree can be given to perform anticlockwise direction

            rotate_first: A boolean flag, default=False
                if True rotation is performed before other transformations

            border_mode: A integer value used in opencv to decide border mode
                BORDER_CONSTANT    = 0
                BORDER_REPLICATE   = 1
                BORDER_REFLECT     = 2
                BORDER_WRAP        = 3
                BORDER_REFLECT_101 = 4
                BORDER_TRANSPARENT = 5

            dtype: type of values to be used for transformations, default=np.float32
        """
        if isinstance(scale_xy, Number):
            scale_xy = (scale_xy, scale_xy)
        self.scale_x, self.scale_y = scale_xy

        if isinstance(shear_xy, Number):
            shear_xy = (shear_xy, shear_xy)
        self.shear_x, self.shear_y = shear_xy

        if isinstance(translate_xy, Number):
            translate_xy = (translate_xy, translate_xy)
        self.translate_x, self.translate_y = translate_xy

        self.radian = np.radians(angle)
        self.rotate_first = rotate_first
        self.border_mode = border_mode
        self.dtype = dtype

        self.scale_shear_T = _get_scale_shear_translate_T(
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            shear_x=self.shear_x,
            shear_y=self.shear_y,
            dtype=self.dtype,
        )
        self.rotation_T = _get_rotation_T(radian=self.radian)

    def _get_src_points(self, h: int, w: int) -> np.ndarray:
        """Initialize 3 points in original image space.

        Args:
            h: height of image

            w: width of image

        Returns:
            src_points: A NumPy array of shape (2, 3)
                np.array(
                    [[x_0, x_1, x_2],
                    [y_0, y_1, y_2]])
        """
        return np.array(
            [[0, w // 2 - 1, 0], [0, 0, h // 2 - 1]], dtype=self.dtype
        )

    def transform(self, img: np.ndarray):
        """Perform affine transformation over image.

        Args:
            img: 3-D NumPy array of shape (H, W, C)
                H: height of image

                W: width of image

                C: number of channels in image

        Returns:
            Affine transformed image of shape (H, W, C)
        """
        _validate_image(img=img)
        h, w, c = img.shape
        src_pts = self._get_src_points(h=h, w=w)

        if self.rotate_first:  # if rotation is requried to be performed first
            dst_pts = self.rotation_T @ src_pts
            dst_pts = self.scale_shear_T @ dst_pts + np.array(
                [self.translate_x, self.translate_y]
            ).reshape(-1, 1)
        else:
            dst_pts = self.scale_shear_T @ src_pts + np.array(
                [self.translate_x, self.translate_y]
            ).reshape(-1, 1)
            dst_pts = self.rotation_T @ dst_pts

        dst_pts = dst_pts.astype(src_pts.dtype)

        M = cv2.getAffineTransform(src_pts.T, dst_pts.T)

        trnsf_img = cv2.warpAffine(
            src=img,
            M=M,
            dsize=(w, h),
            borderMode=self.border_mode,
        )

        if c == 1:
            trnsf_img = np.expand_dims(trnsf_img, axis=-1)

        return trnsf_img
