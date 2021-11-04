from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from numpy.random._generator import Generator

from scipy import ndimage


def _check_image(img: np.ndarray):
    """Check if given image has 3 dimensions or not."""

    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Expected image to be of type numpy.ndarray, "
            f"got {type(img)} instead."
        )

    if img.ndim != 3:
        raise ValueError(
            f"Expected NumPy image with 3 dimension "
            "(height * width * num of channels), "
            f"got {img.ndim} dimensional image instead."
        )

    if img.dtype != np.uint8:
        raise TypeError(
            f"Expected values in image of type uint8, got {img.dtype} instead."
        )


def generate_coordinates(
    img: np.ndarray,
    sigma: float,
    alpha: float,
    rng: Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coordinates for elastic transformation.

    Args:
        height: height of the image.

        width: width of the image.

        num_channels: number of channels in the image.

        sigma: standard deviation for Gaussian kernel.
            The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes.

        alpha: A float value to scale values in the Gaussian kernel.

        rng: The initialized generator object.

    Returns:
        A tuple of arrays required in elastic transformation.
    """
    height, width, num_channels = img.shape
    inputs = rng.random((2, height, width, num_channels)) * 2 - 1

    dx = (
        gaussian_filter(
            input=inputs[0],
            sigma=sigma,
            order=0,
            mode="reflect",
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            input=inputs[1],
            sigma=sigma,
            order=0,
            mode="reflect",
        )
        * alpha
    )
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(
        np.arange(width), np.arange(height), np.ones(num_channels)
    )
    coordinates = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z + dz, (-1, 1)),
    )

    return coordinates


class AffineTransform(object):
    def __init__(
        self,
        M: Optional[np.ndarray] = None,
        angle: float = 0,
        alpha: float = 0,
        seed: Optional[int] = None,
    ):
        """

        Args:
            M: A 2-d NumPy array of shape (2, 3).
                M is required for affine transformation.
                if None, M is computed using parameters angle and alpha.

            angle: A float value in degrees to perform rotation.

            alpha: A float value to compute affine transformation matrix.

            seed: An optional integer value for reproducibility.
        """
        self.M = M
        if self.M is not None:
            assert M.shape == (2, 3)
        self.angle = angle
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _get_affine_matrix(
        self,
        img: np.ndarray,
        rng: Generator,
        alpha: float,
    ) -> np.ndarray:
        """Return affine transformation matrix."""

        height, width, _ = img.shape

        # Affine tranformation require affine/warp transformation matrix
        #   To compute affine transformation matrix,
        #   we require 3 points in original image and 3 points in transformed image.
        #   We use center coordinates to compute reference points in original image space.
        center_coord = np.float32([height, width]) // 2
        ref_point = min((height, width)) // 2
        pts_src = np.float32(
            [
                center_coord - ref_point,
                center_coord + np.array([1.0, -1.0]) * ref_point,
                center_coord + ref_point,
            ]
        )
        pts_dst = pts_src + rng.uniform(
            -alpha, alpha, size=pts_src.shape
        ).astype(pts_src.dtype)

        # get affine transformation matrix to transform (x,y) points
        # from original image space to affine transformed image space
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        return cv2.getAffineTransform(pts_src, pts_dst)

    def transform(self, img):
        """
        Args:
            img: A 3-d (H * W * C) NumPy array of dtype uint8.
                H: height of the image

                W: width of the image

                C: number of channels in image

        Returns:
            img: A 3-d (H * W * C) NumPy array of dtype uint8.
                H: height of the image

                W: width of the image

                C: number of channels in image
        """
        _check_image(img)
        height, width, _ = img.shape

        # rotate image at given angle
        img = ndimage.rotate(
            input=img, angle=self.angle, mode="reflect", reshape=False
        )

        if self.M is None:
            self.M = self._get_affine_matrix(
                img, rng=self.rng, alpha=self.alpha
            )

        # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
        img = cv2.warpAffine(
            src=img,
            M=self.M,
            dsize=(width, height),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # this step is only required after cv2.getAffineTransform
        #   when number of channels in original image is 1
        if img.ndim < 3:
            img = np.expand_dims(img, axis=-1)

        img = np.clip(
            img,
            0,
            255,
        )
        return img.astype(np.uint8)


class ElasticTransform:
    def __init__(
        self,
        sigma: float,
        alpha: float,
        seed: Optional[int] = None,
    ):
        """

        Args:
            sigma: standard deviation for Gaussian kernel.
                The standard deviations of the Gaussian filter
                are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes.

            alpha: A float value to scale values in the Gaussian kernel.

            seed: An optional integer value for reproducibility.
        """
        self.sigma = sigma
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: A 3-d (H * W * C) NumPy array of dtype uint8.
                H: height of the image

                W: width of the image

                C: number of channels in image

        Returns:
            img: A 3-d (H * W * C) NumPy array of dtype uint8.
                H: height of the image

                W: width of the image

                C: number of channels in image
        """
        _check_image(img)
        height, width, num_channels = img.shape
        coordinates = generate_coordinates(
            img, sigma=self.sigma, alpha=self.alpha, rng=self.rng
        )

        img = map_coordinates(
            img,
            coordinates=coordinates,
            order=3,
            mode="reflect",
        ).reshape(height, width, num_channels)

        img = np.clip(
            img,
            0,
            255,
        )
        return img.astype(np.uint8)


def getAffineAndElasticDeform(
    img: np.ndarray,
    affine_transform: bool = True,
    affine_degree: float = 0,
    affine_alpha: float = 0,
    elastic_transform: bool = True,
    elastic_sigma: float = 12.0,
    elastic_alpha: float = 400.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform warp and elastic transformations on input image.

    Reference: https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations

    Args:
        img: A NumPy array of type uint8.
            H * W * C
            H: height of the image.

            W: width of the image.

            C: number of channels in image.

        affine_transform: A boolean flag.
            if True, image is affine transformed, otherwise not.

        affine_degree: A float value in degress to perform rotation.

        affine_alpha: A float value used in warp/affine transformation, default = 40.
            Note: Default value is picked randomly.

        elastic_transform: A boolean flag.
            if True, image is elastic transformed, otherwise not.

        elastic_sigma: standard deviation for Gaussian kernel, default = 12.
            The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes.

        elastic_alpha: A float value used in elastic transformation, default = 400.
            Note: Default value is picked randomly.

        seed: An integer value to be used for random number generator.

    Returns:
        img: A NumPy array of type uint8.
            H * W * C
            H: height of the image.

            W: width of the image.

            C: number of channels in image.
    """

    _check_image_dim(img)
    height, width, num_channels = img.shape
    rng = np.random.default_rng(seed)

    if affine_transform:
        img = getAffineTransform(
            img,
            degree=affine_degree,
            alpha=affine_alpha,
            rng=rng,
            height=height,
            width=width,
            num_channels=num_channels,
        )

    if elastic_transform:
        img = getElasticTransform(
            img,
            sigma=elastic_sigma,
            alpha=elastic_alpha,
            rng=rng,
            height=height,
            width=width,
            num_channels=num_channels,
        )

    img = np.clip(img, a_min=0, a_max=255)

    return img
