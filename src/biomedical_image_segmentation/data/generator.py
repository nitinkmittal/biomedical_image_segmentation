from typing import List, Tuple, Union

import numpy as np
from scipy import ndimage

from ..elastic_deform import custom_2d_elastic_deform


def generate(
    img: np.ndarray,
    mask: np.ndarray,
    rotational_degrees: Union[float, List[float]] = None,
    num_elastic_deforms: int = 0,
    ref_ratio: float = 3.0,
    alpha_affine: Tuple[float, float] = (0.01, 0.2),
    sigma: float = 10.0,
    alpha: float = 1.0,
    adjustment_pixel_range: Tuple[int, int] = None,
    adjusted_pixel: int = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:

    assert img.shape == mask.shape

    if rotational_degrees is None:
        rotational_degrees = []
    elif isinstance(rotational_degrees, float):
        rotational_degrees = [rotational_degrees]
    rotational_degrees += np.linspace(0, 270, 4).tolist()
    rotational_degrees = np.unique(rotational_degrees)

    # rotational and flip transformations
    dataset = []
    for degree in rotational_degrees:
        dataset.append(
            (
                ndimage.rotate(img, angle=degree, mode="reflect"),
                ndimage.rotate(mask, angle=degree, mode="reflect"),
            )
        )
        dataset.append(
            (
                np.flip(dataset[-1][0], axis=0),
                np.flip(dataset[-1][1], axis=0),
            )
        )

    # elastic deformations
    elastic_deformed_dataset = []
    for _ in range(num_elastic_deforms):
        for img, mask in dataset:
            (
                elastic_deformed_img,
                elastic_deformed_mask,
            ) = custom_2d_elastic_deform(
                img=img,
                mask=mask,
                ref_ratio=ref_ratio,
                alpha_affine=alpha_affine,
                sigma=sigma,
                alpha=alpha,
                adjustment_pixel_range=adjustment_pixel_range,
                adjusted_pixel=adjusted_pixel,
            )
            elastic_deformed_dataset.append(
                (elastic_deformed_img, elastic_deformed_mask)
            )

    return dataset + elastic_deformed_dataset
