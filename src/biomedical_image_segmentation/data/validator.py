from typing import List, Tuple, Union

import numpy as np
from scipy import ndimage


def validate(
    dataset: List[Tuple[np.ndarray, np.ndarray]],
    shape: Tuple[int, int] = None,
    check_for_duplicates: bool = False,
):
    """Validate data."""
    if shape is None:
        shape = dataset[0][0].shape

    for img, mask in dataset:

        assert img.shape == shape
        assert mask.shape == shape
        assert np.allclose(np.unique(mask), np.array([0, 255]))

    if check_for_duplicates:
        assert len(np.unique([img for img, _ in dataset], axis=0)) == len(
            dataset
        )
