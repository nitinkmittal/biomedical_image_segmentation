from copy import deepcopy
from typing import Any, List, Tuple

from cv2 import line
from numpy import ndarray
import numpy as np


def insert_grid(
    img: ndarray,
    box_height: int,
    box_width: int,
    color: float = 1.0,
    thickness: int = 1,
) -> ndarray:
    """Insert grid on image."""
    assert img.ndim == 2

    img = deepcopy(img)

    height, width = img.shape

    for x in range(0, width, box_width):
        line(
            img=img,
            pt1=(x, 0),
            pt2=(x, height),
            color=(color,),
            thickness=thickness,
        )

    for y in range(0, height, box_height):
        line(
            img=img,
            pt1=(0, y),
            pt2=(width, y),
            color=(color,),
            thickness=thickness,
        )

    return img


def split(
    a: List[Any],
    ratio: Tuple[int, int, int] = (2 / 3, 1 / 6, 1 / 6),
    replace: bool = False,
):
    """Split input into train, valid and test set."""
    total = len(a)
    train = np.random.choice(
        a, replace=replace, size=int(total * ratio[0])
    ).tolist()
    valid = np.random.choice(
        [b for b in a if b not in train],
        replace=replace,
        size=int(total * ratio[1]),
    ).tolist()
    test = np.random.choice(
        [c for c in a if c not in train + valid],
        replace=replace,
        size=int(total * ratio[2]),
    ).tolist()
    return train, valid, test
