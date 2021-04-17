from copy import deepcopy
from typing import Any, List, Tuple

from cv2 import line
from numpy import ndarray
import numpy as np
from pickle import load
import os


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
    seed: int = 40,
):
    """Split input into train, valid and test set."""
    
    random = np.random.RandomState(seed)
    total = len(a)
    train = random.choice(
        a, replace=replace, size=int(total * ratio[0])
    ).tolist()
    valid = random.choice(
        [b for b in a if b not in train],
        replace=replace,
        size=int(total * ratio[1]),
    ).tolist()
    test = random.choice(
        [c for c in a if c not in train + valid],
        replace=replace,
        size=int(total * ratio[2]),
    ).tolist()
    return train, valid, test

def load_pickle(filename: str) -> Any:
    """Load and return pickle."""
    with open(filename, "rb") as f:
        data = load(f)
    return data


def create_dir(path: str, verbose: bool=False):
    """Create folder if does not exists."""
    if not os.path.exists(path):
        try: 
            os.mkdir(path) 
        except OSError as error: 
            if verbose: print(error)
                
def empty_dir(path: str, verbose: bool=False):
    """Delete content of a directory recursively."""
    assert os.path.exists(path)
    for f in os.listdir(path):
        try:
            os.remove(os.path.join(path, f))
        except Exception as error:
            if verbose: print(error)