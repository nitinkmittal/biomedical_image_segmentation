from cv2 import line
from numpy import ndarray
from copy import deepcopy


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
