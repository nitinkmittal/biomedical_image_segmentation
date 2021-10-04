from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import torch
from cv2 import line
from matplotlib import pyplot as plt


def insert_grid(
    img: np.ndarray,
    grid_size: Tuple[int, int],
    color: Union[Tuple[float, float, float], float] = 1.0,
    thickness: int = 1,
) -> np.ndarray:
    """Insert grid on image.

    Args:
        img: A NumPy array image

        grid_size: A Tuple with box width and height

        color: color for grid lines

        thickness: thickness of grid lines

    Returns:
        img: A NumPy array image with grid lines
    """
    assert img.ndim == 3

    img = deepcopy(img)

    height, width, _ = img.shape
    box_width, box_height = grid_size
    for x in range(0, width, box_width):
        line(
            img=img,
            pt1=(x, 0),
            pt2=(x, height),
            color=color,
            thickness=thickness,
        )

    for y in range(0, height, box_height):
        line(
            img=img,
            pt1=(0, y),
            pt2=(width, y),
            color=color,
            thickness=thickness,
        )

    return img


def plot_results(
    img: torch.Tensor, true_mask: torch.Tensor, pred_mask: torch.Tensor
):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 7))
    ax1.set_title("Image")
    ax1.imshow(img, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title("Original Mask")
    ax2.imshow(true_mask, cmap="gray")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.set_title("Predicted Mask")
    ax3.imshow(pred_mask, cmap="gray")
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.show()
