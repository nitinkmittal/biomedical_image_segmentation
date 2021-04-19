import torch
from matplotlib import pyplot as plt
from cv2 import line
from copy import deepcopy
import numpy as np

def insert_grid(
    img: np.ndarray,
    box_height: int,
    box_width: int,
    color: float = 1.0,
    thickness: int = 1,
) -> np.ndarray:
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


def plot_results(
    img: torch.Tensor,
    true_mask: torch.Tensor, 
    pred_mask: torch.Tensor):
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