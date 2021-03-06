from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize, functional


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        image_transformations=None,
        mask_transformations=None,
    ):
        self.data = dataset
        self.image_transformations = image_transformations
        self.mask_transformations = mask_transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        if self.image_transformations is not None:
            img = self.image_transformations(img)

        if self.mask_transformations is not None:
            mask = self.mask_transformations(mask)

        return img, mask


image_transformations = Compose(
    [
        #         ToTensor(),
        Lambda(
            lambda x: torch.tensor(
                np.expand_dims(x, axis=0), dtype=torch.float32
            )
        ),
        Lambda(
            lambda x: functional.pad(x, padding=94, padding_mode="reflect")
        ),
        Normalize(mean=(0.0,), std=(255.0,)),
    ]
)

mask_transformations = Compose(
    [
        #         ToTensor(),
        Lambda(
            lambda x: torch.tensor(
                np.expand_dims(x, axis=0), dtype=torch.float32
            )
        ),
        #         Lambda(lambda x: x.float()),
        Normalize(mean=(0.0,), std=(255.0,)),
    ]
)


def process_output(output: torch.tensor, threshold: float = 0.5):
    return torch.where(torch.sigmoid(output) >= threshold, 1.0, 0.0)
