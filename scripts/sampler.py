"""This script runs the sampler to produce affine and elastic
    deformed/transformed versions of the given image."""
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from biomedical_image_segmentation.elastic_deform import (
    getWarpAndElasticDeform,
)


AFFINE_REF_RATIO = 3.0
AFFINE_ALPHA = 40.0
ELASTIC_SIGMA = 12.0
ELASTIC_ALPHA_RANGE = (10, 120)


class Sampler:
    def __init__(
        self,
        fp: Union[str, Path],
        save_dir: Union[str, Path],
        num_samples: int = 1,
        affine_transform: bool = True,
        elastic_transform: bool = True,
        seed: Optional[int] = None,
    ):
        """"""
        self.img = np.array(Image.open(fp))
        # This step is required if image is a 2-D image/number of channels=1
        if self.img.ndim == 2:
            self.img = np.expand_dims(self.img, axis=-1)
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.affine_transform = affine_transform
        self.elastic_transform = elastic_transform
        self.seed = seed

    def _save_img(self, img: np.ndarray, fp: Union[str, Path]):
        ToPILImage()(img).save(fp)

    def run(
        self,
    ):
        """"""
        fp = os.path.join(self.save_dir, "original.png")
        self._save_img(self.img, fp)

        rng = np.random.default_rng(self.seed)
        seeds = rng.choice(
            self.num_samples * 1000, size=self.num_samples, replace=False
        )

        for seed in seeds:
            print(seed)
            transformed_img = getWarpAndElasticDeform(
                self.img,
                affine_transform=self.affine_transform,
                affine_ref_ratio=AFFINE_REF_RATIO,
                affine_alpha=AFFINE_ALPHA,
                elastic_transform=self.elastic_transform,
                elastic_sigma=ELASTIC_SIGMA,
                elastic_alpha=rng.uniform(*ELASTIC_ALPHA_RANGE),
                seed=seed,
            )
            fp = os.path.join(self.save_dir, f"{str(seed)}.png")
            self._save_img(transformed_img, fp)


fp = (
    "/home/mittal.nit/projects/biomedical_image_segmentation"
    "/data/samples/train/0.tif"
)

fp = "/home/mittal.nit/projects/biomedical_image_segmentation/notebooks/test.png"
dir = "/home/mittal.nit/projects/biomedical_image_segmentation/data/test"
sampler = Sampler(fp, dir, 10, 1, seed=10)
sampler.run()
