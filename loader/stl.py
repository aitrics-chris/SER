
import torchvision.datasets as datasets
import math
import random
import os
import torchvision
from typing import List
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from typing import Callable, Optional, Union
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from torchvision.datasets import STL10


def get_dataset_stl(args):
    base_transform = v2.Compose([
        v2.Resize((256, 256), interpolation=args.interpolation),
        # v2.RandomResizedCrop(224, scale=(args.crop_min, 1.0), interpolation=args.interpolation),
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])

    aligned_transform = torch.nn.Sequential(
        v2.RandomResizedCrop(224, scale=(args.crop_min, 1.0), interpolation=args.interpolation),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        v2.RandomGrayscale(p=0.2)
    )

    invariant_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5
        )
    ])

    return ImageNet1k_Align(
        root=args.data,
        base_transform=base_transform,
        aligned_transform=aligned_transform,
        invariant_transform=invariant_transform
    ), None



class ImageNet1k_Align(datasets.ImageFolder):
    """
    Custom ImageNet-1k dataset for handling aligned and invariant transformations.
    """
    def __init__(
        self,
        root: Union[str, Path],
        base_transform: Optional[Callable] = None,
        aligned_transform: Optional[Callable] = None,
        invariant_transform: Optional[Callable] = None,
        loader=None,
        is_valid_file=None
    ) -> None:
        super().__init__(
            root=root,
            transform=base_transform,
            loader=loader,
            is_valid_file=is_valid_file
        )
        self.base_transform = base_transform
        self.aligned_transform = aligned_transform
        self.invariant_transform = invariant_transform
        self.loader = datasets.folder.default_loader

    def __getitems__(self, possibly_batched_index):

        half_batch_size = len(possibly_batched_index) // 2
        batched_item = []

        for i in range(half_batch_size):
            idx1 = possibly_batched_index[2 * i]
            idx2 = possibly_batched_index[2 * i + 1]

            img1 = self.loader(self.samples[idx1][0])
            img2 = self.loader(self.samples[idx2][0])

            # img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
            # img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))

            img11, img12 = self.base_transform(img1), self.base_transform(img1)
            img21, img22 = self.base_transform(img2), self.base_transform(img2)

            paired_img1 = torch.stack([img11, img21])
            paired_img2 = torch.stack([img12, img22])

            paired_img1 = self.aligned_transform(paired_img1)
            paired_img2 = self.aligned_transform(paired_img2)

            img11, img12 = paired_img1[0], paired_img2[0]
            img21, img22 = paired_img1[1], paired_img2[1]

            # Invariant transform
            img11, img12 = self.invariant_transform(img11), self.invariant_transform(img12)
            img21, img22 = self.invariant_transform(img21), self.invariant_transform(img22)

            batched_item.append((img11, img12))
            batched_item.append((img21, img22))

        return batched_item