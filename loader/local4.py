
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import os
from typing import List

def get_dataset_local4(args):
    # ColorJitter, RandomGrayscale, GaussianBlur, Solarize will be performed using Kornia at GPU. See build.utils.Aug_equi.py
    
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        # transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]
    augmentation3 = [
        transforms.RandomResizedCrop(96, scale=(0.05,0.14)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    augmentation4 = [
        transforms.RandomResizedCrop(96, scale=(0.05,0.14)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    return datasets.ImageFolder(os.path.join(args.data, 'train'),
        MultiCropsTransform([transforms.Compose(augmentation1), transforms.Compose(augmentation2), transforms.Compose(augmentation3), transforms.Compose(augmentation4)])), None
    

class MultiCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transforms: List[transforms.Compose]):
        self.base_transforms = base_transforms
        # self.base_transform2 = base_transform2

    def __call__(self, x):
        # im1 = self.base_transform1(x)
        # im2 = self.base_transform2(x)
        # return [im1, im2]
        return [_base_transform(x) for _base_transform in self.base_transforms]

