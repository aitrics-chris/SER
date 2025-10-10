import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFilter, ImageOps
import torch
import torchvision.transforms.functional as F
import os
from typing import List
from .inv import MultiCropsTransform

# def get_dataset_augself(args):
#     transform = MultiView(RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation))
#     return datasets.ImageFolder(os.path.join(args.data, 'train'), transform)


# class MultiView:
#     def __init__(self, transform, num_views=2):
#         self.transform = transform
#         self.num_views = num_views

#     def __call__(self, x):
#         # produce self.num_views copies of (tensor, crop-params)
#         return [ self.transform(x) for _ in range(self.num_views) ]

    
# class RandomResizedCrop(transforms.RandomResizedCrop):
#     def forward(self, img):
#         W, H = F.get_image_size(img)
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
#         tensor = F.to_tensor(img)
#         return tensor, torch.tensor([i/H, j/W, h/H, w/W], dtype=torch.float)




def get_dataset_augself(args):
    transform = MultiView(RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation))
    return datasets.ImageFolder(os.path.join(args.data, 'train'), transform)


class MultiView:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        # produce self.num_views copies of (tensor, crop-params)
        return [ self.transform(x) for _ in range(self.num_views) ]

    
class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        W, H = F.get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        tensor = F.to_tensor(img)
        return tensor, torch.tensor([i/H, j/W, h/H, w/W], dtype=torch.float)