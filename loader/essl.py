import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from .inv import MultiCropsTransform
from PIL import Image

def get_dataset_essl(args):
    # ColorJitter, RandomGrayscale, GaussianBlur, Solarize will be performed using Kornia at GPU. See build.utils.Aug_equi.py
    
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.), interpolation=args.interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    augmentation3 = [
        transforms.RandomResizedCrop(96, scale=(0.05,0.14)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    return datasets.ImageFolder(os.path.join(args.data, 'train'),
        MultiCropsTransform([transforms.Compose(augmentation1), transforms.Compose(augmentation2), transforms.Compose(augmentation3)])), None
    
