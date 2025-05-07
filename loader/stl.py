
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from typing import Callable, Optional, Union
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from torchvision.datasets import STL10
from kornia.augmentation.container import ImageSequential
import kornia

def get_dataset_stl(args):
    base_transform = transforms.Compose([
                                            transforms.Resize((224, 224), interpolation=args.interpolation),
                                            transforms.ToTensor(),
                                        ])
    
    post_transform = {}

    post_transform['aligned_transform'] = ImageSequential(            
                                        kornia.augmentation.RandomResizedCrop((224,224), scale=(0.2, 1.0), resample=args.interpolation_kornia, same_on_batch=False),
                                        kornia.augmentation.ColorJiggle(0.4, 0.4, 0.2, 0.1, same_on_batch=False, p=0.8),  # not strengthened                                         
                                        kornia.augmentation.RandomGrayscale(same_on_batch=False, p=0.2),
                                    )   

    post_transform['invariant_transform'] = ImageSequential(            
                                        kornia.augmentation.RandomHorizontalFlip(),
                                        kornia.augmentation.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), same_on_batch=False, p=0.5),
                                    )   

    return datasets.ImageFolder(os.path.join(args.data, 'train'), transform=base_transform), post_transform



# class ImageNet1k_Align(datasets.ImageFolder):
#     """
#     Custom ImageNet-1k dataset for handling aligned and invariant transformations.
#     """
#     def __init__(
#         self,
#         root: Union[str, Path],
#         base_transform: Optional[Callable] = None,
#         aligned_transform: Optional[Callable] = None,
#         invariant_transform: Optional[Callable] = None,
#         loader=None,
#         is_valid_file=None
#     ) -> None:
#         super().__init__(
#             root=root,
#             transform=base_transform,
#             loader=loader,
#             is_valid_file=is_valid_file
#         )
#         self.base_transform = base_transform
#         self.aligned_transform = aligned_transform
#         self.invariant_transform = invariant_transform
#         self.loader = datasets.folder.default_loader

#     def __getitems__(self, possibly_batched_index):

#         half_batch_size = len(possibly_batched_index) // 2
#         batched_item = []

#         for i in range(half_batch_size):
#             idx1 = possibly_batched_index[2 * i]
#             idx2 = possibly_batched_index[2 * i + 1]

#             img1 = self.loader(self.samples[idx1][0])
#             img2 = self.loader(self.samples[idx2][0])

#             # img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
#             # img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))

#             img11, img12 = self.base_transform(img1), self.base_transform(img1)
#             img21, img22 = self.base_transform(img2), self.base_transform(img2)

#             paired_img1 = torch.stack([img11, img21])
#             paired_img2 = torch.stack([img12, img22])

#             paired_img1 = self.aligned_transform(paired_img1)
#             paired_img2 = self.aligned_transform(paired_img2)

#             img11, img12 = paired_img1[0], paired_img2[0]
#             img21, img22 = paired_img1[1], paired_img2[1]

#             # Invariant transform
#             # img11, img12 = self.invariant_transform(img11), self.invariant_transform(img12)
#             # img21, img22 = self.invariant_transform(img21), self.invariant_transform(img22)

#             batched_item.append((img11, img12))
#             batched_item.append((img21, img22))

#         return batched_item