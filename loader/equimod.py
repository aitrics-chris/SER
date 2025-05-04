
import torchvision.datasets as datasets
import math
import random
import os
import torchvision
from typing import List
import torch
import torchvision.transforms.functional as F

def get_dataset_equimod(args):
    dataset = datasets.ImageFolder(os.path.join(args.data, 'train'))
    
    no_transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(256, interpolation=args.interpolation),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor()
                                                ])

    inv1_transform = ParamCompose([
            ParamRandomResizedCrop(pflip=0.5, size=(224, 224), scale=(args.crop_min, 1.), ratio=(3./4., 4./3.), interpolation=args.interpolation),
            ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            ParamGaussianBlur(pblur=1.0, kernel_size=(9, 9), sigma=(0.1, 2.0)),            
            ParamSolarize(threshold=128.0, p=0.0),
        ], [
            torchvision.transforms.ToTensor()
        ])

    inv2_transform = ParamCompose([
            ParamRandomResizedCrop(pflip=0.5, size=(224, 224), scale=(args.crop_min, 1.), ratio=(3./4., 4./3.), interpolation=args.interpolation),
            ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            ParamGaussianBlur(pblur=0.1, kernel_size=(9, 9), sigma=(0.1, 2.0)),
            ParamSolarize(threshold=128.0, p=0.2),
        ], [
            torchvision.transforms.ToTensor()
        ])

    etc = {
            'p_mean': torch.tensor([[6.8162e+01, 9.9199e+01, 2.6933e+02, 2.7457e+02, 4.9905e-01, 8.0054e-01,
                        1.1998e+00, 1.3994e+00, 1.6014e+00, 1.7995e+00, 1.0001e+00, 1.0000e+00,
                        1.0005e+00, 1.5640e-04, 2.0018e-01, 2.7256e-01, 5.2507e-01, 5.0149e-2]]).to(args.gpu),
            'p_std': torch.tensor([[7.7370e+01, 9.8681e+01, 1.3686e+02, 1.4349e+02, 5.0000e-01, 3.9959e-01,
                        1.1661e+00, 1.0201e+00, 1.0201e+00, 1.1657e+00, 4.1347e-01, 4.1323e-01,
                        4.1349e-01, 1.0333e-01, 4.0013e-01, 3.0381e-01, 6.5251e-01, 6.4719e-2]]).to(args.gpu)

            }

    return AugDatasetWrapper(dataset, no_transform, inv1_transform, inv2_transform), etc

    

class AugDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, no_transform, inv1_transform, inv2_transform):
        self.dataset = dataset
        self.no_transform = no_transform
        self.inv1_transform = inv1_transform
        self.inv2_transform = inv2_transform

        self.nb_params = inv1_transform.nb_params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img_0 = self.no_transform(img)
        img_1, param_1 = self.inv1_transform(img)
        img_2, param_2 = self.inv2_transform(img)

        return ((img_0, img_1, param_1, img_2, param_2), label)
    

class ParamRandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, pflip=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pflip = pflip
        self.nb_params = 5

    def get_params(self, img):
        i, j, h, w = super().get_params(img, self.scale, self.ratio)
        flip = int(random.random() < self.pflip)

        return [i, j, h, w, flip]

    def apply(self, img, params):
        i, j, h, w, flip = params

        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        
        if flip:
            img = F.hflip(img)

        params = torch.FloatTensor([i, j, h, w, flip])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, pjitter=0.8, pgray=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pjitter = pjitter
        self.pgray = pgray
        self.nb_params = 10

    def get_params(self, img):
        jitter = int(random.random() < self.pjitter)

        if jitter:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                super().get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                [[0, 1, 2, 3], 1., 1., 1., 0.]
        
        gray = int(random.random() < self.pgray)

        return [jitter, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray]

    def apply(self, img, params):
        jitter, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray = params

        if jitter:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
        
        if gray:
            img = F.rgb_to_grayscale(img, num_output_channels=3)

        params = torch.FloatTensor([jitter, *fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamGaussianBlur(torchvision.transforms.GaussianBlur):
    def __init__(self, pblur=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pblur = pblur
        self.nb_params = 2
    
    def get_params(self, img):
        blur = int(random.random() < self.pblur)

        if blur:
            sigma = super().get_params(self.sigma[0], self.sigma[1])
        else:
            sigma = 0.

        return [blur, sigma]

    def apply(self, img, params):
        blur, sigma = params

        if blur:
            img = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])

        params = torch.FloatTensor([blur, sigma])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamSolarize(torchvision.transforms.RandomSolarize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_params = 1
    
    def get_params(self, img):
        solarize = int(random.random() < self.p)

        return [solarize]

    def apply(self, img, params):
        solarize, = params

        if solarize:
            img = F.solarize(img, self.threshold)

        params = torch.FloatTensor([solarize])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)





class ParamCompose(torch.nn.Module):
    def __init__(self, param_transforms, nonparam_transforms):
        super().__init__()
        self.param_transforms = param_transforms
        self.nonparam_transforms = nonparam_transforms
        self.nb_params = sum([transform.nb_params for transform in self.param_transforms])

    def get_params(self, img):
        return [transform.get_params(img) for transform in self.param_transforms]

    def apply(self, img, params):
        res_params = []

        for transform, transform_params in zip(self.param_transforms, params):
            img, sub_params = transform.apply(img, transform_params)
            res_params.append(sub_params)

        for transform in self.nonparam_transforms:
            img = transform(img)

        return img, torch.cat(res_params)

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)
    

