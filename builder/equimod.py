import random
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms.functional as TF


class ParamRandomResizedCrop(transforms.RandomResizedCrop):
    """Parameterized version of RandomResizedCrop"""

    def __init__(self, pflip=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pflip = pflip
        self.nb_params = 5

    def get_params(self, img):
        """Get parameters for the crop transform"""
        i, j, h, w = super().get_params(img, self.scale, self.ratio)
        flip = int(random.random() < self.pflip)
        return [i, j, h, w, flip]

    def apply(self, img, params):
        i, j, h, w, flip = params

        # Handle size parameter correctly
        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size

        # Apply crop and resize
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:  # C, H, W
                img = img[:, i : i + h, j : j + w]
                img = img.unsqueeze(0)  # Add batch dimension for interpolate
                img = F.interpolate(
                    img, size=size, mode="bilinear", align_corners=False
                )
                img = img.squeeze(0)  # Remove batch dimension
            else:  # Already has batch dimension
                img = img[:, :, i : i + h, j : j + w]
                img = F.interpolate(
                    img, size=size, mode="bilinear", align_corners=False
                )
        else:
            img = F.resized_crop(img, i, j, h, w, size, self.interpolation)

        # Apply flip if needed
        if flip:
            if isinstance(img, torch.Tensor):
                img = torch.flip(img, [-1])  # Flip the last dimension (width)
            else:
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
        self.nb_params = 12  # Updated to match expected parameters

    def get_params(self, img):
        jitter = int(random.random() < self.pjitter)

        if jitter:
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = super().get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = ([0, 1, 2, 3], 1.0, 1.0, 1.0, 0.0)

        gray = int(random.random() < self.pgray)

        # Add additional parameters to match the expected 17 total parameters
        # The additional parameters can be dummy values that match the statistics
        extra_params = [1.0, 1.0]  # Additional parameters to match expected count

        return [
            jitter,
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
            gray,
        ] + extra_params

    def apply(self, img, params):
        (
            jitter,
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
            gray,
            *extra_params,
        ) = params

        # Create a copy of the image to avoid in-place operations
        if isinstance(img, torch.Tensor):
            img = img.clone()

        if jitter:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = TF.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = TF.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = TF.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = TF.adjust_hue(img, hue_factor)

        if gray:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:  # C, H, W
                    img = TF.rgb_to_grayscale(img, num_output_channels=3)
                else:  # B, C, H, W
                    img = TF.rgb_to_grayscale(
                        img.squeeze(0), num_output_channels=3
                    ).unsqueeze(0)
            else:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)

        # Include all parameters in output
        params = torch.FloatTensor(
            [jitter]
            + [float(x) for x in fn_idx]
            + [brightness_factor, contrast_factor, saturation_factor, hue_factor, gray]
            + extra_params
        )

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)


class TensorNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean) if not torch.is_tensor(mean) else mean
        self.std = torch.tensor(std) if not torch.is_tensor(std) else std

    def __call__(self, tensor):
        # Ensure mean and std are on the same device as the input tensor
        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)

        # Ensure mean and std have the right shape for broadcasting
        if self.mean.dim() == 1:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
        else:
            mean = self.mean
            std = self.std

        # Handle both batched and unbatched inputs
        if tensor.dim() == 4:  # batched input (B, C, H, W)
            return (tensor - mean) / std
        else:  # unbatched input (C, H, W)
            return (tensor - mean) / std


class ParamCompose(torch.nn.Module):
    def __init__(self, param_transforms, nonparam_transforms):
        super().__init__()
        self.param_transforms = param_transforms
        self.nonparam_transforms = nonparam_transforms
        self.nb_params = sum(
            [transform.nb_params for transform in self.param_transforms]
        )

    def get_params(self, img):
        return [transform.get_params(img) for transform in self.param_transforms]

    def apply(self, img, params):
        res_params = []

        # Apply parameterized transforms
        for transform, transform_params in zip(self.param_transforms, params):
            img, sub_params = transform.apply(img, transform_params)
            res_params.append(sub_params)

        # Apply non-parameterized transforms
        img_transformed = img.clone()  # Create a copy to avoid memory sharing
        for transform in self.nonparam_transforms:
            img_transformed = transform(img_transformed)

        return img_transformed, torch.cat(res_params)

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)


class EquiModTransformModule(nn.Module):
    """Main transformation module for EquiMod"""

    def __init__(self, embed_dim, args):
        super().__init__()

        # Parameter encoding dimensions (5 from crop + 5 from color)
        self.param_dim = 17

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(self.param_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

        # Feature transformation network
        self.transform_net = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.projector_equiv = nn.Sequential(
            # First layer: FC -> BN -> ReLU
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            
            # Second layer: FC -> BN -> ReLU
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            
            # Third layer: FC -> BN (no ReLU)
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

        self.temperature = args.equimod_temperature

    def transform_features(self, features, encoded_params):
        combined = torch.cat([features, encoded_params], dim=1)
        transformed = self.transform_net(combined)
        return F.normalize(transformed, dim=1)

    def compute_loss(self, features1, features2, params):
        encoded_params = self.param_encoder(params)
        transformed_features = self.transform_features(features1, encoded_params)
        target_features = F.normalize(features2, dim=1)

        batch_size = transformed_features.shape[0]
        similarity_matrix = (
            torch.matmul(transformed_features, target_features.T) / self.temperature
        )
        exp_sim = torch.exp(similarity_matrix)
        mask = torch.eye(batch_size, device=similarity_matrix.device)
        numerator = torch.sum(exp_sim * mask, dim=1)
        denominator = torch.sum(exp_sim, dim=1)
        loss = -torch.log(numerator / denominator).mean()
        return loss

    def project_features_for_equivariance(self, x):
        if len(x.shape) > 2:
            b, t, c = x.shape
            x = x.reshape(-1, c)
            x = self.projector_equiv(x)
            x = x.reshape(b, t, -1)
        else:
            x = self.projector_equiv(x)
        return x


class EquiModWrapper(nn.Module):
    """Wrapper for backbone network with EquiMod functionality"""

    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone

        # Get embedding dimension
        if args.arch == "vit_small":
            embed_dim = backbone.embed_dim
        elif args.arch == "resnet50":
            embed_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        self.transform_module = EquiModTransformModule(embed_dim, args)

        # Projector for invariance learning
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        output = self.projector(features)

        if return_features:
            return output, [features]
        return output

    def forward_transform(self, x, params):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        transformed = self.transform_module.transform_features(
            features, self.transform_module.encode_params(params)
        )

        return transformed


class EquivarianceProjector(nn.Module):
    """
    Projector for equivariance learning that maps features to a space 
    where equivariance properties can be better measured or enforced.
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=256, norm_last_layer=True):
        super().__init__()
        
        # First linear layer with normalization and activation
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Output layer
        if norm_last_layer:
            # Normalized output layer for better representation learning
            self.last_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        else:
            self.last_layer = nn.Linear(hidden_dim, out_dim)
            
    def forward(self, x):
        """
        Forward pass through the projector.
        
        Args:
            x: Input features [batch_size, in_dim]
            
        Returns:
            Projected features [batch_size, out_dim]
        """
        # Handle different input shapes
        if len(x.shape) > 2:
            # If input is [batch, tokens, features], reshape for batch norm
            b, t, c = x.shape
            x = x.reshape(-1, c)  # [batch*tokens, features]
            
            x = self.layer1(x)
            x = self.last_layer(x)
            
            # Reshape back to original format
            x = x.reshape(b, t, -1)  # [batch, tokens, out_dim]
        else:
            # Standard forward pass for [batch, features]
            x = self.layer1(x)
            x = self.last_layer(x)
            
        return x



def build_equimod_transforms(mean, std):
    """Build the complete transform pipeline for EquiMod"""
    # Convert mean and std to lists if they're tensors
    if torch.is_tensor(mean):
        mean = mean.tolist()
    if torch.is_tensor(std):
        std = std.tolist()

    return ParamCompose(
        [
            ParamRandomResizedCrop(
                pflip=0.5,
                size=224,  # Pass as tuple
                scale=(0.15, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            ParamColorJitter(
                pjitter=0.8,
                pgray=0.2,
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2,
            ),
        ],
        [TensorNormalize(mean, std)],
    )


def normalize_image(
    image: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> torch.Tensor:
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {image.dtype}.")

    if image.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}."
        )

    # Convert std to tensor first
    if isinstance(std, (tuple, list)):
        std = torch.tensor(std)

    # Check for zero values in std
    if torch.any(std == 0):
        raise ValueError("std evaluated to zero, leading to division by zero.")

    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    if inplace:
        image = image.sub_(mean)
    else:
        image = image.sub(mean)

    return image.div_(std)


def encode_transformations(trans_params):
    """
    Encodes transformation parameters following the EquiMod paper.

    Args:
        trans_params: Dictionary containing transformation parameters
            - crop: (x, y, width, height)
            - flip: Boolean indicating if horizontal flip was applied
            - color_jitter: Dictionary with brightness, contrast, saturation, hue factors and order
            - grayscale: Boolean indicating if grayscale was applied
            - blur: Sigma value if blur was applied, 0 otherwise

    Returns:
        Tensor of encoded transformations
    """
    batch_size = len(trans_params)
    encoded = []

    for params in trans_params:
        # Binary flags for transformations (except crop which is always applied)
        binary_flags = [
            1 if params.get("flip", False) else 0,
            1 if params.get("color_jitter", {}).get("applied", False) else 0,
            1 if params.get("grayscale", False) else 0,
            1 if params.get("blur", 0) > 0 else 0,
        ]

        # Crop parameters
        crop_params = params.get("crop", (0, 0, 0, 0))

        # Color jitter parameters
        cj = params.get("color_jitter", {})
        jitter_factors = [
            cj.get("brightness", 1.0),
            cj.get("contrast", 1.0),
            cj.get("saturation", 1.0),
            cj.get("hue", 0.0),
        ]
        jitter_order = cj.get("order", [0, 1, 2, 3])

        # Blur parameter
        blur_sigma = params.get("blur", 0)

        # Combine all parameters
        param_vector = (
            binary_flags
            + list(crop_params)
            + jitter_factors
            + jitter_order
            + [blur_sigma]
        )
        encoded.append(param_vector)

    # Convert to tensor and normalize
    encoded_tensor = torch.tensor(encoded, dtype=torch.float32)

    # In practice, you would normalize with precomputed mean and std
    # encoded_tensor = (encoded_tensor - mean) / std

    return encoded_tensor


# Parameter normalization constants
PARAM_MEAN = torch.tensor(
    [
        [
            6.8162e01,
            9.9199e01,
            2.6933e02,
            2.7457e02,
            4.9905e-01,
            8.0054e-01,
            1.1998e00,
            1.3994e00,
            1.6014e00,
            1.7995e00,
            1.0001e00,
            1.0000e00,
            1.0005e00,
            1.5640e-04,
            2.0018e-01,
            5.0030e-01,
            5.2507e-01,
        ]
    ]
)

PARAM_STD = torch.tensor(
    [
        [
            7.7370e01,
            9.8681e01,
            1.3686e02,
            1.4349e02,
            5.0000e-01,
            3.9959e-01,
            1.1661e00,
            1.0201e00,
            1.0201e00,
            1.1657e00,
            4.1347e-01,
            4.1323e-01,
            4.1349e-01,
            1.0333e-01,
            4.0013e-01,
            5.0000e-01,
            6.5251e-01,
        ]
    ]
)


def normalize_parameters(params, device=None):
    """
    Normalize transformation parameters using predefined mean and std values.

    Args:
        params: Tensor of transformation parameters
        device: Device to move the mean and std tensors to

    Returns:
        Normalized parameters
    """
    mean = PARAM_MEAN.to(device if device is not None else params.device)
    std = PARAM_STD.to(device if device is not None else params.device)
    return (params - mean) / std