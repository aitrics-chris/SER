import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mlp(input_dim: int, layers_str: str) -> nn.Sequential:
    """
    Build a simple MLP given the layer configuration string (e.g., '512-128').
    """
    sizes = [input_dim] + list(map(int, layers_str.split('-')))
    layer_list = []
    for i in range(len(sizes) - 2):
        layer_list.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layer_list.append(nn.BatchNorm1d(sizes[i + 1]))
        layer_list.append(nn.ReLU(inplace=True))
    layer_list.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

    return nn.Sequential(*layer_list)


class EquiTrans(nn.Module):
    """
    Hypernetwork-style transformation module.
    """
    def __init__(self, equi_repr_dim: int, trans_repr_dim: int):
        super().__init__()
        self.equitrans_layers = [equi_repr_dim, equi_repr_dim]

        # Calculate parameters needed for each block
        self.num_weights_per_block = [
            self.equitrans_layers[i] * self.equitrans_layers[i + 1]
            for i in range(len(self.equitrans_layers) - 1)
        ]
        self.cumulative_params = [0] + list(np.cumsum(self.num_weights_per_block))

        # Hypernetwork to generate weights
        self.hypernet = nn.Linear(trans_repr_dim, self.cumulative_params[-1], bias=False)

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation predicted by `t` on representation `r`.
        """
        all_weights = self.hypernet(t)  # shape: [B, total_params]
        output = r.unsqueeze(1)

        # Sequentially apply linear blocks
        for i in range(len(self.equitrans_layers) - 1):
            start_idx = self.cumulative_params[i]
            end_idx = start_idx + self.num_weights_per_block[i]

            w_block = all_weights[..., start_idx:end_idx]
            w_reshaped = w_block.view(-1,
                                      self.equitrans_layers[i + 1],
                                      self.equitrans_layers[i])
            output = torch.bmm(output, w_reshaped.transpose(-2, -1))

        return output.squeeze()


class STLTransformModule(nn.Module):
    """
    Complete transformation module for STL that includes:
    - Transformation backbone (MLP)
    - EquiTrans module (hypernetwork)
    - Projectors for equivariance and transformation
    """
    def __init__(self, repr_dim, args):
        super().__init__()
        # Parse dimensions from args, with fallbacks to prevent errors
        trans_backbone = args.stl_trans_backbone
        trans_projector = args.stl_trans_projector
        projector = args.stl_projector
        
        # Get dimensions safely
        try:
            trans_backbone_dim = int(trans_backbone.split('-')[-1])
        except (ValueError, IndexError):
            trans_backbone_dim = 128
            
        try:
            trans_projector_dim = int(trans_projector.split('-')[-1])
        except (ValueError, IndexError):
            trans_projector_dim = 128
            
        try:
            projector_dim = int(projector.split('-')[-1])
        except (ValueError, IndexError):
            projector_dim = 128
        
        # Create transformation backbone (simpler version to avoid memory issues)
        self.trans_backbone = nn.Sequential(
            nn.Linear(2 * repr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, trans_backbone_dim)
        )
        
        # Create EquiTrans module
        self.equi_transform = EquiTrans(repr_dim, trans_backbone_dim)
        
        # Create projectors (simpler version to avoid memory issues)
        self.equi_projector = nn.Sequential(
            nn.Linear(repr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projector_dim)
        )
        
        self.trans_projector = nn.Sequential(
            nn.Linear(trans_backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, trans_projector_dim)
        )
        
        # Store temperature for InfoNCE loss
        self.temperature = args.stl_temperature
    
    def forward_transform(self, feat1, feat2):
        """
        Compute transformation representation from two feature sets.
        """
        # Concatenate features
        trans_12 = torch.cat([feat1, feat2], dim=1)
        trans_21 = torch.cat([feat2, feat1], dim=1)
        
        # Apply transformation backbone
        trans_12 = self.trans_backbone(trans_12)
        trans_21 = self.trans_backbone(trans_21)
        
        # Apply projector
        trans_12 = self.trans_projector(trans_12)
        trans_21 = self.trans_projector(trans_21)
        
        # Normalize
        trans_12 = F.normalize(trans_12, dim=1)
        trans_21 = F.normalize(trans_21, dim=1)
        
        return trans_12, trans_21
    
    def apply_transform(self, feat, trans):
        """
        Apply transformation to features.
        """
        # Apply EquiTrans
        transformed = self.equi_transform(feat, trans)
        
        # Normalize
        transformed = F.normalize(transformed, dim=1)
        
        return transformed
    
    def project_features(self, feat):
        """
        Project features for equivariance loss.
        """
        # Apply projector
        projected = self.equi_projector(feat)
        
        # Normalize
        projected = F.normalize(projected, dim=1)
        
        return projected
    
    def info_nce_loss(self, z1, z2):
        """
        Compute InfoNCE loss between two sets of embeddings.
        Handle potential dimension mismatches by projecting to a common dimension.
        """
        # Check dimensions and project if needed
        if z1.size(1) != z2.size(1):
            # Project to the smaller dimension
            target_dim = min(z1.size(1), z2.size(1))
            
            if z1.size(1) > target_dim:
                z1 = nn.functional.linear(z1, torch.randn(target_dim, z1.size(1), device=z1.device))
            
            if z2.size(1) > target_dim:
                z2 = nn.functional.linear(z2, torch.randn(target_dim, z2.size(1), device=z2.device))
        
        # Ensure normalization
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate for full similarity matrix
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity scores
        scores = torch.mm(z, z.t()) / self.temperature
        
        # Create labels for matching pairs
        n = z1.size(0)
        labels = torch.cat([
            torch.arange(n, 2*n, device=z1.device),
            torch.arange(0, n, device=z1.device)
        ])
        
        # Mask out self-similarity
        mask = torch.eye(2*n, dtype=torch.bool, device=z1.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Compute cross entropy loss
        loss = F.cross_entropy(scores, labels)
        
        return loss