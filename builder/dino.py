import torch
import torch.distributed as dist
import math
import sys
from .utils import concat_all_gather
import torch.nn as nn
import torch.nn.functional as F
from builder.stl import STLTransformModule, load_mlp, build_transforms
from torchvision.transforms import v2


def train_inv(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    student.module.forward = student.module.inv
    teacher_without_ddp.forward = teacher_without_ddp.inv

    ################# Inv: 2-augmentation (color) ###############
    with torch.no_grad():
        images[0] = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        images[1] = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
    ################# images[0], images[1] ################

    # teacher and student forward passes + compute dino loss
    with torch.autocast(device_type="cuda"):
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)
    loss_inv = loss
    if dist.get_rank() == 0:
        loss_list_inv.append(loss_inv.item())
    loss_equiv = torch.tensor([0.0])

    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_erl(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    ################# Inv: 2-augmentation (color) ###############         mean이 이상한거 보면 aug가 제대로 된건지 모르겠음
    with torch.no_grad():
        images_inv_0 = aug_equi.aug_inv1(images[0][:inv_samples_num, ::]).sub_(_mean).div_(_std)
        images_inv_1 = aug_equi.aug_inv2(images[1][:inv_samples_num, ::]).sub_(_mean).div_(_std)
    ################# images_inv_0, images_inv_1 ################

    ############## Equiv: 기본 2-augmentation (color) ############
    # 차이점은 images[0] 에서 둘다 가져온다는 점: images[0]과 images[1]의 기본 aug policy가 동일하기 때문에 가능
    images_equiv_0 = aug_equi.aug_inv1(images[0][inv_samples_num:, ::]).sub_(_mean).div_(_std)
    images_equiv_1 = aug_equi.aug_inv2(images[0][inv_samples_num:, ::]).sub_(_mean).div_(_std)
    ################ images_equiv_0, images_equiv_1 ##############

    ############### Equiv: geometric aug parameters  #############
    w0, h0, degrees0, flips0, num_rot90_pergpu0 = aug_equi.get_params(args.equiv_scale, args.equiv_aspect_ratio, equiv_samples_num)
    w1, h1, _, _, num_rot90_pergpu1 = aug_equi.get_params(args.equiv_scale, args.equiv_aspect_ratio, equiv_samples_num)
    flips1 = torch.logical_not(flips0)
    degrees1 = torch.logical_not(degrees0)
    #################### w, h, degrees, flips ##################
    # print(f'gpu [{args.gpu}]: flips0 [{flips0}], degrees0[{degrees0}], flips1 [{flips1}], degrees1[{degrees1}],')
    ############### Equiv: geometric aug parameters #############
    images_equiv_0 = aug_equi.aug_equiv(images_equiv_0, w0*args.stride, h0*args.stride, degrees0, flips0, num_rot90_pergpu0)
    images_equiv_1 = aug_equi.aug_equiv(images_equiv_1, w1*args.stride, h1*args.stride, degrees1, flips1, num_rot90_pergpu1)
    ############## images_equiv_0, images_equiv_1 #################

    images[0] = images_inv_0
    images[1] = images_inv_1
    images.insert(1, images_equiv_0)
    images.insert(3, images_equiv_1)

    teacher_without_ddp.forward = teacher_without_ddp.hybrid
    student.module.forward = student.module.hybrid
    with torch.autocast(device_type="cuda"):
        teacher_cls, teacher_equiv = teacher(images)             
        student_cls, student_equiv = student(images)

    ################# Equiv: compute equivariance loss ####ß########################
    # if _binary: x, w, h, rot90_inv, rot90_another
    teacher_equiv[0] = aug_equi.aug_equiv_feat(teacher_equiv[0], w1, h1, -num_rot90_pergpu0, num_rot90_pergpu1) # to teacher_equiv[3]
    teacher_equiv[1] = aug_equi.aug_equiv_feat(teacher_equiv[1], w0, h0, -num_rot90_pergpu1, num_rot90_pergpu0) # to teacher_equiv[1], N x 512 x w x h
    # else:
    #     teacher_equiv[1] = aug_equi.aug_equiv(teacher_equiv[1], w1, h1, degrees0, flips0) # to student_equiv[3]
    #     teacher_equiv[3] = aug_equi.aug_equiv(teacher_equiv[3], w0, h0, degrees1, flips1) # to student_equiv[1]
    
    teacher_equiv0 = torch.transpose(teacher_equiv[0], 1, 3).reshape(w1*h1*equiv_samples_num, 512)
    teacher_equiv1 = torch.transpose(teacher_equiv[1], 1, 3).reshape(w0*h0*equiv_samples_num, 512)
    # print(f'before gather [{args.gpu}]: {teacher_equiv0.shape}, nan: {teacher_equiv0.isnan().sum()}')
    teacher_equiv0, n_list0 = concat_all_gather(teacher_equiv0, 2)
    teacher_equiv1, n_list1 = concat_all_gather(teacher_equiv1, 2)
    # print(f'after gather [{args.gpu}]: {teacher_equiv0.shape}, nan: {teacher_equiv0.isnan().sum()}')
    student_equiv0 = torch.transpose(student_equiv[0], 1, 3).reshape(w0*h0*equiv_samples_num, 512)
    student_equiv1 = torch.transpose(student_equiv[1], 1, 3).reshape(w1*h1*equiv_samples_num, 512)
    student_equiv0, _ = concat_all_gather(student_equiv0, 2)
    student_equiv1, _ = concat_all_gather(student_equiv1, 2)

    teacher_equiv0 = torch.nn.functional.normalize(teacher_equiv0, dim=1)
    teacher_equiv1 = torch.nn.functional.normalize(teacher_equiv1, dim=1)
    student_equiv0 = torch.nn.functional.normalize(student_equiv0, dim=1)
    student_equiv1 = torch.nn.functional.normalize(student_equiv1, dim=1)

    # equiv0 = torch.cat([teacher_equiv0, student_equiv1], dim=0)
    # equiv1 = torch.cat([teacher_equiv1, student_equiv0], dim=0)
    # print(f'step 1: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

    equiv0 = torch.mm(teacher_equiv0, torch.transpose(student_equiv1, 0, 1)) / args.temperature
    equiv1 = torch.mm(teacher_equiv1, torch.transpose(student_equiv0, 0, 1)) / args.temperature
    # print(f'step 2: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

    equiv0_numerator = torch.trace(equiv0)
    equiv1_numerator = torch.trace(equiv1)

    equiv0 = torch.exp(equiv0)
    equiv1 = torch.exp(equiv1)
    # print(f'step 3: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

    # print(f'equiv0 max: {equiv0.amax()}, min: {equiv0.amin()} ')
    # print(f'diagonel: {torch.diagonal(equiv0)}')

    mask0 = torch.ones_like(equiv0, device=equiv0.device)
    mask1 = torch.ones_like(equiv1, device=equiv0.device)
    
    for idx_start, idx_end in zip([0]+n_list0, n_list0):
        _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
        _mask0 = mask0[idx_start:idx_end, idx_start:idx_end]
        for idx_sample in range(equiv_samples_num):
            _mask0[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
    mask0.fill_diagonal_(1)

    for idx_start, idx_end in zip([0]+n_list1, n_list1):
        _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
        _mask1 = mask1[idx_start:idx_end, idx_start:idx_end]
        for idx_sample in range(equiv_samples_num):
            _mask1[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
    mask1.fill_diagonal_(1)

    # print(f'gpu[{args.gpu}]: equiv0: {equiv0.shape}, mask0: {mask0.shape},  equiv1: {equiv1.shape}, mask1: {mask1.shape}')
    equiv0_denominator = torch.sum(torch.log(torch.sum(equiv0 * mask0, dim=0, dtype=torch.float32)))
    equiv1_denominator = torch.sum(torch.log(torch.sum(equiv1 * mask1, dim=0, dtype=torch.float32)))
    # print(f'equiv0_denominator: {equiv0_denominator}')
    # print(f'equiv0_numerator: {equiv0_numerator}')

    loss_equiv0 = (equiv0_denominator - equiv0_numerator) / float(equiv0.shape[0])
    loss_equiv1 = (equiv1_denominator - equiv1_numerator) / float(equiv1.shape[0])

    with torch.autocast(device_type="cuda"):
        loss_inv = dino_loss(student_cls, teacher_cls, epoch)
        loss_equiv = 0.5*(loss_equiv0 + loss_equiv1)
        loss = loss_inv + (loss_equiv * args.equiv_lambda)

    if dist.get_rank() == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())

    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_essl(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_stl(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    with torch.no_grad():
        # Apply color augmentations using aug_equi
        aug_images_0 = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        aug_images_1 = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
        
        # Apply geometric augmentations
        w, h, degrees, flips, num_rot90 = aug_equi.get_params(args.equiv_scale, args.equiv_aspect_ratio, images[0].shape[0])
        
        # Create transformed versions using the same geometric parameters
        trans_images_0 = aug_equi.aug_equiv(
            aug_images_0, 
            w*args.stride, h*args.stride, 
            degrees, flips, num_rot90
        )
        trans_images_1 = aug_equi.aug_equiv(
            aug_images_1, 
            w*args.stride, h*args.stride, 
            degrees, flips, num_rot90
        )
    
    with torch.autocast(device_type="cuda"):
        # Get teacher output on augmented views
        teacher_output = teacher([aug_images_0, aug_images_1])
        
        # Forward pass through student with all views
        stl_outputs = student.module.stl_forward(
            aug_images_0, aug_images_1,  # Augmented views
            trans_images_0, trans_images_1  # Transformed views
        )
        
        # Compute invariance loss
        inv_features = stl_outputs['inv_features']
        student_output = torch.cat([inv_features[0], inv_features[1]])
        loss_inv = dino_loss(student_output, teacher_output, epoch)
        
        # Compute STL-specific losses
        stl_losses = student.module.compute_stl_losses(stl_outputs)
        loss_equiv = stl_losses['equi']
        loss_trans = stl_losses['trans']
        
        # Combine losses with weights
        loss = (loss_inv * args.stl_inv_weight + 
                loss_equiv * args.stl_equi_weight + 
                loss_trans * args.stl_trans_weight)
    
    # Update loss tracking
    if dist.get_rank() == 0:
        loss_list_inv.append(loss_inv.item())
        loss_list_equiv.append(loss_equiv.item() + loss_trans.item())
    
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_equimod(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    total_images = len(images[0]) if isinstance(images, list) and len(images) > 0 else 0

    if equiv_samples_num <= 0:
        inv_samples_num = total_images
        equiv_samples_num = 0
    
    # Process invariance images
    with torch.no_grad():
        aug_images_0 = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        aug_images_1 = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
    
    # Calculate invariance loss
    with torch.autocast(device_type="cuda"):
        teacher_output = teacher([aug_images_0, aug_images_1])
        student_output = student([aug_images_0, aug_images_1])
        loss_inv = dino_loss(student_output, teacher_output, epoch)
    
    # Initialize equivariance loss
    loss_equiv = torch.tensor(0.0, device=loss_inv.device)
    
    # Process equivariance images if available and enabled
    if args.equimod_equi_weight > 0 and equiv_samples_num > 0:
        # Create transformed images for equivariance
        with torch.no_grad():
            # Get a subset of images for equivariance
            equiv_img = images[0][:equiv_samples_num]
            
            # Apply augmentations
            aug_equiv_img = aug_equi.aug_inv1(equiv_img).sub_(_mean).div_(_std)
            
            # Generate transformation parameters
            w, h, degrees, flips, num_rot90 = aug_equi.get_params(
                args.equiv_scale, args.equiv_aspect_ratio, equiv_samples_num
            )
            
            trans_equiv_img = aug_equi.aug_equiv(
                aug_equiv_img, w*args.stride, h*args.stride, degrees, flips, num_rot90
            )
        
        with torch.autocast(device_type="cuda"):
            y0, features0 = student(aug_equiv_img, return_features=True)
            
            with torch.no_grad():
                yt, features_t = teacher(trans_equiv_img, return_features=True)
            
            if features0 and features_t:
                f0 = F.normalize(features0[0], dim=1)
                ft = F.normalize(features_t[0], dim=1)
                
                loss_equiv = 1.0 - torch.mean(torch.sum(f0 * ft, dim=1))
    
    # Combine losses
    loss = args.equimod_inv_weight * loss_inv
    if args.equimod_equi_weight > 0:
        loss += args.equimod_equi_weight * loss_equiv
    
    # Update loss lists for tracking
    if dist.get_rank() == 0:
        loss_list_inv.append(loss_inv.item())
        loss_list_equiv.append(loss_equiv.item() if isinstance(loss_equiv, torch.Tensor) else 0)
    
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_augself(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, args, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
            
        self.backbone = backbone
        self.head = head
        self.args = args

        # Determine embedding dimension based on architecture
        if args.arch == 'vit_small':
            embed_dim = backbone.embed_dim
        elif args.arch == 'resnet50':
            embed_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        # Initialize different projectors based on equivariance mode
        if args.equiv_mode == 'erl':
            layers = []
            layers.append(nn.Conv2d(384, 512, kernel_size=1, bias=False))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(512, 512, kernel_size=1, bias=False))
            self.projector_equiv = nn.Sequential(*layers)
            
        elif args.equiv_mode == 'stl':
            # Use the load_mlp function from stl_modules
            self.projector_equiv = load_mlp(embed_dim, args.stl_projector)
            
            # Create the STL transformation module
            self.transform_module = STLTransformModule(embed_dim, args)
            
            # Store loss weights
            self.inv_weight = args.stl_inv_weight
            self.equi_weight = args.stl_equi_weight
            self.trans_weight = args.stl_trans_weight

            self.inv_projector = load_mlp(embed_dim, args.stl_projector)
            self.equi_projector = load_mlp(embed_dim, args.stl_projector)
            self.trans_projector = load_mlp(embed_dim, args.stl_trans_projector)

            # Current simplified version
            self.trans_backbone = load_mlp(2 * embed_dim, args.stl_trans_backbone)

        elif args.equiv_mode in ['essl', 'equimod', 'augself']:
            # For other equivariance modes, create a simple projector
            self.projector_equiv = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128)
            )

    def set_projector_equiv(self, projector_equiv):
        """Set the equivariance projector (used to sync teacher with student)"""
        self.projector_equiv = projector_equiv
        
        # If using STL, also sync the transform module's projectors
        if hasattr(self, 'transform_module'):
            self.transform_module.equi_projector = projector_equiv
            
    def forward(self, x, mask=None, return_features=False):
        """
        Forward pass for standard DINO processing
        """
        if not isinstance(x, list):
            x = [x]
            
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        
        start_idx = 0
        backbone_features = []
        
        for end_idx in idx_crops:
            # Handle different input resolutions
            # Use forward_baseline or forward_inv_ if available
            input_tensor = torch.cat(x[start_idx: end_idx])
            
            if hasattr(self.backbone, 'forward_baseline'):
                _out = self.backbone.forward_baseline(input_tensor)
            elif hasattr(self.backbone, 'forward_inv_'):
                _out = self.backbone.forward_inv_(input_tensor)
            else:
                # Fallback to direct call
                _out = self.backbone(input_tensor)
            
            if isinstance(_out, tuple):
                _out = _out[0]  # Some backbones return (features, attention_maps)
            
            # Store features for equivariance if needed
            if return_features:
                backbone_features.append(_out)
            
            # Apply head to get final output
            _out = self.head(_out)
            
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        
        if return_features:
            return output, backbone_features
        else:
            return output
            
    def inv(self, x):
        """Forward pass for invariance only"""
        feat_inv = []
        for _x in x:
            _cls = self.backbone.forward_inv_(_x)            
            # accumulate outputs
            feat_inv.append(_cls)
        with torch.autocast(device_type="cuda"):
            feat_inv = self.head(torch.cat(feat_inv))
        return feat_inv
 
    def hybrid(self, x, idx_equiv=[1,3], idx_inv=[0,2]):
        """Forward pass for hybrid invariance and equivariance"""
        feat_inv = []
        feat_equiv = []
        for _idx in range(4):
            if _idx in idx_inv:
                _cls = self.backbone.forward_inv_(x[_idx])
                feat_inv.append(_cls)
            elif _idx in idx_equiv:
                _cls, _equiv = self.backbone.forward_equiv(x[_idx])
                feat_inv.append(_cls)
                feat_equiv.append(_equiv)
                    
        with torch.autocast(device_type="cuda"):
            feat_inv = self.head(torch.cat(feat_inv))
            for i in range(len(feat_equiv)):
                feat_equiv[i] = self.projector_equiv(feat_equiv[i])
        return feat_inv, feat_equiv
    
    def stl_forward(self, x1, x2, t1, t2):
        """
        Forward pass for STL mode
        
        Args:
            x1, x2: Original input images
            t1, t2: Transformed versions of the input images
        
        Returns:
            Dictionary containing all necessary outputs for STL loss computation
        """
        feat_x1 = self.backbone.forward_baseline(x1)
        feat_x2 = self.backbone.forward_baseline(x2)
        feat_t1 = self.backbone.forward_baseline(t1)
        feat_t2 = self.backbone.forward_baseline(t2)
        
        # Handle tuple outputs if needed
        if isinstance(feat_x1, tuple):
            feat_x1 = feat_x1[0]
        if isinstance(feat_x2, tuple):
            feat_x2 = feat_x2[0]
        if isinstance(feat_t1, tuple):
            feat_t1 = feat_t1[0]
        if isinstance(feat_t2, tuple):
            feat_t2 = feat_t2[0]
        
        proj_x1 = self.head(feat_x1)
        proj_x2 = self.head(feat_x2)
        proj_t1 = self.head(feat_t1)
        proj_t2 = self.head(feat_t2)
        
        equi_x1 = self.transform_module.project_features(feat_x1)
        equi_x2 = self.transform_module.project_features(feat_x2)
        equi_t1 = self.transform_module.project_features(feat_t1)
        equi_t2 = self.transform_module.project_features(feat_t2)
        
        # Compute transformation representations
        trans_12 = self.trans_backbone(torch.cat([feat_x1, feat_t1], dim=-1))
        trans_21 = self.trans_backbone(torch.cat([feat_x2, feat_t2], dim=-1))

        # Apply transformations (corrected order)
        pred_t1 = self.transform_module.apply_transform(feat_x1, trans_21)
        pred_t2 = self.transform_module.apply_transform(feat_x2, trans_12)

        return {
            'inv_features': [proj_x1, proj_x2, proj_t1, proj_t2],
            'equi_features': [equi_x1, equi_x2, equi_t1, equi_t2],
            'trans_features': [trans_12, trans_21],
            'pred_features': [pred_t1, pred_t2],
            'target_features': [equi_t2, equi_t1]
        }
    
    def compute_stl_losses(self, outputs):
        trans_features = outputs['trans_features']
        pred_features = outputs['pred_features']
        target_features = outputs['target_features']
        
        equi_loss_1 = self.transform_module.info_nce_loss(pred_features[0], target_features[0])
        equi_loss_2 = self.transform_module.info_nce_loss(pred_features[1], target_features[1])
        equi_loss = (equi_loss_1 + equi_loss_2) / 2.0
        
        trans_loss = self.transform_module.info_nce_loss(trans_features[0], trans_features[1])
        
        return {
            'equi': equi_loss,
            'trans': trans_loss
        }
    
    def get_intermediate_layers(self, x, n=1):
        """
        Get intermediate layers from the backbone
        Used for equivariance loss computation
        """
        if hasattr(self.backbone, 'get_intermediate_layers'):
            return self.backbone.get_intermediate_layers(x, n)
        else:
            # Fallback for backbones without this method
            return [self.backbone(x)]