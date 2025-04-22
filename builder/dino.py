import torch
import torch.distributed as dist
import math
import sys
from .utils import concat_all_gather
import torch.nn as nn

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
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_equimod(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, loss_list_inv, loss_list_equiv, inv_samples_num, equiv_samples_num, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


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
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

        if args.equiv_mode == 'erl':
            layers = []
            layers.append(nn.Conv2d(384, 512, kernel_size=1, bias=False))
            layers.append(nn.GELU())
            # layers.append(nn.ReLU())
            layers.append(nn.Conv2d(512, 512, kernel_size=1, bias=False))
            # layers.append(nn.Conv2d(128, 128, kernel_size=1, bias=False))
            self.projector_equiv = nn.Sequential(*layers)
    
    def set_projector_equiv(self, projector_equiv):
        self.projector_equiv = projector_equiv
            
    def inv(self, x):
        feat_inv = []
        for _x in x:
            _cls = self.backbone.forward_inv_(_x)            
            # accumulate outputs
            feat_inv.append(_cls)
        with torch.autocast(device_type="cuda"):
            feat_inv = self.head(torch.cat(feat_inv))
        return feat_inv
 

    def hybrid(self, x, idx_equiv=[1,3], idx_inv=[0,2]):
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