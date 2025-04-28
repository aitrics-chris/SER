import torch
from .moco import concat_all_gather
import torch.nn as nn
import torch.optim as optim

def train_inv(args, images, inv_samples_num, equiv_samples_num, model, aug_equi, _mean, _std, _loss_list_inv, _loss_list_equiv):
                    
    ################# Inv: 2-augmentation (color) ###############
    with torch.no_grad():
        images[0] = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        images[1] = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
    ################# images[0], images[1] ################

    model.module.forward = model.module.forward_inv
    with torch.autocast(device_type="cuda"):
        loss_inv = model(images[0], images[1])
        loss_equiv = torch.tensor([0.0])
    loss = loss_inv

    if args.rank == 0:
        _loss_list_inv.append(loss_inv.item())

    return loss, loss_inv, loss_equiv, _loss_list_inv, _loss_list_equiv 

def train_erl(args, images, inv_samples_num, equiv_samples_num, model, aug_equi, _mean, _std, _loss_list_inv, _loss_list_equiv):
    ################# Inv: 2-augmentation (color) ###############         mean이 이상한거 보면 aug가 제대로 된건지 모르겠음
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

    # images[0] = images_inv_0
    # images[1] = images_inv_1
    # images.insert(1, images_equiv_0)
    # images.insert(3, images_equiv_1)

    # teacher_without_ddp.forward = teacher_without_ddp.hybrid
    # student.module.forward = student.module.hybrid
    model.module.forward = model.module.forward_equiv
    with torch.autocast(device_type="cuda"):
        loss_inv, feat_equiv0, feat_equiv1 = model(images_inv_0, images_inv_1, images_equiv_0, images_equiv_1)

    ################# Equiv: compute equivariance loss ####ß########################
    # if _binary: x, w, h, rot90_inv, rot90_another
    feat_equiv0_posttransformed = aug_equi.aug_equiv_feat(feat_equiv0, w1, h1, -num_rot90_pergpu0, num_rot90_pergpu1)
    # feat_equiv1_posttransformed = aug_equi.aug_equiv_feat(feat_equiv1, w0, h0, -num_rot90_pergpu1, num_rot90_pergpu0)
    
    feat_equiv0_posttransformed = torch.transpose(feat_equiv0_posttransformed, 1, 3).reshape(w1*h1*equiv_samples_num, 512)
    feat_equiv1 = torch.transpose(feat_equiv1, 1, 3).reshape(w1*h1*equiv_samples_num, 512)
    # feat_equiv1_posttransformed = torch.transpose(feat_equiv1_posttransformed, 1, 3).reshape(w0*h0*equiv_samples_num, 512)
    # feat_equiv0 = torch.transpose(feat_equiv0, 1, 3).reshape(w0*h0*equiv_samples_num, 512)

    feat_equiv0_posttransformed, n_list0 = concat_all_gather(feat_equiv0_posttransformed, 2)
    feat_equiv1, n_list1 = concat_all_gather(feat_equiv1, 2)
    # feat_equiv1_posttransformed, n_list0 = concat_all_gather(feat_equiv1_posttransformed, 2)
    # feat_equiv0, n_list1 = concat_all_gather(feat_equiv0, 2)

    feat_equiv0_posttransformed = torch.nn.functional.normalize(feat_equiv0_posttransformed, dim=1)
    feat_equiv1 = torch.nn.functional.normalize(feat_equiv1, dim=1)
    # feat_equiv1_posttransformed = torch.nn.functional.normalize(feat_equiv1_posttransformed, dim=1)
    # feat_equiv0 = torch.nn.functional.normalize(feat_equiv0, dim=1)

    equiv0 = torch.mm(feat_equiv0_posttransformed, torch.transpose(feat_equiv1, 0, 1)) / args.temperature_equiv
    # equiv1 = torch.mm(feat_equiv1_posttransformed, torch.transpose(feat_equiv0, 0, 1)) / args.temperature_equiv

    equiv0_numerator = torch.trace(equiv0)
    # equiv1_numerator = torch.trace(equiv1)

    equiv0 = torch.exp(equiv0)
    # equiv1 = torch.exp(equiv1)

    mask0 = torch.ones_like(equiv0, device=equiv0.device)
    # mask1 = torch.ones_like(equiv1, device=equiv0.device)
    
    for idx_start, idx_end in zip([0]+n_list0, n_list0):
        _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
        _mask0 = mask0[idx_start:idx_end, idx_start:idx_end]
        for idx_sample in range(equiv_samples_num):
            _mask0[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
    mask0.fill_diagonal_(1)

    # for idx_start, idx_end in zip([0]+n_list1, n_list1):
    #     _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
    #     _mask1 = mask1[idx_start:idx_end, idx_start:idx_end]
    #     for idx_sample in range(equiv_samples_num):
    #         _mask1[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
    # mask1.fill_diagonal_(1)

    equiv0_denominator = torch.sum(torch.log(torch.sum(equiv0 * mask0, dim=0, dtype=torch.float32)))
    # equiv1_denominator = torch.sum(torch.log(torch.sum(equiv1 * mask1, dim=0, dtype=torch.float32)))

    loss_equiv = (equiv0_denominator - equiv0_numerator) / float(equiv0.shape[0])
    loss = loss_inv + (loss_equiv * args.equiv_lambda)
    if args.rank == 0:
        _loss_list_inv.append(loss_inv.item())
        _loss_list_equiv.append(loss_equiv.item())

    return loss, loss_inv, loss_equiv, _loss_list_inv, _loss_list_equiv 


def train_essl(args, images, inv_samples_num, equiv_samples_num, model, aug_equi, _mean, _std, loss_list_inv, loss_list_equiv):

    with torch.no_grad():
        images_inv_0 = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        images_inv_1 = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
        images_rotate = aug_equi.aug_rotate(images[2]).sub_(_mean).div_(_std)

        nimages = images_rotate.shape[0]
        n_rot_images = 4 * nimages

        # rotate images all 4 ways at once
        rotated_images = torch.zeros([n_rot_images, images_rotate.shape[1], images_rotate.shape[2], images_rotate.shape[3]]).cuda(args.gpu, non_blocking=True)
        rotated_labels = torch.zeros([n_rot_images]).long().cuda(args.gpu, non_blocking=True)

        rotated_images[:nimages] = images_rotate
        # rotate 90
        rotated_images[nimages:2 * nimages] = images_rotate.flip(3).transpose(2, 3)
        rotated_labels[nimages:2 * nimages] = 1
        # rotate 180
        rotated_images[2 * nimages:3 * nimages] = images_rotate.flip(3).flip(2)
        rotated_labels[2 * nimages:3 * nimages] = 2
        # rotate 270
        rotated_images[3 * nimages:4 * nimages] = images_rotate.transpose(2, 3).flip(3)
        rotated_labels[3 * nimages:4 * nimages] = 3
    
    with torch.autocast(device_type="cuda"):
        loss_inv, logit_equiv = model(images_inv_0, images_inv_1, rotated_images)

        loss_equiv = torch.nn.functional.cross_entropy(logit_equiv, rotated_labels)        
        loss = loss_inv + (loss_equiv * args.equiv_lambda)

    if args.rank == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())
        
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv



def train_augself(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_stl(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_equimod(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args):
    pass
    # return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv




class BarlowTwins(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone

        if args.equiv_mode == 'erl':
            layers = []
            layers.append(nn.Conv2d(args.dim_equiv, 512, kernel_size=1, bias=False))
            layers.append(nn.GELU())
            # layers.append(nn.ReLU())
            layers.append(nn.Conv2d(512, 512, kernel_size=1, bias=False))
            # layers.append(nn.Conv2d(128, 128, kernel_size=1, bias=False))
            self.projector_equiv = nn.Sequential(*layers)
        elif args.equiv_mode == 'essl':
            self.projector_equiv = nn.Sequential(nn.Linear(384, 384),
                                                    nn.LayerNorm(384),
                                                    nn.ReLU(inplace=True),  # first layer
                                                    nn.Linear(384, 384),
                                                    nn.LayerNorm(384),
                                                    nn.ReLU(inplace=True),  # second layer
                                                    nn.Linear(384, 256),
                                                    nn.LayerNorm(256),
                                                    nn.Linear(256, 4))  # output layer    
            self.forward = self.forward_essl

        # projector
        sizes = [args.dim_inv] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward_inv(self, y1, y2):
        z1 = self.projector(self.backbone.forward_inv_(y1))
        z2 = self.projector(self.backbone.forward_inv_(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def forward_equiv(self, images_inv_0, images_inv_1, images_equiv_0, images_equiv_1):
        feat_inv0_0 = self.backbone.forward_inv_(images_inv_0)
        feat_inv1_0 = self.backbone.forward_inv_(images_inv_1)

        feat_inv0_1, feat_equiv0 = self.backbone.forward_equiv(images_equiv_0)
        feat_inv1_1, feat_equiv1 = self.backbone.forward_equiv(images_equiv_1)

        feat_inv0 = torch.cat([feat_inv0_0, feat_inv0_1], dim=0)
        feat_inv1 = torch.cat([feat_inv1_0, feat_inv1_1], dim=0)

        z1 = self.projector(feat_inv0)
        z2 = self.projector(feat_inv1)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_inv = on_diag + self.args.lambd * off_diag

        feat_equiv0 = self.projector_equiv(feat_equiv0)
        feat_equiv1 = self.projector_equiv(feat_equiv1)
        return loss_inv, feat_equiv0, feat_equiv1
    
    def forward_essl(self, y1,y2,x):
        """
        Input:
            x: rotated image
        Output:
            logit
        """

        z1 = self.projector(self.backbone.forward_baseline(y1))
        z2 = self.projector(self.backbone.forward_baseline(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

        logit_equiv = self.projector_equiv(self.backbone.forward_baseline(x))

        # compute features
        return loss, logit_equiv
    

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    if args.arch == 'resnet50':
        optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
        optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases
    elif args.arch == 'vit_small':
        optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
        # print(f'group num: {len(optimizer.param_groups)}')



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()