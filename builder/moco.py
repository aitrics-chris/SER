import torch
import torch.nn as nn
import numpy as np
from .utils import load_mlp_augself, load_mlp_stl, EquiTrans


def train_inv(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    
    model.module.forward = model.module.inv

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        images_inv_0 = aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std)
        images_inv_1 = aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std)
    
    with torch.autocast(device_type="cuda"):
        loss = model(images_inv_0, images_inv_1, moco_m)
        loss_inv = loss
        loss_equiv = torch.tensor([0.0])
    if args.rank == 0:
        loss_list_inv.append(loss_inv.item())
        
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_erl_inv(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    model.module.forward = model.module.equiv

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

    ################# Inv: 2-augmentation (color) ###############
    with torch.no_grad():
        images_inv_0 = aug_equi.aug_inv1(images[0][:inv_samples_num, ::]).sub_(_mean).div_(_std)
        images_inv_1 = aug_equi.aug_inv2(images[1][:inv_samples_num, ::]).sub_(_mean).div_(_std)
    ################# images_inv_0, images_inv_1 ################

    ############## Equiv: 기본 2-augmentation (color) ############
    # 차이점은 images[0] 에서 둘다 가져온다는 점
    images_equiv_0 = aug_equi.aug_inv1(images[0][inv_samples_num:, ::]).sub_(_mean).div_(_std)
    images_equiv_1 = aug_equi.aug_inv2(images[0][inv_samples_num:, ::]).sub_(_mean).div_(_std)
    ################ images_equiv_0, images_equiv_1 ##############
    
    with torch.autocast(device_type="cuda"):
        loss, loss_inv, loss_equiv = model(images_inv_0, images_inv_1, images_equiv_0, images_equiv_1, aug_equi, moco_m, args.equiv_lambda)
    if args.rank == 0:
        loss_list_inv.append(loss_inv.item())
        loss_list_equiv.append(loss_equiv.item())

    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv

def train_inv_essl(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    model.module.forward = model.module.inv_essl

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

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
        loss_inv = model(images_inv_0, images_inv_1, rotated_images.chunk(4), moco_m)
        loss_equiv = torch.tensor([0.0])

    if args.rank == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())
        
    return loss_inv, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv

def train_erl_local4(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    model.module.forward = model.module.erl_local4

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

    image_global = []
    image_local = []
    ################# Inv: 2-augmentation (color) ###############
    with torch.no_grad():
        image_global.append(aug_equi.aug_inv1(images[0]).sub_(_mean).div_(_std))
        image_global.append(aug_equi.aug_inv2(images[1]).sub_(_mean).div_(_std))
    ################# images_inv_0, images_inv_1 ################

    ############## Equiv: 기본 2-augmentation (color) ############
    for _ in range(2):
        image_local.append(aug_equi.aug_inv1(images[2]).sub_(_mean).div_(_std))
    for _ in range(2):
        image_local.append(aug_equi.aug_inv1(images[3]).sub_(_mean).div_(_std))
    ################ image_local with length 4 ##############
    
    with torch.autocast(device_type="cuda"):
        loss, loss_inv, loss_equiv = model(image_global, image_local, aug_equi, moco_m, args.equiv_lambda)
    if args.rank == 0:
        loss_list_inv.append(loss_inv.item())
        loss_list_equiv.append(loss_equiv.item())

    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_essl(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    model.module.forward = model.module.inv

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

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
        loss_inv = model(images_inv_0, images_inv_1, moco_m)

        model.module.forward = model.module.forward_essl
        logit_equiv = model(rotated_images)
        loss_equiv = torch.nn.functional.cross_entropy(logit_equiv, rotated_labels)        
        loss = loss_inv + (loss_equiv * args.equiv_lambda)

    if args.rank == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())
        
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_stl(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    assert len(images) == 2
    model.module.forward = model.module.stl

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        images_0 = images[0].sub_(_mean).div_(_std)
        images_1 = images[1].sub_(_mean).div_(_std)
    
    
    with torch.autocast(device_type="cuda"):
        loss, loss_inv, loss_equiv = model(images_0, images_1, moco_m, args.equiv_lambda)

    if args.rank == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())
        
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_equimod(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):
    assert len(images) == 5
    model.module.forward = model.module.equimod

    for i in range(len(images)):
        images[i] = images[i].cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        images_transform_no = images[0].sub_(_mean).div_(_std)
        images_transform_0 = images[1].sub_(_mean).div_(_std)
        images_transform_1 = images[3].sub_(_mean).div_(_std)
    
    # x = torch.cat([images_transform_no, images_transform_0, images_transform_1], dim=0)
    p = torch.cat([images[2], images[4]], dim=0)
    p = (p - etc['p_mean'])/etc['p_std']
    
    with torch.autocast(device_type="cuda"):
        loss, loss_inv, loss_equiv = model(images_transform_no, images_transform_0, images_transform_1, moco_m, p, args.equiv_lambda)

    if args.rank == 0:
        loss_list_equiv.append(loss_equiv.item())
        loss_list_inv.append(loss_inv.item())
        
    return loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv


def train_augself(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, loss_list_inv, loss_list_equiv, args, etc):        
    raise NotImplementedError('Use augself branch instead!!')



class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, args, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.args = args

        # build encoders
        self.base_encoder = base_encoder(args=args, drop_path_rate=0.0, ssl_type='moco')
        self.momentum_encoder = base_encoder(args=args, drop_path_rate=0.0, ssl_type='moco')

        self._build_projector_and_predictor_mlps(dim, mlp_dim) # both heads changes

        if args.equiv_mode.split('_')[0] == 'erl':
            layers = []
            layers.append(nn.Conv2d(384 if args.arch.startswith('vit') else 2048, 512, kernel_size=1, bias=False))
            layers.append(nn.GELU())
            # layers.append(nn.ReLU())
            layers.append(nn.Conv2d(512, 512, kernel_size=1, bias=False))
            # layers.append(nn.Conv2d(128, 128, kernel_size=1, bias=False))
            self.projector_equiv = nn.Sequential(*layers)        
        elif args.equiv_mode.split('_')[0] == 'essl':
            self.projector_equiv = nn.Sequential(nn.Linear(384, 384),
                                                    nn.LayerNorm(384),
                                                    nn.ReLU(inplace=True),  # first layer
                                                    nn.Linear(384, 384),
                                                    nn.LayerNorm(384),
                                                    nn.ReLU(inplace=True),  # second layer
                                                    nn.Linear(384, 256),
                                                    nn.LayerNorm(256),
                                                    nn.Linear(256, 4))  # output layer    
        elif args.equiv_mode.split('_')[0] == 'augself':
            self.projector_equiv = load_mlp_augself(n_in=384*2, n_hidden=512, n_out=args.moco_dim, num_layers=3, last_bn=False) # args.moco_dim is 256 by default
        elif args.equiv_mode.split('_')[0] == 'equimod':
            y_dim = 128
            p_dim = 18
            self.projector_equiv = nn.Sequential(
                                                    torch.nn.Linear(384, 384, bias=False),
                                                    torch.nn.BatchNorm1d(384),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(384, 384, bias=False),
                                                    torch.nn.BatchNorm1d(384),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(384, y_dim, bias=False), # self.y_dim = 128
                                                    torch.nn.BatchNorm1d(y_dim, affine=False)
                                                )
            self.proj_head_t = torch.nn.Sequential(
                                                    torch.nn.Linear(p_dim, y_dim, bias=False), # p_dim = 15 
                                                    torch.nn.BatchNorm1d(y_dim)
                                                )

            self.predictor_eq = torch.nn.Sequential(
                                                    torch.nn.Linear(y_dim+y_dim, y_dim, bias=False),
                                                    torch.nn.BatchNorm1d(y_dim, affine=False)
                                                )
        elif args.equiv_mode.split('_')[0] == 'stl':
            # Transform backbone
            trans_repr_dim = 128
            repr_dim = 384
            projector = '512-128'
            trans_backbone = '128-128'
            trans_projector = '128-128'
            self.trans_backbone = load_mlp_stl(2 * repr_dim, trans_backbone)
            self.equi_transform = EquiTrans(repr_dim, trans_repr_dim)

            # Projectors
            # self.inv_projector = load_mlp_stl(repr_dim, projector)
            self.projector_equiv = load_mlp_stl(repr_dim, projector)
            self.trans_projector = load_mlp_stl(trans_repr_dim, trans_projector)

            # Index helpers for rearranging
            batch_size = args.batch_size
            self.even_idxs = 2 * torch.arange(batch_size // 2)
            self.odd_idxs = 2 * torch.arange(batch_size // 2) + 1
            self.shifted_idxs = torch.flatten(torch.stack([self.odd_idxs, self.even_idxs], dim=1))


        else:
            raise ValueError(f'Invalid equivarience mode: {args.equiv_mode}')


        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        # if args.equiv_mode == 'erl':
        #     self.forward = self.equiv
        # else:
        #     self.forward = self.inv

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
            # if param_b.shape == torch.Size([4096, 384]):
            #     print(f'whoa')

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    
    def contrastive_loss_temp(self, q, k, temp):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / temp
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * temp)

    def forward_inv(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.base_encoder.forward_inv(x1)
        q2 = self.base_encoder.forward_inv(x2)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder.forward_inv(x1)
            k2 = self.momentum_encoder.forward_inv(x2)

        return q1, q2, k1, k2

    def forward_essl(self, x):
        """
        Input:
            x: rotated image
        Output:
            logit
        """

        # compute features
        return self.projector_equiv(self.base_encoder.forward_inv(x))

    
    def loss_inv(self, q1, q2, k1, k2):
        """
        Input:
            q1: first views of images -> encoder -> CLS
            q2: second views of images -> encoder -> CLS
            k1: first views of images -> momentum encoder -> CLS
            k2: second views of images -> momentum encoder -> CLS
        Output:
            semantic invariance loss
        """

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    

    def forward_equiv(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1, equiv_q1 = self.base_encoder.forward_equiv(x1)
        q2, equiv_q2 = self.base_encoder.forward_equiv(x2)

        equiv_q1 = self.projector_equiv(equiv_q1)
        equiv_q2 = self.projector_equiv(equiv_q2)

        with torch.no_grad():  # no gradient
            # self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1, equiv_k1 = self.momentum_encoder.forward_equiv(x1)
            k2, equiv_k2 = self.momentum_encoder.forward_equiv(x2)
        
        # Controversial
        equiv_k1 = self.projector_equiv(equiv_k1)
        equiv_k2 = self.projector_equiv(equiv_k2)

        return q1, q2, k1, k2, equiv_q1, equiv_q2, equiv_k1, equiv_k2
    
    def equimod(self, images_transform_no, images_inv_0, images_inv_1, moco_m, params, equiv_lambda):
        
        _cls_q0, _cls_q1, cls_k0, cls_k1 = self.forward_inv(images_inv_0, images_inv_1, moco_m)
        cls_q0 = self.predictor(self.base_encoder.head(_cls_q0))
        cls_q1 = self.predictor(self.base_encoder.head(_cls_q1))
        cls_k0 = self.momentum_encoder.head(cls_k0)
        cls_k1 = self.momentum_encoder.head(cls_k1)
        loss_inv = self.loss_inv(cls_q0, cls_q1, cls_k0, cls_k1)

        _cls_q0 = self.projector_equiv(_cls_q0)
        _cls_q1 = self.projector_equiv(_cls_q1)
        yt = torch.cat([_cls_q0, _cls_q1], dim=0)
        y0 = self.projector_equiv(self.base_encoder.forward_inv(images_transform_no))
        y0 = torch.cat([y0, y0], dim=0)

        p = self.proj_head_t(params)
        yt_hat = self.predictor_eq(torch.cat([y0, p], dim=1))

        loss_equiv = 0.5*(self.contrastive_loss_temp(yt, yt_hat, self.args.temperature_equiv) + self.contrastive_loss_temp(yt_hat, yt, self.args.temperature_equiv))
        loss = loss_inv + (loss_equiv * equiv_lambda)
        return loss, loss_inv, loss_equiv


    def inv(self, images_inv_0, images_inv_1, moco_m):
        
        cls_q0, cls_q1, cls_k0, cls_k1 = self.forward_inv(images_inv_0, images_inv_1, moco_m)
        cls_q0 = self.predictor(self.base_encoder.head(cls_q0))
        cls_q1 = self.predictor(self.base_encoder.head(cls_q1))
        cls_k0 = self.momentum_encoder.head(cls_k0)
        cls_k1 = self.momentum_encoder.head(cls_k1)
        return self.loss_inv(cls_q0, cls_q1, cls_k0, cls_k1)
    
    def stl(self, images_0, images_1, moco_m, equiv_lambda):
        _cls_q0, _cls_q1, cls_k0, cls_k1 = self.forward_inv(images_0, images_1, moco_m)
        cls_q0 = self.predictor(self.base_encoder.head(_cls_q0))
        cls_q1 = self.predictor(self.base_encoder.head(_cls_q1))
        cls_k0 = self.momentum_encoder.head(cls_k0)
        cls_k1 = self.momentum_encoder.head(cls_k1)
        loss_inv = self.loss_inv(cls_q0, cls_q1, cls_k0, cls_k1)

        # Transformation backbone
        y_trans122 = self.trans_backbone(torch.cat([_cls_q0, _cls_q1], dim=-1))
        y_trans221 = self.trans_backbone(torch.cat([_cls_q1, _cls_q0], dim=-1))

        # Split into y_trans1, y_trans2
        y_trans1 = torch.cat([y_trans122[self.even_idxs], y_trans221[self.even_idxs]], dim=0)
        y_trans2 = torch.cat([y_trans122[self.odd_idxs], y_trans221[self.odd_idxs]], dim=0)

        # Equivariant transform
        y_pred1 = self.equi_transform(_cls_q1, y_trans221[self.shifted_idxs])
        y_pred2 = self.equi_transform(_cls_q0, y_trans122[self.shifted_idxs])

        # Equivariance loss
        z_equi = torch.cat([self.projector_equiv(_cls_q0), self.projector_equiv(_cls_q1)], dim=0)
        z_equi_pred = torch.cat([self.projector_equiv(y_pred1), self.projector_equiv(y_pred2)], dim=0)
        equi_loss = self.contrastive_loss(z_equi, z_equi_pred) + self.contrastive_loss(z_equi_pred, z_equi) # 0.2 for both moco_t and stl default

        # Transformation loss
        z_trans1, z_trans2 = self.trans_projector(y_trans1), self.trans_projector(y_trans2)
        trans_loss = self.contrastive_loss(z_trans1, z_trans2) + self.contrastive_loss(z_trans2, z_trans1)

        loss = loss_inv + (equi_loss * self.args.stl_lambda_equi) + (trans_loss * self.args.stl_lambda_trans)
        return loss, loss_inv, equi_loss




    def inv_essl(self, images_inv_0, images_inv_1, images_localview, moco_m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            images_small: list of length 4, rotated images
            m: moco momentum
        Output:
            loss
        """
        equiv_samples_num = images_localview[0].shape[0]

        ############### compute semantic inveriance via contrastive loss #############
        cls_q = [-99, -99, -99, -99, -99, -99]    
        cls_k = [-99, -99]     
        
        with torch.autocast(device_type="cuda"):
            # both encoder for global view, need to go through head and predictor
            # cls_q[0], cls_q[1], cls_k[0], cls_k[1] = self.forward_inv(images_inv_0, images_inv_1, moco_m) # momentum for k
            cls_q[0] = self.base_encoder.forward_baseline(images_inv_0)
            cls_q[1] = self.base_encoder.forward_baseline(images_inv_1)

            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(moco_m)  # update the momentum encoder

            # compute momentum features as targets
            cls_k[0] = self.momentum_encoder.forward_inv(images_inv_0)
            cls_k[1] = self.momentum_encoder.forward_inv(images_inv_1)

            # student encoder for local view
            for _equiv_num in range(4):
                cls_q[_equiv_num+2] = self.base_encoder.forward_baseline(images_localview[_equiv_num])

            # head and predictor for CLS from student encoder
            for _aug_num in range(6):
                cls_q[_aug_num] = self.predictor(self.base_encoder.head(cls_q[_aug_num]))

            # only head for CLS from teacher encoder
            for _aug_num in range(2):
                cls_k[_aug_num] = self.momentum_encoder.head(cls_k[_aug_num])
            
            loss_inv = torch.tensor(0.0, requires_grad=True, device=cls_q[0].device)

            for _num_q in range(6):
               for _num_k in range(2):
                    if _num_q == _num_k:
                        continue
                    else:
                        loss_inv = loss_inv + self.contrastive_loss(cls_q[_num_q], cls_k[_num_k])

        return loss_inv

    def erl_local4(self, image_global, image_local, aug_equi, moco_m, lambda_equiv):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        equiv_samples_num = image_local[0].shape[0]
        equiv_feat = 512
        ############### Equiv: geometric aug parameters  #############
        w, h, degrees, flips, num_rot90_pergpu, n_list = [-99, -99, -99, -99], [-99, -99, -99, -99], [-99, -99, -99, -99], [-99, -99, -99, -99], [-99, -99, -99, -99], [-99, -99, -99, -99]
        for _equiv_num in range(4):
            w[_equiv_num], h[_equiv_num], degrees[_equiv_num], flips[_equiv_num], num_rot90_pergpu[_equiv_num] = \
                aug_equi.get_params(equiv_scale=self.args.equiv_scale, ratio=self.args.equiv_aspect_ratio, batch_size=equiv_samples_num, img_size=image_local[_equiv_num].shape[-1])
        flips[1] = torch.logical_not(flips[0])
        degrees[1] = torch.logical_not(degrees[0])
        flips[3] = torch.logical_not(flips[2])
        degrees[3] = torch.logical_not(degrees[2])
        #################### w, h, degrees, flips ##################
        ############### Equiv: geometric aug parameters #############
        for _equiv_num in range(4):
            image_local[_equiv_num] = aug_equi.aug_equiv(image_local[_equiv_num], w[_equiv_num]*self.args.stride, h[_equiv_num]*self.args.stride, degrees[_equiv_num], flips[_equiv_num], num_rot90_pergpu[_equiv_num])
        
        ############## images_equiv_0, images_equiv_1 #################
        ############### compute semantic inveriance via contrastive loss #############
        cls_q = [-99, -99, -99, -99, -99, -99]    
        cls_k = [-99, -99]     
        equiv_q = [-99, -99, -99, -99]    
        equiv_k = [-99, -99, -99, -99] 
        _map = [-99, -99, -99, -99]
        _map_num = [-99, -99, -99, -99]
        _map_denom = [-99, -99, -99, -99]
        _mask = [-99, -99, -99, -99]
        
        with torch.autocast(device_type="cuda"):
            # both encoder for global view, need to go through head and predictor
            cls_q[0], cls_q[1], cls_k[0], cls_k[1] = self.forward_inv(image_global[0], image_global[1], moco_m) # momentum for k       

            # student encoder for local view
            for _equiv_num in range(4):
                cls_q[_equiv_num+2], equiv_q[_equiv_num] = self.base_encoder.forward_equiv(image_local[_equiv_num])
                equiv_q[_equiv_num] = self.projector_equiv(equiv_q[_equiv_num])            

            # DINO's local-to-global strategy do not forward the local view through teacher, but here, we forward the local view through the early portion of teacher encoder
            with torch.no_grad():  # no gradient
                for _equiv_num in range(4):
                    equiv_k[_equiv_num] = self.momentum_encoder.forward_early(image_local[_equiv_num])
                    equiv_k[_equiv_num] = self.projector_equiv(equiv_k[_equiv_num])

            # head and predictor for CLS from student encoder
            for _aug_num in range(6):
                cls_q[_aug_num] = self.predictor(self.base_encoder.head(cls_q[_aug_num]))

            # only head for CLS from teacher encoder
            for _aug_num in range(2):
                cls_k[_aug_num] = self.momentum_encoder.head(cls_k[_aug_num])
            
            loss_inv = torch.tensor(0.0, requires_grad=True, device=cls_q[0].device)

            for _num_q in range(6):
               for _num_k in range(2):
                    if _num_q == _num_k:
                        continue
                    else:
                        loss_inv = loss_inv + self.contrastive_loss(cls_q[_num_q], cls_k[_num_k])

            ################################ loss_inv ###################################

            ################# Equiv: compute equivariance loss ############################
            for _i in range(2):
                equiv_k[2*_i] = aug_equi.aug_equiv_feat(equiv_k[2*_i], w[(2*_i) + 1], h[(2*_i) + 1], -num_rot90_pergpu[2*_i], num_rot90_pergpu[(2*_i)+1]) # to teacher_equiv[3]
                equiv_k[(2*_i)+1] = aug_equi.aug_equiv_feat(equiv_k[(2*_i)+1], w[2*_i], h[2*_i], -num_rot90_pergpu[(2*_i)+1], num_rot90_pergpu[2*_i]) # to teacher_equiv[3]
            
            for i in range(4):
                _h, _w = equiv_k[i].shape[2], equiv_k[i].shape[3]
                equiv_k[i] = torch.transpose(equiv_k[i], 1, 3).reshape(_w*_h*equiv_samples_num, equiv_feat) # equiv_feat = 512                
                equiv_k[i], n_list[i] = concat_all_gather_different_shape(equiv_k[i], 2)

                equiv_q[i] = torch.transpose(equiv_q[i], 1, 3).reshape(w[i]*h[i]*equiv_samples_num, equiv_feat) # equiv_feat = 512
                equiv_q[i], _ = concat_all_gather_different_shape(equiv_q[i], 2)

                equiv_k[i] = torch.nn.functional.normalize(equiv_k[i], dim=1)
                equiv_q[i] = torch.nn.functional.normalize(equiv_q[i], dim=1)

            _map[0] = torch.mm(equiv_k[0], torch.transpose(equiv_q[1], 0, 1)) / self.args.temperature_equiv
            _map[1] = torch.mm(equiv_k[1], torch.transpose(equiv_q[0], 0, 1)) / self.args.temperature_equiv
            _map[2] = torch.mm(equiv_k[2], torch.transpose(equiv_q[3], 0, 1)) / self.args.temperature_equiv
            _map[3] = torch.mm(equiv_k[3], torch.transpose(equiv_q[2], 0, 1)) / self.args.temperature_equiv

            for i in range(4):
                _map_num[i] = torch.trace(_map[i])
                _map[i] = torch.exp(_map[i])
                _mask[i] = torch.ones_like(_map[i], device=_map[i].device)

            # for idx_start, idx_end in zip([0]+n_list0, n_list0):
            #     _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
            #     _mask0 = mask0[idx_start:idx_end, idx_start:idx_end]
            #     for idx_sample in range(equiv_samples_num):
            #         _mask0[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
            # mask0.fill_diagonal_(1)

            for idx_start, idx_end in zip([0]+n_list[0], n_list[0]):
                _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
                __mask = _mask[0][idx_start:idx_end, idx_start:idx_end]
                for idx_sample in range(equiv_samples_num):
                    __mask[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
            _mask[0].fill_diagonal_(1)

            for idx_start, idx_end in zip([0]+n_list[1], n_list[1]):
                _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
                __mask = _mask[1][idx_start:idx_end, idx_start:idx_end]
                for idx_sample in range(equiv_samples_num):
                    __mask[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
            _mask[1].fill_diagonal_(1)

            for idx_start, idx_end in zip([0]+n_list[2], n_list[2]):
                _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
                __mask = _mask[2][idx_start:idx_end, idx_start:idx_end]
                for idx_sample in range(equiv_samples_num):
                    __mask[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
            _mask[2].fill_diagonal_(1)

            for idx_start, idx_end in zip([0]+n_list[3], n_list[3]):
                _idx_per_sample = int((idx_end-idx_start)/float(equiv_samples_num))
                __mask = _mask[3][idx_start:idx_end, idx_start:idx_end]
                for idx_sample in range(equiv_samples_num):
                    __mask[idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample, idx_sample*_idx_per_sample:(idx_sample+1)*_idx_per_sample] = 0
            _mask[3].fill_diagonal_(1)

            loss_equiv = torch.tensor(0.0, requires_grad=True, device=cls_q[0].device)

            for i in range(4):
                # print(f'_map[i].shape: {_map[i].shape}, _mask[i].shape: {_mask[i].shape}')
                _map_denom[i] = torch.sum(torch.log(torch.sum(_map[i] * _mask[i], dim=0, dtype=torch.float32)))
                loss_equiv = loss_equiv + ( (_map_denom[i] - _map_num[i]) / float(_map[i].shape[0]) )

            loss_equiv = loss_equiv / 4.0
            loss = loss_inv + (loss_equiv * lambda_equiv)

        return loss, loss_inv, loss_equiv


    def equiv(self, images_inv_0, images_inv_1, images_equiv_0, images_equiv_1, aug_equi, moco_m, lambda_equiv):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        equiv_samples_num = images_equiv_0.shape[0]
        ############### Equiv: geometric aug parameters  #############
        w0, h0, degrees0, flips0, num_rot90_pergpu0 = aug_equi.get_params(self.args.equiv_scale, self.args.equiv_aspect_ratio, equiv_samples_num)
        w1, h1, _, _, num_rot90_pergpu1 = aug_equi.get_params(self.args.equiv_scale, self.args.equiv_aspect_ratio, equiv_samples_num)
        flips1 = torch.logical_not(flips0)
        degrees1 = torch.logical_not(degrees0)
        #################### w, h, degrees, flips ##################

        ############### Equiv: geometric aug parameters #############

        images_equiv_0 = aug_equi.aug_equiv(images_equiv_0, w0*self.args.stride, h0*self.args.stride, degrees0, flips0, num_rot90_pergpu0)
        images_equiv_1 = aug_equi.aug_equiv(images_equiv_1, w1*self.args.stride, h1*self.args.stride, degrees1, flips1, num_rot90_pergpu1)
        
        ############## images_equiv_0, images_equiv_1 #################
    
        ############### compute semantic inveriance via contrastive loss #############        
        with torch.autocast(device_type="cuda"):
            clsinv_q0, clsinv_q1, clsinv_k0, clsinv_k1 = self.forward_inv(images_inv_0, images_inv_1, moco_m)            
            clsequiv_q0, clsequiv_q1, clsequiv_k0, clsequiv_k1, featequiv_q0, featequiv_q1, featequiv_k0, featequiv_k1 = \
                    self.forward_equiv(images_equiv_0, images_equiv_1)
            cls_q0 = self.predictor(self.base_encoder.head(torch.cat([clsinv_q0, clsequiv_q0], dim=0)))
            cls_q1 = self.predictor(self.base_encoder.head(torch.cat([clsinv_q1, clsequiv_q1], dim=0)))
            cls_k0 = self.momentum_encoder.head(torch.cat([clsinv_k0, clsequiv_k0], dim=0))
            cls_k1 = self.momentum_encoder.head(torch.cat([clsinv_k1, clsequiv_k1], dim=0))
            
            loss_inv = self.loss_inv(cls_q0, cls_q1, cls_k0, cls_k1)
        ################################ loss_inv ###################################

        ################# Equiv: compute equivariance loss ############################

        # featequiv_q0 = aug_equi.aug_equiv_feat(featequiv_q0, w1, h1, degrees0, flips0, num_rot90_pergpu1-num_rot90_pergpu0)
        # featequiv_q1 = aug_equi.aug_equiv_feat(featequiv_q1, w0, h0, degrees1, flips1, num_rot90_pergpu0-num_rot90_pergpu1)
        featequiv_k0 = aug_equi.aug_equiv_feat(featequiv_k0, w1, h1, -num_rot90_pergpu0, num_rot90_pergpu1) # to teacher_equiv[3]
        featequiv_k1 = aug_equi.aug_equiv_feat(featequiv_k1, w0, h0, -num_rot90_pergpu1, num_rot90_pergpu0) # to teacher_equiv[1], N x 512 x w x h


        featequiv_k0 = torch.transpose(featequiv_k0, 1, 3).reshape(w1*h1*equiv_samples_num, 512)
        featequiv_k1 = torch.transpose(featequiv_k1, 1, 3).reshape(w0*h0*equiv_samples_num, 512)
        # print(f'before gather [{args.gpu}]: {teacher_equiv0.shape}, nan: {teacher_equiv0.isnan().sum()}')
        featequiv_k0, n_list0 = concat_all_gather_different_shape(featequiv_k0, 2)
        featequiv_k1, n_list1 = concat_all_gather_different_shape(featequiv_k1, 2)
        # print(f'after gather [{args.gpu}]: {teacher_equiv0.shape}, nan: {teacher_equiv0.isnan().sum()}')
        featequiv_q0 = torch.transpose(featequiv_q0, 1, 3).reshape(w0*h0*equiv_samples_num, 512)
        featequiv_q1 = torch.transpose(featequiv_q1, 1, 3).reshape(w1*h1*equiv_samples_num, 512)
        featequiv_q0, _ = concat_all_gather_different_shape(featequiv_q0, 2)
        featequiv_q1, _ = concat_all_gather_different_shape(featequiv_q1, 2)

        featequiv_k0 = torch.nn.functional.normalize(featequiv_k0, dim=1)
        featequiv_k1 = torch.nn.functional.normalize(featequiv_k1, dim=1)
        featequiv_q0 = torch.nn.functional.normalize(featequiv_q0, dim=1)
        featequiv_q1 = torch.nn.functional.normalize(featequiv_q1, dim=1)

        # equiv0 = torch.cat([teacher_equiv0, student_equiv1], dim=0)
        # equiv1 = torch.cat([teacher_equiv1, student_equiv0], dim=0)
        # print(f'step 1: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

        equiv0 = torch.mm(featequiv_k0, torch.transpose(featequiv_q1, 0, 1)) / self.args.temperature_equiv
        equiv1 = torch.mm(featequiv_k1, torch.transpose(featequiv_q0, 0, 1)) / self.args.temperature_equiv
        # print(f'step 2: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

        equiv0_numerator = torch.trace(equiv0)
        equiv1_numerator = torch.trace(equiv1)

        equiv0 = torch.exp(equiv0)
        equiv1 = torch.exp(equiv1)
        # print(f'step 3: {equiv0.isnan().sum()}, {equiv1.isnan().sum()}')

        # print(f'equiv0 max: {equiv0.amax()}, min: {equiv0.amin()} ')
        # print(f'diagonel: {torch.diagonal(equiv0)}')

        # equiv00_denominator_samesample = []
        # equiv01_denominator_samesample = []
        # equiv10_denominator_samesample = []
        # equiv11_denominator_samesample = []

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

        loss_equiv = 0.5*(loss_equiv0 + loss_equiv1)
        loss = loss_inv + (loss_equiv * self.args.equiv_lambda)

        # with torch.autocast(device_type="cuda"):
        #     loss_equiv, loss_equiv_max = self.loss_equiv(featequiv_q0, featequiv_q1, featequiv_k0, featequiv_k1)

        return loss, loss_inv, loss_equiv
        # return loss_inv, loss_inv, loss_equiv, loss_equiv_max

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.embed_dim
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_different_shape(tensor, size):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor_shape_gather = [torch.empty(size, dtype=torch.int64, device=torch.distributed.get_rank(), requires_grad=False)
        for _rank in range(torch.distributed.get_world_size())]
    tensor_shape = torch.tensor(tensor.shape, device=torch.distributed.get_rank(), requires_grad=False)
    # print(f'gpu [{dist.get_rank()}]: {tensor_shape}')
    torch.distributed.all_gather(tensor_shape_gather, tensor_shape, async_op=False)

    tensors_gather = [torch.empty(tensor_shape_gather[_rank].tolist(), device=torch.distributed.get_rank(), dtype=tensor.dtype)
        for _rank in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # del tensor_shape, tensor_shape_gather
    n_list = np.array([_x.shape[0] for _x in tensors_gather])
    output = torch.cat(tensors_gather, dim=0)
    return output, np.cumsum(n_list).tolist()









class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
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