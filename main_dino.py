import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models

# from datetime import datetime
import builder.utils as utils
import builder.dino as dino
import vision_transformer as vits
import resnet as resnets
from vision_transformer import DINOHead
# from torch.distributed.elastic.utils.data import ElasticDistributedSampler
# from utils import Aug_equi, cosine_scheduler_ascend, cosine_scheduler_descend, constant_scheduler, base_scheduler, gather_from_all, concat_all_gather
from tqdm import trange
import socket
import loader as loaders
import builder.dino as ssls

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--data', default='data/imagenet/', type=str,
        help='Please specify path to the ImageNet training data.')

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_small', 'resnet_50'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.982, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    # parser.add_argument("--warmup_epochs", default=10, type=int,
    #     help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--crop-min', default=0.25, type=float,
                    help='minimum scale for random cropping (0.25 for dino, 0.08 for both moco and barlowtwins)')
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--output_dir', default="codes/erl/results", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    # Equiv
    parser.add_argument('--equiv-mode', default='essl', choices=['erl', 'essl', 'stl', 'equimod', 'augself', 'inv'], type=str, help='equivariance mode, erl is ours')
    parser.add_argument('--equiv-scale', type=float, nargs='+', default=(0.7, 1.3),
            help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for small local view cropping of multi-crop.""")    
    parser.add_argument('--equiv-aspect-ratio', type=float, nargs='+', default=(3./4., 4./3.),
            help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for small local view cropping of multi-crop.""")
    
    parser.add_argument('--equiv-lambda', type=float, default=0.3, help="""lambda for equivariance loss" """)
    parser.add_argument('--equiv-layer', default=12, type=int, help='layer to impose equiv')

    ## For equiv sampler scheduler
    parser.add_argument('--warmup-epochs-scheduler', default=0, type=int, help='number of warmup epochs for scheduler')
    parser.add_argument('--rest-epochs-scheduler', default=0, type=int, help='number of resting epochs where no equivariance loss is imposed')
    parser.add_argument('--equiv-ratio-start', type=float, default=0.02, help="""Ratio of minibatch for equivariance loss""")
    parser.add_argument('--equiv-ratio-end', type=float, default=0.0, help="""Ratio of minibatch for equivariance loss""")
    parser.add_argument('--tag', default='ex', type=str, help='append at the end of the foldername')
    
    parser.add_argument('--temperature', type=float, default=0.4, help="""Temperature for InfoNCE""")
    
    return parser


def train_dino(args):
    args.interpolation = InterpolationMode.BICUBIC

    if 'erl' not in args.equiv_mode:
        assert args.equiv_layer == 12
        
    train_one_step = ssls.__dict__[f'train_{args.equiv_mode}']

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    args.data_mode = args.equiv_mode if args.equiv_mode != 'erl' else 'inv'
    train_dataset = loaders.__dict__[f'get_dataset_{args.data_mode}'](args)
    sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(train_dataset)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch == 'vit_small':
        student = vits.__dict__[args.arch](args, 'dino', args.drop_path_rate)
        teacher = vits.__dict__[args.arch](args, 'dino', 0.0)
        embed_dim = student.embed_dim        
        args.stride = 16
    # if the network is a XCiT
    elif args.arch == 'resnet50':
        student = resnets.resnet50()
        teacher = resnets.resnet50()
        embed_dim = 2048     
        args.stride = 32
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = dino.MultiCropWrapper(student, args, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = dino.MultiCropWrapper(
        teacher, args,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student): # it is false
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    # if ((args.ratio_type_equiv == 'fix') and (args.warmup_epochs_scheduler == 0) and (args.rest_epochs_scheduler == 0)) or (args.ratio_type_equiv == 'base'):
    #     student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # else:
        # student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True if args.equiv_mode == 'erl' else False)
        
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    if args.equiv_mode != 'inv':
        teacher.set_projector_equiv(student.module.projector_equiv)

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    fp16_scaler = torch.GradScaler(device="cuda")

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    start_epoch = to_restore["epoch"]

    _mean = torch.tensor((0.485, 0.456, 0.406)).view(1,3,1,1).cuda(non_blocking=True)
    _std = torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1).cuda(non_blocking=True)
    aug_equi = utils.Aug_equi(args.gpu, args)
   
    if args.equiv_mode == 'erl':
        equi_scheduler = utils.constant_scheduler(
            # args.equiv_lambda,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs_scheduler,
            rest_epochs=args.rest_epochs_scheduler,
            ratio=args.equiv_ratio_start,
        )
        assert args.equiv_ratio_start > 0.0
        assert args.equiv_ratio_end == 0.0
    else:
        equi_scheduler = utils.base_scheduler(
            args.epochs, len(data_loader),
        )


    proj_name = f'DINO_{args.equiv_mode}_{args.arch}_{args.lr}_{args.equiv_mode}_{args.equiv_scale[0]}_{args.equiv_scale[1]}_{round(args.equiv_aspect_ratio[0],2)}_{args.equiv_lambda}_{args.equiv_layer}_{args.warmup_epochs_scheduler}' \
            +f'_{args.rest_epochs_scheduler}_{args.equiv_ratio_start}_{args.equiv_ratio_end}_clipgrad_{args.clip_grad}_{args.temperature}_{socket.gethostname()}_ep{args.epochs}_{args.tag}'

    print(f'proj_name: {proj_name}')
    # proj_name = 'ex'
    args.output_dir = os.path.join(f'ckpts/ckpt_dino', args.equiv_mode, proj_name)
    os.makedirs(args.output_dir, exist_ok=True)

    loss_list_equiv = []
    loss_list_inv = []
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in trange(start_epoch, args.epochs):
        # pass
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats, loss_list_inv, loss_list_equiv = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, train_one_step,
            epoch, fp16_scaler, args, _mean, _std, aug_equi, equi_scheduler, loss_list_inv, loss_list_equiv)

        # ============ writing logs ... ============    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'loss_list_inv': np.array(loss_list_inv),
            'loss_list_equiv': np.array(loss_list_equiv),
        }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
    utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))



def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, train_one_step, epoch,
                    fp16_scaler, args, _mean, _std, aug_equi, equi_scheduler, _loss_list_inv, _loss_list_equiv):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    for _iter, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        step_all = len(data_loader) * epoch + _iter # global training iteration
        
        ################# Split mini-batch into equiv/inv algo ###############
        equiv_samples_num = round(equi_scheduler[step_all] * args.batch_size_per_gpu)
        inv_samples_num = args.batch_size_per_gpu - equiv_samples_num
        ################# inv_samples_num, equiv_samples_num ################            

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[step_all]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[step_all]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        ################# Get loss ###############        
        loss, loss_inv, loss_equiv, _loss_list_inv, _loss_list_equiv = train_one_step(student, teacher, teacher_without_ddp, images, aug_equi, _mean, _std, epoch, dino_loss, \
                                                                                        _loss_list_inv, _loss_list_equiv, inv_samples_num, equiv_samples_num, args)
        ################# loss ###############

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[step_all]  # momentum parameter <- teacher
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_inv=loss_inv.item())
        metric_logger.update(loss_equiv=loss_equiv.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, _loss_list_inv, _loss_list_equiv




class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# class DataAugmentationDINO(object):
#     def __init__(self, global_crops_scale):
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#         ])
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#         ])

#     def __call__(self, image):
#         crops = []
#         crops.extend([tf(image) for tf in [self.global_transfo1, self.global_transfo2]])
#         return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
