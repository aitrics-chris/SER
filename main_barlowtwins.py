from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import InterpolationMode
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import vision_transformer as vits
import builder.barlowtwins as bt
import builder.utils as utils
import loader as loaders
import resnet as resnets
# from utils import Aug_equi, cosine_scheduler_descend, cosine_scheduler_ascend, constant_scheduler, base_scheduler, clip_gradients, concat_all_gather

import socket
# from scp import SCPClient, SCPException
# import paramiko
from tqdm import trange
from builder.utils import AverageMeter, ProgressMeter
import builder.barlowtwins as ssls
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Barlow Twins Training with equivariance loss')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small', choices=['vit_small', 'resnet50'])
parser.add_argument('--data', default='/home/chris/storage/imagenet', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (0.25 for dino, 0.08 for both moco and barlowtwins)')

parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

parser.add_argument('--equiv-scale', type=float, nargs='+', default=(0.7, 1.3),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")    
parser.add_argument('--equiv-aspect-ratio', type=float, nargs='+', default=(3./4., 4./3.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

parser.add_argument('--equiv-lambda', type=float, default=1.0, help="""lambda for equivariance loss" """)
# parser.add_argument('--ratio-type-equiv', default='fix', choices=['fix', 'ascend', 'descend', 'base'], type=str, help='equiv ratio type. "ascend" for cosine ascending')
parser.add_argument('--equiv-mode', default='essl', choices=['erl', 'essl', 'stl', 'equimod', 'augself', 'inv'], type=str, help='equivariance mode, erl is ours')
parser.add_argument('--equiv-layer', default=3, type=int, help='layer to impose equiv')

## For equiv sampler scheduler
parser.add_argument('--warmup-epochs-scheduler', default=0, type=int, help='number of warmup epochs for scheduler')
parser.add_argument('--rest-epochs-scheduler', default=0, type=int, help='number of resting epochs where no equivariance loss is imposed')
parser.add_argument('--equiv-ratio-start', type=float, default=0.01, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--equiv-ratio-end', type=float, default=0.0, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--tag', default='exxxx', type=str, help='append at the end of the foldername')

parser.add_argument('--temperature-equiv', type=float, default=0.4, help="""Temperature for InfoNCE""")
parser.add_argument('--clip_grad', type=float, default=0.0, help="""Maximal parameter gradient norm if using gradient clipping. 
                    Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")


def main():
    args = parser.parse_args()
    args.interpolation = InterpolationMode.BICUBIC
    train_one_step = ssls.__dict__[f'train_{args.equiv_mode}']

    if 'erl' not in args.equiv_mode:
        assert args.equiv_layer == 12

    # if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    utils.setup_for_distributed(args.rank == 0)
##################
    if args.arch == 'vit_small':
        backbone = vits.__dict__[args.arch](args=args, ssl_type='barlowtwins', drop_path_rate=0.0)
        args.dim_equiv = 384
        args.stride = 16
        args.dim_inv = 384
    # if the network is a XCiT
    elif args.arch == 'resnet50':
        backbone = resnets.resnet50()
        args.stride = int(2 * (2**args.equiv_layer))
        dim_el = {1:256, 2:512, 3:1024, 4:2048}
        args.dim_equiv = dim_el[args.equiv_layer]
        args.dim_inv = 2048
    else:
        print(f"Unknow architecture: {args.arch}")

    model = bt.BarlowTwins(args, backbone)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.gpu)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    if args.arch == 'resnet50': # weight_decay=1e-6, moco
        optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    elif args.arch == 'vit_small': # weight_decay=0.1, moco
        optimizer = torch.optim.AdamW(model.parameters(), 0, weight_decay=0.1)

    # dataset = torchvision.datasets.ImageFolder(args.data, Transform())
    args.data_mode = args.equiv_mode if args.equiv_mode != 'erl' else 'inv'
    dataset = loaders.__dict__[f'get_dataset_{args.data_mode}'](args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=True)

    start_time = time.time()
    scaler = torch.GradScaler(device="cuda")

    _mean = torch.tensor((0.485, 0.456, 0.406)).view(1,3,1,1).cuda(non_blocking=True)
    _std = torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1).cuda(non_blocking=True)
    aug_equi = utils.Aug_equi(args.gpu, args)

    if args.equiv_mode == 'erl':
        equi_scheduler = utils.constant_scheduler(
            # args.equiv_lambda,
            args.epochs, len(loader),
            warmup_epochs=args.warmup_epochs_scheduler,
            rest_epochs=args.rest_epochs_scheduler,
            ratio=args.equiv_ratio_start,
        )
    else:
        equi_scheduler = utils.base_scheduler(
            args.epochs, len(loader),
        )
    loss_list_equiv = []
    loss_list_inv = []

    proj_name = f'BarlowTwins_{args.equiv_mode}_{args.arch}_{args.learning_rate_weights}_{args.equiv_scale[0]}_{args.equiv_scale[1]}_{round(args.equiv_aspect_ratio[0],2)}_{args.equiv_lambda}' \
            +f'_{args.equiv_layer}_{args.warmup_epochs_scheduler}_{args.rest_epochs_scheduler}_{args.equiv_ratio_start}_{args.equiv_ratio_end}_clipgrad_{args.clip_grad}_{args.temperature_equiv}_{socket.gethostname()}_ep{args.epochs}_{args.tag}'
    print(f'proj_name: {proj_name}')

    summary_writer = SummaryWriter() if args.rank == 0 else None
    
    for epoch in trange(0, args.epochs):
        # pass
        sampler.set_epoch(epoch)
        losses = AverageMeter('Loss', ':.4e')
        loss_invs = AverageMeter('Loss_inv', ':.4e')
        loss_equivs = AverageMeter('Loss_equiv', ':.4e')
        progress = ProgressMeter(
            len(loader),
            [losses, loss_invs, loss_equivs],
            prefix="Epoch: [{}]".format(epoch))
    
        for step, (images, _) in enumerate(loader):
            # update weight decay and learning rate according to their schedule
            step_all = len(loader) * epoch + step # global training iteration
            ################# Split mini-batch into equiv/inv algo ###############
            equiv_samples_num = round(equi_scheduler[step_all] * per_device_batch_size)
            inv_samples_num = per_device_batch_size - equiv_samples_num
            ################# inv_samples_num, equiv_samples_num ################       

            for i in range(len(images)):
                images[i] = images[i].cuda(args.gpu, non_blocking=True)
            bt.adjust_learning_rate(args, optimizer, loader, step_all)

            loss, loss_inv, loss_equiv, loss_list_inv, loss_list_equiv = train_one_step(args, images, inv_samples_num, equiv_samples_num, \
                                                                                        model, aug_equi, _mean, _std, loss_list_inv, loss_list_equiv)

            losses.update(loss.item(), images[0].size(0))
            loss_invs.update(loss_inv.item(), images[0].size(0))
            loss_equivs.update(loss_equiv.item(), images[0].size(0))

            if args.rank == 0:
                summary_writer.add_scalar("loss", loss.item(), epoch * len(loader) + step)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if args.clip_grad > 0.0:
                scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
                
            scaler.step(optimizer)
            scaler.update()


            if step % 10 == 0:
                progress.display(step)
            
    save_dict = {
                'epoch': epoch + 1,
                'args': args,
                'model':model.module.backbone.state_dict(),
                'projector_inv': model.module.projector.state_dict(),
                'projector_equiv': model.module.projector_equiv.state_dict() if args.equiv_mode != 'inv' else None,
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'loss_list_inv': np.array(loss_list_inv),
                'loss_list_equiv': np.array(loss_list_equiv),
                }
    
    args.output_dir = f'/nfs/thena/chris/icml/ckpt_barlowtwins/{args.equiv_mode}/{proj_name}'
    args.output_dir_local = f'/home/chris/codes/moco3/results/barlowtwins/{args.equiv_mode}/{proj_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_local, exist_ok=True)

    if args.rank == 0:
        # save final model
        torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        torch.save(save_dict, os.path.join(args.output_dir_local, 'checkpoint.pth'))
        summary_writer.close()
    
    # dist.destory_process_group()


if __name__ == '__main__':
    main()
