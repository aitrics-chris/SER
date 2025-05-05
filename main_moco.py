import argparse
import math
import os
import shutil
import time
import warnings
from functools import partial
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import loader as loaders
# import moco.builder
# import moco.loader
# import moco.optimizer
import vision_transformer as vits
import resnet as resnets

import sys
# import vits
import numpy as np
import builder.utils as utils
import builder.moco as moco
import copy
import socket
import builder.moco as ssls
from builder.utils import AverageMeter, ProgressMeter
from torchvision.transforms import InterpolationMode
from kornia.constants import Resample
# from scp import SCPClient, SCPException
# import paramiko


parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
parser.add_argument('--data', default='/home/chris/storage/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                    choices=['vit_small', 'resnet50'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (0.25 for dino, 0.08 for both moco and barlowtwins)')
parser.add_argument('--output_dir', default="/home/chris/codes/erl/results", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--equiv-scale', type=float, nargs='+', default=(0.7, 1.3),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")    
parser.add_argument('--equiv-aspect-ratio', type=float, nargs='+', default=(3./4., 4./3.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
# parser.add_argument('--fixed-ratio', type=float, default=0.5, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--equiv-lambda', type=float, default=1.0, help="""lambda for equivariance loss" """)
# parser.add_argument('--equiv-ratio', type=float, default=0.5, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--equiv-mode', default='stl', choices=['erl_inv', 'essl', 'stl', 'equimod', 'augself', 'inv', 'erl_local4', 'inv_essl'], type=str, help='equivariance mode, erl is ours')
parser.add_argument('--equiv-layer', default=12, type=int, help='layer to impose equiv')

## For equiv sampler scheduler
parser.add_argument('--warmup-epochs-scheduler', default=0, type=int, help='number of warmup epochs for scheduler')
parser.add_argument('--rest-epochs-scheduler', default=0, type=int, help='number of resting epochs where no equivariance loss is imposed')
parser.add_argument('--equiv-ratio-start', type=float, default=0.02, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--equiv-ratio-end', type=float, default=0.0, help="""Ratio of minibatch for equivariance loss""")
parser.add_argument('--tag', default='exxx', type=str, help='append at the end of the foldername')

parser.add_argument('--temperature-equiv', type=float, default=0.2, help="""Temperature for InfoNCE""")
parser.add_argument('--clip_grad', type=float, default=0.0, help="""Maximal parameter gradient norm if using gradient clipping. 
                    Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")

parser.add_argument('--stl-lambda-equi', type=float, default=1.0, help="lambda for STL for equivariant loss")
parser.add_argument('--stl-lambda-trans', type=float, default=0.1, help="lambda for STL for transformation loss")


def main():
    args = parser.parse_args()
    args.interpolation = InterpolationMode.BILINEAR
    args.interpolation_kornia = Resample.BILINEAR.name

    if 'erl' not in args.equiv_mode:
        assert args.equiv_layer == 12

    train_one_step = ssls.__dict__[f'train_{args.equiv_mode}']

    if 'vit' in args.arch:
        args.optimizer = 'adamw'
        args.weight_decay = 0.1
        args.moco_t = 0.2
        args.crop_min = 0.08
    elif 'resnet' in args.arch:
        args.optimizer = 'lars'
        args.weight_decay = 1e-6
        args.moco_t = 1.0
        args.crop_min = 0.2

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
    setup_for_distributed(args.rank == 0)
    cudnn.benchmark = True
    
    # ============ building networks ... ============
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.MoCo_ViT(args,
            vits.__dict__[args.arch],                          
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
        args.stride = 16
    else:
        raise NotImplementedError
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True if 'erl' in args.equiv_mode else False)


    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256
    if args.optimizer == 'lars':
        optimizer = moco.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.GradScaler(device="cuda")
    summary_writer = SummaryWriter() if args.rank == 0 else None


    # ============ preparing data ... ============
    args.batch_size = int(args.batch_size / args.world_size)
    # args.data_mode = args.equiv_mode if args.equiv_mode != 'erl' else 'inv'
    args.data_mode = args.equiv_mode.split('_')[-1]
    train_dataset, etc = loaders.__dict__[f'get_dataset_{args.data_mode}'](args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    _mean = torch.tensor((0.485, 0.456, 0.406)).view(1,3,1,1).cuda(non_blocking=True)
    _std = torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1).cuda(non_blocking=True)
    aug_equi = utils.Aug_equi(args.gpu, args)

    if args.equiv_mode == 'erl_inv':
        equi_scheduler = utils.constant_scheduler(
            # args.equiv_lambda,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs_scheduler,
            rest_epochs=args.rest_epochs_scheduler,
            ratio=args.equiv_ratio_start,
        )
    else:
        equi_scheduler = utils.base_scheduler(
            args.epochs, len(data_loader),
        )
    
    loss_list_equiv = []
    loss_list_inv = []
    proj_name = f'MoCo_{args.equiv_mode}_{args.arch}_{args.lr}_{args.equiv_scale[0]}_{args.equiv_scale[1]}_{round(args.equiv_aspect_ratio[0],2)}_{args.equiv_lambda}' \
            +f'_{args.equiv_layer}_{args.warmup_epochs_scheduler}_{args.rest_epochs_scheduler}_{args.equiv_ratio_start}_{args.equiv_ratio_end}_clipgrad_{args.clip_grad}_{args.temperature_equiv}_{socket.gethostname()}_ep{args.epochs}_{args.tag}'
    print(f'proj_name: {proj_name}')

    for epoch in trange(args.start_epoch, args.epochs):
        # if args.distributed:
        train_sampler.set_epoch(epoch)
        # pass
        # train for one epoch
        loss_list_inv, loss_list_equiv = train(data_loader, model, optimizer, scaler, summary_writer, epoch, args, \
                                               _mean, _std, aug_equi, equi_scheduler, train_one_step, loss_list_inv, loss_list_equiv, etc)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank == 0): # only the first GPU saves checkpoint
    student_head = copy.deepcopy(model.module.base_encoder.head)
    del model.module.base_encoder.head
    teacher_head = copy.deepcopy(model.module.momentum_encoder.head)
    del model.module.momentum_encoder.head
    save_dict = {
                'epoch': epoch + 1,
                'args': args,
                'student': model.module.base_encoder.state_dict(),
                'student_head': student_head,
                'teacher': model.module.momentum_encoder.state_dict(),
                'teacher_head': teacher_head,
                'projector_equiv': model.module.projector_equiv.state_dict() if args.equiv_mode.split('_')[0] != 'inv' else None,
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'loss_list_inv': np.array(loss_list_inv),
                'loss_list_equiv': np.array(loss_list_equiv),
                }    
    
    # proj_name = 'ex'
    
    # dir1 = os.path.join(args.output_dir, ssl_type, proj_name)
    # os.makedirs(dir1, exist_ok=True)
    # save_on_master(save_dict, os.path.join(dir1, f'checkpoint_{args.batch_size}_{args.lr}.pth'))
    dir2 = os.path.join('/nfs/thena/chris/icml/ckpt_moco', args.equiv_mode, proj_name)
    os.makedirs(dir2, exist_ok=True)
    save_on_master(save_dict, os.path.join(dir2, f'checkpoint_{args.batch_size}_{args.lr}.pth'))
    
    if args.rank == 0:
        summary_writer.close()

    
    # if dist.get_rank() == 0:
    #     client = paramiko.SSHClient()
    #     client.load_system_host_keys()
    #     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #     client.connect('43.246.152.183', '6150', 'mai1', 'cvpr2025')

    #     try:
    #         with SCPClient(client.get_transport()) as scp:
    #             scp.put(dir2, f'/mnt/aitrics_ext/ext01/chris/ckpt_moco/{ssl_type}/', recursive=True, preserve_times=True)
    #     except SCPException:
    #         raise SCPException.message
        
    #     client.close()

def train(data_loader, model, optimizer, scaler, summary_writer, epoch, args, _mean, _std, \
          aug_equi, equi_scheduler, train_one_step, _loss_list_inv, _loss_list_equiv, etc):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    loss_invs = AverageMeter('Loss_inv', ':.4e')
    loss_equivs = AverageMeter('Loss_equiv', ':.4e')
    # loss_equivs_max = AverageMeter('Loss_equiv_max', ':.4e')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, learning_rates, losses, loss_invs, loss_equivs],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(data_loader)
    moco_m = args.moco_m
    for _step, (images, _) in enumerate(data_loader):
        
        ############### To determine data portion for equiv and inv ###############
        step_all = len(data_loader) * epoch + _step
        equiv_samples_num = round(equi_scheduler[step_all] * args.batch_size)
        inv_samples_num = args.batch_size - equiv_samples_num
        ############### Output: 1) inv_samples_num, 2) equiv_samples_num ###############

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + _step / iters_per_epoch, args)
        learning_rates.update(lr)
        # if args.moco_m_cos: # always True
        moco_m = adjust_moco_momentum(epoch + _step / iters_per_epoch, args)

        # return overall loss
        loss, loss_inv, loss_equiv, _loss_list_inv, _loss_list_equiv = train_one_step(model, images, inv_samples_num, aug_equi, _mean, _std, moco_m, _loss_list_inv, _loss_list_equiv, args, etc)
        losses.update(loss.item(), images[0].size(0))
        loss_invs.update(loss_inv.item(), images[0].size(0))
        loss_equivs.update(loss_equiv.item(), images[0].size(0))

        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + _step)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if args.clip_grad > 0.0:
            scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            param_norms = utils.clip_gradients(model, args.clip_grad)
        
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if _step % args.print_freq == 0:
            progress.display(_step)
            # print(f'loss: {loss.item()}, loss_inv: {loss_inv.item()}, loss_equiv: {loss_equiv.item()}')

    return _loss_list_inv, _loss_list_equiv
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

if __name__ == '__main__':
    main()
