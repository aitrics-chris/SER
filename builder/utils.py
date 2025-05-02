# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
# from kornia.augmentation import ColorJitter, RandomGrayscale, RandomGaussianBlur
# from kornia.augmentation.container import ImageSequential
import kornia
from torchvision import datasets, transforms
import torch.nn.functional as F
import kornia.augmentation as K
from kornia.enhance import adjust_saturation, adjust_hue

from typing import Dict, Any, Tuple, Optional

from torch import Tensor

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        raise ValueError(f'pretrained weight does NOT exist!!')


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_scheduler_lambda(base_lambda, epochs, niter_per_ep, warmup_epochs=0):
    
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.zeros(warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = base_lambda * 0.5 * (1 - np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

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


def init_distributed_mode(args):
    # launched with torch.distributed.launch
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class LARS_MoCo(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

class LARS_BT(torch.optim.Optimizer):
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

    
def base_scheduler(epochs, niter_per_ep):
    '''
    For baseline
    '''

    return np.zeros(epochs * niter_per_ep)


def constant_scheduler(epochs, niter_per_ep, warmup_epochs=0, rest_epochs=45, ratio=0.5):
    '''
    Gradual decrement from 0 to 1
    '''
    assert epochs > (warmup_epochs + rest_epochs)

    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.zeros(warmup_iters)

    rest_iters = rest_epochs * niter_per_ep
    rest_schedule = np.zeros(rest_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters - rest_iters) * ratio

    schedule = np.concatenate((warmup_schedule, schedule, rest_schedule))
    print(f'{warmup_schedule.shape}, {schedule.shape}, {rest_schedule.shape}, {len(schedule)}, {epochs}, {niter_per_ep}, {epochs * niter_per_ep}')
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_scheduler_ascend(epochs, niter_per_ep, warmup_epochs=0, ratio_start=0.0, ratio_end=0.5):
    '''
    Gradual increment from 0 to 1
    '''
    assert ratio_start < ratio_end

    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.zeros(warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = 0.5 * (1 - np.cos(np.pi * iters / len(iters))) * (ratio_end - ratio_start) + ratio_start

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_scheduler_descend(epochs, niter_per_ep, warmup_epochs=0, rest_epochs=45, ratio_start=0.5, ratio_end=0.0):
    '''
    Gradual decrement from 0 to 1
    '''
    assert ratio_start > ratio_end
    if rest_epochs > 0:
        assert ratio_end == 0.0

    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.zeros(warmup_iters)

    rest_iters = rest_epochs * niter_per_ep
    rest_schedule = np.zeros(rest_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters - rest_iters)
    schedule = 0.5 * (1 + np.cos(np.pi * iters / len(iters))) * (ratio_start - ratio_end) + ratio_end

    schedule = np.concatenate((warmup_schedule, schedule, rest_schedule))
    print(f'{warmup_schedule.shape}, {schedule.shape}, {rest_schedule.shape}, {len(schedule)}, {epochs}, {niter_per_ep}, {epochs * niter_per_ep}')
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v




class Aug_equi(nn.Module):
    def __init__(self, seed, args):
        super(Aug_equi, self).__init__()   

        from kornia.augmentation import ColorJitter, RandomGrayscale, RandomGaussianBlur
        from kornia.augmentation.container import ImageSequential
        self.args = args
        self.inv1 = ImageSequential(            
                                        kornia.augmentation.ColorJiggle(0.4, 0.4, 0.2, 0.1, same_on_batch=False, p=0.8),  # not strengthened                                        
                                        kornia.augmentation.RandomGrayscale(same_on_batch=False, p=0.2),
                                        kornia.augmentation.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), same_on_batch=False, p=1.0),
                                    )        
        
        self.inv2 = ImageSequential(            
                                        kornia.augmentation.ColorJiggle(0.4, 0.4, 0.2, 0.1, same_on_batch=False, p=0.8),  # not strengthened                                        
                                        kornia.augmentation.RandomGrayscale(same_on_batch=False, p=0.2),
                                        kornia.augmentation.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), same_on_batch=False, p=0.1),
                                        kornia.augmentation.RandomSolarize(thresholds=0.5, additions=0, same_on_batch=False, p=0.2) # different from the original implementation (followed Kornia library default)
                                    )
        
        if 'essl' in args.equiv_mode:
            self.inv_rotate = ImageSequential(            
                                        kornia.augmentation.ColorJiggle(0.4, 0.4, 0.2, 0.1, same_on_batch=False, p=0.8),  # not strengthened                                        
                                        kornia.augmentation.RandomGrayscale(same_on_batch=False, p=0.2),
                                        kornia.augmentation.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), same_on_batch=False, p=0.1),
                                        )
                
        self.seed_generator = torch.Generator()
        self.seed_generator.manual_seed(seed)           

    def set_seed(self, seed):
        self.seed_generator.manual_seed(seed)

    # @staticmethod
    def get_params(
        self,
        equiv_scale,
        ratio = [3.0 / 4.0, 4.0 / 3.0],
        batch_size = 128,
        img_size = 224
        ):
        log_ratio = log_ratio = torch.log(torch.tensor((ratio[0], ratio[1])))
        target_area = ((img_size/self.args.stride)**2) * torch.empty(1).uniform_(equiv_scale[0], equiv_scale[1], generator=self.seed_generator).item() # (224/16)^2 = 196.0
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1], generator=self.seed_generator)).item()
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        num_rot90_pergpu = torch.randint(low=0, high=4, size=(1,), generator=self.seed_generator).item()
        if_rot180_persample = torch.rand(size=(batch_size,), generator=self.seed_generator) > 0.5
        flips = torch.rand(size=(batch_size,), generator=self.seed_generator) > 0.5
        
        return w, h, if_rot180_persample, flips, num_rot90_pergpu
    
    def aug_inv1(self, x):
        return self.inv1(x)
    
    def aug_inv2(self, x):
        return self.inv2(x)
    
    def aug_rotate(self, x):
        return self.inv_rotate(x)
    
    def aug_equiv(self, x, w, h, degrees, flips, num_rot90_pergpu):
        x = torch.cat(list(map(self.rotate, torch.split(x, 1), degrees, flips)), dim=0)
        x = nn.functional.interpolate(x, size=(w, h), mode='bicubic') # BxCx6x6 -> BxCxHxW
        x = torch.rot90(x, k=num_rot90_pergpu, dims=[2,3])

        return x
    
    def aug_equiv_feat(self, x, w, h, rot90_inv, rot90_another):
        x = torch.rot90(x, k=rot90_inv+2, dims=[2,3])
        x = transforms.functional.hflip(x)
        x = nn.functional.interpolate(x, size=(w, h), mode='bicubic') # BxCx6x6 -> BxCxHxW
        x = torch.rot90(x, k=rot90_another, dims=[2,3])
        return x
    
    @staticmethod
    def rotate(x, degree, flip):
        x = torch.rot90(x, k=2, dims=[2,3]) if degree else x
        return transforms.functional.hflip(x) if flip else x


@torch.no_grad()
def concat_all_gather(tensor, size):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor_shape_gather = [torch.empty(size, dtype=torch.int64, device=dist.get_rank(), requires_grad=False)
        for _rank in range(torch.distributed.get_world_size())]
    tensor_shape = torch.tensor(tensor.shape, device=dist.get_rank(), requires_grad=False)
    # print(f'gpu [{dist.get_rank()}]: {tensor_shape}')
    torch.distributed.all_gather(tensor_shape_gather, tensor_shape, async_op=False)

    tensors_gather = [torch.empty(tensor_shape_gather[_rank].tolist(), device=dist.get_rank(), dtype=tensor.dtype)
        for _rank in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # del tensor_shape, tensor_shape_gather
    n_list = np.array([_x.shape[0] for _x in tensors_gather])
    output = torch.cat(tensors_gather, dim=0)
    return output, np.cumsum(n_list).tolist()

# def equiv(args, images_equiv_0, teacher_output, aug_equi, lambda_equiv):
#         """
#         Input:
#             x1: first views of images
#             x2: second views of images
#             m: moco momentum
#         Output:
#             loss
#         """
#         equi_samples_num = images_equiv_0.shape[0]
#         ############### Equiv: geometric aug parameters  #############
#         w0, h0, degrees0, flips0 = aug_equi.get_params(args.equiv_scale, args.equiv_aspect_ratio, equi_samples_num)
#         w1, h1, _, _ = aug_equi.get_params(args.equiv_scale, args.equiv_aspect_ratio, equi_samples_num)
#         flips1 = torch.logical_not(flips0)
#         degrees1 = torch.logical_not(degrees0)
#         #################### w, h, degrees, flips ##################

#         ############### Equiv: geometric aug parameters #############
#         images_equiv_0 = aug_equi.aug_equiv(images_equiv_0, w0*16, h0*16, degrees0, flips0)
#         images_equiv_1 = aug_equi.aug_equiv(images_equiv_1, w1*16, h1*16, degrees1, flips1)
#         ############## images_equiv_0, images_equiv_1 #################
    
#         ############### compute semantic inveriance via contrastive loss #############
        
#         with torch.cuda.amp.autocast(True):
#             clsinv_q0, clsinv_q1, clsinv_k0, clsinv_k1 = forward_inv(images_inv_0, images_inv_1, moco_m)            
#             clsequiv_q0, clsequiv_q1, clsequiv_k0, clsequiv_k1, featequiv_q0, featequiv_q1, featequiv_k0, featequiv_k1 = \
#                     self.forward_equiv(images_equiv_0, images_equiv_1)
#             cls_q0 = self.predictor(self.base_encoder.head(torch.cat([clsinv_q0, clsequiv_q0], dim=0)))
#             cls_q1 = self.predictor(self.base_encoder.head(torch.cat([clsinv_q1, clsequiv_q1], dim=0)))
#             cls_k0 = self.momentum_encoder.head(torch.cat([clsinv_k0, clsequiv_k0], dim=0))
#             cls_k1 = self.momentum_encoder.head(torch.cat([clsinv_k1, clsequiv_k1], dim=0))
            
#             loss_inv = self.loss_inv(cls_q0, cls_q1, cls_k0, cls_k1)
#         ################################ loss_inv ###################################      

#         ################# Equiv: compute equivariance loss ############################          
#         featequiv_q0 = aug_equi.aug_equiv(featequiv_q0, w1, h1, degrees0, flips0)
#         featequiv_q1 = aug_equi.aug_equiv(featequiv_q1, w0, h0, degrees1, flips1)
        
#         with torch.cuda.amp.autocast(True):
#             loss_equiv = self.loss_equiv(featequiv_q0, featequiv_q1, featequiv_k0, featequiv_k1)
#         return loss_inv + (loss_equiv * lambda_equiv)

def gather_from_all(tensor: torch.Tensor, dim: int, shape_list):
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)
    
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor, shape_list)
        
        print(f'[3] gpu [{dist.get_rank()}]: output0 ({gathered_tensors[0].get_device()}), output1 ({gathered_tensors[1].get_device()})')
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
        
    print(f'[4] gpu [{dist.get_rank()}]: output0 ({gathered_tensors[0].get_device()}), output1 ({gathered_tensors[1].get_device()})')
    gathered_tensor = torch.cat(gathered_tensors, dim)
    return gathered_tensor

def is_distributed_training_run():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )

def convert_to_distributed_tensor(tensor: torch.Tensor):
        """
        For some backends, such as NCCL, communication only works if the
        tensor is on the GPU. This helper function converts to the correct
        device and returns the tensor + original device.
        """
        orig_device = "cpu" if not tensor.is_cuda else "gpu"
        if (
            torch.distributed.is_available()
            and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
            and not tensor.is_cuda
        ):
            tensor = tensor.cuda()
        return (tensor, orig_device)
    
def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor

class GatherLayer(torch.autograd.Function):
    """https://pytorch.org/docs/stable/autograd.html#function"""
    """Gather tensors from all process, supporting backward propagation."""
    """make sure you are calling the correct methods on ctx and
    validating your backward function using torch.autograd.gradcheck()."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        # output = [torch.empty(0) for _ in range(dist.get_world_size())]
        print(f'[1] gpu [{dist.get_rank()}]: output0 ({output[0].get_device()}), output1 ({output[1].get_device()})')
        dist.all_gather(output, input)
        print(f'[2] gpu [{dist.get_rank()}]: output0 ({output[0].get_device()}), output1 ({output[1].get_device()})')
        return output

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def load_mlp_augself(n_in, n_hidden, n_out, num_layers=3, last_bn=True):
    layers = []
    for i in range(num_layers-1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    reset_parameters_augself(mlp)
    return mlp

def reset_parameters_augself(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)
                

class SSObjective:
    def __init__(self, crop=-1, color=-1, flip=-1, blur=-1, rot=-1, sol=-1, only=False):
        self.only = only
        self.params = [
            ('crop',  crop,  4, 'regression'),
            ('color', color, 4, 'regression'),
            ('flip',  flip,  1, 'binary_classification'),
            ('blur',  blur,  1, 'regression'),
            ('rot',    rot,  4, 'classification'),
            ('sol',    sol,  1, 'regression'),
        ]
# loss_equiv = self.ss_objective(self.projector_equiv, cls_q0, cls_q1, d1, d2)
    def __call__(self, ss_predictor, z1, z2, d1, d2, symmetric=True):
        if symmetric:
            z = torch.cat([torch.cat([z1, z2], 1),
                           torch.cat([z2, z1], 1)], 0)
            d = { k: torch.cat([d1[k], d2[k]], 0) for k in d1.keys() }
        else:
            z = torch.cat([z1, z2], 1)
            d = d1

        losses = { 'total': 0 }
        for name, weight, n_out, loss_type in self.params:
            if weight <= 0:
                continue

            p = ss_predictor[name](z)
            if loss_type == 'regression':
                losses[name] = F.mse_loss(torch.tanh(p), d[name])
            elif loss_type == 'binary_classification':
                losses[name] = F.binary_cross_entropy_with_logits(p, d[name])
            elif loss_type == 'classification':
                losses[name] = F.cross_entropy(p, d[name])
            losses['total'] += losses[name] * weight

        return losses
    

def prepare_training_batch_augself(batch, t1, t2, device):
    ((x1, w1), (x2, w2)) = batch
    with torch.no_grad():
        x1 = t1(x1).detach()
        x2 = t2(x2).detach()
        diff1 = { k: v.to(device) for k, v in extract_diff(t1, t2, w1, w2).items() }
        diff2 = { k: v.to(device) for k, v in extract_diff(t2, t1, w2, w1).items() }

    return x1, x2, diff1, diff2

def extract_diff(transforms1, transforms2, crop1, crop2):
    diff = {}
    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            f1 = t1._params['batch_prob']
            f2 = t2._params['batch_prob']
            break

    center1 = crop1[:, :2]+crop1[:, 2:]/2
    center2 = crop2[:, :2]+crop2[:, 2:]/2
    center1[f1, 1] = 1-center1[f1, 1]
    center2[f1, 1] = 1-center2[f1, 1]
    diff['crop'] = torch.cat([center1-center2, crop1[:, 2:]-crop2[:, 2:]], 1)
    diff['flip'] = (f1==f2).float().unsqueeze(-1)
    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            pass

        elif isinstance(t1, K.RandomGrayscale):
            pass

        elif isinstance(t1, GaussianBlur_augself):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['blur'] = w1-w2

        elif isinstance(t1, K.Normalize):
            pass

        elif isinstance(t1, K.ColorJitter):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['color'] = w1-w2

        elif isinstance(t1, (nn.Identity, nn.Sequential)):
            pass

        elif isinstance(t1, RandomRotation_augself):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['rot'] = (w1-w2+4) % 4

        elif isinstance(t1, K.RandomSolarize):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['sol'] = w1-w2

        else:
            raise Exception(f'Unknown transform: {str(t1.__class__)}')

    return diff


def _extract_w(t):
    if isinstance(t, GaussianBlur_augself):
        m = t._params['batch_prob']
        w = torch.zeros(m.shape[0], 1)
        w[m] = t._params['sigma'].unsqueeze(-1)
        return w

    elif isinstance(t, ColorJitter_augself):
        to_apply = t._params['batch_prob']
        w = torch.zeros(to_apply.shape[0], 4)
        w[to_apply, 0] = (t._params['brightness_factor'] - 1) / (t.brightness[1]-t.brightness[0])
        w[to_apply, 1] = (t._params['contrast_factor'] - 1) / (t.contrast[1]-t.contrast[0])
        w[to_apply, 2] = (t._params['saturation_factor'] - 1) / (t.saturation[1]-t.saturation[0])
        w[to_apply, 3] = t._params['hue_factor'] / (t.hue[1]-t.hue[0])
        return w

    elif isinstance(t, RandomRotation_augself):
        to_apply = t._params['batch_prob']
        w = torch.zeros(to_apply.shape[0], dtype=torch.long)
        w[to_apply] = t._params['degrees']
        return w

    elif isinstance(t, K.RandomSolarize):
        to_apply = t._params['batch_prob']
        w = torch.ones(to_apply.shape[0])
        w[to_apply] = t._params['thresholds_factor']
        return w
    

class RandomRotation_augself(K.AugmentationBase2D):
    # def __init__(self, same_on_batch=False, p=0.5):
    #     super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=False)

    # def generate_parameters(self, batch_shape):
    #     return {"degrees": torch.randint(0, 4, (batch_shape[0],))}

    # def apply_transform(
    #     self,
    #     input: Tensor,
    #     params: Dict[str, Tensor],
    #     flags: Dict[str, Any],
    #     transform: Optional[Tensor] = None
    # ) -> Tensor:
    #     degrees = params['degrees']
    #     # apply torch.rot90 per-sample
    #     return torch.stack(
    #         [torch.rot90(x, int(k), (1, 2)) for x, k in zip(input, degrees.tolist())],
    #         dim=0
    #     )

    def __init__(self, return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        degrees = torch.randint(0, 4, (batch_shape[0], ))
        return dict(degrees=degrees)

    def apply_transform(self, input, params):
        degrees = params['degrees']
        input = torch.stack([torch.rot90(x, k, (1, 2)) for x, k in zip(input, degrees.tolist())], 0)
        return input
    

def load_ss_predictor(n_in, ss_objective, n_hidden=512):
    ss_predictor = nn.ModuleDict()
    for name, weight, n_out, _ in ss_objective.params:
        if weight > 0:
            ss_predictor[name] = load_mlp_augself(n_in*2, n_hidden, n_out, num_layers=3, last_bn=False)

    return ss_predictor

def load_equiv_aug_augself(args):
    t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter_augself(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur_augself(23, (0.1, 2.0)))
    t2 = nn.Sequential(K.RandomHorizontalFlip(),
                        ColorJitter_augself(0.4, 0.4, 0.4, 0.1, p=0.8),
                        K.RandomGrayscale(p=0.2),
                        GaussianBlur_augself(23, (0.1, 2.0)))
    
    return t1, t2

class ColorJitter_augself(K.ColorJitter):
    # def apply_transform(
    #     self,
    #     x: Tensor,
    #     params: Dict[str, Tensor],
    #     flags: Dict[str, Any],
    #     transform: Optional[Tensor] = None
    # ) -> Tensor:
    def apply_transform(self, x, params):
        
        # build your list of ops as before, but refer to params[...] for each
        ops = [
            lambda img: apply_adjust_brightness(img, params),
            lambda img: apply_adjust_contrast(img, params),
            lambda img: adjust_saturation(img, params['saturation_factor']),
            lambda img: adjust_hue(img, params['hue_factor']),
        ]
        out = x
        for idx in params['order'].tolist():
            out = ops[idx](out)
        return out


class GaussianBlur_augself(K.AugmentationBase2D):
    def __init__(self, kernel_size, sigma, border_type='reflect',
                 return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        return dict(sigma=torch.zeros(batch_shape[0]).uniform_(self.sigma[0], self.sigma[1]))

    def apply_transform(self, input, params):
        sigma = params['sigma'].to(input.device)
        k_half = self.kernel_size // 2
        x = torch.linspace(-k_half, k_half, steps=self.kernel_size, dtype=input.dtype, device=input.device)
        pdf = torch.exp(-0.5*(x[None, :] / sigma[:, None]).pow(2))
        kernel1d = pdf / pdf.sum(1, keepdim=True)
        kernel2d = torch.bmm(kernel1d[:, :, None], kernel1d[:, None, :])
        input = F.pad(input, (k_half, k_half, k_half, k_half), mode=self.border_type)
        input = F.conv2d(input.transpose(0, 1), kernel2d[:, None], groups=input.shape[0]).transpose(0, 1)
        return input
    # def __init__(
    #     self,
    #     kernel_size: int,
    #     sigma: Tuple[float, float],
    #     border_type: str = 'reflect',
    #     p: float = 0.5,
    #     same_on_batch: bool = False,
    #     p_batch: float = 1.0,
    #     keepdim: bool = False,
    # ) -> None:
    #     """
    #     Args:
    #       kernel_size: must be odd
    #       sigma: (min, max) std-dev for sampling per image
    #       border_type: any mode supported by F.pad (e.g. 'reflect', 'constant', ...)
    #       p: probability to apply per sample
    #       same_on_batch: apply same sigma to all in batch
    #       p_batch: probability to apply to the batch as a whole
    #       keepdim: if True, return (out, transform); else just out
    #     """
    #     super().__init__(
    #         p=p,
    #         p_batch=p_batch,
    #         same_on_batch=same_on_batch,
    #         keepdim=keepdim
    #     )
    #     assert kernel_size % 2 == 1, "kernel_size must be odd"
    #     self.kernel_size = kernel_size
    #     self.sigma = sigma
    #     # store in flags so apply_transform can read it:
    #     self.flags: Dict[str, Any] = {"border_type": border_type}

    # def __repr__(self) -> str:
    #     return (f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
    #             f"sigma={self.sigma}, border_type={self.flags['border_type']}, "
    #             f"{super().__repr__()})")

    # def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Tensor]:
    #     # sample one sigma per image in [σ_min, σ_max]
    #     return {
    #         "sigma": torch.zeros(batch_shape[0])  # batch_shape[0] == B
    #                       .uniform_(self.sigma[0], self.sigma[1])
    #     }

    # def apply_transform(
    #     self,
    #     input: Tensor,
    #     params: Dict[str, Tensor],
    #     flags: Dict[str, Any],
    #     transform: Optional[Tensor] = None
    # ) -> Tensor:
    #     # 1) pad
    #     k = self.kernel_size
    #     pad = k // 2
    #     padded = F.pad(input, (pad, pad, pad, pad), mode=flags["border_type"])

    #     # 2) build batch-of-kernels
    #     sigma = params["sigma"].to(input.device)                 # [B]
    #     x     = torch.linspace(-pad, pad, steps=k, device=input.device, dtype=input.dtype)
    #     pdf   = torch.exp(-0.5 * (x[None, :] / sigma[:, None]).pow(2))
    #     k1d   = pdf / pdf.sum(dim=1, keepdim=True)               # [B, K]
    #     k2d   = torch.bmm(k1d[:, :, None], k1d[:, None, :])      # [B, K, K]

    #     # 3) grouped conv: swap B<->C so that conv sees batch as its “channel” axis
    #     x = padded.transpose(0, 1)                               # [C, B, H, W]
    #     out = F.conv2d(
    #         x,
    #         k2d[:, None],                                        # [B, 1, K, K]
    #         groups=x.shape[1]                                    # = B, the batch size
    #     )
    #     out = out.transpose(0, 1)                                # [B, C, H, W]

    #     return out
    

def apply_adjust_brightness(img1, params):
    ratio = params['brightness_factor'][:, None, None, None].to(img1.device)
    img2 = torch.zeros_like(img1)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)


def apply_adjust_contrast(img1, params):
    ratio = params['contrast_factor'][:, None, None, None].to(img1.device)
    img2 = 0.2989 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    img2 = torch.mean(img2, dim=(-2, -1), keepdim=True)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)