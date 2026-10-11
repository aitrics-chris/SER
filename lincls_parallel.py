import os
import argparse
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

from builder import utils
import vision_transformer as vits
from concurrent.futures import ThreadPoolExecutor
import ruamel.yaml as yaml
from tqdm import tqdm
import resnet as resnets
import socket
# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
# from kornia.augmentation.container import ImageSequential
# import kornia


def normalize_eval_args(args):
    """Validate the parallel-run config and expand a shared seed per run."""
    fields = ('ckpts', 'lrs', 'gpus', 'arch', 'equiv_layer', 'if_baseline')
    count = len(args.ckpts)
    if not count:
        raise ValueError('Configure at least one checkpoint.')
    for name in fields:
        value = getattr(args, name)
        if not isinstance(value, list) or len(value) != count:
            raise ValueError(f'{name} must contain {count} entries, one per run.')
    if isinstance(args.seed, int):
        args.seed = [args.seed] * count
    if len(args.seed) != count or not all(isinstance(s, int) for s in args.seed):
        raise ValueError('seed must be an integer or one integer per run.')
    args.if_baseline = [v if isinstance(v, bool) else utils.bool_flag(v)
                        for v in args.if_baseline]
    if not 0 <= args.val_epoch < args.epochs:
        raise ValueError('Require 0 <= val_epoch < epochs to select a classifier.')
    if args.evaluate:
        raise ValueError('--evaluate is not implemented; this script fits a linear classifier.')


def build_backbone(args, idx):
    """Adapt a single evaluation run to the public training-model interface."""
    if args.arch[idx] == 'vit_small':
        if args.patch_size != 16:
            raise ValueError('The public vit_small factory supports patch_size=16 only.')
        baseline = args.if_baseline[idx]
        if not isinstance(baseline, bool):
            baseline = utils.bool_flag(baseline)
        layer = 12 if baseline else int(args.equiv_layer[idx])
        if not 1 <= args.n_last_blocks <= 12:
            raise ValueError('n_last_blocks must be between 1 and 12.')
        if not baseline and not 1 <= layer <= 12 - args.n_last_blocks:
            raise ValueError('SER requires all selected blocks to follow CLS insertion: '
                             '1 <= equiv_layer <= 12 - n_last_blocks.')
        model_args = argparse.Namespace(equiv_layer=layer,
                                       equiv_mode='inv' if baseline else 'erl')
        # Initialization is overwritten by the fully checked backbone state below.
        model = vits.vit_small(model_args, ssl_type='moco', drop_path_rate=0.0)
        dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    elif args.arch[idx] == 'resnet50':
        model = resnets.resnet50()
        model.head = nn.Identity()
        dim = 2048
    else:
        raise ValueError(f'Unknown architecture: {args.arch[idx]}')
    return model, dim


def load_backbone_checkpoint(model, path, checkpoint_key='auto'):
    """Load MoCo/DINO/Barlow Twins checkpoints written by this repository."""
    # Repository checkpoints include argparse namespaces; load only trusted files.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    key = checkpoint_key
    if key == 'auto':
        key = 'student' if 'student' in checkpoint else 'model'
    if key not in checkpoint:
        raise ValueError(f'Checkpoint {path} has no {key!r} entry; set --checkpoint_key.')
    state = {k.removeprefix('module.'): v for k, v in checkpoint[key].items()}
    if any(k.startswith('backbone.') for k in state):
        # DINO stores its MultiCropWrapper, including separate projection heads.
        state = {k.removeprefix('backbone.'): v for k, v in state.items()
                 if k.startswith('backbone.')}
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    # Never evaluate a partially initialized backbone silently.
    model.load_state_dict(state, strict=True)
    return key


def extract_features(model, images, n_last_blocks, avgpool_patchtokens):
    """Use identical frozen features in training, validation, and testing."""
    intermediate = model.get_intermediate_layers(images, n_last_blocks)
    if torch.is_tensor(intermediate):
        return intermediate  # ResNet returns pooled B x C features directly.
    features = torch.cat([tokens[:, 0] for tokens in intermediate], dim=-1)
    if avgpool_patchtokens:
        features = torch.cat((features, intermediate[-1][:, 1:].mean(dim=1)), dim=-1)
    return features


def parallel_apply(worker, inputs):
    """Propagate worker failures instead of continuing after a failed thread."""
    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        futures = [pool.submit(worker, *values) for values in inputs]
        for future in futures:
            future.result()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
def set_model(args, device_id, _idx, _model, _linear_classifier, _optimizer, _scheduler, _gradscaler):
    _model, embed_dim = build_backbone(args, _idx)
    key = load_backbone_checkpoint(_model, args.ckpts[_idx], args.checkpoint_key)
    print(f'Loaded {args.ckpts[_idx]} ({key})')

    _model.cuda(device_id)
    _model.requires_grad_(False)
    _model.eval()
    print(f'seed: {args.seed[_idx]}')
    utils.fix_random_seeds(args.seed[_idx])

    _linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    _linear_classifier = _linear_classifier.cuda(device_id)

    # set optimizer
    _optimizer = torch.optim.SGD(
        _linear_classifier.parameters(),
        args.lrs[_idx] * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, args.epochs, eta_min=0)

    # models[device_id] = _model
    # linear_classifiers[device_id] = _linear_classifier
    # optimizers[device_id] = _optimizer
    # schedulers[device_id] = _scheduler
    # gradscalers[device_id] = torch.cuda.amp.GradScaler()
    _gradscaler = torch.GradScaler(device="cuda")

    folder = os.path.join('results', Path(args.ckpts[_idx]).parent.name+f'_seed{args.seed[_idx]}_epoch{args.epochs}{args.tag}')
    os.makedirs(folder, exist_ok=True)

    args.txt_name[_idx] = os.path.join(folder, args.ckpts[_idx].split('/')[-1].split('.')[0])

    return _model, _linear_classifier, _optimizer, _scheduler, _gradscaler


def set_list_per_gpu(args):
    args.process_num = len(args.gpus)
    args.best_acc1 = [float('-inf')] * args.process_num
    args.best_acc5 = [0] * args.process_num
    args.best_epoch = [0] * args.process_num
    args.parameters = [None] * args.process_num
    args.txt_name = [None] * args.process_num

    args.models = [None] * args.process_num
    args.linear_classifiers = [None] * args.process_num
    args.best_classifier = [None] * args.process_num
    args.optimizers = [None] * args.process_num
    args.schedulers = [None] * args.process_num
    args.gradscalers = [None] * args.process_num
    args.acc1_meters = [None] * args.process_num
    args.acc5_meters = [None] * args.process_num
    for i in range(args.process_num):
        args.acc1_meters[i] = AverageMeter('acc1')
        args.acc5_meters[i] = AverageMeter('acc5')

def eval_linear(args):

    cudnn.benchmark = True

    if not os.path.exists('results'):
        os.makedirs('results')

    set_list_per_gpu(args)

    args.mean = torch.tensor((0.485, 0.456, 0.406)).view(1,3,1,1)
    args.std = torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1)

    for idx, device_id in enumerate(args.gpus):
        args.models[idx], args.linear_classifiers[idx], args.optimizers[idx], args.schedulers[idx], args.gradscalers[idx] = \
            set_model(args, device_id, idx, args.models[idx], args.linear_classifiers[idx], args.optimizers[idx], args.schedulers[idx], args.gradscalers[idx])
    
    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    
    for epoch in tqdm(range(0, args.epochs)):
        torch.cuda.empty_cache()
        train(args, train_loader)
        for device_id in range(args.process_num):
            args.schedulers[device_id].step()

        if epoch >= args.val_epoch:
            acc1_list, acc5_list = validate(args, val_loader, epoch) # scoring, save ckpt

            for idx_process in range(args.process_num):
                if (acc1_list[idx_process] > args.best_acc1[idx_process]):
                    args.best_epoch[idx_process] = epoch + 1
                    args.best_acc1[idx_process] = acc1_list[idx_process]
                    args.best_acc5[idx_process] = acc5_list[idx_process]
                    save_dict(args, idx_process, epoch, args.txt_name[idx_process] + f'_lr_{args.lrs[idx_process]}.pth.tar')
                    
            # acc1_list = [_acc1.cpu().item() for _acc1 in acc1_list_tensor]
            acc1_str = ', '.join(str(round(_acc1.item(), 4)) for _acc1 in acc1_list)
            print(f'Epoch {epoch}: {acc1_str}')        

    del train_loader, val_loader

    ####################### get test acc #######################
    for idx_process in range(args.process_num):  
        _save_dict = torch.load(args.txt_name[idx_process] + f'_lr_{args.lrs[idx_process]}.pth.tar', weights_only=False)
        args.linear_classifiers[idx_process].load_state_dict(_save_dict['state_dict'], strict=True)

    dataset_test = datasets.ImageFolder(args.test_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    acc1_list_test, acc5_list_test = validate(args, test_loader, -99)
    # acc1_list_test = [_acc1.cpu().item() for _acc1 in acc1_list_tensor_test]

    for idx_process in range(args.process_num):  
        file_path = args.ckpts[idx_process].split('.pth')[0]+f'_{socket.gethostname()}_{args.seed[idx_process]}_epoch{args.epochs}_lr_{args.lrs[idx_process]}'
        save_dict(args, idx_process, epoch, f'{file_path}.pth.tar')
        with open(file_path + '.txt', 'a' if os.path.isfile(file_path + '.txt') else 'w') as f:
            f.write(f'\nlr_init: {args.lrs[idx_process]}, best_acc1_val: {args.best_acc1[idx_process]}, acc1_test: {acc1_list_test[idx_process]}, acc5_test: {acc5_list_test[idx_process]}, best_epoch: {args.best_epoch[idx_process]}')
    
    # args.ckpts[_idx]
    ####################### end get test acc #######################

    
def save_dict(args, idx_process, epoch, fn):
    _save_dict = {
        "epoch": epoch + 1,
        "state_dict": args.linear_classifiers[idx_process].state_dict(),
        "encoder": args.models[idx_process].state_dict(),
        "optimizer": args.optimizers[idx_process].state_dict(),
        # "scheduler": args.schedulers[idx_process].state_dict(),
        "best_acc1": args.best_acc1[idx_process],
        "best_acc5": args.best_acc5[idx_process],
        "best_epoch": args.best_epoch[idx_process],
    }
    torch.save(_save_dict, fn)


# https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/parallel_apply.py#L30
# https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L153
# only for vit-small
def train(args, loader) -> None:
    
    def _worker(_image, _target, _model, _linear_classifier, _optimizer, _scaler, _gpu, _stream):
        _linear_classifier.train()
        # move to gpu
        _image = _image.cuda(_gpu, non_blocking=True)
        _target = _target.cuda(_gpu, non_blocking=True)

        # forward
        with torch.cuda.device(_gpu), torch.cuda.stream(_stream):
            with torch.autocast(device_type="cuda"):
            # forward
                with torch.no_grad():
                    output = extract_features(_model, _image, args.n_last_blocks,
                                              args.avgpool_patchtokens)
                logit = _linear_classifier(output)
                # compute cross entropy loss
                _loss = nn.CrossEntropyLoss()(logit, _target)

            _optimizer.zero_grad()            
            _loss.backward()
            _optimizer.step()
            # _scaler.scale(_loss).backward()
            # _scaler.step(_optimizer)
            # _scaler.update()
    # end _worker()

    devices = [torch.device(gpu) for gpu in args.gpus]
    streams = [torch.cuda.current_stream(device) for device in devices]

    for (image, target) in loader:
        image = image.sub_(args.mean).div_(args.std)
        parallel_apply(_worker, [
            (image, target, args.models[i], args.linear_classifiers[i],
             args.optimizers[i], args.gradscalers[i], args.gpus[i], streams[i])
            for i in range(args.process_num)
        ])



def validate(args, loader, epoch):
    num_correct = [0] * args.process_num
    num_all = [0] * args.process_num
    def _worker(_image, _target, _model, _linear_classifier, _optimizer, _scaler, _gpu, _stream, i, _acc1_meter, _acc5_meter):
        _linear_classifier.eval()
        # move to gpu
        _image = _image.cuda(_gpu, non_blocking=True)
        _target = _target.cuda(_gpu, non_blocking=True)

        # forward
        with torch.cuda.device(_gpu), torch.cuda.stream(_stream):
            with torch.autocast(device_type="cuda"):
            # forward
                with torch.no_grad():
                    output = extract_features(_model, _image, args.n_last_blocks,
                                              args.avgpool_patchtokens)
                    logits = _linear_classifier(output)

                    # top-1 only
                    # num_correct[i] += _target.eq(torch.argmax(logits, dim=1)).sum()
                    # num_all[i] += logits.shape[0]
                    acc1, acc5 = accuracy(logits, _target, topk=(1, 5))
                    _acc1_meter.update(acc1[0], logits.size(0))
                    _acc5_meter.update(acc5[0], logits.shape[0])

    devices = [torch.device(gpu) for gpu in args.gpus]
    streams = [torch.cuda.current_stream(device) for device in devices]

    for (image, target) in loader:
        image = image.sub_(args.mean).div_(args.std)
        parallel_apply(_worker, [
            (image, target, args.models[i], args.linear_classifiers[i],
             args.optimizers[i], args.gradscalers[i], args.gpus[i], streams[i],
             i, args.acc1_meters[i], args.acc5_meters[i])
            for i in range(args.process_num)
        ])

    acc1 = [meter.avg for meter in args.acc1_meters]
    acc5 = [meter.avg for meter in args.acc5_meters]
    [_meter.reset() for _meter in args.acc1_meters]
    [_meter.reset() for _meter in args.acc5_meters]
    return acc1, acc5

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

    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    config_path = Path(yml)
    if not config_path.is_file():
        config_path = Path(__file__).parent / 'config' / yml
    with config_path.open('r', encoding='utf-8') as f:
        dic = yaml.YAML(typ='safe').load(f)
        for k in dic:
            setattr(args, k, dic[k])


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    # parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="auto", type=str, help='Checkpoint entry: auto chooses student, or model for Barlow Twins; teacher is explicit.')
    parser.add_argument('--seed', default=0, type=int, help='Shared seed; YAML may specify one per run.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--val_epoch', default=60, type=int, help='Number of epochs of training.')

    parser.add_argument('--batch_size_per_gpu', default=2048, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='storage/imagenet_eval', type=str)
    parser.add_argument('--test_path', default='storage/imagenet/val', type=str)
    parser.add_argument('--num_workers', default=24, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='Unsupported legacy option; use linear-probe training instead.')
    # parser.add_argument('--equiv-layer', default=5, type=int, help='layer to impose equiv')
    # parser.add_argument('--if-baseline', default=False, type=utils.bool_flag)

    parser.add_argument('--tag', type=str, default="", help='tag')
    parser.add_argument('--config', type=str, default="eval.yaml", help='config file')


    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    over_write_args_from_file(args, args.config)
    normalize_eval_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError('Linear-probe training requires CUDA. See tests/ for CPU checks.')
    eval_linear(args)
