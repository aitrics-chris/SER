import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import moco.vision_transformer_eval as vits
import threading
import ruamel.yaml as yaml
from pathlib import Path
from tqdm import tqdm
import moco.resnet as resnets
import socket
# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
# from kornia.augmentation.container import ImageSequential
# import kornia

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
    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    
    if args.arch[_idx] == 'vit_small':
        assert args.arch[_idx] in vits.__dict__.keys()
        _model = vits.__dict__[args.arch[_idx]](patch_size=args.patch_size, num_classes=0, cls_layer=int(args.cls_layer[_idx]), if_baseline=args.if_baseline[_idx].upper() == 'TRUE')  
        embed_dim = _model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        
    elif args.arch[_idx] == 'resnet50':
        embed_dim = 2048
        _model = resnets.resnet50()
        _model.head = nn.Identity()
    else:
        raise ValueError('Unknown architecture: {}'.format(args.arch[_idx]))
    
    ssl_type = args.ckpts[_idx].split('ckpt_')[-1].split('/')[0]
    state_dict = torch.load(args.ckpts[_idx], map_location='cpu', weights_only=False)['student']

    # if ssl_type == 'moco':
    #     if 'augself' in args.ckpts[_idx]:
    #         state_dict = torch.load(args.ckpts[_idx], map_location='cpu', weights_only=False)
    #     else:
    #         state_dict = torch.load(args.ckpts[_idx], map_location='cpu', weights_only=False)['student']
    # elif ssl_type == 'dino':
    #     # load weights to evaluate
    #     state_dict = torch.load(args.ckpts[device_id], map_location='cpu', weights_only=False)[args.checkpoint_key]
    #     # remove `module.` prefix
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # remove `backbone.` prefix induced by multicrop wrapper
    #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #     for k in list(state_dict.keys()):
    #         if ('head' in k) or ('projector_equiv' in k):
    #             del state_dict[k]  
    #         if args.if_baseline[_idx].upper() == 'TRUE':
    #             if ('norm_equiv' in k):
    #                 del state_dict[k]
    # elif ssl_type == 'barlowtwins':
    #     state_dict = torch.load(args.ckpts[_idx], map_location='cpu', weights_only=False)['model']
    
    missing_keys, unexpected_keys = _model.load_state_dict(state_dict, strict=False)
    
    for a in unexpected_keys:
        if 'norm' not in a:
            raise ValueError
        else:
            print(f'unexpected_keys: {a}')
        
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

    folder = os.path.join('results', args.ckpts[_idx].split('/')[-2]+f'_seed{args.seed[_idx]}_epoch{args.epochs}{args.tag}')
    if not os.path.exists(folder):
        os.makedirs(folder)    

    args.txt_name[_idx] = os.path.join(folder, args.ckpts[_idx].split('/')[-1].split('.')[0])

    return _model, _linear_classifier, _optimizer, _scheduler, _gradscaler


def set_list_per_gpu(args):
    args.process_num = len(args.gpus)
    args.best_acc1 = [0] * args.process_num
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
        file_path = args.ckpts[idx_process].split('.pth')[0]+f'_{socket.gethostname()}_{args.seed[idx_process]}_epoch{args.epochs}'
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
                    intermediate_output = _model.get_intermediate_layers(_image, args.n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
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
        threads = [
                        threading.Thread(
                            target=_worker, args=(image, target, args.models[i], args.linear_classifiers[i], args.optimizers[i], args.gradscalers[i], args.gpus[i], streams[i])
                        )
                        # for i, (module, input, kwargs, device, stream) in enumerate(
                        #     zip(args.models, args.linear_classifiers, args.optimizers, args.schedulers, inputs, n_last_blocks, devices, streams)
                        # )
                        for i in range(args.process_num)
                    ]     

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


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
                    intermediate_output = _model.get_intermediate_layers(_image, args.n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

                        # False for ViT-small
                    # if args.avgpool_patchtokens:
                    #     output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    #     output = output.reshape(output.shape[0], -1)
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
        threads = [
                        threading.Thread(
                            target=_worker, args=(image, target, args.models[i], args.linear_classifiers[i], args.optimizers[i], args.gradscalers[i], \
                                                  args.gpus[i], streams[i], i, args.acc1_meters[i], args.acc5_meters[i])
                        )
                        # for i, (module, input, kwargs, device, stream) in enumerate(
                        #     zip(args.models, args.linear_classifiers, args.optimizers, args.schedulers, inputs, n_last_blocks, devices, streams)
                        # )
                        for i in range(args.process_num)
                    ]     

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

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
    with open(os.path.join('config', yml), 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    # parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    # parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--val_epoch', default=60, type=int, help='Number of epochs of training.')

    parser.add_argument('--batch_size_per_gpu', default=2048, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='storage/imagenet_eval', type=str)
    parser.add_argument('--test_path', default='storage/imagenet/val', type=str)
    parser.add_argument('--num_workers', default=24, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    # parser.add_argument('--equiv-layer', default=5, type=int, help='layer to impose equiv')
    # parser.add_argument('--if-baseline', default=False, type=utils.bool_flag)

    parser.add_argument('--tag', type=str, default="", help='tag')
    parser.add_argument('--config', type=str, default="eval.yaml", help='config file')


    args = parser.parse_args()

    over_write_args_from_file(args, args.config)
    
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["TORCH_USE_CUDA_DSA"] = '1'

    eval_linear(args)


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONHASHSEED=1 python eval_parallel.py --epochs 50