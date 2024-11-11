import argparse
import os
import random
import time
import warnings
from datetime import datetime
from collections import OrderedDict
import math
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json

import torch
from torch.cuda.amp.autocast_mode import autocast
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.cuda.amp as amp
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
from ipt import ipt_base
from datasets import *
from datetime import datetime
from pytorch_msssim import ms_ssim
import PIL


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--task-name', default='', type=str, metavar='TASK',
                    help='name of the task')
parser.add_argument('-d','--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--eval-data', metavar='DIR', default='./data',
                    help='path to eval dataset')
parser.add_argument('--num-dataset-t', type=lambda s: [int(item) for item in s.split(',')], default=[0, 190],
                    metavar='[start, end]',
                    help='number of samples to use from the training dataset')
parser.add_argument('--num-dataset-v', type=lambda s: [int(item) for item in s.split(',')], default=[190, 200],
                    metavar='[start, end]',
                    help='number of samples to use from the validating dataset')
parser.add_argument('-s','--save-path', metavar='DIR', default='./ckpt',
                    help='path to save checkpoints')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-policy', default='naive',
                    help='lr policy')
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--warmup-lr-multiplier', default=0.1, type=float, metavar='W',
                    help='warmup lr multiplier')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1-4)',
                    dest='weight_decay')
parser.add_argument('--power', default=1.0, type=float,
                    metavar='P', help='power for poly learning-rate decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--reset-epoch', action='store_true', 
                    help='whether to reset epoch')
parser.add_argument('--eval', action='store_true', 
                    help='only do evaluation')         
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--task', default='', type=str, metavar='string',
                    help='specific a task'
                    '["denoise30", "denoise50", "SRx2", "SRx3", "SRx4", "dehaze"] (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16',action='store_true', default=False, help="\
                    use fp16 instead of fp32.")
parser.add_argument('--reset-task', action='store_true',
                    help='reset task and clean the save path')
parser.add_argument('--model-name', default='ImageProcessingTransformer', type=str,
                    help='name of the model to use')
parser.add_argument('--patch-size', default=8, type=int,
                    help='patch size')
parser.add_argument('--crop-size', default=256, type=int,
                    help='crop size')
parser.add_argument('--mid-channels', default=64, type=int,
                    help='mid channels')
parser.add_argument('--heads-tails-num', default=1, type=int,
                    help='heads and tails number')
parser.add_argument('--in-channels', default=1, type=int,
                    help='input channels')
parser.add_argument('--purge-step', default=None, type=int,
                    help='purge step')
parser.add_argument('--use-ms-ssim', action='store_true', default=False, help="use MS-SSIM loss instead of L1 loss.")
parser.add_argument('--alpha-ms-ssim-l1', type=float, default=0.8,
                    metavar='alpha for ms-ssim',
                    help='loss = alpha*ms-ssim + (1-alpha)*l1')
parser.add_argument('--win-size', default=5, type=int, help='window size for ms-ssim')
parser.add_argument('--use-data-aug', action='store_true', default=False, help='whether to use data augmentation')
parser.add_argument('--use-wf', action='store_true', default=False, help='whether to use wf images')

best_acc1 = 0
# set task sets


def main():
    args = parser.parse_args()
    if args.task_name == '':
        # test script

        # args.data = './data/tubulin-sim-800x800-processed.npz'
        # args.eval_data = './data/tubulin-sim-800x800-processed.npz'
        # args.workers = 2
        # args.epochs = int(2e3) # 300
        # args.batch_size = 32
        # args.lr = 5e-5 # batch_size=256
        # args.print_freq = 8
        # args.evaluate = True
        # # args.task = None
        # # args.seed = 0
        # args.patch_size = 4 # 48
        # args.crop_size = 48

        # args.workers = 2
        args.task_name = 'exp_test'
        args.reset_task = True
        args.epochs = int(300)  # 300
        args.batch_size = 1
        args.lr = 5e-5  # batch_size=256
        args.print_freq = int(40/args.batch_size)
        args.evaluate = True
        # args.task = None
        # args.seed = 0
        args.patch_size = int(8)
        args.crop_size = int(256/2)
        args.mid_channels = int(9)
        args.heads_tails_num = int(1)
        args.in_channels = 1 # 3
        args.purge_step = None
        args.model_name='ImageProcessingTransformer'

        args.data = './data/tubulin-sim-800x800-processed-easy.npz'
        args.eval_data = './data/tubulin-sim-800x800-processed-easy.npz'
        args.num_dataset_t = [0, 1]
        args.num_dataset_v = [191, 200]
        args.use_ms_ssim = True
        args.alpha_ms_ssim = 0.8
        args.use_wf = True
        # args.workers = 2
        # args.epochs = int(300)  # 300
        # args.batch_size = 32
        # args.lr = 4e-7  # batch_size=256
        # args.print_freq = int(40 / args.batch_size)
        # args.evaluate = True
        # # args.task = None
        # # args.seed = 0
        # args.patch_size = int(4)
        # args.crop_size = int(256/2)
        # args.mid_channels = int(64)
        # args.heads_tails_num = int(1)
        # args.in_channels = 1  # 3
        # args.purge_step = None
        # args.model_name = "ImageProcessingTransformer_plus"
        # args.lr_policy = 'epoch_poly'
        # args.power = 2

        # args.workers = 2
        # args.epochs = int(300)  # 300
        # args.batch_size = 1
        # args.lr = 4e-5  # batch_size=256
        # args.print_freq = int(40 / args.batch_size)
        # args.evaluate = True
        # # args.task = None
        # # args.seed = 0
        # args.patch_size = int(4)
        # args.crop_size = int(256/2)
        # args.mid_channels = int(64)
        # args.heads_tails_num = int(1)
        # args.in_channels = 1  # 3
        # args.purge_step = None
        # args.model_name = "ImageProcessingTransformer_plus"
        # args.lr_policy = 'naive'
        # args.power = 2


    now = datetime.now()
    timestr = now.strftime("%m-%d-%H_%M_%S")
    # args.save_path = os.path.join(args.save_path, f"{args.task}" if args.task else "train")
    if args.task_name == '':
        args.save_path = os.path.join(args.save_path, timestr)
    else:
        args.save_path = os.path.join(args.save_path, args.task_name)
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif args.reset_task:
        import shutil
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print(f"Reset task and clean the save path: {save_path}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    writer_train = SummaryWriter(log_dir=os.path.join(args.save_path, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(args.save_path, 'val'))
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format("ipt_base"))
    model = ipt_base(in_channels=args.in_channels,patch_size=args.patch_size, crop_size=args.crop_size,
                     mid_channels=args.mid_channels,
                     heads_tails_num=args.heads_tails_num, model_name=args.model_name, use_wf=args.use_wf).cuda()



    # define loss function (criterion) and optimizer

    # IPT uses L1 loss function
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if not args.use_ms_ssim:
        criterion = nn.L1Loss()
    else:
        criterion = MS_SSIM_L1_LOSS(alpha=args.alpha_ms_ssim_l1, win_size=args.win_size)
        # print(f"Using MS-SSIM loss with alpha={args.alpha_ms_ssim_l1}")

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.reset_epoch:
                args.start_epoch = checkpoint['epoch']
            #args.start_epoch = 10
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    input_size = 48

    # Data loading code
    trans_val = transforms.Compose([
        transforms.ToTensor(),  # 0~255 to 0~1
        # transforms.RandomCrop(args.crop_size),
        transforms.CenterCrop(args.crop_size),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    if not args.use_data_aug:
        trans_train = trans_val
    else:
        trans_train = transforms.Compose([
            transforms.ToTensor(),  # 0~255 to 0~1
            # transforms.RandomRotation(degrees=(0,360), resample=PIL.Image.BILINEAR),
            transforms.CenterCrop(args.crop_size+10),
            transforms.RandomCrop(args.crop_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    if args.eval:
        val_dataset = ImageProcessDataset(args.eval_data, num=args.num_dataset_v, transform=trans_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        #raise RuntimeError("evaluate dataloader not implemented")
        validate(val_loader, model, criterion, args)
        return
    
    train_dataset = ImageProcessDataset(args.data, num=args.num_dataset_t, transform=trans_train, use_wf=args.use_wf)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.use_data_aug,
                              num_workers=args.workers)

    val_dataset = ImageProcessDataset(args.eval_data, num=args.num_dataset_v, transform=trans_val, use_wf=args.use_wf)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    args.epoch_size = len(train_loader)
    print(f"Each epoch contains {args.epoch_size} iterations")

    print(f"Using {args.lr_policy} learning rate")

    if args.distributed:
        raise RuntimeError("distributed not implemented")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    scaler = amp.GradScaler() if args.fp16 else None

    print(args)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        iter_num = train(train_loader, model, criterion, optimizer, epoch, args, scaler, writer=writer_train)

        # evaluate on validation set
        validate(val_loader, model, criterion, args, writer_val, iter_num)

    # save model when training is finished
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        model_to_save = getattr(model, "module", model)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, path=args.save_path)
    writer_train.close()
    writer_val.close()


task_map = {"denoise30": 0, "denoise50": 1, "SRx2": 2, "SRx3": 3, "SRx4": 4, "dehaze": 5}

def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None, writer=None):
    # train for one epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnr_out = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.lr_policy == 'naive':
        local_lr = adjust_learning_rate_naive(optimizer, epoch, args)
    elif args.lr_policy == 'step':
        local_lr = adjust_learning_rate(optimizer, epoch, args)
    elif args.lr_policy == 'epoch_poly':
        local_lr = adjust_learning_rate_epoch_poly(optimizer, epoch, args)
        
    
    for i, (target, input_group) in enumerate(train_loader):

        # set random task
        # task_id = random.randint(0, 5) if not args.task else task_map[args.task]
        # input = input_group[task_id]
        task_id = 0
        input = input_group
        input = input[:,:args.in_channels,...]
        if args.use_wf:
            input = torch.cat((input, input_group[:, -1:, ...]), dim=1)

        target = target[:,:args.in_channels,...]
        model.module.set_task(task_id)
        #print(f"Iter {i}, task_id: {task_id}")
        #for m in model.module.modules():
           # if isinstance(m, )
            #print(m.weight.device)
        global_iter = epoch * args.epoch_size + i
        
        if args.lr_policy == 'iter_poly':
            local_lr = adjust_learning_rate_poly(optimizer, global_iter, args)
        elif args.lr_policy == 'cosine':
            local_lr = adjust_learning_rate_cosine(optimizer, global_iter, args)
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        target = target.cuda()
        if scaler is None:
            # compute output
            output = model(input)
            #print(output.device, target.device)
            loss = criterion(output, target)
        else:
            with autocast():
                # compute output
                output = model(input)
                #print(output.device, target.device)
                loss = criterion(output, target)

        # measure accuracy and record loss
        output = (output * 0.5 + 0.5) * 255.
        output = output.clamp(min=0, max=255)
        target = (target * 0.5 + 0.5) * 255.
        psnr = PSNR()(output, target)
        losses.update(loss.item(), input.size(0))
        psnr_out.update(psnr.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if scaler is None:
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'
        #           'LR: {lr: .6f}'.format(
        #            epoch, i, args.epoch_size, batch_time=batch_time,
        #            data_time=data_time, loss=losses, psnr=psnr_out, lr=local_lr))
        #     if writer is not None:
        #         writer.add_scalar('Loss/train', losses.avg, epoch * args.epoch_size + i)
        #         writer.add_scalar('PSNR/train', psnr_out.avg, epoch * args.epoch_size + i)
        #         writer.add_scalar('lr/train', local_lr, epoch * args.epoch_size + i)
        #         writer.add_image('Output', output[0]/255, epoch * args.epoch_size + i)
        #         writer.add_image('Input', (input[0] * 0.5 + 0.5)*255., epoch * args.epoch_size + i)
        #         writer.add_image('Target', target[0]/255, epoch * args.epoch_size + i)

    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Data {data_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'PSNR {psnr.avg:.3f}\t'
          'LR: {lr: .6f}'.format(
        epoch, i, args.epoch_size, batch_time=batch_time,
        data_time=data_time, loss=losses, psnr=psnr_out, lr=local_lr))
    if writer is not None:
        output_show = output[0] / 255
        input_show = (input[0] * 0.5 + 0.5) * 255.
        target_show = target[0] / 255
        iter_num = epoch * args.epoch_size + i
        writer.add_scalar('Loss', losses.avg, iter_num)
        writer.add_scalar('PSNR', psnr_out.avg, iter_num)
        writer.add_scalar('lr', local_lr, iter_num)
        writer.add_image('Output', output_show, iter_num)
        if not args.use_wf:
            writer.add_image('Input', input_show/255, iter_num)
        else:
            writer.add_image('Input/blinking', input_show[:1]/255, iter_num)
            writer.add_image('Input/wf', input_show[1:]/255, iter_num)
        writer.add_image('Target', target_show, iter_num)
        if output_show.shape[0] == 3:
            output_show = output[0, :1, :, :] / 255
            target_show = target[0, :1, :, :] / 255
        # Mix output_show (red) and target_show (green)
        mixed_show = torch.zeros_like(output_show).repeat(3, 1, 1)
        mixed_show[0] = output_show[0]  # Red channel
        mixed_show[1] = target_show[0]  # Green channel
        writer.add_image('Mixed', mixed_show, iter_num)
        return iter_num
    return None


def validate(val_loader, model, criterion, args, writer=None, iter_num=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnr_out = AverageMeter()
    psnr_in = AverageMeter()

    # switch to evaluate mode
    model.eval()
    P = PSNR()
    with torch.no_grad():
        end = time.time()
        for i, (target, input_group) in enumerate(val_loader):
            # task_id = task_map[args.task]
            # input = input_group[task_id]
            task_id = 0
            input = input_group
            input = input[:, :args.in_channels, ...]
            if args.use_wf:
                input = torch.cat((input, input_group[:, -1:, ...]), dim=1)
            target = target[:, :args.in_channels, ...]
            model.module.set_task(task_id)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            output = (output * 0.5 + 0.5) * 255.
            output = output.clamp(min=0, max=255)
            target = (target * 0.5 + 0.5) * 255.
            psnr1 = P(output, target)
            # psnr2 = P(input.cuda(), target)
            losses.update(loss.item(), input.size(0))
            psnr_out.update(psnr1.item(), input.size(0))
            # psnr_in.update(psnr2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # print('Test: [{0}/{1}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'PSNR_Out {psnr1.val:.3f} ({psnr1.avg:.3f})\t'
        #       'PSNR_In {psnr2.val:.3f} ({psnr2.avg:.3f})'.format(
        #        i, len(val_loader), batch_time=batch_time, loss=losses, psnr1=psnr_out, psnr2=psnr_in
        #     ))

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'PSNR_Out {psnr1.avg:.3f}'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses, psnr1=psnr_out
        ))

        if writer is not None and iter_num is not None:
            output_show = output[0] / 255
            input_show = (input[0] * 0.5 + 0.5) * 255.
            target_show = target[0] / 255
            writer.add_scalar('Loss', losses.avg, iter_num)
            writer.add_scalar('PSNR', psnr_out.avg, iter_num)
            writer.add_image('Output', output_show, iter_num)
            if not args.use_wf:
                writer.add_image('Input', input_show/255, iter_num)
            else:
                writer.add_image('Input/blinking', input_show[:1]/255, iter_num)
                writer.add_image('Input/wf', input_show[1:]/255, iter_num)
            writer.add_image('Target', target_show, iter_num)

            if output_show.shape[0] == 3:
                output_show = output[0, :1, :, :] / 255
                target_show = target[0, :1, :, :] / 255
            # Mix output_show (red) and target_show (green)
            mixed_show = torch.zeros_like(output_show).repeat(3, 1, 1)
            mixed_show[0] = output_show[0]  # Red channel
            mixed_show[1] = target_show[0]  # Green channel
            writer.add_image('Mixed', mixed_show, iter_num)

    return psnr_out.avg


def save_checkpoint(state, path='./', filename='checkpoint'):
    saved_path = os.path.join(path, filename+'.pth.tar')
    torch.save(state, saved_path)
    '''
    if is_best:
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        best_path = os.path.join(path, 'model_best.pth')
        for key in state_dict.keys():
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key].cpu()
            else:
                new_state_dict[key] = state_dict[key].cpu()
        torch.save(new_state_dict, best_path)
    '''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate_naive(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr if epoch < 800 else 2/5 * args.lr # <200
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def adjust_learning_rate_epoch_poly(optimizer, epoch, args):
    """Sets epoch poly learning rate"""
    lr = args.lr * ((1 - epoch * 1.0 / args.epochs) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_poly(optimizer, global_iter, args):
    """Sets iter poly learning rate"""
    lr = args.lr * ((1 - global_iter * 1.0 / (args.epochs * args.epoch_size)) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_cosine(optimizer, global_iter, args):
    warmup_lr = args.lr * args.warmup_lr_multiplier
    max_iter = args.epochs * args.epoch_size
    warmup_iter = args.warmup_epochs * args.epoch_size
    if global_iter < warmup_iter:
        slope = (args.lr - warmup_lr) / warmup_iter
        lr = slope * global_iter + warmup_lr
    else:
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (global_iter - warmup_iter) / (max_iter - warmup_iter)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
'''
class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()
'''

class MS_SSIM_L1_LOSS:
    def __init__(self, alpha=0.8, win_size=5):
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.win_size = win_size
        print(f"Using MS-SSIM loss with alpha={alpha}, window size={win_size}")

    def __call__(self, output, target):
        l1_loss = self.l1_loss(output, target)
        ms_ssim_loss = 1 - ms_ssim(output, target, data_range=1.0, size_average=True, win_size=self.win_size)
        return self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss

if __name__ == '__main__':
    main()
