"""
@author:  Tongjia (Tom) Chen
@contact: tomchen@hnu.edu.cn
"""
from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.transforms as T

from model.C3D import C3D
from utils.UCF101 import UCF101
from utils.dataset_loader import VideoDataset
from utils.utils import AverageMeter, Logger, save_checkpoint
from utils.optimizers import init_optim
from IPython import embed

parser = argparse.ArgumentParser(description='Training C3D for action recognition')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 1)")
parser.add_argument('--optim', type=str, default='sgd', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=100, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=64, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=10, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# Misc
parser.add_argument('--print-freq', type=int, default=50, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # data augmentation
    transform_train = T.Compose([
        T.RandomCrop([112, 112]),
        T.RandomHorizontalFlip(p=0.5)
    ])

    transform_test = T.Compose([
        T.RandomCrop([112, 112]),
    ])

    dataset = UCF101()

    trainloader = DataLoader(
        VideoDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    testloader = DataLoader(
        VideoDataset(dataset.test, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model")
    model = C3D(num_classes=101)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, testloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_accuracy = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            accuracy = test(model, testloader, use_gpu)
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'accuracy': accuracy,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Accuracy {:.1%}, achieved at epoch {}".format(best_accuracy, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, (video, label) in enumerate(trainloader):
        if use_gpu:
            video, label = video.cuda(), label.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        outputs = model(video)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), label.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time, data_time=data_time,
                loss=losses))


def test(model, testloader, use_gpu):
    batch_time = AverageMeter()

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (video, label) in enumerate(testloader):
            if use_gpu: video = video.cuda()

            end = time.time()
            output = model(video)
            batch_time.update(time.time() - end)

            prediction = output.max(1, keepdim=True)[1].cpu()

            correct += prediction.eq(label.view_as(prediction)).sum().item()

        end = time.time()

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    print("Results ----------")
    accuracy = correct / len(testloader.dataset)
    print("Correct / all: {} / {} Accuracy: {:.1%}".format(correct ,len(testloader.dataset) ,accuracy))
    print("------------------")

    return accuracy


if __name__ == '__main__':
    main()
