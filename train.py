# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 14:46 
@File : train.py 
@Desc : 
'''
import os
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
# from torchvision.models import resnet
import resnet
from utils.utils import setup_logger, WarmupPolyLR
from utils.io_utils import write_yaml


def init_args():
    params = argparse.ArgumentParser()
    params.add_argument('--model_name', type=str, default='resnet18', help='model_name')
    params.add_argument('--data_root', type=str, default='./data/', help='data root')
    params.add_argument('--epochs', type=int, default=100, help='epochs')
    params.add_argument('--batch_size', type=int, default=128, help='batch size')
    params.add_argument('--lr', type=float, default=1e-3, help='lr')
    params.add_argument('--save_dir', type=str, default='./output', help='save_dir')
    params.add_argument('--log_iter', type=int, default=20, help='log iter')
    params.add_argument('--warmup', type=bool, default=True, help='warmup')
    params.add_argument('--warmup_epoch', type=int, default=1, help='warmup_epoch')
    params.add_argument('--save_latest', type=bool, default=True, help='save latest')
    params.add_argument('--resume', type=str, default='', help='resume')
    args = params.parse_args()
    return args


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(args.save_dir, f'{args.model_name}-{time.strftime("%Y-%m-%d-%H-%M-%S")}')
    # 保存配置
    model_save_dir = os.path.join(save_dir, 'model')
    logs_save_dir = os.path.join(save_dir, 'logs')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_save_dir, exist_ok=True)
    write_yaml(vars(args), os.path.join(save_dir, 'config.yaml'))
    logger_save_path = os.path.join(logs_save_dir, 'train.log')
    logger = setup_logger(logger_save_path)
    logger.info(args)
    writer = SummaryWriter(logs_save_dir)
    # 数据
    logger.info('Prepare Data...')
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, transform=train_transform, download=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=test_transform, download=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_loader_len = len(train_dataloader)
    logger.info('train: {} dataloader, test: {} dataloader'.format(len(train_dataloader), len(test_dataloader)))
    # 模型
    logger.info('Prepare Model...')
    best_acc = 0
    best_epoch = 0
    # 自己写的
    # model = vgg16(num_classes=2, is_dropout=args.is_dropout, is_bn=args.is_bn)
    # torchvision
    if args.resume:
        logger.info(f'Resume From {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model = getattr(resnet, args.model_name)(pretrained=False, num_classes=10)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        best_epoch = ckpt['best_epoch']
    else:
        start_epoch = 0
        model = getattr(resnet, args.model_name)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    # 优化器
    if args.resume:
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # WarmupPolyLR
    if args.warmup:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5,last_epoch=-1)
        warmup_iters = args.warmup_epoch * train_loader_len
        if start_epoch > 1:
            last_epoch = (start_epoch - 1) * train_loader_len
        else:
            last_epoch = -1
        scheduler = WarmupPolyLR(optimizer, max_iters=args.epochs * train_loader_len, warmup_iters=warmup_iters, warmup_epoch=args.warmup_epoch, last_epoch=last_epoch)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    logger.info('Train Begin ...')

    global_step = 0
    for epoch in range(start_epoch+1, args.epochs):
        model.train()
        train_acc = 0
        train_loss = 0
        for idx, (data, targets) in enumerate(train_dataloader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if args.warmup:
                scheduler.step()
            train_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            # 计算准确率
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            acc = (preds == targets).sum() / args.batch_size
            train_acc += (preds == targets).sum()
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', acc, global_step)
            writer.add_scalar('train/lr', lr, global_step)
            global_step += 1
            if (idx + 1) % args.log_iter == 0:
                logger.info(f'[{epoch}/{args.epochs}] [{idx}/{len(train_dataloader)}] global_step: {global_step}, '
                            f'lr: {lr:.6f}, acc: {(preds == targets).sum()/args.batch_size:.4f}, loss: {loss.item():.6f}')
        train_loss_mean = train_loss / (args.batch_size * len(train_dataloader))
        train_acc_mean = train_acc / (args.batch_size * len(train_dataloader))
        writer.add_scalar('train/loss_mean', train_loss_mean, epoch)
        writer.add_scalar('train/acc_mean', train_acc_mean, epoch)
        test_acc, test_loss = val(model, test_dataloader, device, criterion)
        writer.add_scalar('test/acc', test_acc, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        logger.info(f'[{epoch}/{args.epochs}] lr: {optimizer.param_groups[0]["lr"]:.6f}, '
                    f'train_acc: {train_acc_mean:.4f}, train_loss: {train_loss_mean:.6f}, '
                    f'test acc: {test_acc:.4f}, test loss: {test_loss:.6f}')
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            ckpt = {
                'epoch': epoch,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict(),
                'global_step': global_step
            }
            torch.save(ckpt, os.path.join(model_save_dir, f'{args.model_name}-best.pth'))
        if args.save_latest:
            ckpt = {
                'epoch': epoch,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict(),
                'global_step': global_step
            }
            torch.save(ckpt, os.path.join(model_save_dir, f'{args.model_name}-latest.pth'))
        logger.info(f'[{epoch}/{args.epochs}] current best: acc: {best_acc:.4f}, epoch: {best_epoch}')
    writer.close()
    logger.info('Train Finish.')


@torch.no_grad()
def val(model, dataloader, device, criterion):
    model.eval()
    loss, acc, num = 0, 0, 0
    for idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        loss += criterion(logits, targets).item()
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)
        num += targets.size(0)
        acc += (preds == targets).sum()
    return acc.float() / num, loss / num


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = init_args()
    train(args)