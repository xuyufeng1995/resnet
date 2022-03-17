import datetime
import os
import math
import tempfile
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.model import resnet34
from utils.datasets import MyDataSet
from utils.distributed_utils import init_distributed_mode, dist, cleanup, save_on_master
from utils.train_eval_utils import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from utils.utils import read_dataset_and_show
from loguru import logger
from pathlib import Path


def main(args):
    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank, batch_size, output_dir, data_path = args.rank, args.batch_size, args.output_dir, args.data_path

    device = torch.device(args.device)

    # Directories
    save_dir = Path(output_dir) / "train/{}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    save_dir.mkdir(parents=True, exist_ok=True)

    weight = save_dir / "weights"
    weight.mkdir(parents=True, exist_ok=True)
    last, best = weight / 'last.pth', weight / 'best.pth'
    results_file = save_dir / 'results.txt'

    # Tensorboard
    tb_writer = None
    if rank in [-1, 0]:  # 在第一个进程中打印信息，并实例化tensorboard
        logger.info(args)
        logger.info('Start Tensorboard with "tensorboard --logdir=runs --host=0.0.0.0", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=str(save_dir))

    # check num_classes 遍历文件夹，一个文件夹对应一个类别
    train_path, train_label, val_path, val_label, num_classes = read_dataset_and_show(data_path, save_dir, tb_writer)
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_path,
                              images_class=train_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_path,
                            images_class=val_label,
                            transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        logger.info('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    # 实例化模型
    model = resnet34(num_classes=num_classes).to(device)

    # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 接着上次训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint["epoch"] + 1

    best_fitness = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    warmup=True)

        lr_scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)

        acc = sum_num / len(val_sampler)
        if args.output_dir:
            # 只在主节点保存权重
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }

            save_on_master(save_files, last)
            if best_fitness < acc:
                best_fitness = acc
                save_on_master(save_files, best)

            # plot
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/home/xuyufeng/dataset/delivery")
    parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=0, help='start-epoch')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate, 0.01 is the default value for training')
    parser.add_argument('--lrf', type=float, default=0.2)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./runs', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
