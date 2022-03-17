import sys

from tqdm import tqdm
import torch
import utils.distributed_utils as utils
from .loss import ComputeLoss


def train_one_epoch(model, optimizer, data_loader, device, epoch, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    compute_loss = ComputeLoss(model)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mean_loss = torch.zeros(1).to(device)

    for step, [images, labels] in enumerate(metric_logger.log_every(data_loader, 100, header)):
        pred = model(images.to(device))
        loss = compute_loss(pred, labels.to(device))

        # reduce losses over all GPUs for logging purpose
        loss = utils.reduce_value(loss)

        # 记录训练损失
        mean_loss = (mean_loss * step + loss) / (step + 1)  # update mean losses

        if not torch.isfinite(loss):  # 当损失为无穷大时停止训练
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:   # 第一轮使用warmup训练
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for step, [images, labels] in enumerate(data_loader):
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = utils.reduce_value(sum_num, average=False)

    return sum_num.item()
