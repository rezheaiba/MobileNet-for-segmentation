import os
import time
import datetime
import random
from PIL import Image

import torch
from torch import nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import my_modle
from utils.my_dataset import VOCSegmentation
from utils.distributed_utils import ConfusionMatrix
from utils import log
import utils.transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.MaskResize(16),  # 注意此处将目标图缩小了16x
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.MaskResize(16),# 注意此处将目标图缩小了16x
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train):
    base_size = 288
    crop_size = 256
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(crop_size)

def create_lr_scheduler(optimizer, epochs: int = 0):
    '''
    自定义学习率 因子 更新策略
    lr = lr * (gamma^step)
    '''
    def f(step):
        gamma = 0.95
        lr = gamma**step
        lr_min = 0.00001
        lr = lr_min if lr < lr_min else lr
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def criterion(inputs, target):
    '''
    计算交叉熵损失函数
    '''
    losses = {}

    #降低背景权重
    channel = inputs.shape[1]
    weight = torch.ones(channel)
    weight[0] = 0.2
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    losses = nn.functional.cross_entropy(inputs, target, ignore_index=255,weight = weight.to(inputs.device))

    return losses

def main(args):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # read args
    start_epoch=0
    epochs = args.epochs
    lr= args.lr
    # resume="./weights/model_220.pth"
    resume=args.resume
    batch_size = 32
    classes = 20
    # segmentation nun_classes + background
    num_classes = classes + 1

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # dataset
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(transforms=get_transform(train=True),
                                    txt_name="train.txt")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size,
                                            num_workers = num_workers,
                                            shuffle = True,
                                            pin_memory = True,
                                            collate_fn=train_dataset.collate_fn
    )

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(transforms=get_transform(train=False),
                                    txt_name="val.txt")

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=val_dataset.collate_fn)

    # creat model
    model = my_modle(num_classes=num_classes,
        reduced_tail=True,
        backbone="mobilenet_v3_small")
    model.to(device)

    # optimizer
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.SGD(params_to_optimize,lr=lr, momentum=0.9, weight_decay=0.0001)

    # 自定义学习率更新策略 
    lr_scheduler = create_lr_scheduler(optimizer)

    # train loss fun
    loss_func = criterion

    # 创建 混淆矩阵 class 用于评估模型
    confmat = ConfusionMatrix(num_classes)

    # 实例化SummaryWriter对象
    tb_writer=SummaryWriter()
    init_img = torch.zeros((1,3,256,256),device = device)
    tb_writer.add_graph(model,init_img,) # 将模型写入tensorboard 方便查看模型

    # creat logger 用来保存训练以及验证过程中信息
    logger_file = "./runs/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = log.create_logger(logger_file)

    # 载入历史信息
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 不加载 从头开始
        start_epoch = checkpoint['epoch'] + 1

    train_steps = len(train_loader)
    for epoch in range(start_epoch,start_epoch+epochs):
        logger.info(f"----[epoch: {epoch}]")
        # train
        model.train()
        running_loss=0.0
        lr = optimizer.param_groups[0]["lr"]
        for image, target in  train_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        lr_scheduler.step()
        mean_loss = running_loss/train_steps
        logger.info(f"train_loss: {mean_loss:.4f} lr: {lr:.6f}")

        # evaluation
        model.eval()
        # 创建空的混淆矩阵，用于评估模型
        confmat.reset()
        with torch.no_grad():
            for image, target in val_loader:
                image, target = image.to(device), target.to(device)
                output = model(image)
                confmat.update(target.flatten(), output.argmax(1).flatten())
            confmat.reduce_from_all_processes()
        acc_global, acc, iu = confmat.compute()
        logger.info(f"global correct: {acc_global.item() * 100:.1f}  mean IoU: {iu.mean().item() * 100:.1f}'")
        logger.info(f"correct: {['{:.1f}'.format(i) for i in (acc * 100).tolist()]}")
        logger.info(f"  IoU  : {['{:.1f}'.format(i) for i in (iu * 100).tolist()]}")

        # save info
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "lr": lr,
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, "weights/model_{}.pth".format(epoch))

        # tensorborad write
        tb_writer.add_scalar("train/train_loss",mean_loss,epoch)
        tb_writer.add_scalar("train/learning_rate",lr, epoch)
        tb_writer.add_scalar("val/global_correct",acc_global.item(),epoch)
        tb_writer.add_scalar("val/mean_IoU",iu.mean().item(),epoch)

        tb_writer.add_histogram("correct",acc, epoch)
        tb_writer.add_histogram("IoU",iu, epoch)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch lraspp training")
    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--resume', default='',  help='resume from checkpoint')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)