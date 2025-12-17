import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os
import argparse
from datetime import datetime

from model.SeaNet_models import SeaNet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

# 1. 设备配置：更灵活的设备分配
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

# 2. 构建模型并移动到设备
model = SeaNet().to(device)

# 【可选】PyTorch 2.x 特性：编译模型以提高推理和训练速度
# 如果你的环境支持 Triton 且想追求极致速度，可以取消下面这一行的注释：
# model = torch.compile(model)

optimizer = torch.optim.Adam(model.parameters(), opt.lr)

image_root = 'datasets/Image-train/'
gt_root = 'datasets/GT-train/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

# 损失函数移动到设备
CE = torch.nn.BCEWithLogitsLoss().to(device)
MSE = torch.nn.MSELoss().to(device)
IOU = pytorch_iou.IOU(size_average=True).to(device)

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        
        # 3. 移除 Variable。在 Torch 0.4 之后，Tensor 默认支持 autograd
        images, gts = pack
        images = images.to(device)
        gts = gts.to(device)

        # 前向传播
        s12, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(images)

        # 计算损失
        loss1 = CE(s12, gts) + IOU(s12_sig, gts)
        loss2 = CE(s34, gts) + IOU(s34_sig, gts)
        loss3 = CE(s5, gts) + IOU(s5_sig, gts)
        loss4 = MSE(edge1, edge2)

        loss = loss1 + loss2 + loss3 + 0.5 * loss4

        # 反向传播
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # 4. 打印日志：使用 .item() 获取标量值（比 .data 更安全，防止内存增长）
        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, Loss: {:.4f}, L1: {:.4f}, L2: {:.4f}, L3: {:.4f}, L4: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       optimizer.param_groups[0]['lr'], # 直接从优化器读取当前真实LR
                       loss.item(), loss1.item(),
                       loss2.item(), loss3.item(), 0.5 * loss4.item()))

    # 保存模型
    save_path = 'models/SeaNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 每 5 个 epoch 保存一次。PyTorch 2.x 默认使用 zip 序列化，建议保持默认
    if (epoch) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, f'SeaNet_epoch_{epoch}.pth'))

print("Let's go!")
# 5. 循环范围调整为从 1 到 opt.epoch（包含最后一次）
for epoch in range(1, opt.epoch + 1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)