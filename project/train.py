import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime
from tqdm import trange
from tqdm import tqdm

from utils.utils import clip_gradient, adjust_lr, IOU
from utils.data import get_loader

from models.SeaNet import MyNet as SeaNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args(args=[])


model = SeaNet(True).to(device)
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

image_root = 'datasets/Image-train/'

gt_root = 'datasets/GT-train/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

# loss
CE = torch.nn.BCEWithLogitsLoss().to(device)
MSE = torch.nn.MSELoss().to(device)
IOU = IOU(size_average=True).to(device)

history = {
    'total_loss': [],
    'loss1': [],
    'loss2': [],
    'loss3': [],
    'loss4': []
}

def train(train_loader, model, optimizer, epoch):
    model.train()
    
    # 【新增】用于累加一个 epoch 内的 loss，方便计算平均值
    epoch_loss_sum = 0
    epoch_loss1_sum = 0
    epoch_loss2_sum = 0
    epoch_loss3_sum = 0
    epoch_loss4_sum = 0
    
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        
        images, gts = pack
        images = images.to(device)
        gts = gts.to(device)

        s12, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(images)

        loss1 = CE(s12, gts) + IOU(s12_sig, gts)
        loss2 = CE(s34, gts) + IOU(s34_sig, gts)
        loss3 = CE(s5, gts) + IOU(s5_sig, gts)
        loss4 = MSE(edge1, edge2)

        loss = loss1 + loss2 + loss3 + 0.5 * loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        
        # 累加 Loss (使用 .item() 获取数值)
        epoch_loss_sum += loss.item()
        epoch_loss1_sum += loss1.item()
        epoch_loss2_sum += loss2.item()
        epoch_loss3_sum += loss3.item()
        epoch_loss4_sum += loss4.item()

        # if i % 20 == 0 or i == total_step:
        #     tqdm.write(
        #         '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, Loss: {:.4f}'.
        #         format(datetime.now(), epoch, opt.epoch, i, total_step,
        #                optimizer.param_groups[0]['lr'], loss.item()))

    save_path = 'checkpoints/SeaNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, f'SeaNet_epoch_{epoch}.pth'))

    # 【新增】返回本 Epoch 的平均 Loss
    return {
        'total': epoch_loss_sum / total_step,
        'l1': epoch_loss1_sum / total_step,
        'l2': epoch_loss2_sum / total_step,
        'l3': epoch_loss3_sum / total_step,
        'l4': epoch_loss4_sum / total_step
    }
    
tqdm.write("Let's go!")

for epoch in trange(1, opt.epoch + 1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    
    epoch_results = train(train_loader, model, optimizer, epoch)
    
    history['total_loss'].append(epoch_results['total'])
    history['loss1'].append(epoch_results['l1'])
    history['loss2'].append(epoch_results['l2'])
    history['loss3'].append(epoch_results['l3'])
    history['loss4'].append(epoch_results['l4'])

torch.save(model.state_dict(), "./result/" + 'SeaNet.pth')

print("Training finished.")

import matplotlib.pyplot as plt  # 【新增】引入绘图库

print("Plotting results...")
# 创建一个画布
plt.figure(figsize=(12, 8))

# 绘制总 Loss
plt.subplot(2, 1, 1) # 上半部分图
plt.plot(range(1, opt.epoch + 1), history['total_loss'], label='Total Loss', color='red', linewidth=2)
plt.title('Training Loss Curve')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

# 绘制子 Loss (查看细节)
plt.subplot(2, 1, 2) # 下半部分图
plt.plot(range(1, opt.epoch + 1), history['loss1'], label='Loss 1 (Stage 1-2)', linestyle='--')
plt.plot(range(1, opt.epoch + 1), history['loss2'], label='Loss 2 (Stage 3-4)', linestyle='--')
plt.plot(range(1, opt.epoch + 1), history['loss3'], label='Loss 3 (Stage 5)', linestyle='--')
plt.plot(range(1, opt.epoch + 1), history['loss4'], label='Loss 4 (Edge)', linestyle=':')
plt.xlabel('Epoch')
plt.ylabel('Sub-Loss Value')
plt.grid(True)
plt.legend()

# 保存图片到本地
plt.tight_layout()
plt.savefig('loss_curve.png')
print("Loss curve saved as 'loss_curve.png'")