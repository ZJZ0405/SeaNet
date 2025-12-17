import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
import time

from model.SeaNet_models import SeaNet
from data import test_dataset

# 1. 设置设备，增加灵活性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=288, help='testing size')
opt = parser.parse_args()

dataset_path = 'datasets/'

# 2. 加载模型
model = SeaNet()
# 建议加上 map_location，防止在不同显卡配置下加载报错
model.load_state_dict(torch.load('./models/SeaNet/SeaNet.pth.49', map_location=device))

model.to(device)
model.eval()

test_datasets = ['EORSSD']

# 3. 推理阶段不计算梯度，节省显存
with torch.no_grad():
    for dataset in test_datasets:
        save_path = './models/SeaNet/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        image_root = "datasets/Image-test/"
        print(f"Processing dataset: {dataset}")
        gt_root = "datasets/GT-test/"
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        time_sum = 0
        
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            
            # 处理 GT (用于获取尺寸)
            gt = np.asarray(gt, np.float32)
            
            image = image.to(device)

            # 4. 准确的 GPU 计时 (必须同步)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_start = time.time()

            # 前向传播
            res, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(image)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_end = time.time()
            
            time_sum += (time_end - time_start)

            # 5. 使用 interpolate 替代 upsample
            # 注意：gt.shape 通常是 (H, W)，interpolate 可以直接接受
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            
            # 6. 后处理：使用 .detach() 替代 .data
            res = res.sigmoid().detach().cpu().numpy().squeeze()
            
            # 归一化
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype('uint8')
            
            # 7. 保存图片 (imageio 2.x 推荐使用 imwrite)
            # 使用 os.path.join 处理路径更安全
            imageio.imwrite(os.path.join(save_path, name), res)

        # 打印结果
        if test_loader.size > 0:
            avg_time = time_sum / test_loader.size
            fps = test_loader.size / time_sum
            print('Running time per image: {:.5f}s'.format(avg_time))
            print('FPS: {:.5f}'.format(fps))