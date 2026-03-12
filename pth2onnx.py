import torch
from model.SeaNet_models import SeaNet

# 1. 实例化你的网络模型 (需要替换为你自己的模型类)
# 这里假设你的模型类叫做 MyModel
# from your_model_file import MyModel 
# model = MyModel()


# 2. 加载 .pth 权重文件
pth_file_path = "models/SeaNet.pth"  # 替换为你的 pth 文件路径

# 常见情况一：pth 中只保存了 state_dict (推荐方式)
# 使用 map_location 确保在没有 GPU 的机器上也能加载
model = SeaNet()
model.load_state_dict(torch.load(pth_file_path))

# 常见情况二：pth 中保存了整个模型 (网络结构+权重)
# model = torch.load(pth_file_path, map_location=torch.device('cpu'))

# 3. 切换到评估模式 (非常重要！)
# 这会固定住 Dropout 和 BatchNorm 的参数，否则转换后的模型推理结果会不固定
model.cuda().eval()

# 4. 创建一个 Dummy Input (伪造输入)
# 它的形状(Shape)和数据类型必须与你真实推理时喂给模型的输入完全一致
# 这里假设输入是: Batch Size = 1, 通道数 = 3, 高 = 224, 宽 = 224
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')

# 5. 导出为 ONNX
onnx_file_path = "models/SeaNet.pth.onnx"

print("开始导出 ONNX 模型...")
torch.onnx.export(
    model,                         # 正在运行的模型
    dummy_input,                   # 模型的输入 (如果模型有多个输入，可以传入一个元组，如 (dummy_input1, dummy_input2))
    onnx_file_path,                # 保存 ONNX 文件的路径
    export_params=True,            # 是否将训练好的参数权重存储在 ONNX 文件中 (通常设为 True)
    opset_version=17,              # ONNX 的 opset 版本 (11 比较常用且兼容性好，也可以根据需要尝试 12, 13, 14 等)
    do_constant_folding=True,      # 是否执行常量折叠优化 (提升推理速度)
    input_names=['input'],         # 为输入节点指定名称 (可选，但建议指定)
    output_names=['output'],       # 为输出节点指定名称 (可选，但建议指定)
    
    # 动态轴配置 (可选)
    # 如果你希望未来在推理时改变 Batch Size 的大小，可以解除对第 0 维度的固定
    # dynamic_axes={
    #     'input': {0: 'batch_size'},  
    #     'output': {0: 'batch_size'}
    # }
)

print(f"模型已成功导出至: {onnx_file_path}")

import onnx
model = onnx.load(onnx_file_path)
onnx.checker.check_model(model)
print("ONNX 模型自检通过！")