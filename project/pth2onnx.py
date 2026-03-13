import torch
import onnx
from onnxsim import simplify

from models.backbone.MobileNetV3 import mobilenet_v3
from models.backbone.repVGG import create_RepVGG_A0
from models.backbone.mobileone import mobileone

# model = mobilenet_v3(pretrained=True)
model = mobileone(inference_mode=False)

model_name = "mobileone_no_infer"

model.cuda()
# 1. 确保模型处于 eval 模式 (非常重要，否则会导出 Dropout/BatchNorm 的训练状态)
model.eval()

# 2. 创建一个虚拟输入张量
# 假设你的输入图片大小是 288x288，通道数是 3，BatchSize 设为 1
# 如果你的模型在 GPU 上，这个 tensor 也要放到 GPU 上
dummy_input = torch.randn(1, 3, 288, 288).cuda()

# 3. 导出模型
torch.onnx.export(
    model,                      # 你的模型对象
    dummy_input,                # 【修正点】这里必须传虚拟输入，不能直接传文件名
    f"./models/{model_name}.onnx",              # 这里才是保存的文件名
    export_params=True,         # 是否将权重存入文件
    opset_version=18,           # 推荐使用 11 或更高，兼容性更好
    do_constant_folding=False,   # 优化常量折叠
    input_names=['input'],      # 输入节点命名
    output_names=['output'],    # 输出节点命名
    # 如果你想支持动态分辨率（比如输入图片大小不固定），可以加下面这行
    # dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size'}}
)

print("ONNX 模型导出成功！")

print("正在使用 ONNX Simplifier 优化模型...")
onnx_model = onnx.load(f"./models/{model_name}.onnx")
model_simp, check = simplify(onnx_model)

if check:
    onnx.save(model_simp, f"simp_{model_name}.onnx")
    print(f"简化后的 ONNX 模型已成功保存至: simp_{model_name}.onnx")
else:
    print("警告: 简化后的模型未能通过 ONNX 验证！")