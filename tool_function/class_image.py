# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/16 21:42
@Describe ：利用训练好的模型对数据集分类，作为测试集使用
'''
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 1. 导入你的模型类
from model.FC import CharacterNet

# ==================== 1. 配置路径与设备 ====================
image_dir = r"E:\pythondemo\number_dectect_project\imagedata\resized_32x32"
# 修正：去掉了路径字符串末尾的空格
model_path = r"E:\pythondemo\number_dectect_project\result\train_result\character_net.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 2. 图像预处理 (保持与训练一致) ====================
from PIL import Image, ImageOps


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 保持单通道
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==================== 3. 加载模型 ====================
if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到模型文件: {model_path}，请检查路径！")

# 1. 实例化模型对象并转移到计算设备上
model = CharacterNet().to(device)

# 2. 读取保存的 state_dict 参数字典，并加载到模型中
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

# 3. 开启评估模式（关闭 Dropout 和 BatchNorm 的训练行为）
model.eval()

# ==================== 4. 批量预测并重命名 ====================
renamed_count = 0

for filename in os.listdir(image_dir):
    # 过滤非 png 图片以及已经重命名过（数字开头）的文件
    if not filename.endswith('.png') or filename.split('_')[0].isdigit():
        continue

    file_path = os.path.join(image_dir, filename)
    img = Image.open(file_path).convert('L')  # 强制转灰度
    img = ImageOps.invert(img)  # 白底黑字 -> 转为黑底白字
    img.save(file_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_label = torch.argmax(output, dim=1).item()

    # 将 digit_row9_col1.png 替换为 0_row9_col1.png
    clean_name = filename.replace("digit_", "")
    new_filename = f"{pred_label}_{clean_name}"
    new_file_path = os.path.join(image_dir, new_filename)

    os.rename(file_path, new_file_path)
    renamed_count += 1
    print(f"重命名: {filename} -> {new_filename}")

print(f"\n全部完成！共自动打标 {renamed_count} 张图片。")