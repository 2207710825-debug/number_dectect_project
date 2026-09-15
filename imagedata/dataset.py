import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image

# 默认数据路径配置
DEFAULT_TRAIN_DIR = r"E:\pythondemo\number_dectect_project\imagedata\training_img"
DEFAULT_TEST_DIR = r"E:\pythondemo\number_dectect_project\imagedata\test_img"

# 统一的数据预处理流程（避免重复定义）
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class SingleFolderDataset(Dataset):
    """
    根据文件名解析标签的自定义数据集类
    文件名称格式：0_0.png -> 标签为 0
    """

    def __init__(self, data_dir, transform=DEFAULT_TRANSFORM):
        self.data_dir = data_dir
        self.transform = transform

        # 过滤并按文件名获取所有图像文件
        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # 1. 读取图片并转为单通道灰度图
        image = Image.open(img_path).convert('L')

        # 2. 从文件名提取标签（如 "0_5.png" -> 0）
        label = int(img_name.split('_')[0])

        # 3. 图像预处理
        if self.transform:
            image = self.transform(image)

        return image, label, img_name


def get_dataset(mode='train', data_dir=None, train_ratio=0.8, seed=42):
    """
    统一的数据集获取入口函数
    :param mode: 'train' (训练集), 'val' (验证集), 或 'test' (测试集)
    :param data_dir: 数据目录路径，若为 None 则使用默认路径
    :param train_ratio: 训练集划分割比例
    :param seed: 随机种子，保证训练集/验证集划分可复现
    """
    if mode in ['train', 'val']:
        data_dir = data_dir or DEFAULT_TRAIN_DIR
        full_dataset = SingleFolderDataset(data_dir=data_dir)

        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

        return train_ds if mode == 'train' else val_ds

    elif mode == 'test':
        data_dir = data_dir or DEFAULT_TEST_DIR
        return SingleFolderDataset(data_dir=data_dir)

    else:
        raise ValueError("mode 参数必须是 'train', 'val' 或 'test' 之一")