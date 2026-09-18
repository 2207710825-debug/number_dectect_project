# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/17 9:14
@Describe ：通过特征工程（旋转、膨胀扩大、开运算、骨架化）丰富手写数字训练集
'''

import os
import shutil
import cv2
import numpy as np
from skimage.morphology import skeletonize
import random


def remove_outer_frame_floodfill(image: np.uint8) -> np.uint8:
    """
    使用 FloodFill (洪水填充) 算法清除外边框，100% 保护数字内部结构
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_clean = image.copy()
    h, w = img_clean.shape

    # 构建掩码，FloodFill 要求尺寸比原图大 2 个像素
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # 沿着图像四周边缘找白色像素点进行淹没 (变成黑色 0)
    for x in range(w):
        if img_clean[0, x] > 127:  # 上边缘
            cv2.floodFill(img_clean, mask, (x, 0), 0)
        if img_clean[h - 1, x] > 127:  # 下边缘
            cv2.floodFill(img_clean, mask, (x, h - 1), 0)

    for y in range(h):
        if img_clean[y, 0] > 127:  # 左边缘
            cv2.floodFill(img_clean, mask, (0, y), 0)
        if img_clean[y, w - 1] > 127:  # 右边缘
            cv2.floodFill(img_clean, mask, (w - 1, y), 0)

    return img_clean


def rotate_image(image, angle=30):
    """1. 图像旋转 (保持 32x32 尺寸，黑底填充)"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 填充颜色 borderValue 设为 0 (黑色背景)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated


def dilate_image(image, kernel_size=2):
    """2. 特征扩大 (使用形态学膨胀，使笔画变粗/数值放大)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated


def open_operation(image, kernel_size=2):
    """3. 开运算 (先腐蚀后膨胀，用于消除孤立小噪点/平滑边缘)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


def skeletonize_image(image):
    """4. 细化与骨架化 (提取单像素宽度的字符骨架)"""
    # 确保二值化输入：0 或 1 (bool 类型)
    _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary.astype(bool))
    # 转换回 0~255 uint8 格式
    return (skeleton * 255).astype(np.uint8)


def augment_dataset(src_dir, dst_dir):
    """数据集复制与扩充主函数"""
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源训练集目录未找到: {src_dir}")

    # 如果新文件夹不存在，则创建；如果已存在则清空或提示
    if os.path.exists(dst_dir):
        print(f"[提示] 目标目录已存在，正在清理并覆盖: {dst_dir}")
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    # 1. 复制原始训练集图像到新文件夹，避免污染原数据集
    file_list = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"找到原始训练样本: {len(file_list)} 张")

    copied_count = 0
    aug_count = 0

    for fname in file_list:
        src_path = os.path.join(src_dir, fname)
        base_name, ext = os.path.splitext(fname)

        # 复制原图
        #shutil.copy(src_path, os.path.join(dst_dir, fname))
        #copied_count += 1

        # 读取灰度图像
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # ------------------- 特征工程增强 -------------------
        img_rot30 = rotate_image(img, angle=random.randint(-30, 30))
        cv2.imwrite(os.path.join(dst_dir, f"{base_name}_rot30{ext}"), img_rot30 )
        cleaned_img = remove_outer_frame_floodfill(img)
        #cv2.imwrite(os.path.join(dst_dir, f"{base_name}_rot30{ext}"), cleaned_img )

        # 1. 开运算
        img_open = open_operation(cleaned_img, kernel_size=2)
        #cv2.imwrite(os.path.join(dst_dir, f"{base_name}_open{ext}"), img_open)

        # 1. 旋转 30°
        img_rot30 = rotate_image(  img_open , angle=random.randint(-30, 30))
        #cv2.imwrite(os.path.join(dst_dir, f"{base_name}_rot30{ext}"), img_rot30)

        # 2. 特征扩大 (膨胀)
        img_dilated = dilate_image(img_rot30 , kernel_size=2)
        cv2.imwrite(os.path.join(dst_dir, f"{base_name}_dilate{ext}"), img_dilated)



        # 4. 细化与骨架化
        img_skel = skeletonize_image(img_dilated)
        #cv2.imwrite(os.path.join(dst_dir, f"{base_name}_skel{ext}"), img_skel)

        aug_count += 2

    print("\n" + "=" * 50)
    print(f"数据集扩充完成！")
    print(f"原始复制样本: {copied_count} 张")
    print(f"新增增强样本: {aug_count} 张")
    print(f"扩充后总样本数: {copied_count + aug_count} 张")
    print(f"新训练集路径: {dst_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 原训练集目录与新增强训练集目录
    RAW_TRAIN_DIR = r"E:\pythondemo\number_dectect_project\imagedata\resized_32x32"
    AUG_TRAIN_DIR = r"E:\pythondemo\number_dectect_project\imagedata\training_img_augmented"

    augment_dataset(RAW_TRAIN_DIR , AUG_TRAIN_DIR)