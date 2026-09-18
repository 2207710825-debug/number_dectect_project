# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/17 11:56
@Describe ：
'''
# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np


def augment_target_digits(src_dir, dst_dir, target_labels=['1', '7'], augment_factor=3):
    """
    专门针对低准确率类别（如 1 和 7）进行定向数据扩充

    :param src_dir: 源图片文件夹 (如 training_img)
    :param dst_dir: 增强图片保存的目标文件夹 (如 training_img_augmented)
    :param target_labels: 需要定向增强的数字类别列表
    :param augment_factor: 增强倍数（每张原图生成多少张新图）
    """
    os.makedirs(dst_dir, exist_ok=True)

    # 过滤出包含目标类别的图片文件
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]

    # 根据文件名开头判别类别 (假设文件名格式为 "1_xxx.png" 或 "7_xxx.png")
    target_files = [f for f in all_files if any(f.startswith(f"{label}_") for label in target_labels)]

    print(f"找到目标类别 {target_labels} 的图片共 {len(target_files)} 张，准备进行 {augment_factor} 倍定向扩充...")

    count = 0
    for fname in target_files:
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        label = fname.split('_')[0]
        base_name = os.path.splitext(fname)[0]

        h, w = img.shape

        for i in range(augment_factor):
            aug_img = img.copy()

            # 1. 随机小角度旋转 (-12° ~ 12°)，避免旋转过大变成其他数字
            angle = random.randint(-15, 15)
            M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M_rot, (w, h), borderValue=0)

            # 2. 随机微小平移 (上下左右平移 -2 ~ 2 像素)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
            aug_img = cv2.warpAffine(aug_img, M_trans, (w, h), borderValue=0)

            # 3. 针对数字 1 和 7 的概率性形态学微调 (微膨胀/微腐蚀，改变笔画粗细)
            morph_choice = random.random()
            kernel = np.ones((1, 1), np.uint8)
            if morph_choice < 0.3:
                aug_img = cv2.dilate(aug_img, kernel, iterations=1)  # 稍微加粗笔画
            elif morph_choice < 0.6:
                aug_img = cv2.erode(aug_img, kernel, iterations=1)  # 稍微变细笔画

            # 保存增强后的新图片
            new_fname = f"{base_name}_target_aug_{i}{os.path.splitext(fname)[1]}"
            save_path = os.path.join(dst_dir, new_fname)
            cv2.imwrite(save_path, aug_img)
            count += 1

    print(f"定向增强完成！共为类别 {target_labels} 新增了 {count} 张训练样本。")


if __name__ == "__main__":
    # 配置你的路径
    SRC_DIR = r"E:\pythondemo\number_dectect_project\imagedata\training_img_augmented"
    DST_DIR = r"E:\pythondemo\number_dectect_project\imagedata\training_img_augmented"

    # 针对 1 和 7 生成 4 倍的强扩充样本
    augment_target_digits(SRC_DIR, DST_DIR, target_labels=['5'], augment_factor=1)