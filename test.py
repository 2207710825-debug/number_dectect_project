# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/14 17:34
@Describe ：测试脚本 —— 包含准确率评估、混淆矩阵绘制与 0~9 抽样预测可视化
'''
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from imagedata.dataset import get_dataset
from model.Model import CharacterNet

# 中文字体显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(all_targets, all_preds, save_path):
    """
    绘制并保存混淆矩阵热力图
    """
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))

    plt.title('手写数字识别混淆矩阵 (Confusion Matrix)', fontsize=14)
    plt.xlabel('预测类别 (Predicted Label)', fontsize=12)
    plt.ylabel('真实类别 (True Label)', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[已保存] 混淆矩阵热力图已输出至: {save_path}")


def run_test(test_dir=None,
             model_path="result/character_net.pth",
             save_dir="result"):
    """
    加载模型权重并对独立测试集进行推理、绘制混淆矩阵与抽样可视化
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 1. 检查模型权重文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重文件 '{model_path}'，请先运行 train.py 进行训练！")

    # 2. 实例化网络结构并加载权重
    model = CharacterNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"成功加载模型权重: {model_path}")

    # 3. 加载测试数据集
    test_dataset = get_dataset(mode='test', data_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    actual_test_dir = test_dir if test_dir else test_dataset.data_dir

    all_targets = []
    all_preds = []
    samples_per_class = defaultdict(list)

    print("\n" + "=" * 40)
    print(f"开始对测试集进行推理预测: {actual_test_dir}")
    print("=" * 40)

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 记录所有预测值与真实值用于计算混淆矩阵
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 按类别收集前 16 张样本
            for img, label, pred, fname in zip(images, labels, predicted, filenames):
                l_val = label.item()
                if len(samples_per_class[l_val]) < 16:
                    samples_per_class[l_val].append((img.cpu(), pred.item(), fname))

    # 计算总体准确率
    correct = (np.array(all_targets) == np.array(all_preds)).sum()
    total = len(all_targets)
    accuracy = 100.0 * correct / total

    print(f"测试集总样本数: {total}")
    print(f"测试集最终准确率: {accuracy:.2f}%")
    print("=" * 40)

    os.makedirs(save_dir, exist_ok=True)

    # 4. 绘制并保存混淆矩阵
    cm_save_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_targets, all_preds, cm_save_path)

    # 5. 保存详细文本报告（包含分类指标：精确率 Precision, 召回率 Recall, F1-score）
    txt_save_path = os.path.join(save_dir, "test_result.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write("=== 测试集识别结果汇总 ===\n")
        f.write(f"测试集路径: {actual_test_dir}\n")
        f.write(f"测试集总样本数: {total}\n")
        f.write(f"测试集最终准确率: {accuracy:.2f}%\n\n")

        f.write("=== 分类评估报告 (Precision, Recall, F1-Score) ===\n")
        f.write(classification_report(all_targets, all_preds, digits=4))
        f.write("\n" + "=" * 50 + "\n")

        f.write("=== 各类别抽样识别详情 ===\n")
        for cls_label in sorted(samples_per_class.keys()):
            f.write(f"\n数字类别 [{cls_label}] 抽取样本 ({len(samples_per_class[cls_label])} 张):\n")
            for img_t, pred_l, fname in samples_per_class[cls_label]:
                status = "正确" if cls_label == pred_l else "错误"
                f.write(f"文件名: {fname:<15} | 真实标签: {cls_label} | 预测标签: {pred_l} | 结果: {status}\n")

    print(f"[已保存] 详细测试结果报告: {txt_save_path}")

    # 6. 为 0~9 每个类别各导出一张 4x4 网格可视化图
    for cls_label, items in samples_per_class.items():
        plt.figure(figsize=(8, 8))
        plt.suptitle(f"数字类别 '{cls_label}' 抽样预测展示 (16张样本)", fontsize=14)

        for i, (img_tensor, pred_l, fname) in enumerate(items):
            plt.subplot(4, 4, i + 1)
            img = img_tensor.squeeze().numpy() * 0.3081 + 0.1307
            plt.imshow(img, cmap='gray')

            color = 'green' if cls_label == pred_l else 'red'
            plt.title(f"{fname}\n预:{pred_l}", color=color, fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        cls_img_path = os.path.join(save_dir, f"sample_class_{cls_label}.png")
        plt.savefig(cls_img_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"[已保存] 类别 0~9 的可视化预测图已存入: {save_dir}/")


if __name__ == "__main__":
    run_test()