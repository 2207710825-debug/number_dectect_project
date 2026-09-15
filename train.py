# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/14 17:34
@Describe ：训练脚本 —— 负责模型训练、验证、 Loss 曲线保存及模型权重导出
'''
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # 避免后台无界面渲染卡死
import matplotlib.pyplot as plt

from imagedata.dataset import get_dataset
from model.Model import CharacterNet
from loss.loss_function import CustomCrossEntropyLoss

# 中文字体显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer:
    def __init__(self, batch_size=64, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前运行设备: {self.device}")

        # 1. 加载训练集与验证集
        train_dataset = get_dataset(mode='train')
        val_dataset = get_dataset(mode='val')

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

        # 2. 实例化模型、损失函数和优化器
        self.model = CharacterNet(num_classes=10).to(self.device)
        self.criterion = CustomCrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.loss_history = []

    def evaluate(self):
        """验证集评估"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, filenames in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, epochs=80, save_dir="result"):
        """训练主循环"""
        print("\n=== 开始训练模型 ===")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for images, labels, filenames in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                acc = self.evaluate()
                print(f"[Epoch {epoch:02d}/{epochs}] Loss: {avg_loss:.4f} | 验证集准确率: {acc:.2f}%")

        # 1. 保存损失函数曲线图
        self.plot_loss_curve(save_dir=save_dir)

        # 2. 保存模型权重文件
        model_save_path = os.path.join(save_dir, "character_net.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"[已保存] 模型权重已导出至: {model_save_path}\n")

    def plot_loss_curve(self, save_dir="result"):
        """绘制并保存 Loss 曲线图"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', color='b', label='Training Loss')
        plt.title('训练损失变化曲线 (Loss Curve)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[已保存] 损失曲线图已输出至: {save_path}")
        print("=== 模型包含的层及其权重形状 ===")
        state_dict = torch.load("result/character_net.pth", map_location="cpu")
        for layer_name, weight_tensor in state_dict.items():
            print(f"层名称: {layer_name:<30} | 形状(Shape): {list(weight_tensor.shape)}")

        print("\n" + "=" * 50 + "\n")

        # 2. 查看具体某一层的权重参数数值（例如第一卷积层或全连接层）
        # for layer_name, weight_tensor in state_dict.items():
        #     print(f"--- 【{layer_name}】的具体数值 ---")
        #     print(weight_tensor)
        #     print("\n")


if __name__ == "__main__":
    trainer = Trainer(batch_size=128, lr=0.0001)
    # 执行 80 轮训练并导出模型及损失图
    trainer.train(epochs=200, save_dir="result")