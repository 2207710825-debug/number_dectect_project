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

matplotlib.use('Agg')  # 避免后台无界面渲染卡死
import matplotlib.pyplot as plt

from imagedata.dataset import get_dataset
#from model.FC import CharacterNet
from model.CNN import CharacterNet

from loss.loss_function import CustomCrossEntropyLoss

# 中文字体显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer:
    def __init__(self, batch_size=64, lr=0.001, weight_decay=1e-4, patience=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前运行设备: {self.device}")

        # 1. 加载训练集与验证集
        train_dataset = get_dataset(mode='train')
        val_dataset = get_dataset(mode='val')

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

        # 2. 实例化模型、损失函数和优化器
        self.num_classes = 10
        self.model = CharacterNet(num_classes=self.num_classes).to(self.device)
        self.criterion = CustomCrossEntropyLoss().to(self.device)

        # 加入 L2 正则化 (weight_decay) 防止过拟合
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 早停机制相关配置
        self.patience = patience  # 容忍 Val Loss 不下降的最大轮数
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        self.loss_history = []
        self.val_loss_history = []

    def evaluate(self):
        """验证集评估：返回平均验证损失与总准确率"""
        self.model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels, filenames in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_val_loss, accuracy

    def evaluate_per_class(self, class_names=None):
        """逐类别统计准确率并打印输出"""
        self.model.eval()
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.no_grad():
            for images, labels, filenames in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print("\n" + "=" * 45)
        print("           最佳验证集模型各类别准确率           ")
        print("=" * 45)
        print(f"{'类别名称/索引':<15} | {'正确数/总数':<15} | {'准确率 (%)':<10}")
        print("-" * 45)

        for i in range(self.num_classes):
            c_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print(f"{c_name:<15} | {class_correct[i]}/{class_total[i]:<12} | {acc:.2f}%")
            else:
                print(f"{c_name:<15} | N/A (样本数为0)   | N/A")
        print("=" * 45 + "\n")

    def train(self, epochs=80, save_dir="result"):
        """训练主循环（带 Early Stopping 机制）"""
        print("\n=== 开始训练模型 ===")
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_character_net.pth")

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

            val_loss, acc = self.evaluate()
            self.val_loss_history.append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[Epoch {epoch:02d}/{epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | 验证集总体准确率: {acc:.2f}%")

            # 早停逻辑判断：若 Val Loss 刷新最佳值，保存当前最佳权重
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n[Early Stopping] 触发早停机制！验证集 Loss 在连续 {self.patience} 轮内未降低。")
                    print(
                        f"[提示] 训练于 Epoch {epoch} 提前终止，最佳模型诞生在 Epoch {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})")
                    break

        # 1. 保存损失函数曲线图
        self.plot_loss_curve(save_dir=save_dir)

        # 2. 加载性能最好的权重，评估逐类别准确率并导出最终 character_net.pth
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            final_model_path = os.path.join(save_dir, "character_net.pth")
            torch.save(self.model.state_dict(), final_model_path)
            print(f"[已保存] 已将最佳 Epoch {self.best_epoch} 的模型权重覆盖导出至: {final_model_path}\n")

        # 3. 统计并打印最佳模型的各类别准确率
        self.evaluate_per_class()

    def plot_loss_curve(self, save_dir="result"):
        """绘制并保存 Loss 曲线图"""
        plt.figure(figsize=(8, 5))
        epochs_range = range(1, len(self.loss_history) + 1)

        plt.plot(epochs_range, self.loss_history, marker='o', color='b', label='Training Loss')
        plt.plot(epochs_range, self.val_loss_history, marker='s', color='r', linestyle='--', label='Validation Loss')

        # 在曲线上标出最佳 Early Stopping 节点
        if self.best_epoch > 0:
            plt.axvline(x=self.best_epoch, color='g', linestyle=':', label=f'Best Epoch ({self.best_epoch})')

        plt.title('训练与验证损失变化曲线 (Loss Curve)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[已保存] 损失曲线图已输出至: {save_path}")

        pth_path = os.path.join(save_dir, "character_net.pth")
        if os.path.exists(pth_path):
            print("=== 模型包含的层及其权重形状 ===")
            state_dict = torch.load(pth_path, map_location="cpu")
            for layer_name, weight_tensor in state_dict.items():
                print(f"层名称: {layer_name:<30} | 形状(Shape): {list(weight_tensor.shape)}")
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # patience=20 表示连续 20 轮验证集损失不下降则提前终止训练
    trainer = Trainer(batch_size=64, lr=0.0001, weight_decay=1e-4, patience=20)
    # 即使设置了上限 500 轮，达到最优后也会自动触发早停，不会无效训练
    trainer.train(epochs=500, save_dir="result/train_result")