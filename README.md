# 手写数字识别项目 (Handwritten Digit Recognition)

> 基于 PyTorch 的三层全连接神经网络（MLP）实现的手写数字识别项目，覆盖数据加载、模型训练、结果可视化与测试评估全流程。

---

## 目录结构

```
number_dectect_project/
├── imagedata/                  # 数据模块
│   ├── dataset.py              # 自定义 Dataset、训练/验证集划分、数据增强
│   ├── training_img/           # 训练集图像（按 "数字_序号.png" 命名）
│   ├── test_img/               # 测试集图像（按 "数字_序号.png" 命名）
│   └── MNIST_手写数字识别.zip   # 原始数据集压缩包
│
├── model/                      # 模型模块
│   └── Model.py                # CharacterNet 三层全连接网络定义
│
├── loss/                       # 损失函数模块
│   └── loss_function.py        # 自定义 CrossEntropyLoss 封装
│
├── result/                     # 训练/评估产出
│   ├── character_net.pth       # 模型权重文件
│   ├── loss_curve.png          # 训练损失曲线
│   ├── confusion_matrix.png    # 混淆矩阵热力图
│   ├── sample_class_0~9.png    # 各类别抽样预测可视化
│   └── test_result.txt         # 测试报告（准确率、P/R/F1、抽样详情）
│
├── train.py                    # 训练入口脚本
├── test.py                     # 测试与可视化入口脚本
└── README.md                   # 项目说明文档
```

---

## 代码说明

### 1. `imagedata/dataset.py` — 数据加载

- **`SingleFolderDataset`**：继承 `torch.utils.data.Dataset`，读取单文件夹下的所有图像。
  - 文件名格式约定：`{真实标签}_{序号}.png`，例如 `0_5.png` 表示标签为 `0`。
  - 读取时统一转为灰度图（`convert('L')`），并使用 `ToTensor() + Normalize((0.1307,), (0.3081,))` 进行预处理（与 MNIST 标准化参数一致）。
- **`get_dataset(mode, data_dir, train_ratio=0.8, seed=42)`**：统一数据获取入口。
  - `mode='train'`：从训练目录读取，并按 8:2 随机切分为训练集。
  - `mode='val'`：返回对应的验证集子集（与训练集同源，使用固定随机种子保证可复现）。
  - `mode='test'`：从测试目录读取，作为独立评估集。

### 2. `model/Model.py` — 模型定义

**`CharacterNet`**：三层全连接神经网络（MLP），结构如下：

| 层     | 输入维度 | 输出维度 | 激活函数    |
| ----- | ---- | ---- | ------- |
| `fc1` | 1024 | 256  | Sigmoid |
| `fc2` | 256  | 128  | Sigmoid |
| `fc3` | 128  | 64   | Sigmoid |
| `out` | 64   | 10   | —       |

- 输入图像在 `forward` 中被展平为 `[batch_size, 1024]`，对应 32×32 灰度图（或 resize 后尺寸）。
- 输出层维度为 `num_classes=10`，对应 0~9 共 10 个数字类别。
- 使用 Sigmoid 作为隐藏层激活函数（与架构图保持一致）。

### 3. `loss/loss_function.py` — 损失函数

**`CustomCrossEntropyLoss`**：继承 `nn.Module`，内部封装 `nn.CrossEntropyLoss()`，便于后续扩展（如加入 Label Smoothing、正则项等）。

### 4. `train.py` — 训练流程

- **`Trainer`**：封装c训练全流程。
  - 自动检测 CUDA：`torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
  - 数据加载：`DataLoader` 批大小 `batch_size=128`（可调）。
  - 优化器：`Adam`，初始学习率 `lr=0.0001`。
  - 每个 epoch 记录平均 Loss，每 10 个 epoch（以及第 1 个 epoch）在验证集上评估准确率。
  - 训练结束后：
    1. 绘制并保存 `loss_curve.png`；
    2. 保存模型权重 `character_net.pth`；
    3. 打印各层权重形状（便于核对网络结构）。
- 默认配置：`epochs=200`，`batch_size=128`，`lr=0.0001`。

### 5. `test.py` — 测试与可视化

**`run_test(...)`**：加载训练好的权重，对测试集进行评估并产出可视化结果。

- **整体准确率**：统计所有样本的预测正确率。
- **混淆矩阵**：使用 `seaborn.heatmap` 绘制 10×10 矩阵，保存为 `confusion_matrix.png`。
- **分类评估报告**：通过 `sklearn.metrics.classification_report` 输出 Precision / Recall / F1-Score，写入 `test_result.txt`。
- **抽样可视化**：为每个数字类别 0~9 各抽取 16 张样本，4×4 网格展示：
  - 标题包含文件名和预测标签；
  - 正确样本用绿色标题，错误样本用红色标题。

---

## 所需环境

| 项目      | 推荐版本                                   |
| ------- | -------------------------------------- |
| 操作系统    | Windows 10/11、macOS、Linux 均可           |
| Python  | 3.10+（推荐 3.11）                         |
| PyTorch | 2.0+（CPU 或 CUDA 版本均可，无 GPU 时自动回退 CPU）  |
| GPU（可选） | NVIDIA 显卡 + CUDA 11.7+（仅在需要 GPU 加速时安装） |

> 项目默认以 CPU 也能跑通为目标，检测不到 CUDA 时自动切换到 CPU，无需额外配置。

---

## 库与安装包

### 核心依赖

| 包名             | 用途                                         | 安装命令                       |
| -------------- | ------------------------------------------ | -------------------------- |
| `torch`        | 深度学习框架，提供 `nn`、`DataLoader` 等              | `pip install torch`        |
| `torchvision`  | 提供图像变换 `transforms`                        | `pip install torchvision`  |
| `numpy`        | 数值计算（混淆矩阵、准确率统计）                           | `pip install numpy`        |
| `Pillow`       | 图像读取（`PIL.Image`）                          | `pip install Pillow`       |
| `matplotlib`   | 损失曲线、抽样可视化                                 | `pip install matplotlib`   |
| `seaborn`      | 混淆矩阵热力图                                    | `pip install seaborn`      |
| `scikit-learn` | `confusion_matrix`、`classification_report` | `pip install scikit-learn` |

### 一键安装（推荐）

```bash
pip install torch torchvision numpy Pillow matplotlib seaborn scikit-learn
```

> 如果有 NVIDIA 显卡并想启用 GPU 加速，请到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本获取安装命令，例如：
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 验证安装

```bash
python -c "import torch, torchvision, numpy, PIL, matplotlib, seaborn, sklearn; print('环境就绪 OK')"
```

输出 `环境就绪 OK` 即表示所有依赖安装完成。

---

## 快速开始

### 1. 准备数据

将训练图像放入 `imagedata/training_img/`，测试图像放入 `imagedata/test_img/`，命名格式为 `{真实标签}_{任意序号}.png`，例如 `3_42.png`。

### 2. 开始训练

```bash
python train.py
```

训练结束后将在 `result/` 目录下生成：

- `character_net.pth`（模型权重）
- `loss_curve.png`（训练损失曲线）

### 3. 测试与可视化

```bash
python test.py
```

测试完成后将在 `result/` 目录下生成：

- `confusion_matrix.png`（混淆矩阵）
- `sample_class_0.png ~ sample_class_9.png`（各类别抽样可视化）
- `test_result.txt`（评估报告）

---

## 常见问题

1. **中文字体显示为方块**：`train.py` / `test.py` 默认使用 `SimHei` 字体显示中文标题。如系统未安装 SimHei，可改为 `Microsoft YaHei` 或其他已安装的中文字体。
2. **路径硬编码**：`imagedata/dataset.py` 中的 `DEFAULT_TRAIN_DIR` 与 `DEFAULT_TEST_DIR` 已按本机绝对路径写好；如迁移到其他机器，需相应修改这两个常量。
3. **训练太慢**：可在 `train.py` 中调小 `epochs`，或安装 GPU 版 PyTorch 加速。
