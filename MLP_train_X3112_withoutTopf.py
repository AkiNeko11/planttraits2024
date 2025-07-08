import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型保存目录
os.makedirs('X3112_model_MLP_withoutTopf', exist_ok=True)


class PlantTraitsDataset(Dataset):
    """植物性状数据集类"""

    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MLP(nn.Module):
    """多层感知机模型"""

    def __init__(self, input_size, hidden_sizes=[512,256,128, 64], dropout_rate=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=60, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def safe_inverse_transform(predictions, targets, stats):
    """安全的反变换函数，处理NaN和无穷大值"""
    try:
        # 反标准化
        pred_denorm = predictions * stats['y_std'] + stats['y_mean']
        target_denorm = targets * stats['y_std'] + stats['y_mean']

        # 反log1p变换
        pred_original = np.expm1(pred_denorm)
        target_original = np.expm1(target_denorm)

        # 处理NaN和无穷大值
        pred_original = np.nan_to_num(pred_original, nan=0.0, posinf=1e6, neginf=0.0)
        target_original = np.nan_to_num(target_original, nan=0.0, posinf=1e6, neginf=0.0)

        # 确保所有值都是有限的
        pred_original = np.clip(pred_original, 0, 1e6)
        target_original = np.clip(target_original, 0, 1e6)

        return pred_original, target_original

    except Exception as e:
        print(f"反变换过程中出现错误: {e}")
        print(f"预测值范围: {predictions.min():.4f} - {predictions.max():.4f}")
        print(f"目标值范围: {targets.min():.4f} - {targets.max():.4f}")
        return None, None


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")

    # 加载训练数据
    train_data = pd.read_csv('train.csv')
    print(f"训练数据形状: {train_data.shape}")

    # 分离特征和目标变量
    target_col = 'X3112_mean'

    # 检查目标变量是否存在
    if target_col not in train_data.columns:
        print(f"错误: 目标变量 {target_col} 不存在!")
        print(f"可用的目标变量: {[col for col in train_data.columns if 'X' in col and 'mean' in col]}")
        return None, None, None, None

    # 明确指定特征列：第2到164列（索引1到163）
    feature_cols = train_data.columns[1:164].tolist()  # 排除第1列(id)，取第2-164列
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列范围: {feature_cols[0]} 到 {feature_cols[-1]}")

    # 提取特征和目标
    X = train_data[feature_cols].values
    y = train_data[target_col].values

    # 处理缺失值
    print(f"特征缺失值: {np.isnan(X).sum()}")
    print(f"目标变量缺失值: {np.isnan(y).sum()}")

    # 用中位数填充特征缺失值
    X = np.nan_to_num(X, nan=np.nanmedian(X))

    # 移除目标变量缺失值的样本
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"处理后数据形状: X={X.shape}, y={y.shape}")

    # 目标变量变换: log1p + 标准化
    print("正在处理目标变量...")

    # 1. log1p变换
    y_log = np.log1p(y)
    print(f"log1p变换后 - 均值: {y_log.mean():.4f}, 标准差: {y_log.std():.4f}")

    # 2. 标准化
    y_mean = y_log.mean()
    y_std = y_log.std()
    y_normalized = (y_log - y_mean) / y_std

    print(f"标准化后 - 均值: {y_normalized.mean():.4f}, 标准差: {y_normalized.std():.4f}")

    # 保存统计信息用于推理
    stats = {
        'y_mean': float(y_mean),
        'y_std': float(y_std)
    }

    # 特征不进行标准化，直接使用原始值
    print("特征使用原始值，不进行标准化")

    # 保存统计信息
    with open('X3112_model_MLP_withoutTopf/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("统计信息已保存到 X3112_model_MLP_withoutTopf/stats.json")

    return X, y_normalized, y, stats


def train_model(X, y, y_original, stats, epochs=1000, batch_size=64):
    """训练模型"""
    print("开始训练模型...")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = train_test_split(
        X, y, y_original, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")

    # 创建数据加载器
    train_dataset = PlantTraitsDataset(X_train, y_train)
    val_dataset = PlantTraitsDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = MLP(input_size=X.shape[1]).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 早停机制
    early_stopping = EarlyStopping(patience=60)

    # 训练循环
    best_val_r2 = -float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                # 直接处理输出，不使用squeeze
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())

        # 反变换预测结果
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)

        # 使用安全的反变换函数
        val_pred_original, val_target_original = safe_inverse_transform(val_predictions, val_targets, stats)

        if val_pred_original is None:
            print(f"Epoch {epoch}: 反变换失败，跳过验证")
            continue

        # 计算R²
        val_r2 = r2_score(val_target_original, val_pred_original)

        # 打印进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val R²: {val_r2:.6f}")

        # 保存最佳模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), 'X3112_model_MLP_withoutTopf/model.pt')
            print(f"保存最佳模型，验证R²: {val_r2:.6f}")

        # 早停检查
        early_stopping(val_r2)
        if early_stopping.early_stop:
            print(f"早停触发，在第 {epoch} 轮停止训练")
            break

    print(f"训练完成！最佳验证R²: {best_val_r2:.6f}")
    return model


def evaluate_model(model, X, y, y_original, stats):
    """评估模型"""
    print("正在评估模型...")

    model.eval()
    predictions = []

    with torch.no_grad():
        # 分批预测
        batch_size = 64
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i + batch_size]).to(device)
            outputs = model(batch_X)

            # 直接处理输出，不使用squeeze
            predictions.extend(outputs.cpu().numpy().flatten())

    predictions = np.array(predictions)

    # 使用安全的反变换函数
    pred_original, target_original = safe_inverse_transform(predictions, y, stats)

    if pred_original is None:
        print("评估失败：反变换过程中出现错误")
        return 0.0, None, None

    # 计算R²
    r2 = r2_score(target_original, pred_original)

    print(f"最终模型R²: {r2:.6f}")

    # 打印一些预测示例
    print("\n预测示例:")
    for i in range(min(5, len(pred_original))):
        print(f"真实值: {target_original[i]:.4f}, 预测值: {pred_original[i]:.4f}")

    return r2, pred_original, target_original


def main():
    """主函数"""
    print("=== X3112_mean 预测模型训练 ===")

    # 加载和预处理数据
    X, y, y_original, stats = load_and_preprocess_data()

    if X is None:
        return

    # 训练模型
    model = train_model(X, y, y_original, stats)

    # 评估模型
    r2, predictions, targets = evaluate_model(model, X, y, y_original, stats)

    print("\n=== 训练完成 ===")
    print(f"模型文件: X3112_model_MLP_withoutTopf/model.pt")
    print(f"统计信息: X3112_model_MLP_withoutTopf/stats.json")
    print(f"最终R²: {r2:.6f}")


if __name__ == "__main__":
    main()