import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建预测结果保存目录
os.makedirs('X18_MLP_predictions', exist_ok=True)


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

    def forward(self, x):
        return self.network(x)


def safe_inverse_transform(predictions, stats):
    """安全的反变换函数，处理NaN和无穷大值"""
    try:
        # 反标准化
        pred_denorm = predictions * stats['y_std'] + stats['y_mean']

        # 反log1p变换
        pred_original = np.expm1(pred_denorm)

        # 处理NaN和无穷大值
        pred_original = np.nan_to_num(pred_original, nan=0.0, posinf=1e6, neginf=0.0)

        # 确保所有值都是有限的
        pred_original = np.clip(pred_original, 0, 1e6)

        return pred_original

    except Exception as e:
        print(f"反变换过程中出现错误: {e}")
        print(f"预测值范围: {predictions.min():.4f} - {predictions.max():.4f}")
        return None


def load_model_and_stats():
    """加载模型和统计信息"""
    print("正在加载模型和统计信息...")
    
    # 加载统计信息
    with open('X18_model_MLP_withoutTopf/stats.json', 'r') as f:
        stats = json.load(f)
    
    print("统计信息已加载")
    print(f"y_mean: {stats['y_mean']:.6f}")
    print(f"y_std: {stats['y_std']:.6f}")
    
    return stats


def load_test_data():
    """加载测试数据"""
    print("正在加载测试数据...")
    
    # 加载测试数据
    test_data = pd.read_csv('test.csv')
    print(f"测试数据形状: {test_data.shape}")
    
    # 明确指定特征列：第2到164列（索引1到163）
    feature_cols = test_data.columns[1:164].tolist()  # 排除第1列(id)，取第2-164列
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列范围: {feature_cols[0]} 到 {feature_cols[-1]}")
    
    # 提取特征并转换为numpy数组
    X_test = test_data[feature_cols].to_numpy()
    
    # 处理缺失值
    print(f"特征缺失值: {np.isnan(X_test).sum()}")
    
    # 用中位数填充特征缺失值
    median_value = float(np.nanmedian(X_test))
    X_test = np.nan_to_num(X_test, nan=median_value)
    
    print(f"处理后测试数据形状: X_test={X_test.shape}")
    
    return X_test, test_data['id'].values


def predict(model, X_test, stats):
    """使用模型进行预测"""
    print("正在进行预测...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 分批预测
        batch_size = 64
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
            outputs = model(batch_X)
            
            # 直接处理输出，不使用squeeze
            predictions.extend(outputs.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    
    # 使用安全的反变换函数
    pred_original = safe_inverse_transform(predictions, stats)
    
    if pred_original is None:
        print("预测失败：反变换过程中出现错误")
        return None
    
    print(f"预测完成！预测值范围: {pred_original.min():.4f} - {pred_original.max():.4f}")
    print(f"预测值统计: 均值={pred_original.mean():.4f}, 标准差={pred_original.std():.4f}")
    
    return pred_original


def save_predictions(predictions, ids, stats):
    """保存预测结果"""
    print("正在保存预测结果...")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'id': ids,
        'X18_mean': predictions
    })
    
    # 保存到CSV文件
    output_file = 'X18_MLP_predictions/X18_MLP_predictions.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"预测结果已保存到: {output_file}")
    print(f"预测结果形状: {results_df.shape}")
    
    # 保存预测统计信息
    stats_file = 'X18_MLP_predictions/prediction_stats.json'
    prediction_stats = {
        'predictions_count': len(predictions),
        'predictions_mean': float(predictions.mean()),
        'predictions_std': float(predictions.std()),
        'predictions_min': float(predictions.min()),
        'predictions_max': float(predictions.max()),
        'model_stats': stats
    }
    
    with open(stats_file, 'w') as f:
        json.dump(prediction_stats, f, indent=2)
    
    print(f"预测统计信息已保存到: {stats_file}")
    
    return results_df


def main():
    """主函数"""
    print("=== X18_mean MLP模型预测 ===")
    
    # 加载模型和统计信息
    stats = load_model_and_stats()
    
    # 加载测试数据
    X_test, test_ids = load_test_data()
    
    # 创建模型
    model = MLP(input_size=X_test.shape[1]).to(device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load('X18_model_MLP_withoutTopf/model.pt', map_location=device))
    print("模型权重已加载")
    
    # 进行预测
    predictions = predict(model, X_test, stats)
    
    if predictions is not None:
        # 保存预测结果
        results_df = save_predictions(predictions, test_ids, stats)
        
        print("\n=== 预测完成 ===")
        print(f"预测结果文件: X18_MLP_predictions/X18_MLP_predictions.csv")
        print(f"预测统计文件: X18_MLP_predictions/prediction_stats.json")
        
        # 显示前几个预测结果
        print("\n前5个预测结果:")
        for i in range(min(5, len(predictions))):
            print(f"ID: {test_ids[i]}, X18_mean: {predictions[i]:.6f}")
    else:
        print("预测失败！")


if __name__ == "__main__":
    main() 