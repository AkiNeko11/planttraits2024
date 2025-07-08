import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取训练数据...")
train_data = pd.read_csv('train.csv')
print(f"数据形状: {train_data.shape}")

# 目标变量
target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
target_names = {
    'X4_mean': '茎干密度 (SSD)',
    'X11_mean': '叶面积/叶干重 (SLA)',
    'X18_mean': '植物高度',
    'X26_mean': '种子干重',
    'X50_mean': '叶氮含量/叶面积',
    'X3112_mean': '叶面积'
}

# 检查存在的列
existing_columns = [col for col in target_columns if col in train_data.columns]
print(f"找到的目标变量: {existing_columns}")

# 对数变换函数
def log_transform(data, column, method='log'):
    """
    对数变换函数
    method: 'log' (自然对数), 'log10' (常用对数), 'log1p' (log(1+x))
    """
    feature_data = data[column].dropna()
    
    if method == 'log':
        # 检查是否有负值或零值
        if (feature_data <= 0).any():
            print(f"警告: {column} 包含非正值，使用 log1p 变换")
            return np.log1p(feature_data)
        return np.log(feature_data)
    elif method == 'log10':
        if (feature_data <= 0).any():
            print(f"警告: {column} 包含非正值，使用 log10(1+x) 变换")
            return np.log10(feature_data + 1)
        return np.log10(feature_data)
    elif method == 'log1p':
        return np.log1p(feature_data)
    else:
        raise ValueError("method 必须是 'log', 'log10', 或 'log1p'")

# 标准化函数
def standardize_data(data):
    """标准化函数"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# 数据预处理和变换
print("\n=== 数据变换处理 ===")
transformed_data = {}

for col in existing_columns:
    print(f"\n处理 {target_names[col]} ({col})...")
    
    # 原始数据
    original_data = train_data[col].dropna()
    
    # 对数变换 (使用log1p避免负值问题)
    log_data = log_transform(train_data, col, 'log1p')
    
    # 标准化
    standardized_data = standardize_data(log_data)
    
    transformed_data[col] = {
        'original': original_data,
        'log_transformed': log_data,
        'standardized': standardized_data
    }
    
    print(f"  原始数据: 均值={original_data.mean():.4f}, 偏度={original_data.skew():.4f}")
    print(f"  对数变换: 均值={log_data.mean():.4f}, 偏度={log_data.skew():.4f}")
    print(f"  标准化后: 均值={standardized_data.mean():.4f}, 偏度={standardized_data.skew():.4f}")

# 创建可视化
print("\n=== 创建可视化图表 ===")

# 1. 原始数据分布
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('原始数据分布', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    data = transformed_data[col]['original']
    
    # 直方图
    axes[row, col_idx].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # 正态分布曲线
    x = np.linspace(data.min(), data.max(), 100)
    normal_dist = stats.norm.pdf(x, data.mean(), data.std())
    axes[row, col_idx].plot(x, normal_dist, 'r-', linewidth=2, label='正态分布')
    
    # 均值线
    axes[row, col_idx].axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label=f'均值: {data.mean():.4f}')
    
    axes[row, col_idx].set_title(f'{target_names[col]}\n偏度: {data.skew():.3f}', fontsize=11)
    axes[row, col_idx].set_xlabel('原始数值')
    axes[row, col_idx].set_ylabel('密度')
    axes[row, col_idx].legend(fontsize=8)
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_log_normalization_output/original_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 对数变换后分布
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('对数变换后分布 (log1p)', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    data = transformed_data[col]['log_transformed']
    
    # 直方图
    axes[row, col_idx].hist(data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', density=True)
    
    # 正态分布曲线
    x = np.linspace(data.min(), data.max(), 100)
    normal_dist = stats.norm.pdf(x, data.mean(), data.std())
    axes[row, col_idx].plot(x, normal_dist, 'r-', linewidth=2, label='正态分布')
    
    # 均值线
    axes[row, col_idx].axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label=f'均值: {data.mean():.4f}')
    
    axes[row, col_idx].set_title(f'{target_names[col]}\n偏度: {data.skew():.3f}', fontsize=11)
    axes[row, col_idx].set_xlabel('log(1+x) 数值')
    axes[row, col_idx].set_ylabel('密度')
    axes[row, col_idx].legend(fontsize=8)
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_log_normalization_output/log_transformed_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 标准化后分布
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('对数变换 + 标准化后分布', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    data = transformed_data[col]['standardized']
    
    # 直方图
    axes[row, col_idx].hist(data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    
    # 标准正态分布曲线
    x = np.linspace(-4, 4, 100)
    normal_dist = stats.norm.pdf(x, 0, 1)
    axes[row, col_idx].plot(x, normal_dist, 'r-', linewidth=2, label='标准正态分布')
    
    # 均值线
    axes[row, col_idx].axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label=f'均值: {data.mean():.4f}')
    
    axes[row, col_idx].set_title(f'{target_names[col]}\n偏度: {data.skew():.3f}', fontsize=11)
    axes[row, col_idx].set_xlabel('标准化数值')
    axes[row, col_idx].set_ylabel('密度')
    axes[row, col_idx].legend(fontsize=8)
    axes[row, col_idx].grid(True, alpha=0.3)
    axes[row, col_idx].set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('data_log_normalization_output/log_standardized_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 对比图 (原始 vs 对数变换 vs 标准化)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('数据变换对比 (原始 vs 对数变换 vs 标准化)', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    original = transformed_data[col]['original']
    log_transformed = transformed_data[col]['log_transformed']
    standardized = transformed_data[col]['standardized']
    
    # 三个子图
    ax1 = axes[row, col_idx]
    
    # 原始数据
    ax1.hist(original, bins=30, alpha=0.5, color='skyblue', label='原始', density=True)
    ax1.hist(log_transformed, bins=30, alpha=0.5, color='lightgreen', label='对数变换', density=True)
    ax1.hist(standardized, bins=30, alpha=0.5, color='lightcoral', label='标准化', density=True)
    
    ax1.set_title(f'{target_names[col]}', fontsize=11)
    ax1.set_xlabel('数值')
    ax1.set_ylabel('密度')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_log_normalization_output/transformation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 箱线图对比
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# 原始数据箱线图
original_data_for_box = [transformed_data[col]['original'] for col in existing_columns]
bp1 = ax1.boxplot(original_data_for_box, labels=[target_names[col] for col in existing_columns], patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('skyblue')
ax1.set_title('原始数据箱线图', fontsize=14, fontweight='bold')
ax1.set_ylabel('数值')
ax1.tick_params(axis='x', rotation=45)

# 对数变换箱线图
log_data_for_box = [transformed_data[col]['log_transformed'] for col in existing_columns]
bp2 = ax2.boxplot(log_data_for_box, labels=[target_names[col] for col in existing_columns], patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightgreen')
ax2.set_title('对数变换后箱线图', fontsize=14, fontweight='bold')
ax2.set_ylabel('log(1+x) 数值')
ax2.tick_params(axis='x', rotation=45)

# 标准化箱线图
standardized_data_for_box = [transformed_data[col]['standardized'] for col in existing_columns]
bp3 = ax3.boxplot(standardized_data_for_box, labels=[target_names[col] for col in existing_columns], patch_artist=True)
for patch in bp3['boxes']:
    patch.set_facecolor('lightcoral')
ax3.set_title('标准化后箱线图', fontsize=14, fontweight='bold')
ax3.set_ylabel('标准化数值')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data_log_normalization_output/boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 统计摘要
print("\n=== 变换前后统计对比 ===")
summary_data = []

for col in existing_columns:
    original = transformed_data[col]['original']
    log_transformed = transformed_data[col]['log_transformed']
    standardized = transformed_data[col]['standardized']
    
    summary_data.append({
        '特征': target_names[col],
        '原始_均值': original.mean(),
        '原始_偏度': original.skew(),
        '原始_峰度': original.kurtosis(),
        '对数_均值': log_transformed.mean(),
        '对数_偏度': log_transformed.skew(),
        '对数_峰度': log_transformed.kurtosis(),
        '标准化_均值': standardized.mean(),
        '标准化_偏度': standardized.skew(),
        '标准化_峰度': standardized.kurtosis()
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.round(4))

# 保存统计摘要
summary_df.to_csv('data_log_normalization_output/log_normalization_summary.csv', encoding='utf-8-sig', index=False)

print("\n=== 对数标准化完成 ===")
print("生成的可视化文件:")
print("- data_log_normalization_output/original_distribution.png")
print("- data_log_normalization_output/log_transformed_distribution.png")
print("- data_log_normalization_output/log_standardized_distribution.png")
print("- data_log_normalization_output/transformation_comparison.png")
print("- data_log_normalization_output/boxplot_comparison.png")
print("- data_log_normalization_output/log_normalization_summary.csv")