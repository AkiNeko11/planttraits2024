import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取训练数据...")
train_data = pd.read_csv('train.csv')
print(f"数据形状: {train_data.shape}")
print(f"列名: {list(train_data.columns)}")

# 目标变量列名
target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
target_names = {
    'X4_mean': '茎干密度 (SSD)',
    'X11_mean': '叶面积/叶干重 (SLA)',
    'X18_mean': '植物高度',
    'X26_mean': '种子干重',
    'X50_mean': '叶氮含量/叶面积',
    'X3112_mean': '叶面积'
}

# 检查目标变量是否存在
print("\n目标变量信息:")
for col in target_columns:
    if col in train_data.columns:
        print(f"{col}: {target_names[col]} - 存在")
        print(f"  缺失值: {train_data[col].isnull().sum()}")
        print(f"  数据类型: {train_data[col].dtype}")
        print(f"  数值范围: {train_data[col].min():.4f} 到 {train_data[col].max():.4f}")
    else:
        print(f"{col}: {target_names[col]} - 不存在")

# 创建原始数据分布图
print("\n创建原始数据分布图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('植物性状目标变量分布图 (原始数据)', fontsize=16, fontweight='bold')

for i, col in enumerate(target_columns):
    if col in train_data.columns:
        row = i // 3
        col_idx = i % 3
        
        # 创建直方图
        axes[row, col_idx].hist(train_data[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col_idx].set_title(f'{target_names[col]} ({col})', fontsize=12, fontweight='bold')
        axes[row, col_idx].set_xlabel('数值')
        axes[row, col_idx].set_ylabel('频次')
        axes[row, col_idx].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = train_data[col].mean()
        std_val = train_data[col].std()
        axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.4f}')
        axes[row, col_idx].legend()

plt.tight_layout()
plt.savefig('data_analysis_output/target_variables_distribution_original.png', dpi=300, bbox_inches='tight')
plt.show()

# 数据归一化
print("\n进行数据归一化...")
scaler = StandardScaler()

# 准备归一化数据
target_data = train_data[target_columns].copy()
target_data_normalized = pd.DataFrame(
    scaler.fit_transform(target_data),
    columns=target_columns,
    index=target_data.index
)

# 创建归一化后的分布图
print("创建归一化后的分布图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('植物性状目标变量分布图 (归一化后)', fontsize=16, fontweight='bold')

for i, col in enumerate(target_columns):
    row = i // 3
    col_idx = i % 3
    
    # 创建直方图
    axes[row, col_idx].hist(target_data_normalized[col].dropna(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[row, col_idx].set_title(f'{target_names[col]} ({col}) - 归一化', fontsize=12, fontweight='bold')
    axes[row, col_idx].set_xlabel('标准化数值')
    axes[row, col_idx].set_ylabel('频次')
    axes[row, col_idx].grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = target_data_normalized[col].mean()
    std_val = target_data_normalized[col].std()
    axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.4f}')
    axes[row, col_idx].legend()

plt.tight_layout()
plt.savefig('data_analysis_output/target_variables_distribution_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建相关性热力图
print("创建相关性热力图...")
plt.figure(figsize=(10, 8))
correlation_matrix = target_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'label': '相关系数'})
plt.title('目标变量相关性热力图', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data_analysis_output/target_variables_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建箱线图比较
print("创建箱线图比较...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 原始数据箱线图
target_data.boxplot(ax=ax1)
ax1.set_title('原始数据箱线图', fontsize=14, fontweight='bold')
ax1.set_ylabel('数值')
ax1.tick_params(axis='x', rotation=45)

# 归一化数据箱线图
target_data_normalized.boxplot(ax=ax2)
ax2.set_title('归一化数据箱线图', fontsize=14, fontweight='bold')
ax2.set_ylabel('标准化数值')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data_analysis_output/target_variables_boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印统计摘要
print("\n=== 原始数据统计摘要 ===")
print(target_data.describe())

print("\n=== 归一化数据统计摘要 ===")
print(target_data_normalized.describe())

print("\n可视化图表已保存:")
print("- data_analysis_output/target_variables_distribution_original.png")
print("- data_analysis_output/target_variables_distribution_normalized.png") 
print("- data_analysis_output/target_variables_correlation.png")
print("- data_analysis_output/target_variables_boxplot_comparison.png") 