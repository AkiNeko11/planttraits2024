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
print("\n=== 目标变量存在性检查 ===")
existing_columns = []
for col in target_columns:
    if col in train_data.columns:
        existing_columns.append(col)
        print(f"✓ {col}: {target_names[col]} - 存在")
    else:
        print(f"✗ {col}: {target_names[col]} - 不存在")

if not existing_columns:
    print("错误：没有找到任何目标变量！")
    exit()

# 数据诊断函数
def diagnose_feature(data, column, name):
    """诊断单个特征的分布情况"""
    feature_data = data[column].dropna()
    
    if len(feature_data) == 0:
        return None
    
    # 基本统计信息
    stats_info = {
        'count': len(feature_data),
        'mean': feature_data.mean(),
        'median': feature_data.median(),
        'std': feature_data.std(),
        'min': feature_data.min(),
        'max': feature_data.max(),
        'skewness': feature_data.skew(),
        'kurtosis': feature_data.kurtosis(),
        'missing_count': data[column].isnull().sum(),
        'missing_pct': data[column].isnull().sum() / len(data) * 100
    }
    
    # 分位数信息
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats_info[f'p{p}'] = feature_data.quantile(p/100)
    
    # 异常值检测 (IQR方法)
    Q1 = feature_data.quantile(0.25)
    Q3 = feature_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    stats_info['outliers_count'] = len(outliers)
    stats_info['outliers_pct'] = len(outliers) / len(feature_data) * 100
    stats_info['lower_bound'] = lower_bound
    stats_info['upper_bound'] = upper_bound
    
    # Z-score异常值检测
    z_scores = np.abs(stats.zscore(feature_data))
    z_outliers = feature_data[z_scores > 3]
    stats_info['z_outliers_count'] = len(z_outliers)
    stats_info['z_outliers_pct'] = len(z_outliers) / len(feature_data) * 100
    
    return stats_info

# 执行诊断
print("\n=== 详细数据诊断 ===")
diagnosis_results = {}

for col in existing_columns:
    print(f"\n--- {target_names[col]} ({col}) ---")
    result = diagnose_feature(train_data, col, target_names[col])
    
    if result:
        diagnosis_results[col] = result
        
        print(f"数据量: {result['count']:,}")
        print(f"缺失值: {result['missing_count']} ({result['missing_pct']:.2f}%)")
        print(f"均值: {result['mean']:.6f}")
        print(f"中位数: {result['median']:.6f}")
        print(f"标准差: {result['std']:.6f}")
        print(f"最小值: {result['min']:.6f}")
        print(f"最大值: {result['max']:.6f}")
        print(f"偏度: {result['skewness']:.4f}")
        print(f"峰度: {result['kurtosis']:.4f}")
        print(f"IQR异常值: {result['outliers_count']} ({result['outliers_pct']:.2f}%)")
        print(f"Z-score异常值: {result['z_outliers_count']} ({result['z_outliers_pct']:.2f}%)")
        print(f"分位数: P1={result['p1']:.6f}, P5={result['p5']:.6f}, P95={result['p95']:.6f}, P99={result['p99']:.6f}")

# 创建诊断可视化
print("\n=== 创建诊断可视化 ===")

# 1. 分布图 + 统计信息
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('目标变量分布诊断图', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    feature_data = train_data[col].dropna()
    
    # 主直方图
    axes[row, col_idx].hist(feature_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # 添加正态分布曲线
    x = np.linspace(feature_data.min(), feature_data.max(), 100)
    normal_dist = stats.norm.pdf(x, feature_data.mean(), feature_data.std())
    axes[row, col_idx].plot(x, normal_dist, 'r-', linewidth=2, label='正态分布')
    
    # 添加均值和中位数线
    axes[row, col_idx].axvline(feature_data.mean(), color='red', linestyle='--', alpha=0.8, label=f'均值: {feature_data.mean():.4f}')
    axes[row, col_idx].axvline(feature_data.median(), color='orange', linestyle='--', alpha=0.8, label=f'中位数: {feature_data.median():.4f}')
    
    # 添加异常值边界
    result = diagnosis_results[col]
    axes[row, col_idx].axvline(result['lower_bound'], color='green', linestyle=':', alpha=0.6, label=f'下界: {result["lower_bound"]:.4f}')
    axes[row, col_idx].axvline(result['upper_bound'], color='green', linestyle=':', alpha=0.6, label=f'上界: {result["upper_bound"]:.4f}')
    
    axes[row, col_idx].set_title(f'{target_names[col]}\n偏度: {result["skewness"]:.3f}, 峰度: {result["kurtosis"]:.3f}', fontsize=11)
    axes[row, col_idx].set_xlabel('数值')
    axes[row, col_idx].set_ylabel('密度')
    axes[row, col_idx].legend(fontsize=8)
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_diagnosis_output/feature_distribution_diagnosis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 箱线图 + 异常值
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('目标变量箱线图与异常值检测', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    feature_data = train_data[col].dropna()
    result = diagnosis_results[col]
    
    # 箱线图
    bp = axes[row, col_idx].boxplot(feature_data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    
    # 标记异常值
    outliers = feature_data[(feature_data < result['lower_bound']) | (feature_data > result['upper_bound'])]
    if len(outliers) > 0:
        axes[row, col_idx].plot([1] * len(outliers), outliers, 'ro', alpha=0.6, markersize=3)
    
    axes[row, col_idx].set_title(f'{target_names[col]}\n异常值: {result["outliers_count"]} ({result["outliers_pct"]:.1f}%)', fontsize=11)
    axes[row, col_idx].set_ylabel('数值')
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_diagnosis_output/feature_boxplot_diagnosis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Q-Q图 (正态性检验)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('目标变量Q-Q图 (正态性检验)', fontsize=16, fontweight='bold')

for i, col in enumerate(existing_columns):
    row = i // 3
    col_idx = i % 3
    
    feature_data = train_data[col].dropna()
    
    # Q-Q图
    stats.probplot(feature_data, dist="norm", plot=axes[row, col_idx])
    
    # 进行Shapiro-Wilk正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(feature_data)
    
    axes[row, col_idx].set_title(f'{target_names[col]}\nShapiro-Wilk: p={shapiro_p:.4f}', fontsize=11)
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_diagnosis_output/feature_qqplot_diagnosis.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 统计摘要表
print("\n=== 统计摘要表 ===")
summary_df = pd.DataFrame(diagnosis_results).T
summary_df['name'] = [target_names[col] for col in summary_df.index]

# 重新排列列的顺序
columns_order = ['name', 'count', 'missing_count', 'missing_pct', 'mean', 'median', 'std', 
                'min', 'max', 'skewness', 'kurtosis', 'outliers_count', 'outliers_pct', 
                'z_outliers_count', 'z_outliers_pct', 'p1', 'p5', 'p95', 'p99']
summary_df = summary_df[columns_order]

print(summary_df.round(4))

# 保存摘要表
summary_df.to_csv('data_diagnosis_output/feature_diagnosis_summary.csv', encoding='utf-8-sig')
print("\n诊断摘要已保存到: data_diagnosis_output/feature_diagnosis_summary.csv")

print("\n=== 诊断完成 ===")
print("生成的可视化文件:")
print("- data_diagnosis_output/feature_distribution_diagnosis.png")
print("- data_diagnosis_output/feature_boxplot_diagnosis.png") 
print("- data_diagnosis_output/feature_qqplot_diagnosis.png")
print("- data_diagnosis_output/feature_diagnosis_summary.csv") 