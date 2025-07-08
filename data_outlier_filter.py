import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 忽略所有字体相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='fonttools')
warnings.filterwarnings('ignore', message='.*font.*')
warnings.filterwarnings('ignore', message='.*Font.*')
warnings.filterwarnings('ignore', message='.*findfont.*')

# 设置matplotlib不显示字体警告
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 过滤强度设置
FILTER_STRENGTH = 'super_loose'  # 可选: 'strict'(1.5倍IQR), 'moderate'(2.0倍IQR), 'loose'(2.5倍IQR), 'very_loose'(3.0倍IQR)

# 根据强度设置倍数
strength_multipliers = {
    'strict': 1.5,
    'moderate': 2.0, 
    'loose': 2.5,
    'very_loose': 4.0,
    'super_loose':8.0
}

print(f"使用过滤强度: {FILTER_STRENGTH} ({strength_multipliers[FILTER_STRENGTH]}倍IQR)")

# 创建输出目录
output_dir = 'data_outlier_filter_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取原始数据和诊断结果
print("正在读取数据...")
train_data = pd.read_csv('train.csv')
diagnosis_summary = pd.read_csv('data_diagnosis_output/feature_diagnosis_summary.csv', index_col=0)

print(f"原始数据形状: {train_data.shape}")

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

# 异常值过滤函数
def filter_outliers(data, column, strength=None):
    """
    过滤异常值
    strength: 'strict'(1.5倍IQR), 'moderate'(2.0倍IQR), 'loose'(2.5倍IQR), 'very_loose'(3.0倍IQR)
    """
    if strength is None:
        strength = FILTER_STRENGTH  # 使用全局设置
        
    feature_data = data[column].dropna()
    
    if len(feature_data) == 0:
        return data, {}
    
    outlier_info = {
        'column': column,
        'name': target_names[column],
        'original_count': len(feature_data),
        'outliers_removed': 0,
        'outliers_pct': 0,
        'remaining_count': 0
    }
    
    # IQR方法
    Q1 = feature_data.quantile(0.25)
    Q3 = feature_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # 根据强度选择倍数
    multiplier = strength_multipliers[strength]
    print(f"    使用过滤强度: {strength} ({multiplier}倍IQR)")
    lower_bound_iqr = Q1 - multiplier * IQR
    upper_bound_iqr = Q3 + multiplier * IQR
    
    outlier_info['iqr_lower'] = lower_bound_iqr
    outlier_info['iqr_upper'] = upper_bound_iqr
    outlier_info['iqr_outliers'] = feature_data[(feature_data < lower_bound_iqr) | (feature_data > upper_bound_iqr)]
    
    # 移除异常值
    outliers_to_remove = outlier_info['iqr_outliers']
    outlier_indices = data[data[column].isin(outliers_to_remove)].index
    filtered_data = data.drop(outlier_indices)
    
    outlier_info['outliers_removed'] = len(outliers_to_remove)
    outlier_info['outliers_pct'] = len(outliers_to_remove) / len(feature_data) * 100
    outlier_info['remaining_count'] = len(filtered_data)
    outlier_info['outlier_values'] = outliers_to_remove.tolist()
    
    return filtered_data, outlier_info

# 执行异常值过滤
print("\n=== 开始异常值过滤 ===")
filtered_data = train_data.copy()
all_outlier_info = []

for col in target_columns:
    if col in train_data.columns:
        print(f"\n处理 {target_names[col]} ({col})...")
        
        # 使用指定强度过滤异常值
        filtered_data, outlier_info = filter_outliers(filtered_data, col)
        all_outlier_info.append(outlier_info)
        
        print(f"  原始数据量: {outlier_info['original_count']:,}")
        print(f"  移除异常值: {outlier_info['outliers_removed']:,} ({outlier_info['outliers_pct']:.2f}%)")
        print(f"  剩余数据量: {outlier_info['remaining_count']:,}")
        
        if outlier_info['outliers_removed'] > 0:
            print(f"  异常值范围: {outlier_info['iqr_lower']:.6f} ~ {outlier_info['iqr_upper']:.6f}")
            print(f"  异常值示例: {outlier_info['outlier_values'][:5]}...")

# 创建异常值分析报告
print("\n=== 异常值分析报告 ===")
outlier_report = pd.DataFrame(all_outlier_info)
print(outlier_report[['name', 'original_count', 'outliers_removed', 'outliers_pct', 'remaining_count']].round(2))

# 保存异常值分析报告
outlier_report.to_csv(f'{output_dir}/outlier_analysis_report.csv', encoding='utf-8-sig', index=False)

# 保存过滤后的数据
filtered_data.to_csv(f'{output_dir}/train_filtered.csv', index=False)
print(f"\n过滤后的数据已保存到: {output_dir}/train_filtered.csv")
print(f"过滤后数据形状: {filtered_data.shape}")

# 创建过滤前后对比可视化
print("\n=== 创建过滤前后对比图 ===")

# 1. 数据量对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 原始数据量
original_counts = [info['original_count'] for info in all_outlier_info]
filtered_counts = [info['remaining_count'] for info in all_outlier_info]
names = [info['name'] for info in all_outlier_info]

x = np.arange(len(names))
width = 0.35

ax1.bar(x - width/2, original_counts, width, label='原始数据', color='skyblue', alpha=0.8)
ax1.bar(x + width/2, filtered_counts, width, label='过滤后数据', color='lightgreen', alpha=0.8)
ax1.set_xlabel('目标变量')
ax1.set_ylabel('数据量')
ax1.set_title('过滤前后数据量对比')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 异常值比例
outlier_pcts = [info['outliers_pct'] for info in all_outlier_info]
ax2.bar(names, outlier_pcts, color='red', alpha=0.7)
ax2.set_xlabel('目标变量')
ax2.set_ylabel('异常值比例 (%)')
ax2.set_title('各变量异常值比例')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/filtering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 分布对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('过滤前后分布对比', fontsize=16, fontweight='bold')

for i, col in enumerate(target_columns):
    if col in train_data.columns:
        row = i // 3
        col_idx = i % 3
        
        print(f"绘制分布图: {target_names[col]} ({col})")
        
        # 原始数据分布
        original_data = train_data[col].dropna()
        filtered_col_data = filtered_data[col].dropna()
        
        print(f"  原始数据量: {len(original_data):,}, 过滤后数据量: {len(filtered_col_data):,}")
        
        if len(original_data) > 0:
            # 使用对数刻度处理数值范围很大的数据
            if original_data.max() / original_data.min() > 1000:
                # 对数值范围很大的数据使用对数刻度
                axes[row, col_idx].hist(original_data, bins=50, alpha=0.5, color='skyblue', 
                                       label='原始数据', density=True)
                axes[row, col_idx].hist(filtered_col_data, bins=50, alpha=0.5, color='lightgreen', 
                                       label='过滤后数据', density=True)
                axes[row, col_idx].set_xscale('log')
                axes[row, col_idx].set_xlabel('数值 (对数刻度)')
            else:
                # 正常刻度
                axes[row, col_idx].hist(original_data, bins=50, alpha=0.5, color='skyblue', 
                                       label='原始数据', density=True)
                axes[row, col_idx].hist(filtered_col_data, bins=50, alpha=0.5, color='lightgreen', 
                                       label='过滤后数据', density=True)
                axes[row, col_idx].set_xlabel('数值')
            
            axes[row, col_idx].set_title(f'{target_names[col]}\n原始: {len(original_data):,} → 过滤后: {len(filtered_col_data):,}', 
                                        fontsize=10)
            axes[row, col_idx].set_ylabel('密度')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)
        else:
            axes[row, col_idx].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[row, col_idx].transAxes)
            axes[row, col_idx].set_title(f'{target_names[col]} - 无数据', fontsize=10)
    else:
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].text(0.5, 0.5, f'列 {col} 不存在', ha='center', va='center', transform=axes[row, col_idx].transAxes)
        axes[row, col_idx].set_title(f'{target_names.get(col, col)} - 列不存在', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 箱线图对比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('过滤前后箱线图对比', fontsize=16, fontweight='bold')

for i, col in enumerate(target_columns):
    if col in train_data.columns:
        row = i // 3
        col_idx = i % 3
        
        print(f"绘制箱线图: {target_names[col]} ({col})")
        
        # 原始数据和过滤后数据
        original_data = train_data[col].dropna()
        filtered_col_data = filtered_data[col].dropna()
        
        if len(original_data) > 0:
            # 创建对比数据
            comparison_data = pd.DataFrame({
                '原始数据': original_data,
                '过滤后数据': filtered_col_data
            })
            
            # 绘制箱线图
            comparison_data.boxplot(ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{target_names[col]}\n原始: {len(original_data):,} → 过滤后: {len(filtered_col_data):,}', fontsize=10)
            axes[row, col_idx].set_ylabel('数值')
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # 旋转x轴标签
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        else:
            axes[row, col_idx].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[row, col_idx].transAxes)
            axes[row, col_idx].set_title(f'{target_names[col]} - 无数据', fontsize=10)
    else:
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].text(0.5, 0.5, f'列 {col} 不存在', ha='center', va='center', transform=axes[row, col_idx].transAxes)
        axes[row, col_idx].set_title(f'{target_names.get(col, col)} - 列不存在', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 统计摘要对比
print("\n=== 统计摘要对比 ===")
print("原始数据统计摘要:")
original_summary = train_data[target_columns].describe()
print(original_summary)

print("\n过滤后数据统计摘要:")
filtered_summary = filtered_data[target_columns].describe()
print(filtered_summary)

# 保存统计摘要
original_summary.to_csv(f'{output_dir}/original_data_summary.csv', encoding='utf-8-sig')
filtered_summary.to_csv(f'{output_dir}/filtered_data_summary.csv', encoding='utf-8-sig')

# 5. 详细异常值信息
print("\n=== 详细异常值信息 ===")
for info in all_outlier_info:
    print(f"\n{info['name']} ({info['column']}):")
    print(f"  异常值数量: {info['outliers_removed']:,}")
    print(f"  异常值比例: {info['outliers_pct']:.2f}%")
    print(f"  异常值范围: {info['iqr_lower']:.6f} ~ {info['iqr_upper']:.6f}")
    print(f"  异常值示例 (前10个): {info['outlier_values'][:10]}")

# 保存详细异常值信息
detailed_outlier_info = []
for info in all_outlier_info:
    detailed_info = {
        '变量名': info['name'],
        '列名': info['column'],
        '原始数据量': info['original_count'],
        '异常值数量': info['outliers_removed'],
        '异常值比例(%)': round(info['outliers_pct'], 2),
        '剩余数据量': info['remaining_count'],
        'IQR下界': round(info['iqr_lower'], 6),
        'IQR上界': round(info['iqr_upper'], 6),
        '异常值示例': str(info['outlier_values'][:5]) + '...' if len(info['outlier_values']) > 5 else str(info['outlier_values'])
    }
    detailed_outlier_info.append(detailed_info)

detailed_df = pd.DataFrame(detailed_outlier_info)
detailed_df.to_csv(f'{output_dir}/detailed_outlier_info.csv', encoding='utf-8-sig', index=False)

print(f"\n=== 异常值过滤完成 ===")
print(f"输出文件:")
print(f"- {output_dir}/train_filtered.csv (过滤后的数据)")
print(f"- {output_dir}/outlier_analysis_report.csv (异常值分析报告)")
print(f"- {output_dir}/detailed_outlier_info.csv (详细异常值信息)")
print(f"- {output_dir}/original_data_summary.csv (原始数据统计摘要)")
print(f"- {output_dir}/filtered_data_summary.csv (过滤后数据统计摘要)")
print(f"- {output_dir}/filtering_comparison.png (过滤对比图)")
print(f"- {output_dir}/distribution_comparison.png (分布对比图)")
print(f"- {output_dir}/boxplot_comparison.png (箱线图对比)")

print(f"\n数据过滤统计:")
print(f"原始数据量: {len(train_data):,}")
print(f"过滤后数据量: {len(filtered_data):,}")
print(f"总移除数据量: {len(train_data) - len(filtered_data):,}")
print(f"数据保留率: {len(filtered_data) / len(train_data) * 100:.2f}%") 