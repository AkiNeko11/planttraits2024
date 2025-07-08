import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出文件夹
os.makedirs('X11_top20f', exist_ok=True)

print("=== X11_mean 特征重要性分析（目标变量对数标准化） ===")

# === 加载数据 ===
df = pd.read_csv('train.csv')
target_col = 'X11_mean'

# 检查目标变量是否存在
if target_col not in df.columns:
    print(f"错误: 目标变量 {target_col} 不存在!")
    print(f"可用的目标变量: {[col for col in df.columns if 'X' in col and 'mean' in col]}")
    exit()

# 明确指定特征列：第2到164列（索引1到163）
feature_cols = df.columns[1:164].tolist()  # 排除第1列(id)，取第2-164列
print(f"特征数量: {len(feature_cols)}")
print(f"特征列范围: {feature_cols[0]} 到 {feature_cols[-1]}")

X = df[feature_cols].values
y = df[target_col].values

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

# === 目标变量对数标准化 ===
print("\n=== 目标变量对数标准化 ===")

# 检查目标变量的分布
print(f"目标变量统计信息:")
print(f"最小值: {y.min():.6f}")
print(f"最大值: {y.max():.6f}")
print(f"均值: {y.mean():.6f}")
print(f"标准差: {y.std():.6f}")

# 检查是否有负值或零值
negative_count = np.sum(y <= 0)
print(f"负值或零值数量: {negative_count}")

if negative_count > 0:
    print("发现负值或零值，使用 log1p 变换（log(1+x)）")
    # 使用 log1p 变换，可以处理零值
    y_log = np.log1p(y)
    # 记录变换信息用于后续反变换
    log_transform_info = "log1p"
else:
    print("所有值都为正，使用 log 变换")
    y_log = np.log(y)
    log_transform_info = "log"

# 标准化对数变换后的值
from sklearn.preprocessing import StandardScaler
y_scaler = StandardScaler()
y_normalized = y_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()

print(f"\n对数标准化后统计信息:")
print(f"最小值: {y_normalized.min():.6f}")
print(f"最大值: {y_normalized.max():.6f}")
print(f"均值: {y_normalized.mean():.6f}")
print(f"标准差: {y_normalized.std():.6f}")

# 保存变换信息
transform_info = {
    'log_transform': log_transform_info,
    'scaler_mean': y_scaler.mean_[0],
    'scaler_scale': y_scaler.scale_[0]
}

# === 分训练集/验证集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# === 方法1: XGBoost 特征重要性 ===
print("\n=== 方法1: XGBoost 特征重要性分析 ===")

# 创建 DMatrix（XGBoost 特有格式）
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

# 训练 XGBoost 模型
params = {
    'tree_method': 'hist',  # 使用CPU版本，避免GPU问题
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

bst = xgb.train(params, dtrain, num_boost_round=200)

# 预测和评估
y_pred = bst.predict(dtest)
r2_xgb = r2_score(y_test, y_pred)
print(f"XGBoost R²: {r2_xgb:.4f}")

# 特征重要性（基于 Gain）
importance_dict = bst.get_score(importance_type='gain')
importance_series = pd.Series(importance_dict).sort_values(ascending=False)

print("\nXGBoost 特征重要性前20（Gain）:")
for i, (feat, imp) in enumerate(importance_series.head(20).items(), 1):
    print(f"{i:2d}. {feat}: {imp:.4f}")

# === 方法2: 随机森林特征重要性 ===
print("\n=== 方法2: 随机森林特征重要性分析 ===")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"随机森林 R²: {r2_rf:.4f}")

# 特征重要性
rf_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n随机森林特征重要性前20:")
for i, (feat, imp) in enumerate(rf_importance.head(20).items(), 1):
    print(f"{i:2d}. {feat}: {imp:.4f}")

# === 方法3: 排列重要性 ===
print("\n=== 方法3: 排列重要性分析 ===")

perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance_series = pd.Series(perm_importance.importances_mean, index=feature_cols).sort_values(ascending=False)

print("\n排列重要性前20:")
for i, (feat, imp) in enumerate(perm_importance_series.head(20).items(), 1):
    print(f"{i:2d}. {feat}: {imp:.4f}")

# === 方法4: 线性回归系数 ===
print("\n=== 方法4: 线性回归系数分析 ===")

# 标准化特征用于线性回归
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"线性回归 R²: {r2_lr:.4f}")

# 特征重要性（基于系数的绝对值）
lr_importance = pd.Series(np.abs(lr.coef_), index=feature_cols).sort_values(ascending=False)

print("\n线性回归系数绝对值前20:")
for i, (feat, imp) in enumerate(lr_importance.head(20).items(), 1):
    print(f"{i:2d}. {feat}: {imp:.4f}")

# === 方法5: 皮尔逊相关系数 ===
print("\n=== 方法5: 皮尔逊相关系数分析 ===")

# 创建包含标准化目标变量的DataFrame
df_analysis = pd.DataFrame(X, columns=feature_cols)
df_analysis[target_col + '_normalized'] = y_normalized

corrs = df_analysis.corr()[target_col + '_normalized'].drop(target_col + '_normalized')
corr_importance = corrs.abs().sort_values(ascending=False)

print("\n皮尔逊相关系数绝对值前20:")
for i, (feat, corr) in enumerate(corr_importance.head(20).items(), 1):
    print(f"{i:2d}. {feat}: {corr:.4f}")

# === 综合排名 ===
print("\n=== 综合特征重要性排名 ===")

# 创建综合评分
feature_scores = pd.DataFrame({
    'XGBoost_Gain': importance_series,
    'RandomForest': rf_importance,
    'Permutation': perm_importance_series,
    'LinearRegression': lr_importance,
    'Correlation': corr_importance
})

# 填充缺失值
feature_scores = feature_scores.fillna(0)

# 标准化各方法的重要性分数
for col in feature_scores.columns:
    feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / (feature_scores[col].max() - feature_scores[col].min())

# 计算综合得分
feature_scores['综合得分'] = feature_scores.mean(axis=1)
feature_scores = feature_scores.sort_values('综合得分', ascending=False)

print("\n综合特征重要性前20:")
for i, (feat, row) in enumerate(feature_scores.head(20).iterrows(), 1):
    print(f"{i:2d}. {feat}: {row['综合得分']:.4f}")

# === 保存结果 ===
print("\n=== 保存分析结果 ===")

# 保存综合排名
feature_scores.to_csv('X11_top20f/X11_feature_importance_analysis.csv', encoding='utf-8-sig')
print("特征重要性分析结果已保存到: X11_top20f/X11_feature_importance_analysis.csv")

# 保存前20个最重要特征
top_20_features = feature_scores.head(20).index.tolist()
with open('X11_top20f/X11_top_20_features.txt', 'w', encoding='utf-8') as f:
    f.write("X11_mean 最重要的20个特征（目标变量对数标准化）:\n")
    f.write(f"对数变换方法: {log_transform_info}\n")
    f.write(f"标准化均值: {y_scaler.mean_[0]:.6f}\n")
    f.write(f"标准化标准差: {y_scaler.scale_[0]:.6f}\n\n")
    for i, feat in enumerate(top_20_features, 1):
        f.write(f"{i:2d}. {feat}\n")

print("前20个最重要特征已保存到: X11_top20f/X11_top_20_features.txt")

# 保存变换信息
with open('X11_top20f/X11_transform_info.json', 'w', encoding='utf-8') as f:
    json.dump(transform_info, f, indent=2, ensure_ascii=False)
print("变换信息已保存到: X11_top20f/X11_transform_info.json")

# === 可视化 ===
print("\n=== 生成可视化图表 ===")

# 1. 综合重要性前20可视化
plt.figure(figsize=(12, 8))
top_20_scores = feature_scores.head(20)['综合得分']
plt.barh(range(len(top_20_scores)), top_20_scores)
plt.yticks(range(len(top_20_scores)), list(top_20_scores.index))
plt.xlabel('综合重要性得分')
plt.title('X11_mean 最重要的20个特征（目标变量对数标准化）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('X11_top20f/X11_top_20_features.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 各方法重要性对比
plt.figure(figsize=(15, 10))
methods = ['XGBoost_Gain', 'RandomForest', 'Permutation', 'LinearRegression', 'Correlation']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, method in enumerate(methods):
    row = i // 3
    col = i % 3
    top_10 = feature_scores.head(10)[method]
    axes[row, col].barh(range(len(top_10)), top_10)
    axes[row, col].set_yticks(range(len(top_10)))
    axes[row, col].set_yticklabels(list(top_10.index))
    axes[row, col].set_title(f'{method} 前10特征')
    axes[row, col].invert_yaxis()

# 隐藏多余的子图
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('X11_top20f/X11_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 分析完成 ===")
print(f"模型性能对比（基于对数标准化目标变量）:")
print(f"XGBoost R²: {r2_xgb:.4f}")
print(f"随机森林 R²: {r2_rf:.4f}")
print(f"线性回归 R²: {r2_lr:.4f}")
print(f"\n最重要的前5个特征:")
for i, feat in enumerate(top_20_features[:5], 1):
    print(f"{i}. {feat}")

print(f"\n所有输出文件已保存到 X11_top20f 文件夹:")
print("- X11_feature_importance_analysis.csv")
print("- X11_top_20_features.txt")
print("- X11_transform_info.json")
print("- X11_top_20_features.png")
print("- X11_feature_importance_comparison.png")

