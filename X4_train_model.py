import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出文件夹
os.makedirs('X4_model_results', exist_ok=True)

print("=== X4_mean 模型训练 ===")

# === 加载训练数据 ===
print("加载训练数据...")
df_train = pd.read_csv('train.csv')
target_col = 'X4_mean'

# 检查目标变量是否存在
if target_col not in df_train.columns:
    print(f"错误: 目标变量 {target_col} 不存在!")
    print(f"可用的目标变量: {[col for col in df_train.columns if 'X' in col and 'mean' in col]}")
    exit()

# 明确指定特征列：第2到164列（索引1到163）
feature_cols = df_train.columns[1:164].tolist()
print(f"特征数量: {len(feature_cols)}")

X_train_full = df_train[feature_cols].values
y_train_full = df_train[target_col].values

# 处理缺失值
print(f"训练集特征缺失值: {np.isnan(X_train_full).sum()}")
print(f"训练集目标变量缺失值: {np.isnan(y_train_full).sum()}")

# 用中位数填充特征缺失值
X_train_full = np.nan_to_num(X_train_full, nan=np.nanmedian(X_train_full))

# 移除目标变量缺失值的样本
valid_mask = ~np.isnan(y_train_full)
X_train_full = X_train_full[valid_mask]
y_train_full = y_train_full[valid_mask]

print(f"处理后训练数据形状: X={X_train_full.shape}, y={y_train_full.shape}")

# === 目标变量对数标准化 ===
print("\n=== 目标变量对数标准化 ===")

# 检查目标变量的分布
print(f"目标变量统计信息:")
print(f"最小值: {y_train_full.min():.6f}")
print(f"最大值: {y_train_full.max():.6f}")
print(f"均值: {y_train_full.mean():.6f}")
print(f"标准差: {y_train_full.std():.6f}")

# 检查是否有负值或零值
negative_count = np.sum(y_train_full <= 0)
print(f"负值或零值数量: {negative_count}")

if negative_count > 0:
    print("发现负值或零值，使用 log1p 变换（log(1+x)）")
    y_train_log = np.log1p(y_train_full)
    log_transform_info = "log1p"
else:
    print("所有值都为正，使用 log 变换")
    y_train_log = np.log(y_train_full)
    log_transform_info = "log"

# 标准化对数变换后的值
y_scaler = StandardScaler()
y_train_normalized = y_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()

print(f"\n对数标准化后统计信息:")
print(f"最小值: {y_train_normalized.min():.6f}")
print(f"最大值: {y_train_normalized.max():.6f}")
print(f"均值: {y_train_normalized.mean():.6f}")
print(f"标准差: {y_train_normalized.std():.6f}")

# === 分训练集/验证集 ===
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_normalized, test_size=0.2, random_state=42)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")

# === 训练多个模型 ===
print("\n=== 训练模型 ===")

models = {}
results = {}

# 1. XGBoost
print("训练 XGBoost 模型...")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

params = {
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
models['XGBoost'] = bst

# 验证集预测
y_pred_val = bst.predict(dval)
r2_xgb = r2_score(y_val, y_pred_val)
results['XGBoost'] = {'R2': r2_xgb, 'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)), 'MAE': mean_absolute_error(y_val, y_pred_val)}
print(f"XGBoost 验证集 R²: {r2_xgb:.4f}")

# 2. 随机森林
print("训练随机森林模型...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['RandomForest'] = rf

y_pred_rf = rf.predict(X_val)
r2_rf = r2_score(y_val, y_pred_rf)
results['RandomForest'] = {'R2': r2_rf, 'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_rf)), 'MAE': mean_absolute_error(y_val, y_pred_rf)}
print(f"随机森林验证集 R²: {r2_rf:.4f}")

# 3. 线性回归
print("训练线性回归模型...")
# 标准化特征用于线性回归
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['LinearRegression'] = lr

y_pred_lr = lr.predict(X_val_scaled)
r2_lr = r2_score(y_val, y_pred_lr)
results['LinearRegression'] = {'R2': r2_lr, 'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_lr)), 'MAE': mean_absolute_error(y_val, y_pred_lr)}
print(f"线性回归验证集 R²: {r2_lr:.4f}")

# === 保存模型和结果 ===
print("\n=== 保存模型和结果 ===")

# 保存模型
for name, model in models.items():
    with open(f'X4_model_results/{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"{name} 模型已保存到: X4_model_results/{name}_model.pkl")

# 保存预处理器
with open('X4_model_results/target_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
with open('X4_model_results/feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
print("预处理器已保存")

# 保存特征列名
with open('X4_model_results/feature_cols.json', 'w', encoding='utf-8') as f:
    json.dump(feature_cols, f, indent=2, ensure_ascii=False)
print("特征列名已保存")

# 保存变换信息
transform_info = {
    'log_transform': log_transform_info,
    'target_scaler_mean': y_scaler.mean_[0],
    'target_scaler_scale': y_scaler.scale_[0]
}
with open('X4_model_results/transform_info.json', 'w', encoding='utf-8') as f:
    json.dump(transform_info, f, indent=2, ensure_ascii=False)
print("变换信息已保存")

# 保存验证集评估结果
with open('X4_model_results/validation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("验证集评估结果已保存")

# === 可视化验证结果 ===
print("\n=== 生成验证结果可视化 ===")

# 生成验证集预测结果CSV文件
print("生成验证集预测结果CSV文件...")

validation_predictions = {}
for name, model in models.items():
    if name == 'LinearRegression':
        y_pred_val = model.predict(X_val_scaled)
    elif name == 'XGBoost':
        y_pred_val = model.predict(dval)
    else:
        y_pred_val = model.predict(X_val)
    
    # 反变换到原始空间
    y_pred_log = y_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
    if log_transform_info == "log1p":
        y_pred_original = np.expm1(y_pred_log)
    else:
        y_pred_original = np.exp(y_pred_log)
    
    validation_predictions[name] = {
        'normalized': y_pred_val,
        'original': y_pred_original
    }

# 反变换验证集真实值
y_val_log = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
if log_transform_info == "log1p":
    y_val_original = np.expm1(y_val_log)
else:
    y_val_original = np.exp(y_val_log)

# 创建验证集预测结果DataFrame
validation_df = pd.DataFrame({
    'true_normalized': y_val,
    'true_original': y_val_original
})

# 添加各模型的预测结果
for name, pred in validation_predictions.items():
    validation_df[f'{name}_predicted_normalized'] = pred['normalized']
    validation_df[f'{name}_predicted_original'] = pred['original']

# 保存验证集预测结果
validation_df.to_csv('X4_model_results/validation_predictions.csv', index=False)
print("验证集预测结果已保存到: X4_model_results/validation_predictions.csv")

# 生成验证集性能统计
validation_stats = pd.DataFrame({
    'Model': list(results.keys()),
    'R2_Normalized': [results[model]['R2'] for model in results.keys()],
    'RMSE_Normalized': [results[model]['RMSE'] for model in results.keys()],
    'MAE_Normalized': [results[model]['MAE'] for model in results.keys()]
})

# 计算原始空间的指标
for name, pred in validation_predictions.items():
    r2_orig = r2_score(y_val_original, pred['original'])
    rmse_orig = np.sqrt(mean_squared_error(y_val_original, pred['original']))
    mae_orig = mean_absolute_error(y_val_original, pred['original'])
    
    validation_stats.loc[validation_stats['Model'] == name, 'R2_Original'] = r2_orig
    validation_stats.loc[validation_stats['Model'] == name, 'RMSE_Original'] = rmse_orig
    validation_stats.loc[validation_stats['Model'] == name, 'MAE_Original'] = mae_orig

validation_stats.to_csv('X4_model_results/validation_statistics.csv', index=False)
print("验证集统计结果已保存到: X4_model_results/validation_statistics.csv")

plt.figure(figsize=(12, 8))

# 验证集R²对比
models_list = list(results.keys())
r2_values = [results[model]['R2'] for model in models_list]

plt.subplot(2, 2, 1)
plt.bar(models_list, r2_values, color=['blue', 'green', 'red'])
plt.title('各模型验证集R^2对比')
plt.ylabel('R^2')
plt.ylim(0, 1)

# RMSE对比
rmse_values = [results[model]['RMSE'] for model in models_list]

plt.subplot(2, 2, 2)
plt.bar(models_list, rmse_values, color=['blue', 'green', 'red'])
plt.title('各模型验证集RMSE对比')
plt.ylabel('RMSE')

# MAE对比
mae_values = [results[model]['MAE'] for model in models_list]

plt.subplot(2, 2, 3)
plt.bar(models_list, mae_values, color=['blue', 'green', 'red'])
plt.title('各模型验证集MAE对比')
plt.ylabel('MAE')

# 预测vs真实值散点图
plt.subplot(2, 2, 4)
for i, (name, model) in enumerate(models.items()):
    if name == 'LinearRegression':
        y_pred_val = model.predict(X_val_scaled)
    elif name == 'XGBoost':
        y_pred_val = model.predict(dval)
    else:
        y_pred_val = model.predict(X_val)
    
    # 反变换到原始空间
    y_pred_log = y_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
    if log_transform_info == "log1p":
        y_pred_original = np.expm1(y_pred_log)
    else:
        y_pred_original = np.exp(y_pred_log)
    
    # 反变换验证集真实值
    y_val_log = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    if log_transform_info == "log1p":
        y_val_original = np.expm1(y_val_log)
    else:
        y_val_original = np.exp(y_val_log)
    
    plt.scatter(y_val_original, y_pred_original, alpha=0.6, label=name)

plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'k--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('验证集预测vs真实值')
plt.legend()

plt.tight_layout()
plt.savefig('X4_model_results/validation_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 训练完成 ===")
print(f"所有模型和结果已保存到 X4_model_results 文件夹")

print(f"\n验证集性能对比:")
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")

best_model = max(results.keys(), key=lambda x: results[x]['R2'])
best_r2 = results[best_model]['R2']
print(f"\n最佳模型（基于验证集R2）: {best_model} (R² = {best_r2:.4f})")

print(f"\n生成的文件:")
print("- 模型文件: XGBoost_model.pkl, RandomForest_model.pkl, LinearRegression_model.pkl")
print("- 预处理器: target_scaler.pkl, feature_scaler.pkl")
print("- 配置信息: feature_cols.json, transform_info.json")
print("- 验证结果: validation_results.json, validation_predictions.csv, validation_statistics.csv, validation_performance.png") 