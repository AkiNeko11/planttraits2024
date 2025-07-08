import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

print("=== X26_mean 模型预测 ===")

# === 检查模型文件是否存在 ===
model_dir = 'X26_model_results'
if not os.path.exists(model_dir):
    print(f"错误: 模型文件夹 {model_dir} 不存在!")
    print("请先运行 X26_train_model.py 训练模型")
    exit()

required_files = [
    'XGBoost_model.pkl', 'RandomForest_model.pkl', 'LinearRegression_model.pkl',
    'target_scaler.pkl', 'feature_scaler.pkl', 'feature_cols.json', 'transform_info.json'
]

missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join(model_dir, file)):
        missing_files.append(file)

if missing_files:
    print(f"错误: 缺少以下模型文件:")
    for file in missing_files:
        print(f"  - {file}")
    print("请先运行 X26_train_model.py 训练模型")
    exit()

# === 加载模型和配置 ===
print("加载模型和配置...")

# 加载模型
models = {}
for model_name in ['XGBoost', 'RandomForest', 'LinearRegression']:
    with open(f'{model_dir}/{model_name}_model.pkl', 'rb') as f:
        models[model_name] = pickle.load(f)
    print(f"已加载 {model_name} 模型")

# 加载预处理器
with open(f'{model_dir}/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)
with open(f'{model_dir}/feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
print("已加载预处理器")

# 加载配置
with open(f'{model_dir}/feature_cols.json', 'r', encoding='utf-8') as f:
    feature_cols = json.load(f)
with open(f'{model_dir}/transform_info.json', 'r', encoding='utf-8') as f:
    transform_info = json.load(f)
print("已加载配置信息")

print(f"特征数量: {len(feature_cols)}")
print(f"对数变换方法: {transform_info['log_transform']}")

# === 加载测试数据 ===
print("\n=== 加载测试数据 ===")
try:
    df_test = pd.read_csv('test.csv')
    print(f"测试数据形状: {df_test.shape}")
    
    # 检查测试数据是否有目标变量
    target_col = 'X26_mean'
    if target_col in df_test.columns:
        print("测试数据包含目标变量，可以进行完整评估")
        has_target = True
        y_test_original = df_test[target_col].values
    else:
        print("测试数据不包含目标变量，只进行预测")
        has_target = False
        
    # 提取特征
    X_test = df_test[feature_cols].values
    
    # 处理缺失值
    print(f"测试集特征缺失值: {np.isnan(X_test).sum()}")
    X_test = np.nan_to_num(X_test, nan=np.nanmedian(X_test))
    
    if has_target:
        print(f"测试集目标变量缺失值: {np.isnan(y_test_original).sum()}")
        # 移除目标变量缺失值的样本
        valid_mask = ~np.isnan(y_test_original)
        X_test = X_test[valid_mask]
        y_test_original = y_test_original[valid_mask]
        
        # 对测试集目标变量进行相同的变换
        if transform_info['log_transform'] == "log1p":
            y_test_log = np.log1p(y_test_original)
        else:
            y_test_log = np.log(y_test_original)
        
        y_test_normalized = target_scaler.transform(y_test_log.reshape(-1, 1)).flatten()
        
except FileNotFoundError:
    print("未找到 test.csv 文件")
    exit()

# === 进行预测 ===
print("\n=== 进行预测 ===")

predictions = {}
test_results = {}

for name, model in models.items():
    print(f"\n预测 {name} 模型...")
    
    if name == 'LinearRegression':
        X_test_scaled = feature_scaler.transform(X_test)
        y_pred_test = model.predict(X_test_scaled)
    elif name == 'XGBoost':
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        y_pred_test = model.predict(dtest)
    else:
        y_pred_test = model.predict(X_test)
    
    # 反变换到原始空间
    y_pred_log = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    if transform_info['log_transform'] == "log1p":
        y_pred_original = np.expm1(y_pred_log)
    else:
        y_pred_original = np.exp(y_pred_log)
    
    predictions[name] = {
        'normalized': y_pred_test,
        'original': y_pred_original
    }
    
    if has_target:
        # 计算标准化空间的指标
        r2_test = r2_score(y_test_normalized, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_normalized, y_pred_test))
        mae_test = mean_absolute_error(y_test_normalized, y_pred_test)
        
        # 计算原始空间的指标
        r2_original = r2_score(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        
        test_results[name] = {
            'normalized_space': {'R2': r2_test, 'RMSE': rmse_test, 'MAE': mae_test},
            'original_space': {'R2': r2_original, 'RMSE': rmse_original, 'MAE': mae_original}
        }
        
        print(f"{name} 测试集结果:")
        print(f"  标准化空间 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
        print(f"  原始空间 - R²: {r2_original:.4f}, RMSE: {rmse_original:.4f}, MAE: {mae_original:.4f}")
    else:
        print(f"{name} 预测完成")
        print(f"  预测样本数: {len(y_pred_original)}")
        print(f"  预测值范围: {y_pred_original.min():.6f} 到 {y_pred_original.max():.6f}")
        print(f"  预测值均值: {y_pred_original.mean():.6f}")

# === 保存预测结果 ===
print("\n=== 保存预测结果 ===")

# 创建输出文件夹
os.makedirs('X26_predictions', exist_ok=True)

# 保存各模型的预测结果
for name, pred in predictions.items():
    predictions_df = pd.DataFrame({
        'predicted_normalized': pred['normalized'],
        'predicted_original': pred['original']
    })
    predictions_df.to_csv(f'X26_predictions/{name}_predictions.csv', index=False)
    print(f"{name} 预测结果已保存到: X26_predictions/{name}_predictions.csv")

# 保存综合预测结果
all_predictions_df = pd.DataFrame({
    'XGBoost_original': predictions['XGBoost']['original'],
    'RandomForest_original': predictions['RandomForest']['original'],
    'LinearRegression_original': predictions['LinearRegression']['original']
})
all_predictions_df.to_csv('X26_predictions/all_models_predictions.csv', index=False)
print("综合预测结果已保存到: X26_predictions/all_models_predictions.csv")

if has_target:
    # 保存测试集评估结果
    with open('X26_predictions/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print("测试集评估结果已保存到: X26_predictions/test_results.json")

# === 可视化结果 ===
print("\n=== 生成可视化图表 ===")

if has_target:
    # 1. 模型性能对比
    plt.figure(figsize=(12, 8))
    
    # 原始空间R²对比
    models_list = list(test_results.keys())
    r2_original_values = [test_results[model]['original_space']['R2'] for model in models_list]
    
    plt.subplot(2, 2, 1)
    plt.bar(models_list, r2_original_values, color=['blue', 'green', 'red'])
    plt.title('各模型测试集R²对比（原始空间）')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    
    # 标准化空间R²对比
    r2_normalized_values = [test_results[model]['normalized_space']['R2'] for model in models_list]
    
    plt.subplot(2, 2, 2)
    plt.bar(models_list, r2_normalized_values, color=['blue', 'green', 'red'])
    plt.title('各模型测试集R²对比（标准化空间）')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    
    # RMSE对比
    rmse_original_values = [test_results[model]['original_space']['RMSE'] for model in models_list]
    
    plt.subplot(2, 2, 3)
    plt.bar(models_list, rmse_original_values, color=['blue', 'green', 'red'])
    plt.title('各模型测试集RMSE对比（原始空间）')
    plt.ylabel('RMSE')
    
    # MAE对比
    mae_original_values = [test_results[model]['original_space']['MAE'] for model in models_list]
    
    plt.subplot(2, 2, 4)
    plt.bar(models_list, mae_original_values, color=['blue', 'green', 'red'])
    plt.title('各模型测试集MAE对比（原始空间）')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig('X26_predictions/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 预测vs真实值散点图
    plt.figure(figsize=(15, 5))
    
    for i, (name, pred) in enumerate(predictions.items(), 1):
        plt.subplot(1, 3, i)
        
        plt.scatter(y_test_original, pred['original'], alpha=0.6)
        plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{name} 预测vs真实值')
        
        # 添加R²标注
        r2 = test_results[name]['original_space']['R2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('X26_predictions/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    # 当没有目标变量时，只生成预测结果分布图
    plt.figure(figsize=(15, 5))
    
    for i, (name, pred) in enumerate(predictions.items(), 1):
        plt.subplot(1, 3, i)
        
        plt.hist(pred['original'], bins=30, alpha=0.7, color=['blue', 'green', 'red'][i-1])
        plt.xlabel('预测的X26_mean值')
        plt.ylabel('频数')
        plt.title(f'{name} 预测结果分布')
        
        # 添加统计信息
        mean_val = pred['original'].mean()
        std_val = pred['original'].std()
        plt.text(0.05, 0.95, f'均值: {mean_val:.4f}\n标准差: {std_val:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('X26_predictions/prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 各模型预测结果对比
    plt.figure(figsize=(12, 6))
    
    # 箱线图对比
    plt.subplot(1, 2, 1)
    all_preds = [pred['original'] for pred in predictions.values()]
    plt.boxplot(all_preds, labels=list(predictions.keys()))
    plt.title('各模型预测结果分布对比')
    plt.ylabel('预测的X26_mean值')
    
    # 散点图对比
    plt.subplot(1, 2, 2)
    for i, (name, pred) in enumerate(predictions.items()):
        plt.scatter(range(len(pred['original'])), pred['original'], alpha=0.6, label=name)
    plt.xlabel('样本索引')
    plt.ylabel('预测的X26_mean值')
    plt.title('各模型预测结果散点图')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('X26_predictions/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n=== 预测完成 ===")
print(f"所有结果已保存到 X26_predictions 文件夹")

if has_target:
    print(f"\n最佳模型（基于测试集R²）:")
    best_model = max(test_results.keys(), key=lambda x: test_results[x]['original_space']['R2'])
    best_r2 = test_results[best_model]['original_space']['R2']
    print(f"{best_model}: R² = {best_r2:.4f}")
    
    print(f"\n生成的文件:")
    print("- 预测结果: *_predictions.csv, all_models_predictions.csv")
    print("- 评估结果: test_results.json")
    print("- 可视化: model_performance_comparison.png, prediction_vs_actual.png")
else:
    print(f"\n预测完成！")
    print(f"所有模型都已用于预测测试集")
    
    print(f"\n生成的文件:")
    print("- 预测结果: *_predictions.csv, all_models_predictions.csv")
    print("- 可视化: prediction_distributions.png, model_comparison.png")
    
    print(f"\n预测结果说明:")
    print("- predicted_normalized: 标准化空间的预测值")
    print("- predicted_original: 原始空间的预测值（已反变换）")
    print("- 可以直接使用 predicted_original 列作为最终预测结果") 