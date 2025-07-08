# PlantTraits2024 - FGVC11

## 项目简介

来源于kaggle的题目 [PlantTraits2024 - FGVC11](https://www.kaggle.com/competitions/planttraits2024)

这是一个植物性状预测竞赛，我们团队将其作为机器学习的课程设计项目。使用多种模型来预测植物的性状，进行不同模型的对比分析。项目包含多个目标变量的多种预测模型，每个模型都经过精心训练和优化。

## 项目结构

```
planttraits2024/
├── MLP_train_X*.py              # 各种目标变量的MLP训练脚本
├── X*_predict.py                # 各种目标变量的预测脚本
├── X*_model_MLP_*/              # 训练好的MLP模型文件夹
├── X*_model_results/            # 训练好的其他模型文件夹
├── X*_predictions/              # 预测结果文件夹
├── X*_top20f/                   # 目标变量的top20影响参数文件夹
├── X*_topfeatures.py/           # 分析目标变量的top影响参数
├── data_*.py                    # 数据处理和分析脚本
├── train.csv                    # 训练数据(过滤异常值数据)
├── train(origin).csv            # 训练数据(原始数据)
├── test.csv                     # 测试数据
├── sample_submission.csv        # 提交格式示例
├── test-sub-*.csv               # 不同模型的测试提交数据
├── target_name_meta.tsv         # 目标变量元数据
├── requirememt.txt              # 项目需求
└── README.md                    # 项目说明文档
```

## 目标变量

项目支持预测以下植物性状：
- X4_mean
- X11_mean
- X18_mean
- X26_mean
- X50_mean
- X3112_mean

## 模型架构

### XGBoost
- **算法**: 梯度提升决策树
- **目标函数**: reg:squarederror (均方误差回归)
- **评估指标**: RMSE
- **主要参数**:
  - max_depth: 6 (树的最大深度)
  - learning_rate: 0.1 (学习率)
  - subsample: 0.8 (样本采样比例)
  - colsample_bytree: 0.8 (特征采样比例)
  - num_boost_round: 200 (提升轮数)
  - early_stopping_rounds: 20 (早停轮数)
- **树方法**: hist (直方图算法)

### RandomForest
- **算法**: 随机森林回归
- **主要参数**:
  - n_estimators: 100 (决策树数量)
  - random_state: 42 (随机种子)
  - n_jobs: -1 (使用所有CPU核心)
- **特征选择**: 随机特征子集
- **集成方式**: 平均预测

### LinearRegression
- **算法**: 线性回归
- **特征预处理**: StandardScaler标准化
- **正则化**: 无正则化项
- **求解方法**: 最小二乘法
- **适用场景**: 线性关系较强的数据

### MLP
- **网络结构**: 512 → 256 → 128 → 64 → 1
- **激活函数**: ReLU
- **正则化**: BatchNorm + Dropout (0.3)
- **优化器**: Adam (lr=0.001)
- **损失函数**: MSE
- **早停机制**: patience=60

### 数据预处理
- **特征**: 使用第2-164列作为特征（163个特征）
- **目标变量**: log1p变换 + StandardScaler标准化
- **缺失值处理**: 中位数填充
- **特征标准化**: 仅线性回归使用StandardScaler，其他模型使用原始特征值

## 使用方法

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac(可选):
source .venv/bin/activate

# 安装依赖
pip install torch pandas numpy scikit-learn
```

### 2. 数据处理

```bash
# 数据分析
python data_analysis.py

# 数据诊断
python data_diagnosis.py

# 异常值过滤
python data_outlier_filter.py

# 对数标准化
python data_log_normalization.py

# 降维处理（以X11为例）
python X11_topfeatures.py
```

### 3. 训练模型

```bash
# 以X11为例
# 训练XGBoost、RandomForest、LinearRegression模型
python X11_train_model.py

# 训练MLP模型
python MLP_train_X11_withoutTopf.py

# 训练MLP模型（降维后选择top20参数）
python MLP_train_X11_Topf.py
```

### 4. 进行预测

```bash
# 以X11为例
# 采用XGBoost、RandomForest、LinearRegression模型进行推理
python X11_predict.py

# 使用MLP模型进行预测
python X11_MLP_predict.py
```

## 模型性能

### 模型对比策略
项目采用多种机器学习算法进行对比分析：
- **传统机器学习**: XGBoost、RandomForest、LinearRegression
- **深度学习**: MLP (多层感知机)
- **集成学习**: 随机森林和XGBoost的集成优势

### 优化策略
每个模型都经过以下优化：
- **XGBoost**: 早停机制防止过拟合，直方图算法提高训练效率
- **RandomForest**: 随机特征选择减少过拟合，并行训练提高效率
- **LinearRegression**: 特征标准化提高数值稳定性
- **MLP**: 早停机制、批量归一化、Dropout正则化防止过拟合

### 评估指标
- **R² (决定系数)**: 衡量模型解释方差的能力
- **RMSE (均方根误差)**: 衡量预测误差的大小
- **MAE (平均绝对误差)**: 衡量预测误差的绝对值

### 数据变换策略
- **目标变量**: log1p变换处理偏态分布，StandardScaler标准化
- **特征处理**: 中位数填充缺失值，保持原始特征值
- **验证策略**: 80%训练集，20%验证集，固定随机种子确保可重现性

## 文件说明

### 训练脚本
- `MLP_train_X*.py`: 训练各种目标变量的MLP模型
  - 自动保存最佳模型到对应的模型文件夹
  - 保存数据预处理统计信息
  - 支持早停机制和模型验证
- `X*_train_model.py`: 训练各种目标变量的XGBoost、RandomForest、LinearRegression模型
  - 同时训练三种传统机器学习模型
  - 自动保存模型、预处理器和配置信息
  - 生成验证集性能对比和可视化图表

### 预测脚本
- `X*_predict.py`: 使用训练好的XGBoost、RandomForest、LinearRegression模型进行预测
  - 自动加载模型、预处理器和配置信息
  - 输出预测结果到对应的预测文件夹
  - 支持批量预测和结果统计
- `X*_MLP_predict.py`: 使用训练好的MLP模型进行预测
  - 自动加载模型和统计信息
  - 安全的反变换函数处理异常值
  - 输出预测结果和统计信息

### 数据处理脚本
- `data_analysis.py`: 数据探索性分析
  - 数据分布可视化
  - 特征相关性分析
  - 缺失值统计
- `data_diagnosis.py`: 数据诊断和清洗
  - 异常值检测
  - 数据质量评估
  - 清洗建议
- `data_outlier_filter.py`: 异常值检测和处理
  - 多种异常值检测方法
  - 自动过滤异常样本
  - 生成清洗后的数据集
- `log_normalization.py`: 对数变换处理
  - 目标变量分布分析
  - 对数变换效果评估
  - 变换参数优化
- `X*_topfeatures.py`: 目标变量的降维处理分析top影响参数
  - 特征重要性分析
  - 特征选择策略
  - 降维效果评估

### 输出文件结构
```
X*_model_results/
├── XGBoost_model.pkl              # XGBoost模型文件
├── RandomForest_model.pkl         # 随机森林模型文件
├── LinearRegression_model.pkl     # 线性回归模型文件
├── target_scaler.pkl              # 目标变量标准化器
├── feature_scaler.pkl             # 特征标准化器
├── feature_cols.json              # 特征列名配置
├── transform_info.json            # 数据变换信息
├── validation_results.json        # 验证集评估结果
├── validation_predictions.csv     # 验证集预测结果
├── validation_statistics.csv      # 验证集统计指标
└── validation_performance.png     # 性能对比可视化图

X*_model_MLP_*/
├── model.pt                       # MLP模型权重
└── stats.json                     # 数据统计信息

X*_predictions/
├── X*_predictions.csv             # 预测结果文件
└── prediction_stats.json          # 预测统计信息
```

## 注意事项

1. **数据文件**: 大型数据文件(如模型、训练数据集等)已被git排除，需单独获取
2. **硬件设备**: 本次训练主要设备 NVIDIA GeForce RTX 4060 Laptop GPU，使用本人笔记本电脑进行本地训练
3. **GPU支持**: 代码自动检测GPU可用性，支持CUDA加速
4. **内存使用**: 大数据集可能需要较大内存，建议分批处理

## 依赖包

主要依赖包：
- torch 2.6.0+cu124
- pandas  2.3.0
- numpy 2.3.1
- scikit-learn 1.7.0

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。 
