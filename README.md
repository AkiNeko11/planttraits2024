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
- **todo网络结构**: 512 → 256 → 128 → 64 → 1  


### RandomForest
- **todo网络结构**: 512 → 256 → 128 → 64 → 1


### LinearRegression
- **todo网络结构**: 512 → 256 → 128 → 64 → 1


### MLP
- **网络结构**: 512 → 256 → 128 → 64 → 1
- **激活函数**: ReLU
- **正则化**: BatchNorm + Dropout (0.3)
- **优化器**: Adam (lr=0.001)
- **损失函数**: MSE
- **早停机制**: patience=60

### 数据预处理

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

每个模型都经过以下优化：
- 早停机制防止过拟合
- 批量归一化提高训练稳定性
- Dropout正则化减少过拟合
- 安全的反变换函数处理异常值

## 文件说明

### 训练脚本
- `MLP_train_X*.py`: 训练各种目标变量的MLP模型
- `X*_train_model.py`: 训练各种目标变量的XGBoost、RandomForest、LinearRegression模型
- 自动保存最佳模型到对应的模型文件夹
- 保存数据预处理统计信息

### 预测脚本
- `X*_predict.py`: 使用训练好的XGBoost、RandomForest、LinearRegression模型进行预测
- `X11_MLP_predict.py`: 使用训练好的MLP模型进行预测
- 自动加载模型和统计信息
- 输出预测结果到对应的预测文件夹

### 数据处理脚本
- `data_analysis.py`: 数据探索性分析
- `data_diagnosis.py`: 数据诊断和清洗
- `data_outlier_filter.py`: 异常值检测和处理
- `data_log_normalization.py`: 对数变换处理
- `X11_topfeatures.py`:目标变量的降维处理分析top影响参数

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