"""
Run a (local) vertical federated learning
process on Titanic dataset using
LogisticRegression models
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import get_datasets, partition_data, scale

# 从CSV文件加载训练数据集和测试数据集
x_train, y_train, x_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)

# 记录训练数据集的样本数量
n_data = x_train.shape[0]

# 对训练数据集和测试数据集进行分区，以便进行联邦学习
split_train = partition_data(x_train)
split_test = partition_data(x_test)

# 用于存储每个分区的模型预测结果、准确率和模型本身的字典
outputs = dict()
accuracies = dict()
models = dict()

# 遍历每个数据分区
for i, (_train, _test) in enumerate(zip(split_train, split_test)):
    # 对数据进行特征缩放
    _train, _test = scale(_train, _test)

    # 使用逻辑回归模型进行训练
    _model = LogisticRegression()
    _model.fit(_train, y_train)

    # 记录模型在训练集上的预测结果
    outputs[i] = _model.predict_proba(_train)

    # 记录模型在训练集上的准确率
    accuracies[i] = 100 * accuracy_score(_model.predict(_train), y_train)

    # 记录模型本身
    models[i] = _model

# 将每个分区的模型预测结果合并成一个新的训练数据集
train_combined = np.empty((n_data, 0))

for _train in outputs.values():
    train_combined = np.hstack((train_combined, _train))

# 使用组件服务器(MLPClassifier)对合并后的训练数据进行训练
comp_server = MLPClassifier(
    hidden_layer_sizes=(500, 500,), learning_rate_init=0.001, verbose=True
)
comp_server.fit(train_combined, y_train)

# 对合并后的训练数据进行预测
pred_train_combined = comp_server.predict(train_combined)

# 计算合并后的训练数据的准确率
train_acc_combined = accuracy_score(pred_train_combined, y_train)

# 输出每个数据分区的训练准确率以及整体合并后的训练准确率
for i, acc in accuracies.items():
    print(f"Holder {i} train accuracy: {acc:.3f}%")

print(f"Combined accuracy: {100*train_acc_combined:.3f}%")
