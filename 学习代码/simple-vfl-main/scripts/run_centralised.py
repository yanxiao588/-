"""
Train a LogisticRegression model
on the Titanic dataset
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 将上一级目录添加到系统路径中，以便可以导入src目录下的模块
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 从src目录下的feature_engineering模块导入get_datasets和scale函数
from src.feature_engineering import get_datasets, scale

# 读取训练集和测试集数据，然后使用get_datasets函数进行处理
x_train, y_train, x_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)
# 使用scale函数对训练集和测试集的特征进行标准化
x_train, x_test = scale(x_train, x_test)

# 创建逻辑回归模型的实例
lr = LogisticRegression()

# 使用训练数据拟合模型
lr.fit(x_train, y_train)

# 对训练集数据进行预测
pred_train = lr.predict(x_train)

# 对测试集数据进行预测
pred_test = lr.predict(x_test)

# 打印训练集的准确率，格式化为百分比并保留三位小数
train_acc = accuracy_score(pred_train, y_train)
print(f"Train accuracy: {100*train_acc:.3f}%")
