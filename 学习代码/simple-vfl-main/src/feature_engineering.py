"""
Functions to create a useable dataset from the raw titanic data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 预处理数据，以便更好地适应机器学习模型。



def _bin_age(value):
    if np.isnan(value):
        return "Unknown"
    elif value <= 10:
        return "Child"
    elif value <= 40:
        return "Adult"
    else:
        return "Elderly"


def create_features(df):
    # 将年龄分箱处理，应用自定义的 _bin_age 函数
    df["Age"] = df["Age"].apply(_bin_age)

    # 提取船舱号的首字母，如果船舱号为 'C123'，则提取为 'C'
    df["Cabin"] = df["Cabin"].apply(lambda x: x[0])

    # 从姓名中提取称谓，如 'Mr', 'Miss', 'Mrs' 等
    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

    # 将不常见的称谓统一替换为 'Rare'
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )

    # 将法语称谓 'Mlle' 和 'Ms' 替换为 'Miss', 将 'Mme' 替换为 'Mrs'
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # 对性别、船舱等级、登船地点、称谓、船舱号和年龄进行独热编码
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )

    # 删除不再需要的列，如乘客ID、姓名和票号
    df = df.drop(columns=["PassengerId", "Name", "Ticket",], axis=1,)

    return df


def clean(df):
    # 移除 'Embarked' 列中有缺失值的行，因为只有少数几个缺失，所以直接移除
    df = df.loc[df.Embarked.notnull()]

    # 移除 'Fare' 列中有缺失值的行，同样因为缺失值很少
    df = df.loc[df.Fare.notnull()]

    # 对于 'Cabin' 列中的缺失值，将其填充为 "Unknown"
    df.loc[df.Cabin.isna(), "Cabin"] = "Unknown"

    return df


def get_datasets(df, df_test):
    # 将训练集和测试集合并，以确保在进行独热编码时能够获得正确数量的特征
    df_combined = pd.concat([df, df_test], ignore_index=True)

    # 对合并后的数据集进行清理和特征创建
    df_combined = clean(df_combined)
    df_combined = create_features(df_combined)

    # 从合并后的数据集中分离出训练集和测试集
    # 训练集包含 'Survived' 列（即已知是否存活的数据）
    df = df_combined.loc[df_combined.Survived.notnull()]
    df_test = df_combined.loc[df_combined.Survived.isna()]

    train_y = df["Survived"].values
    train_x = df.drop("Survived", axis=1)

    test_x = df_test.drop("Survived", axis=1)

    return train_x, train_y, test_x


def scale(train, test):
    # 创建 StandardScaler 对象
    s = StandardScaler()

    # 使用训练数据来拟合标准化器，然后对训练数据进行转换
    train = s.fit_transform(train)
    test = s.transform(test)

    return train, test


def partition_data(df):
    # 定义第一部分数据的关键词
    partition_1_keywords = ("Parch", "Cabin", "Pclass")
    partition_1_columns = []

    # 根据关键词，从数据帧中选择包含这些关键词的列
    for kw in partition_1_keywords:
        partition_1_columns.extend([c for c in df.columns if kw in c])

    # 定义第二部分数据的关键词
    partition_2_keywords = ("Sex", "Title")
    partition_2_columns = []

    for kw in partition_2_keywords:
        partition_2_columns.extend([c for c in df.columns if kw in c])

    # 定义第三部分数据，它包含除了第一和第二部分之外的所有列
    partition_3_columns = list(
        set(df.columns) - set(partition_1_columns) - set(partition_2_columns)
    )

    return df[partition_1_columns], df[partition_2_columns], df[partition_3_columns]
