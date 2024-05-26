# 引入相关库
from sklearn.datasets import load_boston
import pandas as pd 

# 使用 load_boston 函数加载波士顿房价数据集。
boston_dataset = load_boston()

# 将数据集中的特征数据转换为一个数据框，并使用特征名称作为列名。
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# 对数据进行标准化，使每个特征的均值为0，标准差为1。
boston = (boston-boston.mean())/(boston.std()) 

col_names = boston.columns.values.tolist()


# 创建一个空字典用于存储新的列名。
columns = {}

# 遍历特征名称列表
for idx, n in enumerate(col_names):
	columns[n] = "x%d"%idx 
    # 将特征名称重命名为 “x0”, “x1”, …，并存储在字典中。

#  使用新的列名重命名数据框的列
boston = boston.rename(columns=columns)	

# 将波士顿房价数据集的目标变量（房价）添加到数据框中。
boston['y'] = boston_dataset.target

# 添加一个索引列，表示每个样本的唯一标识
boston['idx'] = range(boston.shape[0])

# 这一行代码将 boston 数据框中的 idx 列提取出来，存储在变量 idx 中。
idx = boston['idx']

# 这一行代码从 boston 数据框中删除了名为 idx 的列。参数 labels=['idx'] 表示要删除的列名，axis=1 表示按列删除，inplace=True 表示在原数据框上直接进行修改。
boston.drop(labels=['idx'], axis=1, inplace = True)

# 这一行代码将之前提取的 idx 列重新插入到 boston 数据框的第一列位置。
boston.insert(0, 'idx', idx)

# 从第406行开始，将数据划分为训练集和测试集。
eval = boston.iloc[406:]

# 随机抽样80个样本作为第一个数据集。
df1 = eval.sample(80)
df2 = eval.sample(85)

# 提取第一个数据集的特定特征列。
housing_1_eval = df1[["idx", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]]

# 将数据框 housing_1_eval 中的数据保存到名为 “housing_1_eval.csv” 的 CSV 文件中。
housing_1_eval.to_csv('housing_1_eval.csv', index=True, header=True)

# 提取第二个数据集的特定特征列。
housing_2_eval = df2[["idx", "y", "x8", "x9", "x10", "x11", "x12"]]

housing_2_eval.to_csv('housing_2_eval.csv', index=True, header=True)
