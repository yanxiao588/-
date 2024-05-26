
from sklearn.datasets import load_boston
import pandas as pd 


boston_dataset = load_boston()

# 将数据集中的特征数据转换为一个数据框，并使用特征名称作为列名。
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# 对数据进行标准化，使每个特征的均值为0，标准差为1。
boston = (boston-boston.mean())/(boston.std()) 

# 获取数据框的列名列表。
col_names = boston.columns.values.tolist()


# 创建一个空字典用于存储新的列名。
columns = {}
# 遍历特征名称列表
for idx, n in enumerate(col_names):
	columns[n] = "x%d"%idx # 将特征名称重命名为 “x0”, “x1”, …，并存储在字典中。

# 使用新的列名重命名数据框的列。
boston = boston.rename(columns=columns)	

# 将波士顿房价数据集的目标变量（房价）添加到数据框中。
boston['y'] = boston_dataset.target

# 添加一个索引列，表示每个样本的唯一标识。
boston['idx'] = range(boston.shape[0])

idx = boston['idx']

boston.drop(labels=['idx'], axis=1, inplace = True)

boston.insert(0, 'idx', idx)

# 从数据框中提取前406行作为训练集。
train = boston.iloc[:406]

# 随机抽样360个样本作为第一个数据集。
df1 = train.sample(360)

# 随机抽样380个样本作为第二个数据集。
df2 = train.sample(380)

# 提取第一个数据集的特定特征列。
housing_1_train = df1[["idx", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]]

# 将数据集保存为 CSV 文件，命名为 “housing_1_train.csv” 。
housing_1_train.to_csv('housing_1_train.csv', index=False, header=True)

# 提取第二个数据集的特定特征列。
housing_2_train = df2[["idx", "y", "x8", "x9", "x10", "x11", "x12"]]

# 将数据集保存为 CSV 文件，命名为 “housing_2_train.csv”。
housing_2_train.to_csv('housing_2_train.csv', index=False, header=True)

