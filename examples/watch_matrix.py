import torch
import numpy as np
import pandas as pd

# 加载 .pt 文件
file_path = "/share/disk/hzb/dataset/assistment2009/adjacency_matrix.pt"

# 加载内容
data = torch.load(file_path)

# 检查数据类型和基本信息
print("数据类型:", type(data))
if isinstance(data, torch.Tensor):
    print("张量形状:", data.shape)
    print("数据内容:", data)
elif isinstance(data, torch.sparse.Tensor):
    print("稀疏张量形状:", data.shape)
    print("稀疏张量内容:")
    print(data)
else:
    print("文件内容:", data)

# 转换为密集矩阵（稀疏矩阵时）
dense_data = data.to_dense()

# 只取左上角 300×300 的部分
dense_data_cropped = dense_data[:1000, :1000]

# 转为 NumPy 数组
numpy_data = dense_data_cropped.numpy()

# 转为 Pandas DataFrame
df = pd.DataFrame(numpy_data)

# 保存到 CSV 文件
df.to_csv("output_matrix_cropped.csv", index=False)

# 检查 DataFrame 的前几行
print(df.head())
