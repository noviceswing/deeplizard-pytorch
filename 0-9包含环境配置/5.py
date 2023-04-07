# 张量可以是多维的，因此可以表示和处理高维度数据。
# 在PyTorch中，可以使用张量进行高效计算，包括在GPU上进行计算。

# 张量的具体实例 6 种 2 类

# 第一类:计算机术语
# number: 一维数组，表示标量
# array: 一维数组，表示向量
# 2d-array: 二维数组，可以表示矩阵

# 第二类:数学术语
# scalar: 标量，只有一个数值
# vector: 向量，有多个数值，但只有一个维度
# matrix: 矩阵，有多个数值，有两个维度

# 是一一对应的关系, number对应scalar......
# 每一组的关系都与访问特定元素的索引数有关
# number    scalar  不用索引    0维张量
# array     vector  用一个索引  1维张量
# 2d-array  matrix  用两个索引  2维张量

### 注意 ###
# 当需要指定超过两个索引来访问特定元素时
# 使用通用语言 n维张量

# 最终我们是使用n维**张量**来确定我们正在处理的维度数量


d = [1, 2, 3, 4]
print(d[2])

dd = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
print(dd[0][2])

# 使用pytorch

import torch

t = torch.tensor([1, 2, 3, 4])
print(t[2].item())

tt = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
print(tt[0][2].item())

# .item()只能打印一个数字
# def item(self) -> Number: ...

# 创建dd后也可以直接转换
tt = torch.tensor(dd)
print(type(tt))