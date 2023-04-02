# 秩、轴数和形状是深度学习中使用张量的三个重要的属性

import torch
tt = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 秩的含义是张量中元素的维度数,也就是张量的轴的总和数。
# 张量的秩告诉我们需要用多少个索引来访问张量中的特定元素。
# 在PyTorch中，可以使用dim()方法获取张量的秩。
rank = tt.dim()

# 轴的含义是张量中的一个维度，也可以称为张量的某个方向。
# 元素被认为是存在一个轴上的，可以使用索引访问张量的某个轴。
# **张量的秩数就是张量中轴的数量**，可以使用len()函数获取。
# 在PyTorch中，可以使用索引访问张量的某个轴，例如tensor[:, 0]表示访问张量的第一个轴的所有元素。 
shape = tt.shape
num_axes = len(shape)

# 形状的含义是张量中每个轴的长度，也就是每个维度上的元素数量。
# 张量的形状告诉我们张量的大小和维度。
# 在PyTorch中，可以使用shape属性获取张量的形状。
shape = tt.shape

# 打印张量的秩、形状和轴数
print("张量的秩：", rank)
print("张量的轴数：", num_axes)
print("张量的形状：", shape)



# 重塑（reshape）是指改变张量的形状，而不改变其元素的数量和值。
# 只改变了元素的分组
# 在神经网络中，重塑操作非常常见，因为我们需要将输入数据转换为适合网络的形状。

# 张量形状为 6 * 1 
# number: 一维数组，表示标量
# scalar: 标量，只有一个数值

# array: 一维数组，表示向量
# vector: 向量，有多个数值，但只有一个维度

# 2d-array: 二维数组，可以表示矩阵
# matrix: 矩阵，有多个数值，有两个维度

# reshape

# 张量形状为 3 * 2 
# number    scalar  不用索引    0维张量

# array     vector  用一个索引  1维张量

# 2d-array  matrix  用两个索引  2维张量
tt = torch.tensor([
    [1],
    [2],
    
    [3],
    [4],
    
    [5],
    [6]
])
print(tt)
tt_reshaped = tt.reshape(3, 2)
print(tt_reshaped)