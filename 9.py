"""
    将数据转换为Pytorch张量主要方法之间的区别,以及怎么选择最佳的方法
"""

# 张量是数学中的概念，表示多维数组，可以进行加、减、乘、除等运算。
# PyTorch张量是PyTorch框架中的数据结构，也表示多维数组，可以使用PyTorch提供的函数进行加、减、乘、除等运算。# PyTorch张量是抽象张量的一种具体实现，它提供了丰富的操作和函数，方便用户进行深度学习相关的计算和操作。

import torch
import numpy as np

data = np.array([1, 2, 3]) 
print(type(data))

# 下面比较这4种方法的差异
# 1
t1 = torch.Tensor(data)
print(t1)
print(t1, t1.dtype) # 类构造函数

# 2
t2 = torch.tensor(data)
print(t2, t2.dtype) # 工厂函数
# 后面三种都是工厂函数,在此处类构造函数和工厂函数区别不明显，
# 将重心放在返回的对象的dtype上, 更倾向于选择工厂函数
# The difference between these two methods is that 
# torch.Tensor() uses the default dtype which is float32,
print(torch.get_default_dtype())
# while torch.tensor() infers the dtype from the input data.
print(torch.tensor([1, 2, 3]))
print(torch.tensor([1.0, 2.0, 3.0]))
# 有趣的是工厂函数也可以直接指定数据的类型
print(torch.tensor([1, 2, 3], dtype=torch.float64))


# 初始化data
data = np.array([1, 2, 3]) 

t1 = torch.Tensor(data)
t2 = torch.tensor(data)

# 3
t3 = torch.as_tensor(data)
print(t3, t3.dtype)
# 4
t4 = torch.from_numpy(data)
print(t4, t4.dtype)

# 修改data的值看看会发生什么
data[0] = 0
data[1] = 0
data[2] = 0
# 在这一步并没有改变张量


print(t1)
print(t2)

print(t3)
print(t4)
# tensor([1., 2., 3.])
# tensor([1, 2, 3], dtype=torch.int32)

# tensor([0, 0, 0], dtype=torch.int32)
# tensor([0, 0, 0], dtype=torch.int32)

# The output results are different because the methods used to create the tensors are different.
# t1 is created using the torch.Tensor() constructor which creates a new tensor with the same data type as the input data.
# t2 is created using the torch.tensor() factory function which infers the data type from the input data.

# t3 is created using the torch.as_tensor() factory function which shares the same memory as the input data.
# t4 is created using the torch.from_numpy() factory function which also shares the same memory as the input data.

# Therefore, if the input data is modified after the tensor is created, the behavior of the different methods will be different.

# Share Data            Copy data
# torch.as_tensor()     torch.tensor()
# torch.from_numpy()    torch.Tensor()

### the best choice ###
"""
    torch.tensor() 
    torch.as_tensor()
"""


