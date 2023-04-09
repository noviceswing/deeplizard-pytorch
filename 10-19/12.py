"""
    学习元素操作
"""

# 元素操作是指对张量中的元素进行操作，这些张量元素在张量中有对应或相同的索引位置
# 元素操作是在对应元素上进行的，如果两个元素在张量中占据同样的位置，我们可称两个元素是对应的
# 元素的位置由索引决定

# 元素操作包括加、减、乘、除等


import torch 
import numpy as np

t1 = torch.tensor([
    [1, 2],
    [3, 4]
], dtype = torch.float32)

t2 = torch.tensor([
    [9, 8],
    [7, 6]
], dtype = torch.float32)

print(t1.shape, t2.shape)

# 第一个轴是数组，第二个轴为数字

print(t1[0])
print(t1[0][0])
print(t2[0][0])

# 对应关系有 1-->9 2-->8 等

# 两个张量必须具有相同的形状才能进行元素操作，这样才能确保元素操作中的元素位置对应

# addition 加是一种元素操作
# 对应元素相加
print(t1 + t2)
# 同样的减、乘、除也是元素操作

# 使用标量值的算术运算
# 标量是0维，那么可以使用标量与t1可以进行元素操作吗？

print(t1)

print(t1 + 2)
print(t1.add(2))
# add是addition的缩写

print(t1 - 2)
print(t1.sub(2))
# sub是subtract的缩写

print(t1 * 2)
print(t1.mul(2))
# mul是multiple的缩写

print(t1 / 2)
print(t1.div(2))
# div是divide的缩写

# 答案是可以的，这就是我们时常听到的广播机制

# 广播机制是一种强大的机制，它允许PyTorch对不同形状的张量执行元素操作
# 当两个张量形状不同时，PyTorch将自动使用广播机制
# 广播机制的工作方式是，首先将两个张量的形状调整为相同的形状，然后对它们执行元素操作
# PyTorch将自动将较小的张量广播到较大的张量的形状，以便它们具有相同的形状


# 通过对numpy模拟广播的函数的广播，看看在标量值上的广播是什么样子
# np.broadcast函数是用于模拟广播的函数，它返回一个对象，该对象封装了将一个数组广播到另一个数组的结果
# np.broadcast_to() function creates an array.创建的是数组
print(np.broadcast_to(2, t1.shape))
# 该函数不会进行任何实际的操作，而是返回一个可以用于广播的迭代器
# 该迭代器可以用于迭代广播的结果，以便在每个元素上执行操作

# 迭代器是一种可以遍历容器中元素的对象，它可以用于遍历序列中的元素
# 在Python中，迭代器是一种对象，它可以用于遍历可迭代对象中的元素
# 可迭代对象是指可以返回一个迭代器的对象，例如列表、元组、字符串等
# 迭代器对象可以使用next()函数来获取下一个元素，如果没有更多元素，则引发StopIteration异常

# 在PyTorch中，张量是可迭代对象，可以使用迭代器来遍历张量中的元素
# 例如，可以使用for循环遍历张量中的元素，也可以使用iter()函数和next()函数来手动遍历张量中的元素

# 下面是一个使用迭代器遍历张量中元素的示例
# 首先，我们使用iter()函数获取张量的迭代器对象
# 然后，我们使用next()函数来获取下一个元素，直到没有更多元素为止

# 获取张量的迭代器对象
tt = np.broadcast_to(2, t1.shape)
t_iter = iter(tt)
# 使用next()函数遍历张量中的元素
print(next(t_iter))
print(next(t_iter))

# 我们可以看到标量2被广播成了跟t1一样的维度
print(t1 + 2)
# 是这样实现的
print(t1 + torch.tensor(
    np.broadcast_to(2, t1.shape), # 传入的是数组
    dtype = torch.float32 # 这里并不是非torch.float32不可，只是the default dtype which is float32
))
# 这里，我们使用np.broadcast_to()函数来创建一个与t1具有相同形状的数组，
# 该数组的所有元素都是标量值2。
# 然后，我们使用torch.tensor()函数将该数组转换为PyTorch张量，并将其添加到t1中。
# 这样，我们就可以在t1中对所有元素进行标量值的广播操作，而无需使用迭代器来遍历张量中的元素。

# 一个更复杂的例子

t1 = torch.tensor([
    [1, 1],
    [1, 1]
], dtype=torch.float32)

t2 = torch.tensor([2, 4], dtype=torch.float32)

# 使用np.broadcast_to()函数创建一个与t1具有相同形状的数组，该数组的所有元素都是t2，使用t2.numpy()将t2转换为NumPy数组。
broadcasted_t2 = np.broadcast_to(t2.numpy(), t1.shape) 
print(broadcasted_t2)

# 直接使用t2
broadcasted_t2 = np.broadcast_to(t2, t1.shape)
print(broadcasted_t2)
# 由于PyTorch张量和NumPy数组之间可以相互转换，因此这两句的结果是相同的。

# 将t2添加到t1中
result = t1 + broadcasted_t2
print(result)

# 使用广播机制可以避免手动遍历张量中的元素，从而提高代码的效率和可读性。
# 特别是在标准化的过程中，广播机制可以帮助我们将标量值添加到张量中，而无需使用迭代器来遍历张量中的元素。

# 数据预处理的标准化
# 标准化是指将数据转换为具有零均值和单位方差的数据的过程
# 标准化是数据预处理的一种常见方法，它可以帮助我们更好地理解数据，并提高模型的性能
# 标准化的过程是将数据减去其均值，然后除以其标准差
# 均值是指数据的平均值，标准差是指数据的离散程度
# 标准化后的数据具有零均值和单位方差，这意味着数据的平均值为0，标准差为1
# 标准化后的数据分布在[-1, 1]之间，这有助于提高模型的性能并加速训练过程

# 对广播机制需要有深刻的理解！！！


### 比较运算 ###


# 比较运算
# PyTorch还支持比较运算，例如等于（==）、大于（>）、小于（<）等
# 比较运算的结果是一个布尔张量，其中每个元素都是True或False

# False if the operation is False
# True if the operation is True

t = torch.tensor([
    [0, 5, 7],
    [6, 0, 7],
    [0, 8, 0]
], dtype = torch.float32)

# 比较运算
# 也是用于元素操作的
# PyTorch还支持比较运算，例如等于（==）、大于（>）、小于（<）等
# 比较运算的结果是一个布尔张量，其中每个元素都是True或False

# return False if the operation is False
# return True if the operation is True

# t.eq()函数用于比较张量t中的元素是否等于0，返回一个布尔张量
print(t.eq(0))

# t.ge()函数用于比较张量t中的元素是否大于或等于0，返回一个布尔张量
print(t.ge(0))

# t.gt()函数用于比较张量t中的元素是否大于0，返回一个布尔张量
print(t.gt(0))

# t.lt()函数用于比较张量t中的元素是否小于0，返回一个布尔张量
print(t.lt(0))

# t.le()函数用于比较张量t中的元素是否小于或等于7，返回一个布尔张量
print(t.le(7))

# 以最后一个为例，再次理解广播机制
print(t <= torch.tensor(
    np.broadcast_to(7, t.shape)
    ,dtype = torch.float32
))

print(t <= torch.tensor([
    [7, 7, 7],
    [7, 7, 7],
    [7, 7, 7]
]))

# 使用函数进行元素操作
# 我们可以认为这个函数被应用到了张量的每个元素上

# t.abs()函数用于获取张量t中每个元素的绝对值
print(t.abs())

# t.sqrt()函数用于获取张量t中每个元素的平方根
print(t.sqrt())

# t.neg()函数用于获取张量t中每个元素的相反数
print(t.neg())

# t.neg().abs()函数用于获取张量t中每个元素的相反数的绝对值
print(t.neg().abs())


# PyTorch中可以使用元素、分量或者点操作来引用一个张量中的元素
# 元素操作是指使用张量的索引来引用张量中的单个元素
# 分量操作是指使用张量的索引来引用张量中的某个分量
# 点操作是指使用张量的属性来引用张量中的某个分量或属性

