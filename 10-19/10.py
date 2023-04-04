"""
    使用所了解到的张量知识
    覆盖神经网络中的基本张量操作
"""

# 包含的四大操作
# 1 重塑操作
# 2 元素操作
# 3 还原操作
# 4 访问操作

# 重塑操作：reshape
# 重塑操作是指改变张量的形状，但是不改变张量中元素的数量或值。
# 重塑操作可以用于改变张量的形状，以适应不同的计算需求。

# 元素操作是指对张量中的元素进行操作。
# 这些操作包括加、减、乘、除等。

# 还原操作：squeeze、unsqueeze
# 还原操作是指减少或增加张量的维度。
# 这些操作可以用于减少或增加张量的形状，以适应不同的计算需求。


# 访问操作：索引、切片
# 访问操作是指访问张量中的特定元素或元素子集。
# 这些操作可以用于访问张量中的特定元素或元素子集，以适应不同的计算需求。


"""
    reshape 重塑操作
"""

import torch
t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)
# 两种方式查看形状
print(t.size())
# 或
print(t.shape)
# 通过形状查看秩
print(len(t.shape))

# 该张量的秩为2，轴数为2，形状为3 * 4
# 第一个轴的长度为3，第二个轴长度为4

# 就重塑而言，我们会关注张量中元素的数量
# 进行重塑操作时候需要考虑的元素数量的原因是，重塑操作改变张量的形状，
# 但是不改变张量中元素的数量或值。
# 因此，重塑操作的目的是为了适应不同的计算需求，而不是改变张量中的元素数量。

# 获取元素数量
print(torch.tensor(t.shape).prod())
print(torch.tensor([3, 4]).prod())
print(torch.tensor([    
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]]).prod())
# 我最开始在这里犯了一个错误
# 没有准确理解为什么需要将形状转换为一个张量后再使用.prod()才能得到正确的元素数量
# 是因为.prod()返回张量中所有元素的乘积

# 或使用
print(t.numel()) 

# 下面是一些符合规范的重塑操作
# 也就是12个位置的变换
print(t.reshape(4, 3))
print(t.reshape(6, 2))
print(t.reshape(12, 1))
# 以上三种变换是没有改变张量维度的，都是2维张量

# 我们也可以通过reshape操作改变张量的维度
print((t.reshape(2, 2, 3)).dim())


# 使用squeeze操作可以去掉张量中维度为1的轴
# 去掉维度为1的轴可以简化张量的形状，使其更易于处理
# 但是注意squeeze ()函数只能删除维度为1的轴，若不为1，该操作无效，但不会报错
tt = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(tt.shape)

# 删除所有维度为1的轴
print(tt.reshape(1, 9, 1).squeeze().shape)

# 指定删除维度为1的轴
print(tt.reshape(1, 9, 1).squeeze(dim=0).shape) 
print(tt.reshape(1, 9, 1).squeeze(dim=2).shape)

# 使用unsqueeze的操作可以增加张量的维度
# unsqueeze()函数可以在指定位置插入一个维度为1的轴
# 这个操作可以用于在神经网络中添加新的维度，以便更好地处理数据
tt = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(tt.shape)
# unsqueeze()函数必须传入一个参数，即dim，用于指定在哪个位置插入一个维度为1的轴。
# unsqueeze() missing 1 required positional arguments: "dim"
print(tt.reshape(1, 9, 1).squeeze(dim=0).unsqueeze(dim=0).shape)


# 使用flatten函数可以将张量展平为一维张量
# 这个操作可以用于将多维张量转换为一维张量，以便更好地处理数据
tt = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(tt.shape)

# 将张量展平为一维张量
print(tt.flatten()) # torch.Size([9])


# 在卷积层之后，我们需要将张量展平为一维张量，以便输入到全连接层中进行处理
# 这个操作就是flatten操作，是从卷积层过渡到全连接层过程中必须发生的操作
# 在PyTorch中，我们可以使用flatten()函数来实现这个操作
# 例如，如果我们有一个形状为[batch_size, channels, height, width]的张量，
# 我们可以使用flatten()函数将其展平为形状为[batch_size, channels * height * width]的张量
# 因此，flatten()函数可以用于将多维张量转换为一维张量，以便更好地处理数据
# 全连接层是神经网络中的一种基本层，也称为密集层或线性层。
# 在全连接层中，每个神经元都与前一层中的所有神经元相连。
# 因此，全连接层可以用于学习输入数据中的任何模式或特征。
# 在PyTorch中，我们可以使用nn.Linear()函数来创建全连接层。

# 不管是什么造型的神经网络，只要它最后一层是全连接层，那这个全连接层必然只会承担以下两个任务当中的一个：
# 分类
# 回归


# 从头开始实现flatten
def flatten(t):
    # 获取张量的元素数量
    t = t.reshape(1, -1) # -1指的是自动计算该维度的大小，以适应张量中的元素数量
    # 删除维度为1的轴
    t = t.squeeze()
    return t


# 在PyTorch中，我们可以使用cat()函数来实现张量的拼接操作
# cat()函数可以将多个张量沿着指定的维度拼接在一起

t1 = torch.tensor([
    
    [1, 2],
    [3, 4]
])

t2 = torch.tensor([
    [5, 6],
    [7, 8]
])

# 拼接这两个张量
print(torch.cat((t1, t2), dim=0))
print(torch.cat((t1, t2), dim=1))

# 以下是错误的写法
# print(torch.cat(t1, t2), dim=0)
# print(torch.cat(t1, t2), dim=1)
# 原因如下:
#  * (tuple of Tensors tensors, int dim, *, Tensor out)

# 就两个张量的形状而言，计算它们的方式会影响输出的张量结果的形状

# 每当我们改变张量的形状时，我们就可以认为我们是在做重塑操作


# 重塑张量的意义在于适应不同的计算需求，而不是改变张量中的元素数量。
# 重塑操作可以改变张量的形状，但不改变张量中元素的数量或值。
# 重塑操作可以用于将多维张量转换为一维张量，以便更好地处理数据，也可以用于改变张量的维度。