# 探索Pytorch张量深入研究Pytorch

# 数据的预处理(是必须的部分)
# 其最终目标是将正在处理的数据转换为能用于神经网络的张量

# 数据预处理是卷积神经网络中非常重要的一步，它可以帮助我们减少噪声、增强特征、提高模型的泛化能力等。
# 数据预处理的具体步骤包括数据清洗、数据增强、数据归一化等，可以根据具体任务的需求进行选择和组合。
# 数据清洗可以帮助我们去除异常值、缺失值等不合理的数据，从而提高模型的鲁棒性和准确性。
# 数据增强可以通过对原始数据进行旋转、翻转、裁剪等操作，生成更多的训练样本，从而提高模型的泛化能力。
# 数据归一化可以将数据缩放到相同的范围内，避免不同特征之间的差异对模型的影响，从而提高模型的训练效果。

# Class torch.Tensor类的介绍
import torch
import numpy as np

t = torch.Tensor() # 注意是大写T
print(type(t)) 
# <class 'torch.Tensor'>
# torch.Tensor是torch包中的一个类，可以用来创建张量
# torch.tensor是一个函数，可以将Python列表或NumPy数组转换为张量
# torch.Tensor和torch.tensor都可以创建张量，但是它们的行为略有不同
# torch.Tensor会尝试根据输入数据的类型和形状创建一个张量
# 而torch.tensor会根据输入数据的类型和形状创建一个新的张量
# 因此，torch.Tensor可能会返回一个空张量，而torch.tensor不会
# 另外，torch.Tensor默认使用float32数据类型，而torch.tensor会根据输入数据的类型自动选择数据类型


# 之前了解到的秩,轴,形状适用于所有的张量

# 介绍三个Pytorch张量的相关属性
print(t.dtype) # 表示数据类型
print(t.device)# 表示张量在什么设备上进行计算
print(t.layout)# 表示张量的存储顺序
# torch.strided表示普通的张量存储方式，即按照一定的步幅在内存中存储元素
# torch.sparse_coo表示稀疏张量的COO（Coordinate）存储方式，即按照坐标的方式存储非零元素的位置和值
# torch.sparse_csr表示稀疏张量的CSR（Compressed Sparse Row）存储方式，即按照行的方式存储非零元素的位置和值

# 设备决定了张量计算的位置
# 在pytorch中可以使用torch.device创建设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 上述代码将创建一个名为device的torch.device对象，如果当前环境支持CUDA，则将其指定为第一个GPU，否则指定为CPU
print(device)

# 张量包含统一类型的数据

# 首先测试张量在同一设备上进行计算
# 在之前的pytorch版本中,张量的计算应该发生在同一类型的数据
# 目前的pytorch版本若张量运算发生在不同数据类型,会自动将低精度提升至高精度

# PyTorch is able to perform the addition operation between tensors of different data types 
# by automatically promoting the lower precision tensor to the higher precision tensor. 

t1 = torch.tensor([1, 2, 3])
print(t1.dtype)

t2 = torch.tensor([1., 2., 3.])
print(t2.dtype)

try:
    result = t1 + t2
    print(result)
    print(result.dtype)
    
except Exception as e:
    print("An error occurred while printing:", e)

# 现在测试在不同设备进行的张量运算
t1 = torch.tensor([1, 2, 3])
print(t1.dtype)

# 将张量t1移动到GPU设备上
t2 = t1.cuda()

print(t1.device)
print(t2.device)

try:
    result = t1 + t2
    print(result)
    print(result.dtype)
    
except Exception as e:
    print("An error occurred while printing:", e)
# 在不同设备间不可以进行张量运算

"""
    在Pytorch中创建张量对象
"""

# 有初始数据的情况下创建,一般讲有4种方法

data = np.array([1, 2, 3]) 
print(type(data))

# 都接受numpy数组这样的数据结构
# 1
print(torch.Tensor(data))
# 2
print(torch.tensor(data))
# 3
print(torch.as_tensor(data))
# 4
print(torch.from_numpy(data))


# 没有初始数据的情况下创建
# 单位矩阵张量
print(torch.eye(2))
# The torch.eye() function takes an integer argument n and returns an n x n identity matrix. 
# The diagonal elements of the matrix are 1, and all other elements are 0.

# 全0张量
print(torch.zeros(2,2))
# 一般会用于初始化模型参数。在深度学习中，模型参数的初始化非常重要，可以影响模型的训练效果和收敛速度。
# 全0矩阵张量可以作为一种简单的初始化方式,但是在某些情况下可能会导致模型无法收敛。

# 全1张量
print(torch.ones(2,2))
# 一般会用于初始化模型参数。在深度学习中，模型参数的初始化非常重要，可以影响模型的训练效果和收敛速度。
# 全1矩阵张量可以作为一种简单的初始化方式，但是在某些情况下可能会导致梯度消失或爆炸的问题。

# 具有随机值的张量
print(torch.rand(2,2))
# the values in the tensor created using torch.rand() have a range of [0, 1). 
# 一般会用于模型参数的初始化，可以作为一种简单的初始化方式。