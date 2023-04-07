# 验证torch是否安装完成


import torch
# This code will print the version of PyTorch and then print the string 

print(torch.__version__)

# 验证cuda
print(torch.cuda.is_available())

# 查看CUDA版本
print(torch.version.cuda)