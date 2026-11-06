import os
import pandas as pd
import torch
from matplotlib_inline import backend_inline
from d2l import torch as d2l  
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

# data = pd.read_csv(data_file)

# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean(numeric_only = True))
# inputs = pd.get_dummies(inputs, dummy_na=True, dtype='float')
# print(inputs)

# x = torch.tensor(inputs.to_numpy(dtype=float))
# y = torch.tensor(outputs.to_numpy(dtype=float))

# print(x,y)
 
# y = torch.eye(4,4,dtype = float)
# x = torch.ones(4,5,dtype = float)
# z = torch.mm(y,x)
# print(z)

# def f(x):
#     return 3*x**2-4*x

# def numerical_lim(f,x,h):
#     return (f(x + h) - f(x)) / h

# h = 0.1
# for i in range(5):
#     print(f'h = {h:.5f},numerical limit = {numerical_lim(f,1,h):.5f}')
#     h *= 0.1

# x = torch.arange(4.0,requires_grad = True)
# y = 2*torch.dot(x,x)

# y.backward()
# # print(x.grad)

# import torch
# import matplotlib.pyplot as plt
# import math

# # 生成数据
# x = torch.linspace(-2 * math.pi, 2 * math.pi, 200, requires_grad=True)
# y = torch.sin(x)

# # 用自动求导计算 dy/dx
# # 因为 y 是向量，需要先 sum 成标量再 backward，或者用 autograd.grad
# y.sum().backward()
# grad = x.grad   # 这就是自动求导得到的 df/dx

# # 转换为 numpy 用于绘图
# x_np = x.detach().numpy()
# y_np = y.detach().numpy()
# grad_np = grad.detach().numpy()

# # 绘图
# plt.figure(figsize=(10, 5))
# plt.plot(x_np, y_np, label='f(x) = sin(x)')
# plt.plot(x_np, grad_np, label="df/dx (autograd)", linestyle='--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('f(x) = sin(x) 与自动求导得到的导数')
# plt.legend()
# plt.grid(True)
# plt.show()

# from torch.distributions import multinomial
# import matplotlib.pyplot as plt
# fait_probs = torch.ones(6) / 6 
# counts = multinomial.Multinomial(1000000,fait_probs).sample((500,))
# sum_counts = counts.cumsum(dim = 0)
# estimates = sum_counts / sum_counts.sum(dim = 1,keepdim = True)
# estimates_np = estimates.numpy()
# x = range(1, 501)

# plt.figure(figsize=(10, 6))
# for i in range(6):
#     plt.plot(x, estimates_np[:, i], label=f'Face {i+1}', linewidth=1.2)

# plt.axhline(y=1/6, color='black', linestyle='--', linewidth=1.5,
#             label='True probability = 1/6')

# plt.xlabel('Number of experiment groups (each group = 1,000,000 rolls)')
# plt.ylabel('Estimated probability')
# plt.title('Convergence of estimated probabilities')
# plt.legend(loc='upper right')
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


