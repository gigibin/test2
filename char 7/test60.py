import torch
from torch import nn
from d2l import torch as d2l

# conv2d = nn.Conv2d(1, 1, kernel_size=(1,3), bias=False)
# X = torch.ones((6, 8))
# X[:, 2:6] = 0
# X = X.reshape

X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4,4)
pool2d = nn.MaxPool2d(3,padding=1,stride=1)
print(pool2d(X))









