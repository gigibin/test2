import torch
import torch.nn.functional as F
from torch import nn

#不带出参数的层
class CentereLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X):
        return X - X.mean()

layer = CentereLayer()
# print(layer(torch.FloatTensor([1,2,3,4,5])))

# net = nn.Sequential(nn.Linear(8, 128), CentereLayer())
# Y = net(torch.rand(4, 8))
# print(Y)


#带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

# linear = MyLinear(5, 3)
# print(linear(torch.rand(3, 5)))


#练习
import torch
from torch import nn

class Reduction1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, in_features, out_features))
    def forward(self, X):
        y = torch.einsum('bi,bj,ijk->bk',X ,X, self.W)
        return y

layer = Reduction1(5, 3)
X = torch.randn(4, 5)
Y = layer(X)
print(Y)

