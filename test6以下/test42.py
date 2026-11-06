import torch
from torch import nn
from d2l import torch as d2l
from test6以下.test36 import train_ch3
from test6以下.test36 import predict_ch3

# batch_size = 256

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# num_inputs, num_outputs, num_hiddens = 784, 10, 1024

# W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))


# params = [W1, b1, W2, b2]

# #手动实现relu代码
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X,a)


# #定义模型s
# def net(X):
#     X = X.reshape((-1,num_inputs))
#     H = relu(X@W1 + b1)
#     return (H@W2 + b2)

# #损失函数
# loss = nn.CrossEntropyLoss(reduction='none')

# num_epochs, lr = 10, 0.1

# updater = torch.optim.SGD(params, lr=lr)

# train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# predict_ch3(net, test_iter)

#简洁实现多层感知机
net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,256),
        nn.ReLU(),
        nn.Linear(256,10)
        )

#给输入矩阵赋值权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)


batch_size, lr ,num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


