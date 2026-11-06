import random
import torch
from d2l import torch as d2l

def syn(w,b,num):
    x = torch.normal(0,1,(num,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = syn(true_w,true_b,1000)
print('features:', features[0],'\nlabel:', labels[0])


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(
#     features[:, 0].detach().numpy(),
#     features[:, 1].detach().numpy(),
#     labels.detach().numpy(),
#     s=1
# )
# ax.set_xlabel('feature 0')
# ax.set_ylabel('feature 1')
# ax.set_zlabel('label')
# plt.show()


#数据划分
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]


batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break


#初始化模型参数
w = torch.normal(0,0.01,size = (2,1),requires_grad = True)
b = torch.zeros(1, requires_grad=True)

def linreg(X,w,b):
    return torch.matmul(X,w) + b

def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features ,labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()

        sgd([w, b],lr,batch_size)

    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')



















