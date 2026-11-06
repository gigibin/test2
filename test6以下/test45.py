import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l


# ============ 自定义工具函数 ============
def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum().item(), l.numel())
    return metric[0] / metric[1]


class Animator:
    """训练完一次性显示"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

    def show(self):
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.show()


# ============ 生成数据 ============
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)

test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# ============ 初始化参数 ============
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# # ============ 训练 ============
# def train(lambd):
#     w, b = init_params()
#     net = lambda X: linreg(X, w, b)
#     loss = squared_loss

#     num_epochs, lr = 100, 0.003

#     animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                         xlim=[5, num_epochs], legend=['train', 'test'])

#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             l = loss(net(X), y) + lambd * l2_penalty(w)
#             l.sum().backward()
#             sgd([w, b], lr, batch_size)

#         if (epoch + 1) % 5 == 0:
#             animator.add(epoch + 1,
#                          (evaluate_loss(net, train_iter, loss),
#                           evaluate_loss(net, test_iter, loss)))

#     animator.show()   # ← 训练完显示曲线
#     print('w的L2范数是：', torch.norm(w).item())


# train(lambd=5)


#简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    trainer = torch.optim.SGD([
        {"params":net[0].weight, 'weight_decay':wd},
        {"params":net[0].bias}], lr = lr
    )
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()

        if(epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
            (d2l.evaluate_loss(net, train_iter, loss),
            d2l.evaluate_loss(net, test_iter, loss)))
    animator.show()
    print('w的L2范数：', net[0].weight.norm().item())

train_concise(10)