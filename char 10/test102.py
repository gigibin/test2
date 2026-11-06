import torch
from torch import nn
import matplotlib.pyplot as plt

# ================= 1. 补全模型定义 =================
class NWKernelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习的参数 w，初始化为随机值
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # 将 queries 复制，以便和所有的 keys 做交叉距离计算
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 计算注意力权重
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1
        )
        # 加权求和得到输出
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# ================= 2. 自定义画图函数 =================
def plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train):
    plt.figure(figsize=(6, 4))
    
    def to_numpy(data):
        return data.detach().numpy() if isinstance(data, torch.Tensor) else data
    
    plt.plot(to_numpy(x_test), to_numpy(y_truth), label='Truth', linestyle='-')
    plt.plot(to_numpy(x_test), to_numpy(y_hat), 'm--', label='Pred')
    plt.plot(to_numpy(x_train), to_numpy(y_train), 'o', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 5)
    plt.ylim(-1, 5)
    plt.legend()
    plt.show()

# ================= 3. 生成数据 =================
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)

# ================= 4. 模型训练 =================
# 构造训练集的 key 和 value (剔除自己)
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))

# (1 - eye) 生成一个对角线为 0，其他为 1 的布尔掩码，把对角线（自己）剔除
mask = (1 - torch.eye(n_train)).type(torch.bool)
keys = X_tile[mask].reshape((n_train, -1))
values = Y_tile[mask].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

loss_history = []
print("开始训练...")
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    loss_val = float(l.sum())
    loss_history.append(loss_val)
    print(f'epoch {epoch + 1}, loss {loss_val:.6f}')

# ================= 5. 模型预测与可视化 =================
# 构造测试集的 key 和 value
keys_test = x_train.repeat((n_test, 1))
values_test = y_train.repeat((n_test, 1))

# 预测
with torch.no_grad():
    y_hat = net(x_test, keys_test, values_test).unsqueeze(1)

# 调用画图函数
plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)