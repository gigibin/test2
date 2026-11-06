import torch
import torchvision
from torch import nn
import matplotlib
# 关键修改：不要用 TkAgg，用默认后端或者 Agg
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l


# 1. 自定义 Animator (增加一个保存图片的方法)
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(6, 4)):
        if legend is None: legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes, ]
        self.axes[0].set_xlabel(xlabel)
        self.axes[0].set_ylabel(ylabel)
        if xlim: self.axes[0].set_xlim(xlim)
        if ylim: self.axes[0].set_ylim(ylim)
        self.legend_labels = legend
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                
    def save(self, filename="result.png"):
        """把图保存下来"""
        self.axes[0].cla()
        if self.legend_labels: self.axes[0].legend(self.legend_labels)
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ 训练曲线已保存到: {filename}")


# 2. 数据集类
class HuggingFaceToPyTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item['img'] if 'img' in item else item['image'] 
        label = item['label']
        if not isinstance(img, Image.Image): img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        return img, label


# 3. 数据加载函数 (强制 num_workers=0)
def load_cifar10(is_train, augs, batch_size):
    split_name = "train" if is_train else "test"
    hf_dataset = load_dataset("./cifar10_data", split=split_name)
    dataset = HuggingFaceToPyTorch(hf_dataset, transform=augs)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train,
        num_workers=0  # 👈 这里强制设为0，防止多进程段错误
    )
    return dataloader


# 4. 核心训练函数
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list): X = [x.to(devices[0]) for x in X]
    else: X = X.to(devices[0])
    y = y.to(devices[0])
    
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    
    return l.sum(), d2l.accuracy(pred, y)


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    
    if len(devices) > 1: net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else: net = net.to(devices[0])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            # 控制打印频率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        
        print(f'epoch {epoch + 1}, loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    
    print(f'Done! {metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
    animator.save() # 结束后保存图片


# 5. 主程序
if __name__ == '__main__':
    batch_size = 256
    devices = d2l.try_all_gpus()
    
    net = d2l.resnet18(10, 3)
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    
    # 关键修改：全小写 'none'
    loss = nn.CrossEntropyLoss(reduction='none') 
    trainer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)