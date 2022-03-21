# coding=utf-8
"""
PyCharm Editor
Author @git cyrille-kone & geoffroyO
"""
import tqdm
import numpy as np
import torch as th
from pathlib import Path
import torch.utils.data as thd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mnist.setup import loss_, validate
from torchvision import datasets, transforms

FIG_DIR = Path("../data/fig")
DATA_DIR = Path("../data/.mnist")
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
print("------MNIST------")
print("------Loading data------")
S = 1000
train_ds = datasets.MNIST(DATA_DIR, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST(DATA_DIR, download=True, train=False, transform=transforms.ToTensor())
trainloader = thd.DataLoader(train_ds, batch_size=S, shuffle=True)
testloader = thd.DataLoader(test_ds, batch_size=10000, shuffle=True)

# setup constants
DIM = 784
K = 10
lambd = None
D = DIM * K + (1 if lambd is None else 0)  # no lambda
total_steps = len(trainloader)
N = len(train_ds)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

print("------Initializing------")
# initialize the model
w = th.zeros((D, 1), device=device)
#w = th.nn.init.xavier_uniform_(w)
param = Variable(w.view(-1), requires_grad=True)

# define training parameters
B = 4 * th.diag(th.randn((D,)))
C = B @ B.T + 1e-7 * th.eye(D)  # B @ B.T
eps = 2 * D * S / N / th.trace(C)  # eq17
H = 2 * S * th.inverse(C) / eps / N  # eq19
# normalize
H /= th.linalg.eigvals(H).abs().max()
H = H.to(device)
print("epsilon = %s" % eps)
# define utils
losses = []
val_losses = []
epochs = 10
log_freq = 2
criterion = loss_(D, K, lambd)
opt = th.optim.SGD([param], lr=eps)
print("------Training the model------")
for e in range(epochs):
    # compute loss
    for i, (data, target) in enumerate(trainloader):
        # putting data on device
        data = data.to(device).view(-1, DIM)  # - mean
        # data = data / scale
        target = target.to(device).view(-1)
        # making mini batch
        idx = np.random.choice(data.size(0), S)  # TODO cas mbsz >= data.size(0)
        # theta.grad.data.zero_()
        # loglambd.grad.data.zero_()
        loss = criterion(data, target, param)
        # back prop
        opt.zero_grad()
        loss.backward()
        # print(theta.grad[0])
        # param = Variable(param.detach() - eps * H @ param.grad.data, requires_grad=True)
        # loglambd = loglambd.detach() - eps*loglambd.grad.data
        # loglambd.requires_grad = True
        # print(theta)
        # opt.step()
        param.data -= eps * H @ param.grad.data
        if i % (total_steps // log_freq) == 0:
            val_loss = validate(param, testloader, criterion)
            print(
                f"Epoch:{e + 1:3d}/{epochs} step {i + 1:3d} / {total_steps} loss: {loss.item():9.2f} validation loss : #{val_loss:9.2f}")
            losses.append(loss.item())
            val_losses.append(val_loss)

print("------Plot figure------")
# plot results
plt.style.use("seaborn-ticks")
plt.semilogy(val_losses, label="val")
plt.semilogy(losses, label="train")
plt.legend()
plt.show()
