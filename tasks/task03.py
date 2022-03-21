# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
from utils import ToyDataset

import torch
import tqdm
import numpy as np
import torch as th
from pathlib import Path
import torch.utils.data as thd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.decomposition import PCA
from covertype import ForestCoverType
from mnist.setup import loss_, validate
from ou_process import ornstein_uhlenbeck_nd
from torch.autograd.functional import hessian

FIG_DIR = Path("../data/fig")
FIG_DIR.mkdir(exist_ok=True)
train_ds, test_ds = ToyDataset(n_samples=50000, n_features=10, n_informative=2).random_split()  # 80% 20%
print("------ToyDataset------")
print("------Loading data------")
S = 10
trainloader = thd.DataLoader(train_ds, batch_size=S, shuffle=True)
testloader = thd.DataLoader(test_ds, batch_size=10000, shuffle=True)

# setup constants
DIM = 10
K = 2
lambd = 1e-4
D = DIM * K + (1 if lambd is None else 0)  # no lambda
total_steps = len(trainloader)
N = len(train_ds)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

print("------Initializing------")
# initialize the model
w = th.zeros((D, 1), device=device)
w = th.nn.init.normal_(w)
param = Variable(w.view(-1), requires_grad=True)

# define training parameters
B = th.diag(th.randn((D,))) / 4
C = B @ B.T + 1e-7 * th.eye(D)  # B @ B.T
eps = 2 * D * S / N / th.trace(C)  # eq17
H = 2 * S * th.inverse(C) / eps / N  # eq19
# normalize
H /= th.linalg.eigvals(H).abs().max()

# H = th.diag(H).to(device)
print("epsilon = %s" % eps)

# define utils
losses = []
val_losses = []
epochs = 10
log_freq = 2
criterion = loss_(D, K, lambd)
print("------Training the model------")
params = param.data.unsqueeze(0)
opt = th.optim.SGD([param], lr=eps)
for e in range(epochs):
    # compute loss
    for i, (data, target) in enumerate(trainloader):
        # putting data on device
        data = data.to(device).view(-1, DIM)  # - mean
        # data = data / scale
        target = target.to(device).view(-1)
        # making mini batch
        # idx = np.random.choice(data.size(0), S)  # TODO cas mbsz >= data.size(0)
        # theta.grad.data.zero_()
        # loglambd.grad.data.zero_()
        loss = criterion(data, target, param)
        # back prop
        opt.zero_grad()
        loss.backward()
        # print(param.grad)
        # print(theta.grad[0])
        param.data -= eps * H @ param.grad.data
        # loglambd = loglambd.detach() - eps*loglambd.grad.data
        # loglambd.requires_grad = True
        # print(theta)
        # opt.step()
        params = th.cat((params, param.data.view(1, -1)))
        if i % (total_steps // log_freq) == 0:
            val_loss = validate(param, testloader, criterion, DIM)
            print(
                f"Epoch:{e + 1:3d}/{epochs} step {i + 1:3d} / {total_steps} loss: # {loss.item():9.2f} validation "
                f"loss : #{val_loss:9.2f}")
            losses.append(loss.item())
            val_losses.append(val_loss)

print("------Plot figure------")
# plot results
# TODO
plt.style.use("seaborn-ticks")
plt.semilogy(val_losses, label="val")
plt.semilogy(losses, label="train")
plt.legend()
plt.show()

print("------Simulating OU Process------")
n = params.size(0)  # int(params.size(0)*0.1)
t = np.linspace(1e-5, 1, n)
# OU
param_mean = params.mean(0)
X_0 = params[0].detach().cpu().numpy()
X = th.from_numpy(train_ds.dataset.data).to(device)
Y = th.from_numpy(train_ds.dataset.targets).to(device)
A = hessian(lambda p: criterion(X, Y, p), param_mean)
Xt = ornstein_uhlenbeck_nd(t,
                           X_0=X_0,
                           H=H.cpu().numpy(),
                           A=A.cpu().numpy(),
                           B=B.cpu().numpy(),
                           eps=eps.cpu().numpy(),
                           S=S)
print("------Computing PCA------")
pca = PCA(2)
Xt_pca, params_pca = np.split(pca.fit_transform(np.concatenate((Xt, params))), 2)
print("------Plot results------")
plt.scatter(*Xt_pca[:, :2].T, label="OU", alpha=0.8, c=t, s=3, cmap="Blues")
plt.scatter(*params_pca[:, :2].T, label="SGD", c=t, s=2, cmap="Greens")
plt.legend()
plt.show()
