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
from sklearn.decomposition import PCA
from covertype import ForestCoverType
from mnist.setup import loss_, validate
from ou_process import ornstein_uhlenbeck_nd
from torch.autograd.functional import hessian

FIG_DIR = Path("../data/fig/task2")
FIG_DIR.mkdir(exist_ok=True)
train_ds, test_ds = ForestCoverType().random_split()  # 80% 20%
print("------ForestCoverType------")
print("------Loading data------")
S = 100
trainloader = thd.DataLoader(train_ds, batch_size=S, shuffle=True)
testloader = thd.DataLoader(test_ds, batch_size=10000, shuffle=True)

# setup constants
DIM = 54
K = 7
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
print("-" * 20, "Training the model", "-" * 20)
# saving SGD process
param_history = param.data.unsqueeze(0)
opt = th.optim.SGD([param], lr=eps)
# pre-conditioning
pre_cond_hook = param.register_hook(lambda grad: H @ grad)
for e in range(epochs):
    # compute loss
    for i, (data, target) in enumerate(trainloader):
        # putting data on device
        data = data.to(device)
        target = target.to(device).view(-1)
        loss = criterion(data, target, param)
        # back prop
        opt.zero_grad()
        loss.backward()
        opt.step()
        # saving params
        param_history = th.cat((param_history, param.data.view(1, -1)))
        if i % (total_steps // log_freq) == 0:
            val_loss = validate(param, testloader, criterion, DIM)
            print(
                f"Epoch: {e + 1:3d}/{epochs} step {i + 1:3d} / {total_steps} loss: # {loss.item():9.2f} validation "
                f"loss : #{val_loss:9.2f}")
            losses.append(loss.item())
            val_losses.append(val_loss)
        if lambd is None:
            # clamp value of lambda if it is being optimized
            with th.no_grad():
                param[-1].clamp_(1e-6, 1)
                # print(param[-1], param.grad,)

# We remove the Hook
pre_cond_hook.remove()
print("------Plot figure------")
# plot results
plt.style.use("seaborn-ticks")
plt.semilogy(val_losses, label="val")
plt.semilogy(losses, label="train")
plt.legend()
plt.show()

print("------Simulating OU Process------")
n = param_history.size(0)
t = np.linspace(0, 1, n)
# OU
param_mean = param_history.mean(0)
X_0 = param_history[0].detach().cpu().numpy()
X = th.from_numpy(train_ds.dataset.data)[:20000].to(device)
Y = th.from_numpy(train_ds.dataset.targets)[:20000].to(device)
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
# 10 percent
params_pca = pca.fit_transform(param_history.detach().cpu().numpy())
Xt_pca = pca.transform(Xt)
print("------Plot results------")
plt.plot(*Xt_pca.T, label="OU", alpha=0.5)
plt.plot(*params_pca.T, label="SGD", alpha=0.7)
plt.legend(frameon=True, title="PCA realization of")
plt.savefig(FIG_DIR/"OUvsSGD_PCA.png")
plt.show()
