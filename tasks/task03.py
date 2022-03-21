# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import tqdm
import numpy as np
import torch as th
from pathlib import Path
from utils import ToyDataset
import torch.utils.data as thd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.decomposition import PCA
from mnist.setup import loss_, validate
from ou_process import ornstein_uhlenbeck_nd
from torch.autograd.functional import hessian

FIG_DIR = Path("../data/fig")
FIG_DIR.mkdir(exist_ok=True)
# we choose two informative features to perform PCA latter on
train_ds, test_ds = ToyDataset(n_samples=50000, n_features=50, n_informative=2).random_split()  # 80% 20%

print("-" * 20, "ToyDataset", "-" * 20)
print("-" * 20, "Loading data", "-" * 20)
# batch size
S = 1000
# device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# should mimic SGD process
trainloader = thd.DataLoader(train_ds, batch_size=S, shuffle=True)
testloader = thd.DataLoader(test_ds, batch_size=10000, shuffle=True)

X = th.from_numpy(train_ds.dataset.data).to(device)
Y = th.from_numpy(train_ds.dataset.targets).to(device)
# setup constants
DIM = 50
K = 2
lambd = 1e-3
D = DIM * K + (1 if lambd is None else 0)  # whether is given or optimized
total_steps = len(trainloader)
N = len(train_ds)

print("-" * 20, "Initializing", "-" * 20)
# initialize the model
w = th.zeros((D, 1), device=device)
w = th.nn.init.kaiming_uniform_(w, a=np.sqrt(5))
param = Variable(w.view(-1), requires_grad=True)
criterion = loss_(D, K, lambd)
# compute the true gradient
print("-" * 20, "Computing the true gradient", "-" * 20)
param_true = Variable(w.view(-1), requires_grad=True)
true_loss = criterion(X, Y, param_true)
true_loss.backward()
true_gradient = param_true.grad

# Estimation of the noise matrix for a given learning rate
# twin param to estimate  the gradient noise matrix
param_twin = Variable(w.view(-1), requires_grad=True)
print("-" * 20, "Estimation of the Gradient noise matrix C", "-" * 20)
param_twin_history = param_twin.data.unsqueeze(0)
opt_twin = th.optim.SGD([param_twin], lr=1e-3)
for i, (data, target) in enumerate(trainloader):
    # putting data on device
    data = data.to(device)
    target = target.to(device).view(-1)
    loss = criterion(data, target, param_twin)
    # back prop
    opt_twin.zero_grad()
    loss.backward()
    opt_twin.step()
    # saving params
    param_twin_history = th.cat((param_twin_history, param_twin.data.view(1, -1)))

# Estimating C
# shift the mean since PyTorch will use it for the computation of the covariance matrix
param_twin_history += - param_twin_history.mean(0) + true_gradient
C = th.cov(param_twin_history.T) + 1e-7
# Estimating parameter B
print("-" * 20, "Computing SVD to estimate B", "-" * 20)
# since C is psd and symmetric we should have
# C = P D^(1/2) D^(1/2) P ^T
V, P = th.linalg.eig(C)
B = P @ th.sqrt(th.diag(V))

# B = th.diag(th.randn((D,))) / 4
# gradient noise matrix
# C = B @ B.T + 1e-7 * th.eye(D)  # B @ B.T
# optimal learning rate
eps = 2 * D * S / N / th.trace(C)  # eq17
# optimal pre-conditioning matrix
H = 2 * S * th.inverse(C) / eps / N  # eq19
# normalize
# spectral normalization (to avoid exploding values)
H /= th.linalg.eigvals(H).abs().max()
print("-" * 20, "Computing optimal epsilon", "-" * 20)
print("eps = %s" % eps.item())

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
# We remove the Hook
pre_cond_hook.remove()

print("------Plot training figure------")
# plot results
plt.style.use("seaborn-ticks")
plt.semilogy(val_losses, label="val")
plt.semilogy(losses, label="train")
plt.title(r"Training metrics")
plt.legend()
plt.show()

print("------Simulating OU Process------")
n = param_history.size(0)  # int(params.size(0)*0.1)
t = np.linspace(1e-5, 1, n)

#  Parameters for the OU process
param_mean = param_history[-100:].mean(0)
X_0 = param_history[0].detach().cpu().numpy()

# Computing the Hessian for the whole dataset at the mean SGD point
A = hessian(lambda p: criterion(X, Y, p), param_mean)

# Simulate an OU process with same parameters as SGD
Xt = ornstein_uhlenbeck_nd(t,
                           X_0=X_0,
                           H=H.cpu().numpy(),
                           A=A.cpu().numpy(),
                           B=B.cpu().numpy(),
                           eps=eps.cpu().numpy(),
                           S=S)
print("-"*20, "Computing PCA", "-"*20)
# Perform PCA for visualization
pca = PCA(2)
Xt_pca, params_pca = np.split(pca.fit_transform(np.concatenate((Xt, param_history))), 2)
print("-"*20, "Plot results", "-"*20)
plt.plot(*Xt_pca[:, :2].T, "o", label="OU", alpha=0.8)
plt.plot(*params_pca[:, :2].T, "+", label="SGD")
plt.legend(["OU", "SGD"], title="PCA Realization of", frameon=True)
plt.show()
# End of task
