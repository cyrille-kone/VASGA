import torch
from tqdm import tqdm
import numpy as np


def sgd_sampler(X, y, loss_fun, lr=1, precond=None, max_iter=1000, mini_batch_size=100):
    # Size of the data
    N = len(y)

    # Dimension of the parameter
    size = X.shape[1] + 2

    # Set the preconditioning matrix to identity if no preconditioning
    if precond is None: precond = torch.eye(size, dtype=torch.float64)

    # Keep trace of the iterates
    theta_sg = torch.zeros(size=(max_iter, size), dtype=torch.float64)

    # Keep trace of the loss
    val_loss = []

    # Initialization of the parameter
    θ = torch.rand(size, dtype=torch.float64, requires_grad=True)

    for k in tqdm(range(max_iter)):

        # Draw a random batch
        S = np.random.randint(N, size=mini_batch_size, dtype=int)

        # Compute the approximated loss on the batch
        loss = torch.zeros(1, dtype=torch.float64)
        for index in S:
            loss += loss_fun(θ, X[index], y[index], N)

        # Set gradients to 0 to avoid accumulation
        θ.grad = None

        # Update and computation of the gradients
        loss.backward()
        θ.data -= lr * (precond @ θ.grad)

        # Add the new iterate
        theta_sg[k] = θ.data.clone()

        # Add the loss value to the historic
        if k % 5 == 0:
            val_loss.append(loss.item() / mini_batch_size)
    return theta_sg, val_loss
