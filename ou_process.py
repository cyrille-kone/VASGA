# coding=utf-8
"""
PyCharm Editor
Author @git cyrille-kone & geoffroyO
"""
import numpy as np
import matplotlib.pyplot as plt


def ornstein_uhlenbeck_1d(t, x_0=None, theta=1.1, mu=0, sigma=0.1):
    r"""
    Numerical resolution of the 1d Ornstein-Uhlenbeck process
    dX_t = -\theta*(X_t - \mu)dt + \sigma*dW_t
    Parameters
    ----------

    :param t: timesteps
    :param x_0: initial condition
    :param theta
    :param mu
    :param sigma
    Returns
    -------
    A realization of X_t
    """
    dt = np.gradient(t)
    xs = np.zeros_like(t)
    x_0 = np.random.normal() if x_0 is None else x_0
    for idx, _ in enumerate(t):
        x_1 = x_0 + theta * (mu - x_0) * dt[idx] + sigma * np.random.normal() * np.sqrt(dt[idx])
        xs[idx] = x_1
        x_0 = x_1
    return xs


def ornstein_uhlenbeck_nd(t, X_0=None, H=None, A=None, B=None, eps=0.001, S=10):
    r""" Numerical solution of the n-dimensional Ornstein-Uhlenbeck process
    as SGD update.
    dX_t = -\epsilon HA X_t + \sqrt{\frac{\epsilon}{S}}HBdW_t
    Parameters
    ----------
    :param t: timesteps
    :param X_0: initial condition
    :param H: preconditioning matrix
    :param A: loss matrix eq.10
    :param B: gradient noise matrix
    :param eps: learning rate
    :param S: SGD mini batch size

    Returns
    -------
    A realization of X_t
    """
    X_0 = np.random.multivariate_normal(np.zeros(2), np.eye(2)) if X_0 is None else X_0
    assert X_0.size >= 2, "X_0 should be at least 2-dimensional vector but is %d  \
    dimension. Consider using ornstein_uhlenbeck_1d" % X_0.ize
    dim = X_0.size
    dt = np.gradient(t)
    Xs = np.zeros((dt.size, dim))
    H = np.eye(dim) if H is None else H
    A = np.diag(np.random.uniform(size=dim)) if A is None else A
    B = 0.5 * np.eye(dim) if B is None else B
    X_0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim)) if X_0 is None else X_0
    dW = np.random.multivariate_normal(mean=np.zeros(dim), cov=eps * np.eye(dim), size=t.size)
    for idx, _ in enumerate(t):
        X_1 = X_0 - eps * dt[idx] * H @ A @ X_0 + np.sqrt(eps / S) * H @ B @ dW[idx]
        Xs[idx] = X_0
        X_0 = X_1
    return Xs


if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)  # from 0 to 1 with 1000 points
    ou_1d = ornstein_uhlenbeck_1d(t)
    ou_2d = ornstein_uhlenbeck_nd(t)
    plt.style.use("seaborn-ticks")
    plt.figure(1)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$X_t$")
    plt.scatter(t, ou_1d, c=t, cmap="Blues", s=3)
    plt.title("Ornstein-Uhlenbeck process 1d")
    plt.legend(title=r"$dX_t = -\theta(X_t - \mu)dt + \sigma dW_t$")
    plt.show()
    plt.figure(2)
    plt.xlabel(r"$X^0_t$")
    plt.ylabel(r"$X^1_t$")
    plt.scatter(*ou_2d.T, c=t)
    plt.title("Ornstein-Uhlenbeck process 2d")
    plt.legend(title=r"$dX_t = -\epsilon HA X_t + \sqrt{\frac{\epsilon}{S}}HBdW_t$")
    plt.show()
