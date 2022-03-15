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


if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)  # from 0 to 1 with 1000 points
    ou_1d = ornstein_uhlenbeck_1d(t)
    plt.figure(1)
    plt.style.use("seaborn-ticks")
    plt.scatter(t, ou_1d, c=t, cmap="Blues", s=3)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$X_t$")
    plt.title("Ornstein-Uhlenbeck process")
    plt.text(0.5, 0.5, r"$dX_t = -\theta*(X_t - \mu)dt + \sigma*dW_t$")
    plt.show()

