# coding=utf-8
"""
PyCharm Editor
Author @git cyrille-kone & geoffroyO
"""
import torch as th
import numpy as np

DIM = 784


# TODO verifier la loss f
def loss_(D, K, lambd=None):
    r'''Loss function wrapper
    Parameters
    ----------
    D: Dimension of the parameter vector
    K: Number of classes
    lambd: Regularization parameter
    Returns
    -------
    Callable loss '''

    def L(x, y, theta, lambd):
        r'''Loss function eq.27'''
        return (lambd / 2) * th.sum(th.square(theta)) - D * K * th.log(lambd) / 2 + D*K*np.log(2*np.pi)/2 + (1) * (
                th.sum(th.logsumexp(x @ theta, 1)) - th.trace(x @ theta[:, y]))

    if lambd is not None:
        def loss(x, y, param):
            r''' loss function
            Parameters
            --------
            x : input data
            y: target
            param: model's parameters
            Returns
            -------
            loss @Autograd
            '''
            theta = param.view(-1, K)
            return L(x, y, theta, th.tensor(lambd))

    else:
        def loss(x, y, param):
            r''' loss function
           Parameters
           --------
           x : input data
           y: target
           param: model's parameters
           Returns
           -------
           loss @Autograd
           '''
            theta, lambd = th.split(param, [D - 1, 1])
            theta = theta.view(-1, K)
            return L(x, y, theta, th.sigmoid(lambd))
    return loss


@th.no_grad()
def validate(param, testloader, criterion, d=None, device=None):
    r'''
    Parameters
    ----------
    :param testloader: Loader for the test set
    :param param: Parameters of the model
    :param criterion: loss function
    :param d: resize vector dim
    :param device
    Returns
    -------
    Validation loss
    '''
    device = param.device if device is None else device
    param = param.to(device)
    total_loss = 0.
    d = DIM if d is None else d
    for (data, target) in testloader:
        # putting data on device
        data = data.to(device).view(-1, d)
        target = target.to(device).view(-1)
        loss = criterion(data, target, param)
        total_loss += loss.item()
    return total_loss
