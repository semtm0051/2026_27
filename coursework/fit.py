"""fit.py:  a small neural network for one-dimensional regression.

This file gives you everything you need to train a network on
data.csv: loading and splitting the data, the network, and a training
function that records the training loss and, optionally, the
validation loss after every epoch. For the basic coursework should not
need to change anything here, but you must read it: the exercises ask
you to call these functions with different arguments and to explain
what they do. For the bonus questions you will have to change some arguments!

    import fit
    x, y = fit.load("data.csv")
    (xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=60, n_val=30)
    net, history = fit.train(xtr, ytr, hidden=20, epochs=2000, xval=xva, yval=yva)
    yhat = fit.predict(net, xgrid)

"""

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# data
def load(path="data.csv"):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


############################################################
# seed set to make the runs reproduciable!

def split(x, y, n_train, n_val, seed=0):

    """Shuffle the data and cut it into train / validation / test pieces.

    Returns ((xtr, ytr), (xva, yva), (xte, yte)); the test piece is whatever
    is left over. Set n_val=0 if you do not want a validation set.
    """
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return (x[tr], y[tr]), (x[va], y[va]), (x[te], y[te])


#numpy array of shape (n,) -> torch tensor of shape (n, 1)
def _t(a):
    
    return torch.as_tensor(np.asarray(a)).reshape(-1, 1)


############################################################
# model

def make_net(hidden):
    """One hidden layer of `hidden` sigmoid units; input and output are scalars.

    hidden=0 gives a network with no hidden layer at all, i.e. a straight line.
    """
    if hidden == 0:
        return nn.Linear(1, 1)
    return nn.Sequential(nn.Linear(1, hidden), nn.Sigmoid(), nn.Linear(hidden, 1))


def n_params(net):
    return sum(p.numel() for p in net.parameters())


###############################################################
# training

# what does no_grad() do?

def mse(net, x, y):
    """Mean squared error of the network on (x, y), as a plain float."""
    with torch.no_grad():
        return ((net(_t(x)) - _t(y)) ** 2).mean().item()


def train(x, y, hidden, epochs=2000, lr=0.01, weight_decay=0.0,
          xval=None, yval=None, optimiser="adam", seed=0):
    """Train a network on (x, y) with full-batch Adam.

    Returns (net, history), where history is a dict with keys 'train' and,
    if a validation set was given, 'val'; each is a list with the mean
    squared error after every epoch.
    """
    torch.manual_seed(seed)
    net = make_net(hidden)


    if optimiser == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimiser == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimiser must be 'adam' or 'sgd'")

    X, Y = _t(x), _t(y)
    history = {"train": []}
    if xval is not None:
        history["val"] = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        history["train"].append(loss.item())
        if xval is not None:
            history["val"].append(mse(net, xval, yval))
    return net, history


# Network output on a numpy array x, returned as a numpy array.

def predict(net, x):

    with torch.no_grad():
        return net(_t(x)).numpy().ravel()
