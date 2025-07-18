import numpy as np

def kernel_linear(x, y):
    return np.dot(x, y.T)

def kernel_poly(x, y, degree=3):
    return np.dot(x, y.T) ** degree

def kernel_rbf(x, y, gamma=1.0):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    dists = np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * dists) 