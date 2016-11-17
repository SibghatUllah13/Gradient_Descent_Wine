import numpy as np


def gendata(w, n):
    """Generate a noisy dataset"""
    d = len(w)
    x = np.ones((n,d))
    x[:,1:] = 10 * np.random.rand(n, d-1)
    y = innprod(x, w) + np.random.randn(n)
    return (x, y)


def innprod(x, y):
    """Compute the inner product <x,y>."""
    if x.shape[-1] == y.shape[0]:
        return (x * y).sum(axis=-1)
    else:
        return np.nan



def grad(b, a, x):
    """Compute the gradient of (y - <c,z>)^2, with respect to z,
    evaluated at z=x."""
    m = len(b)  # number of data points
    d = x.shape[0]  # number of dimensions
    gr = np.zeros(d, dtype=float) # gradient vector
    for j in range(d):  # compute the j-th component of the gradient
        for k in range(m):  # sum over all m data points
            gr[j] += (innprod(x, a[k, :]) - b[k]) * a[k, j]
    return gr


def normalize(A):
    """Normalized a vector so to have mean 0 and variance 1."""
    return (A - A.mean(axis=0))/A.std(axis=0)


def descent(y, x, alpha=1e-3, itr=100, eps=1e-6):
    """Perform a linear regression through gradient descent."""
    d = x.shape[1] # number of dimensions
    theta = np.zeros(d) # start from an arbitrary point -- e.g. [0,0]
    oldTheta = np.ones(d)
    i = 0 # iteration counter
    var = np.infty
    while var >= eps and i < itr:
        g = grad(y, x, theta)
        theta -= alpha * g
        i += 1
        var = np.sum(np.abs(theta - oldTheta))/np.sum(np.abs(oldTheta))
        oldTheta = theta.copy()  # beware of numpy's views!
    return theta


def sqerr(y, c, x):
    """Compute the squared error (y - <c,x>)^2"""
    return ((y - innprod(x, c))**2).sum()


def r2(y, c, x):
    """Compute the coefficient of determination of a linear regression model."""
    sse = sqerr(y, c, x)
    sst = sqerr(y, np.array([y.mean()]), np.ones((len(y),1)))
    return 1.0 - sse/sst
