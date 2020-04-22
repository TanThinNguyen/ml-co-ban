import numpy as np
from time import time   # for comparing running time

d, N = 1000, 10000  # dimension, number of training points
X = np.random.randn(N, d)   # N d-dimensional points
z = np.random.randn(d)

# naively compute square distance between two vector; return square of Euclid distance between z, x
def dis_pp(z, x):
    d = z - x.reshape(z.shape)  # force x and z to have the same dims
    return np.sum(d*d)

# from one point to each point in a set, naive;
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dis_pp(z, X[i])
    return res

# from one point to each point in a set, fast;
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, axis=1)    # square of l2 norm of each X[i], can be precomputed
    z2 = np.sum(z*z)    # square of l2 norm of z
    return X2 + z2 - 2*X.dot(z)     # z2 can be ignored


t1 = time()
D1 = dist_ps_naive(z, X)
print("naive point2set, running time: ", time() - t1, "s")

t2 = time()
D2 = dist_ps_fast(z, X)
print("fast point2set, running time: ", time() - t2, "s")
print("Result difference: ", np.linalg.norm(D1 - D2))


Z = np.random.randn(100, d)
# from each point in one set to each point in another set, half fast
def dist_ss_0(Z, X):
    M, N = Z.shape[0], X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res

# from each point in one set to each point in another set, fast
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, axis=1)    # square of l2 norm each row of X
    Z2 = np.sum(Z*Z, axis=1)    # square of l2 norm each row of Z
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)
