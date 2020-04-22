import numpy as np
from scipy.spatial.distance import cdist

#
# X = np.array([[-2, -2], [-1, 4], [2, 3]])
# cov = np.cov(X.T)
# print(cov)

# cov = [[1, 0], [0, 1]]
# N = 500
# X0 = np.random.multivariate_normal([2, 2], cov, N)
# print(X0)
# s = np.sum(X0[:, 1])
# a = s/500
# print(a)
# print(np.sum((X0[:, 1] - a)**2)/499)

# X = np.array([[1, 2],
#              [1, 1],
#              [2, 2],
#              [4, 5],
#              [2, 6]])
# idx = np.random.choice(5, 3, replace=False)
# print(idx)
# print([X[idx]])

# X = np.array([[1, 2, 1],
#              [3, 5, 1]])
# Y = np.array([[1, 3, 1],
#              [2, 4, 2],
#              [1, 2, 2]])
#
# res = cdist(X, Y)
# print(res)
# print(np.min(res, axis=1))
# print(np.argmin(res, axis=1))

# X = np.array([[1, 2],
#               [1, 1],
#               [3, 4],
#               [5, 6],
#               [1, 5]])
# labels = np.array([0, 1, 1, 0, 0])
# Xk = X[labels == 0, :]
# Xkmean = np.mean(Xk, axis=0)
# print(Xk)
# print(Xkmean)

# A = np.array([[1, 1, 2],
#              [1, 1, 1],
#              [2, 2, 2]])
# x = [a[0] for a in A]
# print(set(x))


