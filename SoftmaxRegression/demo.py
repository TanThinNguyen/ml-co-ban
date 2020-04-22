import numpy as np

# X = np.array([[1, 2, 2],
#               [4, 2, 1]])
# A = np.sum(X, axis=1, keepdims=True)
# B = np.sum(X, axis=1)
# print(A)
# print(B)

# Y = np.array([[6, 3, 1, 5],
#               [4, 4, 3, 2],
#               [1, 5, 1, 4],
#               [2, 3, 2, 1]])
# a = np.array([1, 2, 3, 2])
# b = np.array([2, 3, 2, 0])
# print(Y[a, b])
# print(Y[a, b[a]])

A = np.array([[1, 2, 3],
              [2, 1, 3]])
B = np.array([[1, 2],
              [1, 1],
              [2, 1]])
# print(A.dot(B))
C = A.copy()
A[0, 0] = 0
print(A)
print(C)
print(A.size)

