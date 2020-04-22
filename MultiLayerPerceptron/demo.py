import numpy as np

# A = np.array([[1, 2, 1, 2],
#               [1, 1, 1, 1],
#               [2, 3, 2, 3]])
# B = np.array([[1],
#               [1],
#               [1]])
# C = np.array([[1, 1, 1, 1]])
# print(A + B)
# print(A + C)

# a = np.array([1, 2, 3, 4])
# b = np.array([1])
# print(a + b)

Z = np.array([[1, -2, 1, 3],
              [2, 3, -1, 2],
              [-1, 3, 3, -1]])
print(np.maximum(Z, 0))

