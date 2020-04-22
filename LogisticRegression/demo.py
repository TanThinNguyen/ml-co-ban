import numpy as np
import matplotlib.pyplot as plt

# y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
# idx_pass = np.where(y == 1)[0]
# idx_fail = np.where(y == 0)[0]
# print(idx_pass)
# print(idx_fail)


# plt.figure(1)
# plt.plot([0, 1], [2, 2], "r--")
#
# plt.figure(2)
# plt.plot([0, 1], [2, 2], "b")
# plt.show()


# a = np.array([[1, 2, 1, 2, 1, 2],
#               [3, 4, 3, 4, 3, 4]])
# b = [[6, 7, 6, 7, 6, 7],
#      [8, 9, 8, 9, 8, 9]]
# a1 = np.asarray(a)
# c = np.concatenate((a, b), axis=0)
# print(a > 1)
# print(a[a > 1])


A = np.array([[1, 2, 1, 2],
              [3, 4, 3, 4],
              [1, 1, 2, 2],
              [3, 3, 3, 4],
              [5, 5, 5, 6],
              [6, 7, 8, 9]])
y = np.array([2, 1, 4, 1, 5, 1])
B = A[np.where(y == 1)[0]]
C = A[y == 1]
print(B)
print(C)


