import numpy as np

A = np.array([[1, 3],
              [2, 4],
              [5, 6]])
x = np.array([[1],
              [2]])
b = np.array([[2],
              [4],
              [6]])

print(np.dot(x.transpose().dot(A.transpose()), b))
print(np.dot(b.transpose().dot(A), x))
print(np.dot(A, A.transpose()))
