import numpy as np
from sklearn.model_selection import train_test_split

# d = np.random.randn(5)
# d = np.array([1, 2, 3, 4, 5])
# print(d)
# print(d*d)
# print(np.sum(d*d))

# X = np.array([[1, 2, 3],
#               [4, 5, 6]])
# print(X*X)
# X2 = np.sum(X*X, axis=1)
# print(X2)

# a = np.array([1, 2, 3, 4])
# b = np.array([9, 8, 7, 6])
# print(np.linalg.norm(a - b))
# c = a - b
# print(np.sqrt(np.sum(c*c)))

# X = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])
# z = np.array([1, 5])
# print(np.dot(X[0], z))
# print(np.dot(X[1], z))
# print(np.dot(X[2], z))
# print(X.dot(z))

# a = np.array([1, 1, 2])
# b = np.array([2, 2, 1])
# print(a*b)
# print(np.dot(a, b))
# A = np.array([[2, 1, 1],
#               [1, 1, 3]])
# print(np.dot(a, A))
# print(np.dot(A, a))

# A = np.array([[1, 1, 2],
#               [1, 2, 2]])
# B = np.array([[2, 1, 3],
#               [1, 2, 1]])
# print(A/B)
# print(np.dot(A.T, B))

# A = np.array([[0,  1,  2],
#               [3,  4,  5],
#               [6,  7,  8],
#               [9, 10, 11]])
# b = np.array([1, 3, 4])
# c = np.array([1, 3, 4, 5])
# print(np.dot(A, b))
# print(np.dot(c, A))
# print(A*b)

# a = np.array([1, 2, 1])
# A = np.array([[1, 1, 1],
#               [2, 2, 2]])
# print(np.linalg.norm(a))
# print(np.sqrt(np.sum(a*a)))
# print(A.dot(a))
# X = np.sum(A*A, axis=1)
# print(X)
# print(X.reshape(-1, 1))


# data = np.array([[1, 1, 2, 1],
#                  [2, 2, 1, 1],
#                  [1, 0, 1, 0],
#                  [0, 0, 0, 1],
#                  [1, 1, 1, 1],
#                  [0, 1, 0, 1]])
#
# train, test = train_test_split(data, test_size=2)
# print(train)
# print(test)

data = np.array([1, 2, 1, 0, 0, 1, 3, 2])
train, test = train_test_split(data, test_size=5)
print(train)
print(test)

