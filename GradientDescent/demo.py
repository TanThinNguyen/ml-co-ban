import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# x = np.array([1, 2, 3, 4, 5, 6, 7])
# y = np.array([2.1, 4.01, 6.2, 8.12, 10.05, 12.01, 14.1])
#
# a = [1, 2]
# b = [10, 12]
# c = [4, 5]
#
# plt.plot(x, y, "ro")
# plt.plot(a, b, c, "ro")
# plt.show()


# ------------------------------------------
# Demo hồi quy tuyến tính đồng thời 2 hàm số (chung X)
# X = np.random.rand(1000)
# y1 = 4 + 3*X + 0.5*np.random.randn(1000)
# y2 = 5 + 6*X + 0.5*np.random.randn(1000)
# Y1 = y1.reshape(1000, 1)
# Y2 = y2.reshape(1000, 1)
# # print("y1: ", y1)
# # print("y2: ", y2)
# Y = np.concatenate((Y1, Y2), axis=1)
#
# model = LinearRegression()
# model.fit(X.reshape(-1, 1), Y)
# print("coef: ", model.coef_)
# print("intercept: ", model.intercept_)


# ------------------------------------------
# A = np.array([[1, 2, 3],
#               [1, 4, 5]])
# X = np.array([[1, 2],
#               [2, 1],
#               [1, 1]])
# B = np.dot(A, X)
# print(B)
# print(np.trace(B))

w = np.array([1, 2, 2, 1])
res = np.linalg.norm(w)/len(w)
print(np.linalg.norm(w))
print(res)

