# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# truyền vào tuple (row, col) = (X.shape[0], 1) tạo ra vector dọc toàn 1
one = np.ones((X.shape[0], 1))
# Building Xbar, each row is one data point
Xbar = np.concatenate((one, X), axis=1)

# Calculating weights of the linear regression model
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
# giả nghịch đảo của A dc tính bằng np.linalg.pinv(A)
w = np.dot(np.linalg.pinv(A), b)

# Hồi quy: cân nặng = w1*(chiều cao) + w0
w0, w1 = w[0], w[1]
# print(w0)
# print(w1)

# Dự đoán cân nặng với chiều cao 155, 160
y1 = w1 * 155 + w0
y2 = w1 * 160 + w0
print("Input 155cm, true output 52kg, predicted output %.2f kg" % y1)
print("Input 160cm, true output 56kg, predicted output %.2f kg" % y2)


# Vẽ đồ thị dữ liệu huấn luyện và mô hình hồi quy
xhat = np.arange(np.min(X), np.max(X), 0.01)
yhat = w1 * xhat + w0
plt.plot(xhat, yhat, label="Mo hinh tim duoc", color="red")
plt.scatter(X.T[0], y, label="Du lieu huan luyen")
plt.xlabel("x - height (cm)")
plt.ylabel("y - weight (kg)")
plt.legend()
plt.show()

# ------------------------------------------------------
# Nghiệm theo thư viên scikit-learn
# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y)
print("coef: ", regr.coef_)
print("intercept: ", regr.intercept_)
print("Scikit-learn solution: w1 = ", regr.coef_[0], "w0 = ", regr.intercept_)
print("Our solution:          w1 = ", w1, "w0 = ", w0)

