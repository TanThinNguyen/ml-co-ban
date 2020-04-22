import numpy as np
from sklearn.linear_model import LinearRegression

# Demo hồi quy tuyến tính đồng thời 2 hàm số (chung X)
X = np.random.rand(1000)
y1 = 4 + 3*X + 0.5*np.random.randn(1000)
y2 = 5 + 6*X + 0.5*np.random.randn(1000)
Y1 = y1.reshape(1000, 1)
Y2 = y2.reshape(1000, 1)
# print("y1: ", y1)
# print("y2: ", y2)
Y = np.concatenate((Y1, Y2), axis=1)

model = LinearRegression()
model.fit(X.reshape(-1, 1), Y)
print("coef: ", model.coef_)
print("intercept: ", model.intercept_)