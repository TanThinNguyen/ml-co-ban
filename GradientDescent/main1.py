# Ví dụ áp dụng gradient descend (GD) với bài toán hồi quy tuyến tính
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(21)
# Tạo 1000 điểm dữ liệu gần đường thẳng y = 4 + 3x
X = np.random.rand(1000)
y = 4 + 3*X + 0.25*np.random.randn(1000)  # noise added

# Tìm nghiệm bằng công thức nghiệm (thông qua thư viện)
model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b, w])
print("coef: ", model.coef_)
print("intercept: ", model.intercept_)
print("Solution found by sklearn: ", sol_sklearn)

# --------------------------------
# Vẽ đồ thị biểu diễn dữ liệu
plt.plot(X, y, "b.", label="Input data")
# Vẽ đồ thị biểu diễn kết quả
x0 = np.array([0, 1])
y0 = b + w*x0
plt.plot(x0, y0, color="r", linewidth=2, label="Sklearn solution")
plt.axis([0, 1, 0, 10])
plt.legend()
plt.show()


# --------------------------------
# Tìm nghiệm bằng GD: nghiệm w chứa cả hệ số điều chỉnh b
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis=1)

# Tính gradient của hàm số
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# Tính giá trị của hàm mất mát
def cost(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linalg.norm(y - Xbar.dot(w))**2

# Hàm thực hiện gradient descend
def myGD(w_init, grad, eta):
    w_res = [w_init]
    for it in range(100):
        w_new = w_res[-1] - eta*grad(w_res[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w_res.append(w_new)
    return w_res, it

w_init = np.array([2, 1])
w1, it1 = myGD(w_init, grad, 1)
print("Solution found by GD: w = ", w1[-1].T, ", after %d iterations." % it1)
