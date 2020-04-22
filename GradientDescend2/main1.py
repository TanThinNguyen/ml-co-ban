import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(21)
# Tạo 1000 điểm dữ liệu gần đường thẳng y = 4 + 3x
X = np.random.rand(1000)
y = 4 + 3*X + 0.25*np.random.randn(1000)  # noise added

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis=1)

# Stochastic Gradient Decend
# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = xi.dot(w) - yi
    return xi*a

def SGD(w_init, sgrad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    count = 0
    N = X.shape[0]

    # Lặp khoảng 10 epoch
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count % iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
                    return w, count
                w_last_check = w_this_check
    return w, count

# Tìm nghiệm bằng thư viện
model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
w, b = model.coef_[0][0], model.intercept_[0]
print("Solution by sklearn: ", [b, w])

# Tìm nghiệm bằng SGD
w_init = np.array([2, 1])
w1, count1 = SGD(w_init, sgrad, 0.1)
print("Solution by SGD: ", w1[-1], " after %d iterations" % count1)


# Vẽ đồ thị minh họa
# Dữ liệu đầu vào
plt.plot(X, y, "b.", label="Input data")
plt.axis([0, 1, 0, 10])
# Kết quả bằng sklearn
x0 = np.array([0, 1])
y_sklearn = b + w*x0
plt.plot(x0, y_sklearn, color="r", linewidth="2", label="Sklearn solution")
# Kết quả bằng SGD
y_SGD = w1[-1][0] + w1[-1][1] * x0
plt.plot(x0, y_SGD, color="y", linewidth="2", label="SGD solution")

plt.legend()
plt.show()
