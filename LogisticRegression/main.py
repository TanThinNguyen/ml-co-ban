import numpy as np
import matplotlib.pyplot as plt

def sigmoid(S):
    """
    :param S: an numpy array
    :return sigmoid funtion of each element of S
    """
    return 1/(1 + np.exp(-S))

# Hàm ước lượng xác suất đầu ra cho mỗi điểm dữ liệu
def prob(w, X):
    """
    :param w: a 1d numpy array of shape(d)
    :param X: a 2d numpy array of shape(N, d); N datapoint, each with size d
    """
    return sigmoid(X.dot(w))

# Hàm mất mát với weight decay (suy giảm trọng số)
def loss(w, X, y, lam):
    # y: a 1d numpy array of shape(N); each element = 0 or 1
    a = prob(w, X)
    loss_0 = -np.mean(y*np.log(a) + (1-y)*np.log(1-a))
    weight_decay = 0.5*lam*np.sum(w*w)
    return loss_0 + weight_decay

def logistic_regression(w_init, X, y, lam, lr=0.1, nepoches=2000):
    # lam: regulariza parameter, lr: learning rate, nepoches: epoches
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    # store history of loss in loss_hist
    loss_hist = [loss(w, X, y, lam)]
    ep = 0
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)  # stochastic
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            ai = sigmoid(xi.dot(w))
            # update
            w = w - lr*((ai - yi)*xi + lam*w_old)

            if np.linalg.norm(w - w_old)/d < 1e-6:
                break
            w_old = w
        # store lost after an epoch
        loss_hist.append(loss(w, X, y, lam))
    return w, loss_hist


# Tạo dữ liệu đầu vào
np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00,
               4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
# Bias trick
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hist = logistic_regression(w_init, Xbar, y, lam, lr=0.05, nepoches=500)
print("Solution of Logistic Regression: ", w)
print("Final loss:", loss(w, Xbar, y, lam))


# Dự đoán đầu ra cho điểm dữ liệu mới
def predict(w, X, threshold=0.5):
    """
    predict output for each row of X
    :param w:
    :param X: a numpy array of shape (N, d)
    :param threshold: 0 < threshold < 1
    """
    res = np.zeros(X.shape[0])
    # Đánh dấu những điểm pass là 1
    res[np.where(prob(w, X)) > threshold] = 1
    return res


# -----------------------------------------
# Vẽ đồ thị minh họa
# Dữ liệu đầu vào
plt.figure(1)
idx_pass = np.where(y == 1)[0]
idx_fail = np.where(y == 0)[0]
plt.plot(X[idx_pass, 0], y[idx_pass], "bs", label="Pass")
plt.plot(X[idx_fail, 0], y[idx_fail], "ro", label="Fail")
# Kết quả Logistic Regression
x0 = np.arange(0, 6, 0.01)
y0 = sigmoid(w[0]*x0 + w[1])
plt.plot(x0, y0, color="c", label="Logistic Regression")
plt.xlabel("studying hours")
plt.ylabel("predicted probability of pass")
plt.axis([0, 6, -0.5, 1.5])
plt.legend()

# Hàm mất mát (loss function) sau mỗi epoch
plt.figure(2)
plt.plot(range(len(loss_hist)), loss_hist, "r")

plt.show()

