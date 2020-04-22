import numpy as np
import matplotlib.pyplot as plt

# Hàm xác định nhãn của các điểm dữ liệu (các hàng của X) khi đã tìm dc vector trọng số w
def predict(w, X):
    """
    predict label of each row of X, given w
    :param w: a 1-d numpy array of shape (d)
    :param X: a 2-d numpy array of shape (N, d), each row is a datapoint
    """
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    """
    perform perceptron learning algorithm
    :param X: a 2-d numpy array of shape (N, d), each row is a datapoint
    :param y: a 1-d numpy array of shape (N), label of each row of X; y[i] = 1 or -1
    :param w_init: a 1-d numpy array of shape (d)
    """
    w = w_init
    for i in range(100):
        pred = predict(w, X)
        # find indexes of misclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]  # np.where() trả về dạng tuple (a,)
        # number of misclassified points
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:   # no more misclassified points
            return w, i
        # randomly pick one misclassified point
        random_id = np.random.choice(mis_idxs, 1)[0]  # có [0] để kết quả là kiểu int chứ ko phải kiểu list
        # update w
        w = w + y[random_id]*X[random_id]
    return w, i

#
means = [[-1, 0], [1, 0]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1*np.ones(N)), axis=0)

Xbar = np.concatenate((np.ones((2*N, 1)), X), axis=1)
w_init = np.random.randn(Xbar.shape[1])
w, i = perceptron(Xbar, y, w_init)
print("Solution: ", w, ", after %d iterations" % i)


# Vẽ đồ thị minh họa
# Dữ liệu input
plt.plot(X0[:, 0], X0[:, 1], "r^")
plt.plot(X1[:, 0], X1[:, 1], "bs")

# Kết quả perceptron
x0 = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
y0 = -w[0]/w[2] + -w[1]/w[2] * x0
plt.plot(x0, y0, color="c")
plt.show()
