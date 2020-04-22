import numpy as np
import matplotlib.pyplot as plt

# Tính hàm softmax
def softmax(Z):
    """
    compute softmax values for each sets of scores in Z.
    each column of Z is a set of scores.
    :param Z: a numpy array of shape (N, C)
    :return: a numpy array of shape (N, C)
    """
    e_Z = np.exp(Z)
    A = e_Z/np.sum(e_Z, axis=1, keepdims=True)
    return A

# Hàm softmax phiên bản ổn định
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = e_Z/np.sum(e_Z, axis=1, keepdims=True)
    return A

# Hàm mất mát
def softmax_loss(X, y, W):
    """
    :param X: 2d numpy array of shape (N, d), each row is one data point
    :param y: 1d numpy array, label of each row of X
    :param W: 2d numpy array of shape (d, C), each column corresponding to one output node
    """
    A = softmax_stable(X.dot(W))
    id0 = range(X.shape[0])  # indexes in axis 0, indexes in axis 1 are in y
    return -np.mean(np.log(A[id0, y]))

# Hàm tính gradient
def softmax_grad(X, y, W):
    """
    :param X: 2d numpy array of shape (N, d), each row is one data point
    :param y: 1a numpy array, label of each row of X
    :param W: 2d numpy array of shape (d, C), each column corresponding to one output node
    :return:
    """
    A = softmax_stable(X.dot(W))  # shape of (N, C)
    id0 = range(X.shape[0])
    A[id0, y] -= 1  # A <- A - Y, shape of (N, C)
    return X.T.dot(A) / (X.shape[0])

# Hàm huấn luyện hồi quy softmax
def softmax_fit(X, y, W, lr=0.01, nepochs=100, tol=1e-5, batch_size=10):
    W_old = W.copy()
    ep = 0
    loss_hist = [softmax_loss(X, y, W)]  # store history of loss
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    while ep < nepochs:
        ep += 1
        mix_ids = np.random.permutation(N)  # stochastic
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W = W - lr*softmax_grad(X_batch, y_batch, W)
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old)/W.size < tol:
            break
        W_old = W
    return W, loss_hist

# Hàm dự đoán nhãn của dữ liệu mới
def pred(W, X):
    return np.argmax(X.dot(W), axis=1)

# Tạo dữ liệu có 5 lớp
C, N = 5, 500   # numer of classes and number of points per class
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)
X = np.concatenate((X0, X1, X2, X3, X4), axis=0)  # each row is a datapoint
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((X, one), axis=1)  # bias trick

y = np.asarray([0]*N + [1]*N + [2]*N + [3]*N + [4]*N)  # label
W_init = np.random.randn(Xbar.shape[1], C)
W, loss_hist = softmax_fit(Xbar, y, W_init, lr=0.05)
print("Solution: W\n", W)


# -----------------------------------------------
# Vẽ đồ thị minh họa
# Dữ liệu input
plt.figure(1)
plt.plot(X0[:, 0], X0[:, 1], "ro", label="Class 1")
plt.plot(X1[:, 0], X1[:, 1], "b^", label="Class 2")
plt.plot(X2[:, 0], X2[:, 1], "gv", label="Class 3")
plt.plot(X3[:, 0], X3[:, 1], "cs", label="Class 4")
plt.plot(X4[:, 0], X4[:, 1], "yp", label="Class 5")
# plt.legend()

# Hàm mất mát qua các epoch
plt.figure(2)
plt.plot(range(len(loss_hist)), loss_hist, color="r")

plt.show()


