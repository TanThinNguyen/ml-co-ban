import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import random

# Tạo ngẫu nhiên 3 cụm, mỗi cụm 500 điểm theo phân phối chuẩn
np.random.seed(18)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis=0)
K = 3   # 3 clusters
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# Vẽ đồ thị
def kmeans_display(X, labels):
    N = np.amax(labels) + 1
    X0 = X[labels == 0, :]
    X1 = X[labels == 1, :]
    X2 = X[labels == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], "b^", markersize="4")
    plt.plot(X1[:, 0], X1[:, 1], "go", markersize="4")
    plt.plot(X2[:, 0], X2[:, 1], "rs", markersize="4")

    plt.axis("equal")
    plt.show()

# Khởi tạo các tâm cụm
def kmeans_init_centroids(X, k):
    # randomly pick k rows of X as initial centroids
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Tìm nhãn mới cho các điểm khi biết các tâm cụm
def kmeans_assign_labels(X, centroids):
    # calculate pairwise distance between data and centroids
    D = cdist(X, centroids)
    # return index of the closet centroid
    return np.argmin(D, axis=1)

# Cập nhật các tâm cụm khi biết nhãn của từng điểm
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points that are assigned to the k-th cluster
        Xk = X[labels == k, :]
        centroids[k, :] = np.mean(Xk, axis=0)   # take average
    return centroids

# Kiểm tra điều kiện dừng của thuật toán
def has_converged(centroids, new_centroids):
    # return True if two sets of centroids are the same
    return (set([tuple(a) for a in centroids])) == (set([tuple(a) for a in new_centroids]))
 
# Phần chính của K-means clustering
def kmeans(X, K):
    # Lưu lịch sử các centroids trong 1 list
    centroids = [kmeans_init_centroids(X, K)]
    # Lưu lịch sử các labels trong 1 list
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return centroids, labels, it

# Vẽ đồ thị dữ liệu ban đầu
# kmeans_display(X, original_label)
# Áp dụng thuật toán vào dữ liệu ban đầu
centroids, labels, it = kmeans(X, K)
print("Centers found by our algorithm:\n", centroids[-1])

# Sử dụng scikit-learn
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)
print("Centers found by scikit-learn\n", model.cluster_centers_)
pred_label = model.predict(X)
kmeans_display(X, pred_label)
