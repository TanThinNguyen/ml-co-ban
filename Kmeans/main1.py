# Áp dụng K-means phân loại chữ số viết tay từ bộ dữ liệu MNIST
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datadir = "E:/Dulieu(znz)/Python_Pycharm/datasets"
mndata = MNIST(datadir)
mndata.load_testing()
X = mndata.test_images
K = 10

kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)
centroids = kmeans.cluster_centers_

X1 = np.asarray(mndata.test_images)
print(X1.shape)

