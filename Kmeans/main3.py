# Áp dụng K-means nén ảnh
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Đọc file ảnh lưu vào 1 ma trận
img = mpimg.imread("girl.jpg")
# Hiển thị ảnh gốc
# plt.figure(1)
plt.subplot(2, 3, 1)
imgplot = plt.imshow(img)
plt.axis("off")


# Biến đổi bức ảnh thành một ma trận mà mỗi hàng là 3 giá trị màu của 1 điểm ảnh (số hàng là số điểm ảnh)
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
plot_count = 2
for K in [5, 10, 15, 20]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img_new = np.zeros_like(X)
    # replace each pixel by its centroid
    for k in range(K):
        img_new[label == k] = kmeans.cluster_centers_[k]
    # reshape image
    img_res = img_new.reshape(img.shape[0], img.shape[1], img.shape[2])
    # display output image
    # plt.figure(plot_count)
    plt.subplot(2, 3, plot_count)
    plt.imshow(img_res)
    plt.axis("off")
    plot_count += 1
plt.show()
