# Áp dụng K-means tách vật thể trong ảnh
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Đọc file ảnh lưu vào 1 ma trận
img = mpimg.imread("girl.jpg")
# Hiển thị ảnh gốc
# plt.figure(1)
plt.subplot(2, 1, 1)
imgplot = plt.imshow(img)
plt.axis("off")


# Biến đổi bức ảnh thành một ma trận mà mỗi hàng là 3 giá trị màu của 1 điểm ảnh (số hàng là số điểm ảnh)
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
# Áp dụng phân cụm K-means tìm tập 3 điểm màu đỏ, đen và màu da
K = 3  # 3 màu
kmeans = KMeans(n_clusters=K).fit(X)
label = kmeans.predict(X)
img_new = np.zeros_like(X)
for k in range(K):
    # replace each pixel by its center
    img_new[label == k] = kmeans.cluster_centers_[k]  # np.where(label == k)[0]

# Biển đổi ma trận về lại ảnh
img_res = img_new.reshape((img.shape[0], img.shape[1], img.shape[2]))
# Hiển thị ảnh sau khi phân cụm theo màu
# plt.figure(2)
plt.subplot(2, 1, 2)
plt.imshow(img_res, interpolation="nearest")
plt.title("K = 3")
plt.axis("off")
plt.show()
