# Ví dụ áp dụng Logistic Regression phân loại 2 số viết tay 0, 1 (phân loại nhị phân)
import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from display_network import *

# load MNIST data
datadir = "E:\\Dulieu(znz)\\Python_Pycharm\\datasets"
mnist = MNIST(datadir)
mnist.load_testing()
mnist.load_training()

# Gộp chung lại thành bộ 70000 điểm dữ liệu,
X_all = np.asarray(np.concatenate((mnist.train_images, mnist.test_images), axis=0))
y_all = np.asarray(np.concatenate((mnist.train_labels, mnist.test_labels), axis=0))
# Lấy ra tất cả điểm ứng với chữ sô 0, 1,
# sau đó chọn ngẫu nhiên 2000 điểm làm tập kiểm tra, còn lại là tập huấn luyện
X0 = X_all[np.where(y_all == 0)[0]]  # all digit 0
X1 = X_all[np.where(y_all == 1)[0]]  # all digit 1
y0 = np.zeros(X0.shape[0])  # class 0 label
y1 = np.ones(X1.shape[0])   # class 1 label

X = np.concatenate((X0, X1), axis=0)  # all digits 0 and 1
y = np.concatenate((y0, y1))  # all labels
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000)

model = LogisticRegression(C=1e5)   # C is inverse of lamda
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy %.2f %%" % (100*accuracy_score(y_test, y_pred)))

# Tìm những điểm bị phân loại sai
# mis = np.where((y_pred - y_test) != 0)[0]
# Xmis = X_test[mis, :]
# plt.axis("off")
# A = display_network(Xmis.T, 1, Xmis.shape[0])
# f2 = plt.imshow(A, interpolation="nearest")
# plt.gray()
# plt.show()

