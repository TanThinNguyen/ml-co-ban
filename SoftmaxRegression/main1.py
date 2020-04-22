import numpy as np
from sklearn.linear_model import LogisticRegression
from mnist import MNIST
from sklearn.metrics import accuracy_score

# load MNIST data
datadir = "E:\\Dulieu(znz)\\Python_Pycharm\\datasets"
mnist = MNIST(datadir)
mnist.load_training()
mnist.load_testing()

X_train = np.asarray(mnist.train_images)
y_train = np.asarray(mnist.train_labels)
X_test = np.asarray(mnist.test_images)
y_test = np.asarray(mnist.test_labels)

model = LogisticRegression(C=1e5, solver="lbfgs", multi_class="multinomial")  # C is inverse of lamda
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy %.2f %%" % (100*accuracy_score(y_test, y_pred)))
