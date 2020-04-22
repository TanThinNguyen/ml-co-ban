# Thuật toán KNN dự đoán trên iris dataset
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split    # for splitting data
from sklearn.metrics import accuracy_score  # for evaluate results

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print("Labels: ", np.unique(iris_y))

# split train and test (train_size = 20, test_size = 130)
np.random.seed(7)   # 7 is a arbitrary number
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)
print("Training size:", X_train.shape[0], ", test size:", X_test.shape[0])

# result with 1-NN
# p = 2 tương ứng với l2 norm để tính khoảng cách
# weights tức là đánh trọng số cho từng điểm lân cận, điểm gần hơn có tầm ảnh hưởng lớn hơn
model = neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights="distance")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 7-NN: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))

# trọng số tự định nghĩa
def myweight(distances):
    sigma2 = 0.4    # we can change this number
    return np.exp(-distances**2/sigma2)


model2 = neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights=myweight)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("Accuracy of 7-NN: %.2f %%" % (100 * accuracy_score(y_test, y_pred2)))
