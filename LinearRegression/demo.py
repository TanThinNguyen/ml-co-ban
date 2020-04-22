import numpy as np
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data
plt.plot(X, y, "ro", label="Du lieu huan luyen")
plt.axis([140, 190, 45, 75])
plt.xlabel("x - height (cm)")
plt.ylabel("y - weight (kg)")
plt.legend()
plt.show()

# Linear Regression model
# weight = w1 * (height) + w0

# building Xbar
one = np.ones((1, X.shape[0]))
Xbar = np.concatenate((one, X.T), axis=0)
# calculating weights of the fitting line
A = np.dot(Xbar, Xbar.T)
b = np.dot(Xbar, y)
w = np.dot(np.linalg.pinv(A), b)
w0, w1 = w[0], w[1]

# Predict




