import numpy as np

a = np.array([1, 4, 3, 2, 5])
b = np.array([1, 1, 3, 6, 5])
c = np.array([[1, 5, 3],
              [6, 2, 1]])
id = np.where(np.equal(a, b) == True)[0]
# print(np.equal(a, b))
# print(id)
# idx = np.where(c < 5)
# print(c[idx])

# x = np.array([-1.2, 3.2, 2.5, -0.1])
# print(np.sign(x))

# print(np.random.permutation(10))
# print(np.random.choice(id, 1)[0])
# print(np.random.choice(5, 3, replace=False))


