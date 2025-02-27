# Ví dụ áp dụng gradient descend cho hàm 1 biến
# Lấy hàm f(x) = x^2 + 5sin(x)
import numpy as np

# Tính đạo hàm hàm số tại x
def grad(x):
    return 2*x + 5*np.cos(x)

# TÍnh giá trị hàm số tại x
def cost(x):
    return x**2 + 5*np.sin(x)

# Hàm thực hiện gradient descend
def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:     # just a small number
            break
        x.append(x_new)
    return x, it


(x1, it1) = myGD1(-5, 0.1)
(x2, it2) = myGD1(5, 0.1)
print("Solution x1 = %f, cost = %f, after %d iterations" % (x1[-1], cost(x1[-1]), it1))
print("Solution x2 = %f, cost = %f, after %d iterations" % (x2[-1], cost(x2[-1]), it2))
