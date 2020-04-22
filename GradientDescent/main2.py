# Ví dụ kiểm tra gradient bằng pp xấp xỉ gradient (phần kiểm tra gradient trong mục giải tích ma trận ebook tr52)
import numpy as np

# Hàm kiểm tra gradient với các tham số:
#   + fn: hàm tính giá trị của hàm số f(X) (truyền function như 1 parameter), về mặt toán hàm số nhận đầu vào
# là vector hoặc ma trận, trả về số thực
#   + gr: hàm tính đạo hàm của hàm số f(X) (truyền function như 1 parameter)
def check_grad(fn, gr, X):
    X_flat = X.reshape(-1)  # convert X to an 1d-array, 1 for loop needed
    shape_X = X.shape       # original shape of X
    num_grad = np.zeros_like(X)         # numerical gradient, shape = shape of X
    grad_flat = np.zeros_like(X_flat)   # 1d version of grad
    eps = 1e-6      # a small number, 1e-10 -> 1e-6 is ussually good
    numElems = X_flat.shape[0]      # number of elements in X
    # calculate numerical gradient
    for i in range(numElems):
        Xp_flat = X_flat.copy()
        Xn_flat = X_flat.copy()
        Xp_flat[i] -= eps
        Xn_flat[i] += eps
        Xp = Xp_flat.reshape(shape_X)
        Xn = Xn_flat.reshape(shape_X)
        grad_flat[i] = (fn(Xn) - fn(Xp)) / (2*eps)

    num_grad = grad_flat.reshape(shape_X)
    diff = np.linalg.norm(num_grad - gr(X))
    print("Difference between two methods should be small:", diff)


# ---- Check if grad(trace(AX)) = A^T ----
m, n = 10, 20
A = np.random.rand(m, n)
X = np.random.rand(n, m)

# Tính giá trị hàm số f(X) = trace(AX)
def fn1(X):
    return np.trace(np.dot(A, X))

# Tính giá trị đạo hàm hàm số f(X) = trace(AX)
def gr1(X):
    return A.T

check_grad(fn1, gr1, X)


# ---- Check if grad(x^T * A * x) = (A + A^T) * x ----
A = np.random.rand(m, m)
x = np.random.rand(m, 1)

def fn2(x):
    return x.T.dot(A).dot(x)

def gr2(x):
    return (A + A.T).dot(x)

check_grad(fn2, gr2, x)

