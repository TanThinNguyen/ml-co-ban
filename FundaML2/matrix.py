import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(A)

# tạo ma trận đơn vị
print(np.eye(3))
print(np.eye(3, k=1))
print(np.eye(3, k=-2))

# Tạo ma trận đường chéo
print(np.diag([1, 3, 5]))
# Lấy ra đường chéo chính của ma trận
print(np.diag(A))
# Lấy ra đường chéo phụ của ma trận, thêm chỉ số k=1
print(np.diag(A, k=1))

# BT tạo ma trận (n+1)*(n+1) có đường chéo phụ ngay dưới đường chéo chính nhận giá trị từ 1->n
n = 5
B = np.diag(np.arange(1, n + 1), k=-1)
print(B)

# Lấy kích thước ma trận bằng shape, trả về 1 tuple là (numRow, numCol)
C = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Row: ", C.shape[0])
print("Col: ", C.shape[1])

# Truy cập vào một phần từ hàng i, cột j: A[i][j] hoặc A[i, j]
print(A[1, 2])

# Truy cập vào hàng/cột
# Hàng đầu tiên: A[0] hoặc A[0,:] hoặc A[0][:]
print(A[0])
# Cột đầu tiên: A[:, 0]
print(A[:, 0])


# BT viết hàm tỉnh tổng các phần tử trên cột có chỉ số chẵn
def even_col_sum(x):
    s = 0
    for i in range(0, x.shape[1], 2):
        s += np.sum(x[:, i])
    return s


print(even_col_sum(A))
