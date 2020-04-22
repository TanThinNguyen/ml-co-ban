import numpy as np

# Khai báo vector (mảng một chiều)
# mảng kiểu int32
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# mảng kiểu float64
y = np.array([1.3, 1.4])
# ngoài ra cũng có thể ép kiểu như sau
z = np.array([1, 3, 5], dtype=np.float64)
print(x)
print(type(x[0]))
print(y)
print(type(y[0]))
print(z)
print(type(z[0]))

# Tạo vector 0
vectorzero = np.zeros(3)
# Tạo vector 0 với số phần từ giống vector x
vectorzero1 = np.zeros_like(x)
print(vectorzero)
print(vectorzero1)
print(type(vectorzero1[0]))

# Tạo vector toàn giá trị 1
vectorone = np.ones(3)
vectorone1 = np.ones_like(y)
print(vectorone)
print(vectorone1)
print(type(vectorone1[0]))

# Tạo mảng các số nguyên từ m đến < n
intarr = np.arange(5, 6.5)
print(intarr)

# Tạo cấp số cộng: mảng các sô nguyên từ m đến < n, công sai d
intarr1 = np.arange(5, 10, 0.5)
print(intarr1)
intarr2 = np.arange(15, 7.6, -0.5)
print(intarr2)

# BT: xây dựng mảng các lũy thừa của 2 nhỏ hơn 1025 bao gồm cả 1 = 2**0
expo = np.arange(0, 11)
res = 2**expo
print(res)

# Kích thước của mảng, trả về dạng tuple
print(expo.shape)
# với mảng một chiều thì tuple có 1 phần tử dạng (d, )
print(expo.shape[0])


# Truy cập nhiều phần tử của mảng
arr = 0.5 * np.arange(10)
ids = [1, 5, 2, 7]
# đọc nhiều phần tử
print(arr[ids])
# đọc 3 số đầu (index 0, 1, 2)
print(arr[:3])
# đọc 3 số cuối (index -3, -2, -1)
print(arr[-3:])
# đọc 3 số có index (1, 2, 3)
print(arr[1:4])

# Gán 1 cho arr[1], arr[3], arr[5]
arr[[1, 3, 5]] = 1
# Gán 3 phần tử cuối cho mảng [0, 1, 2]
arr[-3:] = np.array([0, -1, -2])
# Lấy các phần từ có index chẵn
print(arr[::2])
# Đảo ngược mảng
print(arr[::-1])
