import numpy as np

arr = np.array([1, 4, 7, 2, 5, 10])
print(arr)
print(arr[::-1])

# Phép toán giữa 2 vector cùng số lượng phần tử
arr1 = np.array([2, 3, 4, 5, 6, 7])
print(arr * arr1)
print(arr + arr1)
print(arr ** arr1)

x = np.array([1, -5, 9, -12, 2, -3])
y = np.pi / 2 - x
z = np.cos(x) - np.sin(x)
print(np.sum(z))


# Tính norm 1 của vector x bằng tổng gttđ của các phần tử của vector
def sum_abs(a):
    return np.sum(np.abs(a))


print(sum_abs(x))

# Tích vô hướng của 2 vector: tích từng phần từ r cộng lại
print(np.sum(arr * arr1))
# Tích vô hướng của 2 vector: dùng hàm dot()
print(arr.dot(arr1))
print(np.dot(arr, arr1))

# Tính norm 2 của vector: căn bậc hai tổng bình phương các phần tử
print(np.sqrt(np.sum(x * x)))

# Tìm min, max
print(x.min())
print(x.max())
# Tìm index của phần tử min, max
print(x.argmax())
print(x.argmin())

# Hàm softmax cho mảng 1 chiều
vector = np.array([1, 5, 2, 7, 1])
softmax = np.exp(vector)/np.sum(np.exp(vector))
