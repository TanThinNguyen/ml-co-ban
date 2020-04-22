import numpy as np
# Ví dụ gradient với momentum
# Các tham số:
#   + grad: hàm tính gradient
#   + theta_init: điểm xuất phát
def GD_momentum(grad, theta_init, eta, gamma):
    # Lưu lại lịch sử các theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = eta*grad(theta[-1]) + gamma*v_old
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new))/np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v_old = v_new
    return theta


# Cải tiến momentum bằng Nesterov Accelerated Gradient (NAG)
def GD_NAG(grad, theta_init, eta, gamma):
    # Lưu lại lịch sử các theta, v
    theta = [theta_init]
    v = [np.zeros_like(theta_init)]
    for it in range(100):
        v_new = eta*grad(theta[-1] - gamma*v[-1]) + gamma*v[-1]
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new))/np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v.append(v_new)
    return theta


