import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cost(x):
    m = A.shape[0]
    return 0.5/m*np.linalg.norm(A.dot(x)-b)**2

def grad(x):
    m = A.shape[0]
    return 1/m*A.T.dot(A.dot(x)-b)

def check_grad(x):
    eps = 1e-4
    g_check = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g_check[i] = (cost(x1) - cost(x2))/(2*eps)
        # thay doi i 1 vi tri thi norm cac vi tri con lai =0
    if np.linalg.norm(g_check-grad(x)) >1e-5:
        print('warming')
def gradient_descent(x0_init, learning_rate, iteration):
    m = A.shape[0]
    x_list = [x0_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        if np.linalg.norm(grad(x_new))/m<0.7:
            break
        # dùng độ thay đổi của a, b trong x --> ko nên, ta ko tưởng tưởng độ thay đổi này.
        # nên dùng cost thay vì dùng grad làm điều kiện dừng (cost giống độ lệch khoảng cách của
        # mỗi điểm, grad là độ thay đổi hệ số a,b )
        x_list.append(x_new)
    return x_list


# Data
# Data
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

# Draw data
fig = plt.figure("GD vs sklearn")
ax = plt.axes(xlim = (-10, 60), ylim = (0,50))
plt.plot(A,b,'ro')

# add ones, square to A:
ones = np.ones((A.shape[0],1))
square = A**2
A = np.concatenate((ones,A,square),axis=1)
print(A)

# Line created by Linear Regression formula (sklearn)
lr = LinearRegression()
lr.fit(A,b)
x0_gd = np.linspace(2,60,1000)
y0_sklearn = lr.intercept_[0] + lr.coef_[0][1]*x0_gd + lr.coef_[0][2]*x0_gd*x0_gd
plt.plot(x0_gd,y0_sklearn,color = 'green')
# plt.show()

# x = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
# x0_gd = np.linspace(0,30,1000)
# y0_lr = x[0][0] + x[1][0]*x0_gd + x[2][0]*x0_gd*x0_gd
# plt.plot(x0_gd,y0_lr,color = 'green')
# plt.show()

# Line created by gradient descent (random initial)


x0_init = np.array([[1.,2.,-0.1]]).T
y0_init = x0_init[0][0] + x0_init[1][0]*x0_gd + x0_init[2][0]*x0_gd**2
plt.plot(x0_gd,y0_init, color = 'black')
# plt.show()
# # check_grad(x0_init)

learning_rate = 0.000001
iteration = 1000
x_list = gradient_descent(x0_init, learning_rate, iteration)
for i in range(len(x_list)):
    y0_x_list = x_list[i][0] +  x_list[i][1]*x0_gd + x_list[i][2]*x0_gd**2
    plt.plot(x0_gd,y0_x_list,color = 'black',alpha = 0.3)
print(len(x_list))
print(x_list[-1])
# plt.show()
#
#
line, = ax.plot([], [], color ='blue')

# animation function.  This is called sequentially
def animate(i):
    y = x_list[i][0][0] + x_list[i][1][0] * x0_gd + x_list[i][2][0]*x0_gd**2
    line.set_data(x0_gd, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=110, interval=50, blit=True)

plt.legend(('Value in each GD iteration', 'Solution by formular', 'Inital value for GD'))
plt.show()