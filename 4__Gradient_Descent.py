import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
def cost(x):
    m = A.shape[0]
    0.5/m*np.linalg.norm(A.dot(x)-b)**2
def grad(x):
    m = A.shape[0]
    return 1/m*A.T.dot(A.dot(x)-b)


def Gradient_Descent(x_init,learning_rate,iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        x_list.append(x_new)
    return x_list


# Data
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

# Plot data
fig = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(0,30),ylim=(-10,50))
plt.plot(A,b,'ro')

# Add ones+ square to A
ones = np.ones((A.shape[0],1), dtype = np.int8)
square = A**2
A = np.concatenate((ones,A,square),axis=1)

# 1.1 Find x by Linear Regression formula --> plot
x_lr = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
x_plt = np.linspace(2,30,100)
y_plt_lr = x_lr[0][0] + x_lr[1][0] * x_plt +x_lr[2][0] * x_plt**2
plt.plot(x_plt,y_plt_lr,color='green')

# 2.Find x by GD --> plot
x_init = np.array([[1.,3.,-2]]).T
y_plt_init = x_init[0][0] + x_init[1][0]*x_plt +x_init[2][0]*x_plt**2
plt.plot(x_plt,y_plt_init,color='black')

learning_rate = 0.000001
iteration = 100
x_list = Gradient_Descent(x_init,learning_rate,iteration)
for i in range(len(x_list)):
    y_plt_gd = x_list[i][0][0] + x_list[i][1][0] * x_plt +x_list[i][2][0] * x_plt**2
    plt.plot(x_plt,y_plt_gd,color='black')

# Draw animation
ln, = plt.plot([], [], color='blue')
def update(i):
    y_animation_plt = x_list[i][0][0] + x_list[i][1][0]*x_plt +x_list[i][2][0] * x_plt**2
    ln.set_data(x_plt, y_animation_plt)
    return ln,

ani = FuncAnimation(fig, update, frames=100, interval= 50,blit=True)


plt.show()