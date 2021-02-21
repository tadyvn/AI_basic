import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Random data
A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
plt.plot(A,b,'ro')

model = linear_model.LinearRegression()
model.fit(A,b)

print(model.coef_)
print(model.intercept_)
x0 = np.array([[1,46]]).T
print(x0)
y0 = x0*model.coef_ + model.intercept_

plt.plot(x0,y0)
plt.show()

