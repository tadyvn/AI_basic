import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# giai thich predict : data train lam moi truong --> tim ra k, data test chi de test
digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target


# shuffle by index
randIndex = np.arange(len(digits_y))
np.random.shuffle(randIndex)

digits_X = digits_X[randIndex]
digits_y = digits_y[randIndex]

X_train, X_test, y_train, y_test = train_test_split(digits_X,digits_y,test_size=360)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
acc= accuracy_score(y_predict,y_test)

print(acc)
print(knn.predict(X_test[0].reshape(1, -1)))
plt.gray()
plt.imshow(X_test[0].reshape(8,8))
plt.show()

