from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = KNeighborsClassifier()
model.fit(X, y)
model.predict([3, 5, 4, 2], [5, 4, 2, 3])
