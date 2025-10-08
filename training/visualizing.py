import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load data (uses only 2 features for easy 2D visulisatoin)
iris = load_iris()
X = iris.data[:, :2] #use sepal lenght and sepal width only
y = iris.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Models
tree = DecisionTreeClassifier().fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

# create a meshgrid (to plot all possible x-y conmbination)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# predict for each pont on the grid
Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# reshape for plotting
Z_tree = Z_tree.reshape(xx.shape)
Z_knn = Z_knn.reshape(xx.shape)

# plot decision tree
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_tree, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
plt.title("Decision Tree Decision Boundary")


# Plot KNN
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_knn, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:,0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
plt.title("KNN Decision Boundary (k=3)")

plt.show()
