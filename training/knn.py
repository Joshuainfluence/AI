from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
x= iris.data
y= iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
# Train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=3) #looks at 3 nearest neighbors
knn_model.fit(X_train, y_train)

# predict 
knn_pred = knn_model.predict(X_test)

# Evaluate
print("KNN accuracy:", accuracy_score(y_test, knn_pred))