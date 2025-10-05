from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load a sample dataset (flowers data)
iris = load_iris()
x = iris.data
y = iris.target

# split data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train a decision tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# predict on test data
y_pred = model.predict(x_test)

# Evaluate 
print("Model accuracy:", accuracy_score(y_test, y_pred))