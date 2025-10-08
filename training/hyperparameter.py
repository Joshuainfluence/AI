# HYPERPARAMETER TUNING WITH GRIDSEARCHCV

# import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load datasets and split it
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the model and parameter grid
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5, # 5-fold cross-validation
    scoring='accuracy', # measure performance using accuracy
    n_jobs=1 # use all CPU cores for speed
)

# fit the model
grid_search.fit(X_train, y_train)

# print the best results
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

best_knn = grid_search.best_estimator_

y_pred = best_knn.predict(X_test)

print("Test acccuracy:", accuracy_score(y_test, y_pred))

