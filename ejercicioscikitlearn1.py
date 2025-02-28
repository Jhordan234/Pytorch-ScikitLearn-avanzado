#Ejercicio Número 1
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5)

print("Precisión en cada fold:", scores)
print("Precisión media:", scores.mean())

#Ejercicio Número 2 

from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)
print("Precisión del modelo Ridge:", score)

#Ejercicio Número 3

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': [1, 0.1, 0.01, 0.001], 
    'kernel': ['rbf']
}

grid = GridSearchCV(model, param_grid, cv=5, verbose=1)
grid.fit(X, y)

print("Mejores hiperparámetros:", grid.best_params_)
print("Mejor precisión obtenida:", grid.best_score_)

