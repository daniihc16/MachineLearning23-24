import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

Path = ""
# Función para dibujar la superficie de separación de un clasificador con 2 clases y 2 atributos
def dibujar_clasificador(clf, X_train, y_train, X_test, y_test):
    colores = ListedColormap(["r", "g"])
    DecisionBoundaryDisplay.from_estimator(clf, X_train, eps = 0.15, cmap=colores, grid_resolution=1000, response_method="predict", alpha=0.4)
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c= "r", edgecolors="k", label='0 - Train')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c= "g", edgecolors="k", label='1 - Train')
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],  c= "r", marker = 'v', edgecolors="k", label='0 - Test')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],  c= "g", marker = 'v', edgecolors="k", label='1 - Test')
    plt.legend()

# Leer los ficheros de datos y pasarlos a matrices de numpy
data_train = pd.read_csv(Path+'wine_train.csv', sep = ";")
x1_train = data_train["f6"].to_numpy().reshape(-1, 1)
x2_train = data_train["f10"].to_numpy().reshape(-1, 1)
y_train  = data_train["Clase"].to_numpy()
X_train = np.c_[x1_train, x2_train]
y_train[y_train!=1]=0  # Transformar y en 0/1 Para clasificación binaria de la clase 1

data_test  = pd.read_csv(Path+'wine_test.csv',  sep = ";")
x1_test = data_test["f6"].to_numpy().reshape(-1, 1)
x2_test = data_test["f10"].to_numpy().reshape(-1, 1)
y_test  = data_test["Clase"].to_numpy()
X_test  = np.c_[x1_test, x2_test]
y_test[y_test!=1]=0   # Transformar y en 0/1 Para clasificación binaria de la clase 1

# Ejemplo de dibujar una superficie de separación
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1)
clf.fit(X_train, y_train)
dibujar_clasificador(clf, X_train, y_train, X_test, y_test)
plt.title("Ejemplo de Superficie de Separación")
plt.show()

