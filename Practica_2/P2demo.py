#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solución P2: Regresión polinómica de precios de coches

@author: Juan D. Tardós
@date: Jan 19, 2024
@version: 1.0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

plt.close('all')

# Leer los ficheros de datos y pasarlos a matrices de numpy
coches_train = pd.read_csv('CochesTrain.csv', sep = ";")
x1_train = coches_train["Agnos"].to_numpy().reshape(-1, 1)
x2_train = coches_train["Km"].to_numpy().reshape(-1, 1)
x3_train = coches_train["CV"].to_numpy().reshape(-1, 1)
y_train  = coches_train["Precio"].to_numpy()

coches_test  = pd.read_csv('CochesTest.csv',  sep = ";")
x1_test = coches_test["Agnos"].to_numpy().reshape(-1, 1)
x2_test = coches_test["Km"].to_numpy().reshape(-1, 1)
x3_test = coches_test["CV"].to_numpy().reshape(-1, 1)
y_test  = coches_test["Precio"].to_numpy()

# Mostrar los datos de entrenamiento
seaborn.pairplot(coches_train)
plt.show()

