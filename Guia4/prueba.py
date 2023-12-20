import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Datos de ejemplo
categoria1 = ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'A', 'B']
categoria2 = ['X', 'Y', 'X', 'Y', 'X', 'Z', 'Z', 'Y', 'X', 'X']

# Crear un DataFrame a partir de las listas
df = pd.DataFrame({'Categoria1': categoria1, 'Categoria2': categoria2})

# Convertir las categorías en números usando LabelEncoder de scikit-learn
label_encoder = LabelEncoder()
df['Categoria1'] = label_encoder.fit_transform(df['Categoria1'])
df['Categoria2'] = label_encoder.fit_transform(df['Categoria2'])

# Calcular la matriz de contingencia usando Pandas
matriz_contingencia = pd.crosstab(df['Categoria1'], df['Categoria2'])

# Convertir la matriz de contingencia a una matriz NumPy
matriz_contingencia_numpy = matriz_contingencia.to_numpy()

# Imprimir la matriz de contingencia NumPy
print(matriz_contingencia_numpy)
