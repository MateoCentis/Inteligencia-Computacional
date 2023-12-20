import numpy as np
from k_medias import k_medias
from somMejorado import SOM
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from leer_archivo import leer_archivo
[x,_] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia4/irisbin_trn.csv', 4,3)

neuronas = [10,10]
epocaMax = 1000
[_,centroidesPertenece,centroides] = k_medias(x,9)
neuronasPertence = SOM(x,[3,3],1000,False)
# kmeans = KMeans(16, random_state=42)
# centroidesPertenece = kmeans.fit_predict(x)
print(centroidesPertenece)
print(neuronasPertence)
contingency = contingency_matrix(centroidesPertenece, neuronasPertence)

print("Matriz de Contingencia:")
print(contingency)
