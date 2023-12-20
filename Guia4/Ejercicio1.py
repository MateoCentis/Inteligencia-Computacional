import numpy as np
from leer_archivo import leer_archivo
# from SOM import SOM
from somMejorado import SOM
[xTe,_] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia4/te.csv',2,0)
[xCir,_] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia4/circulo.csv',2,0)

neuronas = [100,1]
# neuronas = [25,25]
epocaMax = 1000
SOM(xTe, neuronas, epocaMax,True)  