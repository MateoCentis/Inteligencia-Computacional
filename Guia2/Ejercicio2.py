import numpy as np
from perceptronMulticapaTobiP import perceptronMulticapa
from leer_archivo import leer_archivo

[x, yD] = leer_archivo("Guia2/concent_trn.csv", 2,1)
[xP, yP] = leer_archivo("Guia2/concent_tst.csv", 2,1)
# x = np.square(x)
# xP = np.square(xP)
capas = [4,1] #con 3 capas se logra 10% de error
epocaMaxima = 1000
velocidadAprendizaje =  0.59 #con 0.58 o menos no aprende
criterioError = 0.01  
#Velocidad aprendiza 0.7 y criterio error 0.07 funciona con [4,1] en 75 epocas approx
#[20,1] en 2112 epocas 0.03 de error 
perceptronMulticapa(x, yD, xP, yP, capas, epocaMaxima, velocidadAprendizaje, criterioError)
#Puede ser que se estanque con [4,1] si no tiene más de 0.58 de velocidad de aprendizaje
#Entradas cuadráticas: elevando al cuadrado las entradas con 4 neuronas 0.014 de error de test