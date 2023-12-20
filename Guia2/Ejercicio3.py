import numpy as np
from perceptronMulticapaTobiP import perceptronMulticapa
from leer_archivo import leer_archivo

[x, yD] = leer_archivo("Guia2/irisbin_trn.csv", 4,3)
[xP, yP] = leer_archivo("Guia2/irisbin_tst.csv", 4,3)
capas = [1,3]       
epocaMaxima = 3000   
velocidadAprendizaje =  0.01
criterioError = 0.01    
perceptronMulticapa(x, yD, xP, yP, capas, epocaMaxima, velocidadAprendizaje, criterioError)
#velocidadAPrendizaje = 0.001, criterioError = 0.01, capas = [4,16,3], epocaMaxima = 3000 (epocas 132 1 error entrenamiento)
