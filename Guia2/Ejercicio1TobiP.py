import numpy as np
from perceptronMulticapaTobiP import perceptronMulticapa
from leer_archivo import leer_archivo

[x, yD] = leer_archivo("Guia2/XOR_trn.csv", 2,1)    
[xP, yP] = leer_archivo("Guia2/XOR_tst.csv", 2,1)
capas = [2, 1]
epocaMaxima = 100
velocidadAprendizaje =  0.01
criterioError = 0.001
perceptronMulticapa(x, yD, xP, yP, capas, epocaMaxima, velocidadAprendizaje, criterioError)
# [
#     cantEpocasTr,
#     cantErroresTr,
#     errorPorcentualTr,
#     cantErroresPrueba,
#     errorPorcentualPrueba,
# ] = perceptronMulticapa(x, yD, xP, yP, capas, 10, 0.0001, 0.05)
