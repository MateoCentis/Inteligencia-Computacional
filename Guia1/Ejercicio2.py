import numpy as np
from perceptron import perceptron   
from leer_archivo import leer_archivo
#Recta: x_2= w0/w2 - w_1/w_2 x_1

#---------------------------------------------OR-------------------------------------------------
[x,yD] = leer_archivo('Guia1/OR_trn.csv',2)
[xP,yP] = leer_archivo('Guia1/OR_tst.csv',2)
[cantEpocasTr,cantErroresTr,errorPorcentualTr,cantErroresPrueba,errorPorcentualPrueba] = perceptron(x,yD,xP,yP,20,0.01,0.2,1)

#---------------------------------------------XOR-------------------------------------------------
[x,yD] = leer_archivo('Guia1/XOR_trn.csv',2)
[xP,yP] = leer_archivo('Guia1/XOR_tst.csv',2)
# [cantEpocasTr,cantErroresTr,errorPorcentualTr,cantErroresPrueba,errorPorcentualPrueba] = perceptron(x,yD,xP,yP,100,0.01,0.50,1)