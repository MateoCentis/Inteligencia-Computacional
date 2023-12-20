import numpy as np
from leer_archivo import leer_archivo
from perceptron import perceptron
[x50,yD50] = leer_archivo('Guia1/OR_50_trn.csv',2)
[x50P,yD50P] = leer_archivo('Guia1/OR_50_tst.csv',2)

                                                                                    
# [cantEpocasTr,cantErroresTr,errorPorcentualTr,cantErroresPrueba,errorPorcentualPrueba] = perceptron(x50,yD50,x50P,yD50P,20,0.1,0.001,0)

[x90,yD90] = leer_archivo('Guia1/OR_90_trn.csv',2)
[x90P,yD90P] = leer_archivo('Guia1/OR_90_tst.csv',2)

[cantEpocasTr,cantErroresTr,errorPorcentualTr,cantErroresPrueba,errorPorcentualPrueba] = perceptron(x90,yD90,x90P,yD90P,20,0.01,0.1,1)