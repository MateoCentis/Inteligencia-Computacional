import numpy as np
from algoritmoGenetico import algoritmoGenetico
from sklearn import svm #Maquina de vectores
from sklearn.metrics import balanced_accuracy_score
from leer_archivo import leer_archivo
[x_train,y_train] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia6/leukemia_train.csv', 7129,1)
[x_test,y_test] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia6/leukemia_test.csv', 7129,1)


def funcionAptitud(x):
    clasificador = svm.SVC(C=5,kernel='rbf')
    clasificador.fit(x_train[:,x == 1], np.ravel(y_train))

    yPredecida = clasificador.predict(x_test[:,x == 1])
    accuracyBalanceada = balanced_accuracy_score(np.ravel(y_test), yPredecida)
    alpha = 0.7
    beta = 0.3
    parametrosIndividuo = sum(x)
    totalParametros = len(x)
    return alpha*accuracyBalanceada - beta*(parametrosIndividuo/totalParametros)

sizeIndividuo = 7129
N = 100
generacionesCortar = 10000
paciencia = 50
genotipo = True
mejorIndividuo, historico = algoritmoGenetico(sizeIndividuo,N,funcionAptitud,generacionesCortar,0,0,paciencia,genotipo)

clasificador =  svm.SVC(C=5,kernel='rbf')
clasificador.fit(x_train[:, mejorIndividuo == 1], np.ravel(y_train))
yPredecida = clasificador.predict(x_test[:, mejorIndividuo == 1])
accuracyBalanceada = balanced_accuracy_score(np.ravel(y_test), yPredecida)
print("Accuracy balanceada", accuracyBalanceada)
print("Cantidad total de parámetros", len(mejorIndividuo))
print("Cantidad restante de parámetros", sum(mejorIndividuo))
