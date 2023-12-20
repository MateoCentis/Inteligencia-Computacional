import numpy as np
from leer_archivo import leer_archivo
#Carga del archivo csv 
#---------------------------------------------Datos para entrenamiento-------------------------------------------------
[x,yD] = leer_archivo('Guia1/OR_trn.csv',2)


x = np.insert(x, 0, -1, axis=1) #meto una columna de -1's
epoca = 0
epocaMaxima = 100
cantErrores = 0
velocidadAprendizaje = 0.001
errorPorcentual = 0
criterioParada = 0
erroresPorEpoca = np.zeros(epocaMaxima)
w = np.random.uniform(-0.5, 0.5, len(x[1])) #Es por patron

while (epoca < epocaMaxima):
    #Cálculo de pesos por patrón----------------------------
    for n in range(len(x)):
        y = np.sign(np.dot(x[n],w))
        error = yD[n] - y
        w = w + (velocidadAprendizaje*error)*x[n]
    #Cálculo de error de época------------------------------
    cantErrores = 0
    for n in range(len(x)):
        y = np.sign(np.dot(x[n],w))
        error = yD[n] - y
        if (error != 0):
            cantErrores += 1
    errorPorcentual = cantErrores/len(x)
    erroresPorEpoca[epoca] = errorPorcentual
    if(errorPorcentual < criterioParada):#criterio de parada
        break
    #hacer sort random de los datos para que el orden cambie
    epoca += 1


print("Cantidad de épocas: ",epoca)
print("Cantidad de errores entrenamiento: ", cantErrores)
print("Error porcentual entrenamiento: ", errorPorcentual)

#---------------------------------------------Datos de prueba------------------------------------------------- 
[xP,yP] = leer_archivo('Guia1/OR_tst.csv',2)

xP = np.insert(xP, 0, -1, axis=1) #meto una columna de -1's
cantErroresPrueba = 0
for n in range(len(xP)):
    y = np.sign(np.dot(xP[n],w))
    error = yP[n] - y
    if (error != 0):
        cantErroresPrueba += 1

errorPorcentualPrueba = cantErroresPrueba/len(xP)

print("Cantidad de errores de testeo: ", cantErroresPrueba)
print("Error porcentual testeo: ", errorPorcentualPrueba)


#Lógica entrenamiento:
#mientras epoca < epocaMaxima
    #para cada patron
        #Paso patron
        #calculo error
        #corrijo pesos
        #paso al siguiente patron
    #evaluo criterio de parada =>  se pasan de vuelta todos los patrones 
                                    #sin actualizar pesos y se cuentan los errores
        #si no es lo suficientemente bueno corto

#Lógica prueba:
    #para cada patron
        #pasarlo por la neurona
        #evaluar desempeño (error)
    #retportar desempeño (ej: error promedio)
