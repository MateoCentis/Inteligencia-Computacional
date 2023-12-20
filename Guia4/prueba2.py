#Ejercicio 2 => K-medias:
import numpy as np
def k_medias(x,k):
    #---------------------------------------------Inicialización--------------------------------------------
    #1. Inicializarlo al azar (asigno valores cualquiera a los centroides)
    #2. Asignar grupos al azar (vector que dice cada patrón a que centroide va y ahí armo centroides)(no valido para simétricos)
    #3. Otra forma es agarrar patrones al azar (si se puede que este lejos) y esos los uso como centroide 
    #Opción 1:
    centroides = np.random.uniform(-2, 2, (k,len(x[0])))
#---------------------------------------------Algoritmo-------------------------------------------------
    centroidesPerteneceAnterior = np.zeros(len(x))
    bandera = True
    centroidesPertenece = np.zeros(len(x))
    while (bandera != False):
        #-REASIGNAR PATRONES:
        patronesPorCentroide = np.zeros(len(centroides))
        distancias = np.zeros(len(centroides))
        #Paso por todos los patrones y decido cual es el centroide más cercano a ese patron, en base a eso defino su grupo
        for n in range(len(x)):
            for i in range(len(centroides)):
                distancias[i] = np.sum((x[n] - centroides[i]) ** 2)
            disMenor = np.min(distancias)
            indices = np.where(distancias == disMenor)
            if len(indices[0]) > 0:
                iMin = indices[0][0]
            else:
                iMin = -1
            centroidesPertenece[n] = iMin #Asigno al patron n el centroide de indice i
            patronesPorCentroide[iMin] += 1
        #-REASIGNO CENTROIDES:
        #Una vez tengo definidos todos los patrones calculo los centroides de cada grupo con el promedio
        if np.array_equal(centroidesPertenece,centroidesPerteneceAnterior):
            bandera = False
        else:
            # print(centroidesPertenece)
            centroidesPerteneceAnterior = np.copy(centroidesPertenece)
            #Cuando ningun patrón cambie de grupo termina la iteración
            centroidesAux = np.zeros((len(x),len(x[0])))
            for n in range(len(x)):
                indiceCentroide = int(centroidesPertenece[n])
                centroidesAux[indiceCentroide,:] +=  x[n]
            for i in range(len(centroides)):
                cantPatrones = patronesPorCentroide[i]    
                if cantPatrones != 0:
                    centroides[i,:] = centroidesAux[i,:]/cantPatrones

    return [x , centroidesPertenece, centroides] 