#Ejercicio 2 => K-medias:
import numpy as np
def k_medias(x,k):
    #---------------------------------------------Inicialización--------------------------------------------
    #1. Inicializarlo al azar (asigno valores cualquiera a los centroides)
    #2. Asignar grupos al azar (vector que dice cada patrón a que centroide va y ahí armo centroides)(no valido para simétricos)
    #3. Otra forma es agarrar patrones al azar (si se puede que este lejos) y esos los uso como centroide 
    #Opción 1:
#---------------------------------------------Algoritmo-------------------------------------------------
    # centroides = np.random.uniform(-0.5, 0.5, (k,len(x[0])))
    centroides = np.zeros((k,len(x[0])))
    sizeX = len(x)
    for i in range(k):
        centroides[i,:] = x[np.random.randint(0,sizeX)]
    bandera = True
    centroidesPertenece = np.zeros(len(x))
    distancias = np.zeros(len(centroides))
    while (bandera != False):
        #-REASIGNAR PATRONES:
        patronesPorCentroide = np.zeros(len(centroides))
        #Paso por todos los patrones y decido cual es el centroide más cercano a ese patron, en base a eso defino su grupo
        cont = 0
        for n in range(len(x)):
            for i in range(len(centroides)):
                distancias[i] = np.sum((x[n] - centroides[i]) ** 2)
            # disMenor = np.argmin(distancias)
            indices = np.argmin(distancias)
            iMin = indices
            if (centroidesPertenece[n] == iMin):
                cont += 1
            centroidesPertenece[n] = iMin #Asigno al patron n el centroide de indice i
            patronesPorCentroide[iMin] += 1
        #-REASIGNO CENTROIDES:
        #Una vez tengo definidos todos los patrones calculo los centroides de cada grupo con el promedio
        if cont == len(x):#ninguno cambió
            bandera = False
        else:
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