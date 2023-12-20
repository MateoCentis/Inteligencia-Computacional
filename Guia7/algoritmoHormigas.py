import numpy as np
import random
import matplotlib.pyplot as plt

def probabilidadDeNodo(Distancias,noVisitados,feromonas,posicionActual,camino):
    nodoActual = camino[posicionActual]
    probabilidades = np.zeros(len(noVisitados))
    alpha = 1
    beta = 1
    suma = 0
    for u in range(len(noVisitados)):
        conocimientoAdicional = 1/Distancias[nodoActual,noVisitados[u]]##Deseo de la hormiga de moverse del nodo (se pesa con la distancia)
        suma += (feromonas[nodoActual,noVisitados[u]]**alpha)*(conocimientoAdicional**beta)
    # print(suma) 
    for j in range(len(noVisitados)):
        conocimientoAdicional = 1/Distancias[nodoActual,noVisitados[j]]##Deseo de la hormiga de moverse del nodo (se pesa con la distancia)
        probabilidades[j] = ((feromonas[nodoActual,noVisitados[j]]**alpha)*(conocimientoAdicional**beta))/suma 
    nodoSeleccionado = noVisitados[random.choices(range(len(probabilidades)),probabilidades)[0]]
    return nodoSeleccionado
    
        
def longitudCamino(camino, Distancias):
    distanciaTotal = 0
    for i in range(len(camino)-1): #sumo todas las conexiones
        j = (i+1)  
        distanciaTotal += Distancias[camino[i],camino[j]]
    return distanciaTotal


def algoritmoHormigas(Distancias,N,tasaEvaporacion,metodoDeposito,maxIteraciones,Q):
    ##Inicialización al azar de matriz feromonas
    feromonas = np.ones((Distancias.shape))
    sizeMatriz = len(Distancias[0,:])+1
    caminosHormigas = np.zeros((N,sizeMatriz), dtype=int)
    caminosHormigas[0,:] = np.ones((1,sizeMatriz))##pongo el primer camino en uno para que pase el while
    cantIteraciones = 0
    C = set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    while not(np.all(caminosHormigas[0] == caminosHormigas)) and cantIteraciones < maxIteraciones:##todas pasen por el mismo camino (compara vector de caminos a ver si son todos iguales)
        longitudCaminos = np.zeros(N)
        for k in range(N): 
            
            camino = np.zeros(sizeMatriz, dtype=int) ## camino vacio de la cant de nodos que hay 
            cont = 0 #1
            noVisitados = list(C-set(camino))
            
            while (noVisitados): ##mientras no vuelva al origen
                i = probabilidadDeNodo(Distancias,noVisitados,feromonas,cont,camino)
                camino[cont+1] = i #[cont]
                cont += 1
                noVisitados = list(C-set(camino))
            # camino = np.append(camino,0)
            # print(lencamino)
            caminosHormigas[k,:] = camino.copy()
            longitudCaminos[k] = longitudCamino(camino,Distancias).copy()
        
        feromonas = feromonas*(1-tasaEvaporacion) 
        deltaFeromonas = np.zeros(feromonas.shape)
        for k in range(N):
            camino = np.copy(caminosHormigas[k,:])
            for i in range(len(camino)-1):
                camI = camino[i].copy()
                camJ = camino[i+1].copy()
                if (metodoDeposito == 0):#global 
                    deltaFeromonas[camI,camJ] += Q/longitudCaminos[k]
                if (metodoDeposito == 1):#uniforme
                    deltaFeromonas[camI,camJ] += Q
                if (metodoDeposito == 2):#local
                    deltaFeromonas[camI,camJ] += Q / Distancias[camI,camJ]
        feromonas += deltaFeromonas
        cantIteraciones += 1
        if cantIteraciones % 10 == 0:
            print("Iteración: ", cantIteraciones)
            print(caminosHormigas)
    indiceMejorCamino = np.argmin(longitudCaminos).copy()
    mejorCamino = caminosHormigas[indiceMejorCamino,:].copy()
    plt.imshow(feromonas)
    plt.show()
    return longitudCamino(mejorCamino, Distancias), mejorCamino
            