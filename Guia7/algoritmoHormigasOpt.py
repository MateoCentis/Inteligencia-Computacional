import numpy as np
import random
import matplotlib.pyplot as plt

def probabilidadDeNodo(Distancias, camino, feromonas, posicionActual):
    nodoActual = int(camino[posicionActual])
    probabilidades = np.zeros(len(Distancias[0,:]))
    alpha = 2
    beta = 2

    nodos_no_visitados = np.where(~np.isin(np.arange(len(probabilidades)), camino))[0]

    conocimientoAdicional = 1 / Distancias[nodoActual, nodos_no_visitados]
    suma = np.sum((feromonas[nodoActual, nodos_no_visitados]**alpha) * (conocimientoAdicional**beta))
    probabilidades[nodos_no_visitados] = ((feromonas[nodoActual, nodos_no_visitados]**alpha) * (conocimientoAdicional**beta)) / suma

    nodoSeleccionado = random.choices(nodos_no_visitados, probabilidades[nodos_no_visitados])[0]

    return nodoSeleccionado

def longitudCamino(camino, Distancias):
    indices = np.arange(len(camino))
    suma = np.sum(Distancias[ camino[indices[:-1]] , camino[indices[1:]] ])
    distanciaTotal = suma + Distancias[camino[-1], camino[0]]
    return distanciaTotal

def algoritmoHormigasOpt(Distancias, N, tasaEvaporacion, metodoDeposito, maxIteraciones):
    feromonas = np.ones(Distancias.shape)
    sizeMatriz = len(Distancias[0, :])
    caminosHormigas = np.zeros((N, sizeMatriz), dtype=int)
    caminosHormigas[0, :] = np.arange(sizeMatriz)
    cantIteraciones = 0

    while not np.all(caminosHormigas[0] == caminosHormigas) and cantIteraciones < maxIteraciones:
        longitudCaminos = np.zeros(N)
        cantidadPorNodo = np.zeros(sizeMatriz)
        
        for k in range(N):
            camino = np.zeros(sizeMatriz, dtype=int)
            cont = 1
            while cont < len(camino) - 1:
                i = probabilidadDeNodo(Distancias, camino, feromonas, cont)
                camino[cont] = i
                cantidadPorNodo[i] += 1
                cont += 1
            caminosHormigas[k, :] = camino
            longitudCaminos[k] = longitudCamino(camino, Distancias)
        
        feromonas *= (1 - tasaEvaporacion)
        deltaFeromonas = np.zeros(feromonas.shape)
        
        for i in range(sizeMatriz):
            for j in range(i + 1, sizeMatriz):
                nodos_no_visitados = np.where(~np.isin(np.arange(sizeMatriz), caminosHormigas[:, [i, j]]))
                if metodoDeposito == 0: #0: global
                    deltaFeromonas[i, j] = np.sum(cantidadPorNodo[i] / longitudCaminos)
                    deltaFeromonas[j, i] = deltaFeromonas[i, j]
                elif metodoDeposito == 1: #1: uniforme
                    deltaFeromonas[i, j] = np.sum(cantidadPorNodo[i])
                    deltaFeromonas[j, i] = deltaFeromonas[i, j]
                elif metodoDeposito == 2: #local
                    deltaFeromonas[i, j] = np.sum(cantidadPorNodo[i] / Distancias[i, j])
                    deltaFeromonas[j, i] = deltaFeromonas[i, j]
        
        feromonas += deltaFeromonas
        cantIteraciones += 1
        print("Iteración:", cantIteraciones)
    indiceMejorCamino = np.argmin(longitudCaminos)
    mejorCamino = caminosHormigas[indiceMejorCamino,:]
    plt.imshow(feromonas)
    plt.show()
    return longitudCamino(mejorCamino, Distancias)
