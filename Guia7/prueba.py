def longitudCamino(camino, Distancias):
    distanciaTotal = 0
    for i in range(len(camino)): #sumo todas las conexiones
        j = (i+1) % len(camino)
        distanciaTotal += Distancias[camino[i],camino[j]]
    return distanciaTotal

def longitudCamino()