import numpy as np
import matplotlib.pyplot as plt

def distancia_hamming(matriz1, matriz2):
    return np.sum(matriz1 != matriz2)


def plot_neuronas(pesosPorNeurona, iteracion):
    plt.clf()
    plt.title('Vecindad para la iteración ' + str(iteracion))
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

    # Graficar uniones entre nodos vecinos
    for i in range(len(pesosPorNeurona)):
        for j in range(len(pesosPorNeurona[0])):
            if i > 0:
                plt.plot([pesosPorNeurona[i, j][0], pesosPorNeurona[i - 1, j][0]],
                         [pesosPorNeurona[i, j][1], pesosPorNeurona[i - 1, j][1]], 'r-', linewidth=1)
            if j > 0:
                plt.plot([pesosPorNeurona[i, j][0], pesosPorNeurona[i, j - 1][0]],
                         [pesosPorNeurona[i, j][1], pesosPorNeurona[i, j - 1][1]], 'r-', linewidth=1)

    # Graficar nodos
    for i in range(len(pesosPorNeurona)):
        for j in range(len(pesosPorNeurona[0])):
            plt.plot(pesosPorNeurona[i, j][0], pesosPorNeurona[i, j][1], 'ko')

    plt.show()

def SOM(x, neuronas, epocaMax):
    epoca = 0
    radioVecindad = 0
    sizeEntrada = len(x[0])
    neuronasX = neuronas[0]
    neuronasY = neuronas[1]
    #-------------------------------------Incialización de pesos w-----------------------------------------
    #neuronas: matriz que en cada celda tiene un vector de pesos
    pesosPorNeurona = np.empty((neuronasX,neuronasY),dtype=object)
    dimensionMayor = max(neuronasX,neuronasY) #Resuelve caso bidimensional o unidimensional
    for i in range(neuronasX):
        for j in range(neuronasY):
            pesos = np.random.uniform(-0.5, 0.5, [sizeEntrada]) #vector j con los pesos
            pesosPorNeurona[i,j] = pesos

    cambioEtapaA = epocaMax/4
    cambioEtapaB = epocaMax*0.7
    neuronaPertenece = np.zeros(len(x))
    while (epoca < epocaMax):
        #Etapa ordenamiento global (1)
        if (epoca < cambioEtapaA):
            radioVecindad = int(dimensionMayor/2)
            velocidadAprendizaje = 0.75
        #Etapa de transición (2)
        if (epoca < cambioEtapaB and epoca >= cambioEtapaA):#Decrecen linealmente
            cociente = (cambioEtapaB-epoca)/(cambioEtapaB-cambioEtapaA)
            radioVecindad = int(4*cociente+1)
            velocidadAprendizaje = 0.6*cociente+0.1
        #Etapa de ajuste fino (3)
        if epoca >= epocaMax*0.7:
            radioVecindad = 0
            velocidadAprendizaje = 0.01
        for n in range (len(x)): 
            #-----------------------------Cálculo de distancia---------------------------------
            distancias = np.zeros((neuronasX,neuronasY))
            for i in range(neuronasX):
                for j in range(neuronasY):
                    distancias[i,j] = np.sum((x[n] - pesosPorNeurona[i,j])**2)#distancia euclidea al cuadrado
            disMenor = np.min(distancias)
            indices = np.where(distancias == disMenor)
            iMin = indices[0][0]
            jMin = indices[1][0]
            neuronaPertenece[n] = iMin*neuronasX + jMin #acá guardamos el numero de neurona para luego comparar con centroides
            #------------------------------Ajuste de pesos-----------------------------------------
            for i in range(iMin - radioVecindad, iMin + radioVecindad + 1):
                for j in range(jMin - radioVecindad, jMin + radioVecindad + 1):
                    if (
                        abs(i - iMin) + abs(j - jMin) <= radioVecindad
                        and i >= 0 and i < neuronasX
                        and j >= 0 and j < neuronasY
                    ):
                        pesosPorNeurona[i,j] = pesosPorNeurona[i,j] + velocidadAprendizaje * (x[n] - pesosPorNeurona[i,j])
        if epoca % 50 == 0:
            print("Época: ", epoca)
        if epoca % 333 == 0:
            plot_neuronas(pesosPorNeurona,epoca)
        epoca += 1
    return neuronasPertenece



