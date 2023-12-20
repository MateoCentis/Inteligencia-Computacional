import numpy as np
import matplotlib.pyplot as plt

def distancia_euclidiana(matriz1, matriz2):
    return np.sum((matriz1 - matriz2)**2)

def plot_neuronas(pesosPorNeurona, iteracion):
    plt.clf()
    plt.title('Vecindad para la iteración ' + str(iteracion))
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    
    # Graficar nodos
    for i in range(len(pesosPorNeurona)):
        for j in range(len(pesosPorNeurona[0])):
            plt.plot(pesosPorNeurona[i, j, 0], pesosPorNeurona[i, j, 1], 'ko')

    # Graficar uniones entre nodos vecinos
    for i in range(len(pesosPorNeurona)):
        for j in range(len(pesosPorNeurona[0])):
            if i > 0:
                plt.plot([pesosPorNeurona[i, j, 0], pesosPorNeurona[i - 1, j, 0]],
                         [pesosPorNeurona[i, j, 1], pesosPorNeurona[i - 1, j, 1]], 'r-', linewidth=1)
            if j > 0:
                plt.plot([pesosPorNeurona[i, j, 0], pesosPorNeurona[i, j - 1, 0]],
                         [pesosPorNeurona[i, j, 1], pesosPorNeurona[i, j - 1, 1]], 'r-', linewidth=1)

    plt.show()

def SOM(x, neuronas, epocaMax, graficar):
    epoca = 0
    radioVecindad = 0
    sizeEntrada = len(x[0])
    neuronasX, neuronasY = neuronas
    
    # Inicialización de pesos w
    pesosPorNeurona = np.random.uniform(-0.5, 0.5, (neuronasX, neuronasY, sizeEntrada))
    
    cambioEtapaA = epocaMax / 4
    cambioEtapaB = epocaMax * 0.7
    neuronaPertenece = np.zeros(len(x))
    while epoca < epocaMax:
        if epoca < cambioEtapaA:
            radioVecindad = int(max(neuronasX, neuronasY) / 2)
            velocidadAprendizaje = 0.8
        elif cambioEtapaA <= epoca < cambioEtapaB:
            radioVecindad = int(4 * (cambioEtapaB - epoca) / (cambioEtapaB - cambioEtapaA) + 1)
            velocidadAprendizaje = 0.6 * (cambioEtapaB - epoca) / (cambioEtapaB - cambioEtapaA) + 0.1
        else:
            radioVecindad = 0
            velocidadAprendizaje = 0.1
        
        for n in range(len(x)):
            # Cálculo de distancia
            distancias = np.sum((x[n] - pesosPorNeurona) ** 2, axis=2)
            iMin, jMin = np.unravel_index(np.argmin(distancias), distancias.shape)
            neuronaPertenece[n] = iMin*neuronasX + jMin
            # Ajuste de pesos
            i_ini = max(0, iMin - radioVecindad)
            i_fin = min(neuronasX, iMin + radioVecindad + 1)
            j_ini = max(0, jMin - radioVecindad)
            j_fin = min(neuronasY, jMin + radioVecindad + 1)
            
            delta_pesos = velocidadAprendizaje * (x[n] - pesosPorNeurona[i_ini:i_fin, j_ini:j_fin])
            pesosPorNeurona[i_ini:i_fin, j_ini:j_fin] += delta_pesos
            
        if epoca % 50 == 0:
            print("Época: ", epoca)
        if epoca % 333 == 0 and graficar == True:
            plt.close()
            plot_neuronas(pesosPorNeurona,epoca)
        plt.show()
        epoca += 1
    return neuronaPertenece
