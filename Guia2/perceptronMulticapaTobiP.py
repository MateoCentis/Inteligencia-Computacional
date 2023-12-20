import numpy as np
from leer_archivo import leer_archivo
from sigmoid import sigmoid
from signo import signo
import matplotlib.pyplot as plt

def calcularError(yD,Y):
    cantErrores = 0
    if (len(Y) == 1):
        error = yD - signo(Y)
        if np.any(error != 0):
            cantErrores = 1
    else:#Se elige al mayor como la salida del clasificador
        indiceMayor = np.argmax(Y)  #miro los dos indices máximos y espero que coincidan
        indiceMayorEsperada = np.argmax(yD)
        if indiceMayor != indiceMayorEsperada:
            cantErrores = 1
        # if not((yD == signo(Y)).all()):
        #     cantErrores = 1
    return cantErrores



# [x,yD] = leer_archivo('Guia1/OR_trn.csv',2)
def perceptronMulticapa(x, yD, xP, yDP, capas, epocaMaxima, velocidadAprendizaje, criterioError):
    x = np.insert(x, 0, -1, axis=1)
    cantCapas = len(capas)
    constanteSigmoid = 1
    W = np.empty(cantCapas, dtype=object)
    epoca = 0
    cantErrores = 0
    errorPorcentual = 0
    erroresPorEpoca = np.zeros(epocaMaxima)
    errorCuadraticoPorEpoca = np.zeros(epocaMaxima)
    valor = 0 #Bandera para graficar el ejercicio 2
    seguir = 1  # en vez del break
    # -----------------------------------------Cálculo de pesos-------------------------------------------------
    sizeEntrada = len(x[0])
    for c in range(len(capas)):
        if c == 0:  # se inicializa W para todos los patrones
            W[c] = np.random.uniform(
                -0.5, 0.5, [capas[c], sizeEntrada]
            )  # caso de primer matriz
        else:
            W[c] = np.random.uniform(-0.5, 0.5, [capas[c], capas[c - 1] + 1])#El +1 es por la entrada x_0
    if valor == 1:
        plt.ion()
        figura = plt.figure()
        plt.xlim(0,1)
        plt.ylim(0,1)
    while (epoca < epocaMaxima and seguir == 1):  # Por época
        for n in range(len(x)):  # Por patrón
            Y = []
            # --------------------------------Propagación hacia delante-----------------------------------------
            entradas = x[n].reshape(-1,1)  # COLUMNA
            for i in range(cantCapas):  # Por capa
                salidaLineal = W[i] @ entradas #vector columna
                y = sigmoid(salidaLineal, constanteSigmoid)
                y = y.flatten().reshape(-1,1)
                Y.append(y)#El primer x[n] ya tiene el -1
                # pongo como entrada de la capa siguiente la salida y le agrego la columna de -1's
                entradas = Y[-1]
                entradas = np.insert(entradas, 0, -1).reshape(-1,1)#Se le pone el -1 a cada salida
            # Una vez calculadas
            e_j = yD[n].reshape(-1,1) - Y[-1] #Compara contra la salida final y[-1]
            deltas = []
            # ------------------------------------Propagación hacia atrás------------------------------------------
            for i in range(cantCapas - 1, -1, -1):
                derivadaFunActivacion = ((1 / 2) * (1 + Y[i]) * (1 - Y[i]))
                if (i == cantCapas-1):
                    deltas.insert(0,derivadaFunActivacion * e_j) #El ultimo delta es
                else:
                    #w: pesos transpuestos sin el -1
                    pesos = W[i+1]  # Obtenemos los pesos de la capa siguiente
                    w = np.transpose(pesos[:, 1:]) 
                    a = (w @ deltas[0])
                    deltas.insert(0,derivadaFunActivacion * a)
            # ------------------------------------Actualización de pesos-------------------------------------------------
            for i in range(cantCapas):
                delta_w = np.zeros_like(W[i])
                if i == 0:
                    entrada0 = x[n].reshape(-1,1)  #reshape(1,deltas[i].shape[0])
                    delta_w = velocidadAprendizaje * ( deltas[i] @ entrada0.T)
                else:
                    #Acá se agrega el -1 a las salidas ya que actúan como entradas
                    salidaSesgada = np.insert(Y[i-1], 0, -1).flatten().reshape(-1,1).T
                    delta_w = velocidadAprendizaje * (deltas[i] @ salidaSesgada)
                W[i] = W[i] + delta_w
        # -------------------------------------Ver error y criterio de parada-------------------------------------------------
        cantErrores = 0
        errorCuadratico = 0
        salida = np.zeros(len(x))
        for k in range(len(x)):
            Y = []
            entradas = x[k].reshape(-1,1)  # COLUMNA
            for i in range(cantCapas):  # Por capa
                salidaLineal = W[i] @ entradas #vector columna
                y = sigmoid(salidaLineal, constanteSigmoid)
                y = y.flatten().reshape(-1,1)
                Y.append(y)#El primer x[n] ya tiene el -1
                # pongo como entrada de la capa siguiente la salida y le agrego la columna de -1's
                entradas = Y[-1]
                entradas = np.insert(entradas, 0, -1).reshape(-1,1)#Se le pone el -1 a cada salida
            if (valor == 1):
                salida[k] = Y[-1]
            errorAux = calcularError(yD[k],Y[-1])
            cantErrores += errorAux
            errorCuadratico += np.sum((yD[k].reshape(-1,1) - Y[-1])**2)/len(yD[k])
        erroresPorEpoca[epoca] = cantErrores
        errorCuadraticoPorEpoca[epoca] = errorCuadratico
        errorPorcentual = cantErrores / len(x)
        if (epoca % 50 == 0):
            print("Época ", epoca, "| Cantidad errores: ", cantErrores)
        # if( epoca % 20 == 0 and valor == 1 and epoca < 500 and epoca > 200):
        if( epoca % 50 == 0 and valor == 1):
            x1 = x[:,1]
            x2 = x[:,2]
            figura.clear()
            for g in range(0,len(salida)):
                if (salida[g] > 0):
                    plt.scatter(x1[g],x2[g],color='green',marker='o')
                else:
                    plt.scatter(x1[g],x2[g],color='red',marker='o')
            plt.pause(0.1)
        if errorPorcentual < criterioError:  # criterio de parada
            seguir = 0
        epoca += 1    
    plt.ioff()
    print("Cantidad de épocas: ", epoca)
    cantEpocasTr = epoca
    print("Cantidad de errores entrenamiento: ", cantErrores)
    cantErroresTr = cantErrores
    print("Error porcentual entrenamiento: ", errorPorcentual)
    errorPorcentualTr = errorPorcentual
    epocas = np.arange(1,epoca+1)
    fig1 = plt.figure()
    plt.plot(epocas,errorCuadraticoPorEpoca[0:epoca])
    plt.title('Error cuadrático medio por época')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático')
    plt.grid()
    fig2 = plt.figure()
    plt.plot(epocas,erroresPorEpoca[0:epoca])
    plt.title('Cantidad errores por época')
    plt.xlabel('Épocas')
    plt.ylabel('Cantidad de errores')
    plt.grid()
    plt.show()
    
    # ---------------------------------------------Datos de prueba-------------------------------------------------
    xP = np.insert(xP, 0, -1, axis=1)  # meto una columna de -1's
    cantErroresPrueba = 0
    errorCuadraticoPrueba = 0
    for k in range(len(xP)):
        Y = []
        entradas = xP[k].reshape(-1,1)  # COLUMNA
        for i in range(cantCapas):  # Por capa
            salidaLineal = W[i] @ entradas #vector columna
            y = sigmoid(salidaLineal, constanteSigmoid)
            y = y.flatten().reshape(-1,1)
            Y.append(y)#El primer x[n] ya tiene el -1
            entradas = Y[-1]
            entradas = np.insert(entradas, 0, -1).reshape(-1,1)#Se le pone el -1 a cada salida
        errorAux = calcularError(yDP[k],Y[-1])
        cantErroresPrueba += errorAux
    errorPorcentualPrueba = cantErroresPrueba / len(xP)
    print("------------------TESTEO--------------------")   
    print("Cantidad de errores de testeo: ", cantErroresPrueba)
    print("Error porcentual testeo: ", errorPorcentualPrueba)
    return [
        cantEpocasTr,
        cantErroresTr,
        errorPorcentualTr,
        cantErroresPrueba,
        errorPorcentualPrueba,
    ]

#EJEMPLO PLOT CUADRÁTICO
# cantidad = 50
# rango = 2
# variacionesAlpha1 = variacionCoeficientes(alpha1, cantidad, rango)
# variacionesAlpha3 = variacionCoeficientes(alpha3, cantidad, rango)

# fig = plt.figure(figsize = (12,10))
# ax = plt.axes(projection='3d')

# X, Y = np.meshgrid(variacionesAlpha1, variacionesAlpha3)

# Z = ArmoMatrizECM(t,X,Y)
# surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# # Set axes label
# ax.set_xlabel('alpha1', labelpad=20)
# ax.set_ylabel('alpha3', labelpad=20)
# ax.set_zlabel('Error Cuadratico total', labelpad=10)

# fig.colorbar(surf, shrink=0.5, aspect=8)

# plt.show()