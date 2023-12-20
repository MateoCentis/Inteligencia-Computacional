import numpy as np
import matplotlib.pyplot as plt
from signo import signo
def perceptron(x,yD,xP,yP,epocaMaxima,velocidadAprendizaje,criterioParada,graficaRecta):
    if(graficaRecta != 0):
        w = np.random.uniform(-0.5, 0.5, 3) #Es por patron
        plt.ion()
        figure, ax = plt.subplots(figsize=(10, 8))
        x1 = np.linspace(-2,2,50)
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        x2 = (w[0]/w[2]) - (w[1]/w[2])*x1
        line1, = ax.plot(x1,x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.scatter(x[:,0],x[:,1],30,[f"C{int(i+1)}" for i in yD])
        plt.title("Evolución recta",fontsize=20)
    x = np.insert(x, 0, -1, axis=1) #meto una columna de -1's
    w = np.random.uniform(-0.5, 0.5, len(x[1])) #Es por patron
    epoca = 0
    cantErrores = 0
    errorPorcentual = 0
    erroresPorEpoca = np.zeros(epocaMaxima)
    
    while (epoca < epocaMaxima):
        #Cálculo de pesos por patrón----------------------------
        for n in range(len(x)):
            y = signo(np.dot(x[n],w))
            error = yD[n] - y
            w = w + (velocidadAprendizaje*error)*x[n]
            if (graficaRecta != 0):
                x2 = (w[0]/w[2]) - (w[1]/w[2])*x1       
                line1.set_xdata(x1)
                line1.set_ydata(x2)
                # re-drawing the figure
                figure.canvas.draw()
                # to flush the GUI events
                figure.canvas.flush_events()
                # time.sleep(0.0001)
        #Cálculo de error de época------------------------------
        cantErrores = 0
        for n in range(len(x)):
            y = signo(np.dot(x[n],w))
            error = yD[n] - y
            if (error != 0):
                cantErrores += 1
        errorPorcentual = cantErrores/len(x)
        erroresPorEpoca[epoca] = errorPorcentual
        if(errorPorcentual < criterioParada):#criterio de parada
            break #ojo con el break
        #hacer sort random de los datos para que el orden cambie
        epoca += 1
    print("Cantidad de épocas: ",epoca)
    cantEpocasTr = epoca
    print("Cantidad de errores entrenamiento: ", cantErrores)
    cantErroresTr = cantErrores
    print("Error porcentual entrenamiento: ", errorPorcentual)
    errorPorcentualTr = errorPorcentual

    #---------------------------------------------Datos de prueba------------------------------------------------- 
    xP = np.insert(xP, 0, -1, axis=1) #meto una columna de -1's
    cantErroresPrueba = 0
    for n in range(len(xP)):
        y = signo(np.dot(xP[n],w))
        error = yP[n] - y
        if (error != 0):
            cantErroresPrueba += 1

    errorPorcentualPrueba = cantErroresPrueba/len(xP)

    print("Cantidad de errores de testeo: ", cantErroresPrueba)
    print("Error porcentual testeo: ", errorPorcentualPrueba)
    return [cantEpocasTr,cantErroresTr,errorPorcentualTr,cantErroresPrueba,errorPorcentualPrueba]

