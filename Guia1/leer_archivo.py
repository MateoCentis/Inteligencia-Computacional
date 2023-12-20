import numpy as np
#cantidad de columnas es cantidad de comas (en este ejemplo serían 2)
def leer_archivo(nombre,cantColumnas):
    data = np.loadtxt(nombre,delimiter=",")
    x = np.array(data[:,0:cantColumnas])
    y = np.array(data[:,cantColumnas])
    y = np.transpose(y)
    return [x,y]
    #x da una matriz donde cada fila va a ser la entrada i-ésima
    #y va a ser un vector con las salidas