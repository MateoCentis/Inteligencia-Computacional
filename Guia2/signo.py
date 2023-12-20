import numpy as np


def signo(x):
    aux = 1000
    if isinstance(x,np.ndarray):
        aux = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] >= 0:
                aux[i] = 1  
            else:
                aux[i] = -1
    else:
        if x >= 0:
            aux = 1  
        else:
            aux = -1
    return aux
