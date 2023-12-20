from signo import signo
from perceptronMulticapaTobiP import calcularError
import numpy as np
x = np.array([-1,1,-1])
y = np.array([0.4,0.5,0.5])
print(calcularError(x,y))