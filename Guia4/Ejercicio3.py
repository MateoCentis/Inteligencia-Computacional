import numpy as np
import matplotlib.pyplot as plt
from leer_archivo import leer_archivo
from k_medias import k_medias
from somMejorado import SOM
[x,_] = leer_archivo('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia4/irisbin_trn.csv', 4,3)
# definir uan métrica
# iterar entre varios k
# graficar la métrica para esos k
# analizar el codo de la gráfica

#--cosas a tener en cuenta:
    #  la inicialización cambiar o probar otra 
    #  Probar con otra medida (mixta o cohesión)

Sem = np.zeros(9)
for k in range(2,11):
  [patrones,centroidesPertenece,centroides] = k_medias(x,k)

  #Cálculo de la métrica (separación)
  suma = 0
  for i in range(k):
    for j in range(k):
        if i != j:
            suma += np.linalg.norm(centroides[i,:]-centroides[j,:])
    
  Sem[k-2] = (2/(k**2-k))*suma

aux = [2,3,4,5,6,7,8,9,10]
plt.plot(aux,Sem,'b-o')
plt.grid(True)
plt.xlabel('k')
plt.ylabel('Separación')
plt.title('Búsqueda del k óptimo')
plt.show()