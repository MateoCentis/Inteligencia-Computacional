import numpy as np
import matplotlib.pyplot as plt
from algoritmoEnjambre import algoritmoEnjambre
from algoritmoGenetico import algoritmoGenetico 
from algoritmoGenetico import binarioADecimal
def f1(x):
    ret = 0
    if np.any(x == np.inf):
        ret = np.inf
    else:
        ret = -x*np.sin(np.sqrt(abs(x)))
    return ret

def f1G(x):
    return -1*(-x*np.sin(np.sqrt(abs(x))))

#---------------------------------------------Iniciso a)-------------------------------------------------
cantParticulas = 20
sizeParticula = 1
sizeIndividuo = 20
xMin1 = -512
xMax1 = 512
cantIteraciones = 100
paciencia = 5
mejorIndividuo1, historico1 = algoritmoEnjambre(cantParticulas,sizeParticula,xMin1,xMax1,f1,cantIteraciones,paciencia)
mejorIndividuoG1, historicoG1 = algoritmoGenetico(sizeIndividuo,cantParticulas,f1G,cantIteraciones,xMin1,xMax1,paciencia,False)

historico1 = np.array(historico1)
historicoG1 = np.array(historicoG1)
mejorIndividuoG1 = binarioADecimal(mejorIndividuoG1,xMin1,xMax1)

print("Mejor individuo enjambre: ",mejorIndividuo1)
print("Mejor individuo genético: ",mejorIndividuoG1)
##Gráficos
fig,ax = plt.subplots()

#Esto es para graficar el punto encima de la función
x1 = np.arange(xMin1, xMax1+1)
y1 = f1(x1)
ax.plot(x1,y1,label='f1',linewidth=2)
ax.scatter(mejorIndividuo1,f1(mejorIndividuo1),color='black',marker='x',label='Algoritmo enjambre',s=50,zorder=10)
ax.scatter(mejorIndividuoG1,f1(mejorIndividuoG1),color='red',label='Algoritmo genético',s=50,zorder=5)
plt.xlim([-512,512])
ax.grid(True)
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.legend()

fig2,ax2 = plt.subplots()
ax2.plot(historico1,label='historico Enjambre',linewidth=2)
ax2.plot(historicoG1,label='historico Genético',linewidth=2)
plt.xlim([0,max(len(historico1),len(historicoG1))])
ax2.grid(True)
ax2.set_xlabel('Iteraciones')
ax2.set_ylabel('funcionAptitud')
ax2.legend()
#---------------------------------------------Iniciso b)-------------------------------------------------
def f2(x,y):
    ret = 0
    if np.any(x == np.inf) or np.any(y == np.inf):
        ret = np.inf
    else:
        ret = (x**2+y**2)**0.25*((np.sin(50*(x**2+y**2)**0.1))**2+1)
    return ret

def f2G(x,y):
    return -1*((x**2+y**2)**0.25*((np.sin(50*(x**2+y**2)**0.1))**2+1))

Min2 = -100
Max2 = 100 
cantParticulas = 100
sizeParticula = 2
sizeIndividuo2 = 20 
cantIteraciones = 100
paciencia = 15

mejorIndividuo2, historico2 = algoritmoEnjambre(cantParticulas,sizeParticula,Min2,Max2,f2,cantIteraciones,paciencia)
mejorIndividuoG2, historicoG2 = algoritmoGenetico(sizeIndividuo2,cantParticulas,f2G,cantIteraciones,Min2,Max2,paciencia,False)

historico2 = np.array(historico2)
historicoG2 = np.array(historicoG2)
# mejorIndividuoG2 = binarioADecimal(mejorIndividuoG2,Min2,Max2)


##Para genético
mejorIndividuo2xG = binarioADecimal(mejorIndividuoG2[:int(sizeIndividuo2/2)],Min2,Max2)
mejorIndividuo2yG = binarioADecimal(mejorIndividuoG2[int(sizeIndividuo2/2):],Min2,Max2)

x2 = np.arange(Min2, Max2+1)
y2 = np.arange(Min2, Max2+1)

X2,Y2 = np.meshgrid(x2,y2)
ZG2 = f2(X2,Y2)
##Para genético
mejorIndividuo2zG = -f2(mejorIndividuo2xG,mejorIndividuo2yG)
##Para enjambre
mejorIndividuo2x = mejorIndividuo2[0]
mejorIndividuo2y = mejorIndividuo2[1]
mejorIndividuo2z = f2(mejorIndividuo2x,mejorIndividuo2y)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

surf = ax3.plot_surface(X2, Y2, ZG2, cmap='plasma')

barra_colores = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

ax3.scatter(mejorIndividuo2x,mejorIndividuo2y,mejorIndividuo2z,color='black',marker='x',s=100,label='Algoritmo enjambre')
ax3.scatter(mejorIndividuo2xG, mejorIndividuo2yG, mejorIndividuo2zG, color='red', marker='o', s=100, label='Algoritmo genético')
ax3.set_xlabel('X2')
ax3.set_ylabel('Y2')
ax3.set_zlabel('f2(X2, Y2)')

ax3.legend()

fig4,ax4 = plt.subplots()
ax4.plot(f2(historico2[0],historico2[1]),label='historico Enjambre',linewidth=2)
ax4.plot(historicoG2,label='historico Genético',linewidth=2)
plt.xlim([0,max(len(historico2),len(historicoG2))])
ax4.grid(True)
ax4.set_xlabel('Iteraciones')
ax4.set_ylabel('funcionAptitud')
ax4.legend()

plt.show()