import numpy as np
import matplotlib.pyplot as plt
from algoritmoGenetico import algoritmoGenetico
from algoritmoGenetico import binarioADecimal
from mpl_toolkits.mplot3d import Axes3D

genotipo = False
def gradienteDescendente2D(funcion, derivada, velocidadAprendizaje, cantIteraciones, x_inicial, y_inicial):
    x_inicial = np.array(x_inicial)
    y_inicial = np.array(y_inicial)
    historial_x = np.zeros((cantIteraciones+1,len(x_inicial)))
    historial_y = np.zeros((cantIteraciones+1,len(y_inicial)))
    historial_x[0,:] = x_inicial
    historial_y[0,:] = y_inicial

    for i in range(cantIteraciones):
        df_dx, df_dy = derivada(x_inicial, y_inicial)
        x_inicial -= velocidadAprendizaje * df_dx
        y_inicial -= velocidadAprendizaje * df_dy
        historial_x[i+1,:] = x_inicial
        historial_y[i+1,:] = y_inicial
    return x_inicial, y_inicial, historial_x, historial_y

def f1(x):
    return -1*(-x*np.sin(np.sqrt(abs(x))))

def f2(x,y):
    return -1*((x**2+y**2)**0.25*((np.sin(50*(x**2+y**2)**0.1))**2+1))

N = 20
generacionesCortar = 60
menor1 = -512
mayor1 = 512
sizeIndividuo1 = 30
paciencia = 30
#---------------------------------------------Ejercicio 1-------------------------------------------------
def gradienteDescendente1D(funcion, derivada, velocidadAprendizaje, cantIteraciones, x_inicial):
    # historial_x = [x_inicial]
    x_inicial = np.array(x_inicial)
    historial_x = np.zeros((cantIteraciones+1,len(x_inicial)))
    historial_x[0,:] = x_inicial
    for i in range(cantIteraciones):
        df_dx = derivada(x_inicial)
        x_inicial -= velocidadAprendizaje * df_dx
        historial_x[i+1,:] = x_inicial

    return x_inicial, historial_x

def f1G(x):
    return -x*np.sin(np.sqrt(abs(x)))

def df1G(x):
    return -np.sin(np.sqrt(np.abs(x)))-(x**2*np.cos(np.sqrt(np.abs(x))))/(2*np.abs(x)*np.sqrt(np.abs(x)))
    

x1 = np.arange(menor1, mayor1+1)
mejorIndividuo, historico1 = algoritmoGenetico(sizeIndividuo1,N,f1,generacionesCortar,menor1,mayor1,paciencia,genotipo)
historico1 = np.array(historico1)
minimo1 = binarioADecimal(mejorIndividuo,menor1,mayor1)
fig,ax = plt.subplots()
y1 = -f1(x1)
ax.plot(x1,y1,label='f1',linewidth=2)
velocidadAprendizaje = 0.01
# numIteraciones = len(historico1)
numIteraciones = generacionesCortar
x_inicial = np.random.uniform(-512, 512, len(historico1))
xGV, historial_xG = gradienteDescendente1D(f1G,df1G,velocidadAprendizaje,numIteraciones,x_inicial)
# print("Gradiente: ", historial_xG)
print("Histórico: ", historico1)
indiceMinimo = np.argmin(-f1(xGV))
xG = xGV[indiceMinimo]
ax.scatter(xG,-f1(xG),color='black',marker='x',label='Gradiente descendiente',s=50,zorder=10)
ax.scatter(minimo1,-f1(minimo1),color='red',label='Algoritmo genético',s=50,zorder=5)
plt.xlim([-512,512])
ax.grid(True)
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.legend()

# fig2,ax2 = plt.subplots()
# ax2.grid(True)
# ax2.set_xlabel('Generaciones')
# ax2.set_ylabel('Eje Y')
# generaciones = np.arange(len(historico1))
# print(historico1)
# print(-f1(historial_xG[:,indiceMinimo]))
# ax2.plot(generaciones,-historico1, color='blue',label='Algoritmo genético')
# ax2.plot(generaciones,-f1(historial_xG[:len(historico1),indiceMinimo]), color='red',label='Gradiente descendiente')
# ax2.legend()
# plt.show()

#---------------------------------------------Ejercicio 2-------------------------------------------------
def f2Gradiente(x, y):
    return ((x**2+y**2)**0.25*((np.sin(50*(x**2+y**2)**0.1))**2+1))

def df2G(x, y):
    df_dx = (x * (np.sin(50 * (x**2 + y**2)**0.1)**2 + 1)) / (2 * (x**2 + y**2)**0.75)
    df_dy = (20 * x * np.cos(50 * (x**2 + y**2)**0.1) * np.sin(50 * (x**2 + y**2)**0.1)) / (x**2 + y**2)**(13/20)
    return df_dx, df_dy

menor2 = -100
mayor2 = 100
x2 = np.arange(menor2, mayor2+1)
y2 = np.arange(menor2, mayor2+1)
N = 300
sizeIndividuo2 = 40
minimo2, historico2 = algoritmoGenetico(sizeIndividuo2,N,f2,generacionesCortar,menor2,mayor2,paciencia,genotipo)
historico2 = np.array(historico2)
minimo2x = binarioADecimal(minimo2[:int(sizeIndividuo2/2)],menor2,mayor2)
minimo2y = binarioADecimal(minimo2[int(sizeIndividuo2/2):],menor2,mayor2)
X2,Y2 = np.meshgrid(x2,y2)
Z2 = -f2(X2,Y2)
minimo2z = -f2(minimo2x,minimo2y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X2, Y2, Z2, cmap='plasma')

barra_colores = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

velocidadAprendizaje = 0.01
numIteraciones = len(historico2)
x_inicial = np.random.uniform(-100, 100, len(historico2))
y_inicial = np.random.uniform(-100, 100, len(historico2))
[xGV, yGV, historial_x, historial_y] = gradienteDescendente2D(f2Gradiente,df2G,velocidadAprendizaje,numIteraciones,x_inicial,y_inicial)
indiceMinimo = np.argmin(-f2(xGV, yGV))
xG = xGV[indiceMinimo]
yG = yGV[indiceMinimo]
zG = -f2(xG,yG)
# print(zG)
ax.scatter(xG,yG,zG,color='black',marker='x',s=100,label='Gradiente descendente')
ax.scatter(minimo2x, minimo2y, minimo2z, color='red', marker='o', s=100, label='Algoritmo Genético')
ax.set_xlabel('X2')
ax.set_ylabel('Y2')
ax.set_zlabel('f2(X2, Y2)')

ax.legend()


# fig2,ax2 = plt.subplots()
# ax2.grid(True)
# ax2.set_xlabel('Generaciones')
# ax2.set_ylabel('Eje Z')
# generaciones = np.arange(len(historico2))
# ax2.plot(generaciones,-historico2, color='blue',label='Algoritmo genético')
# ax2.plot(generaciones,-f2(historial_x[:,indiceMinimo],historial_y[:,indiceMinimo]), color='red',label='Gradiente descendiente')

# # ax2.legend()
plt.show()