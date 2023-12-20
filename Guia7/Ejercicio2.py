import numpy as np
from algoritmoHormigas import algoritmoHormigas
from numpy import genfromtxt
rutaArchivo = 'C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia7/gr17.csv'
D = genfromtxt('C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/EjerciciosPractica/Guia7/gr17.csv', delimiter=",")
#D es la matriz de distancias
tasaEvaporacion = 0.1 #ver que con tasa de evaporación alta rompe despues de muchas iteraciones
                        #porque la matriz de feromonas se hace muy chica y esto hace luego 
                        #que las probabilidades sean infinitas
#metodos 0:global, 1:local, 2: uniforme
#considerar tiempo de búsqueda y longitud de caminos encontrados
cantHormigas = 20
maxIteraciones = 500
metodo = 0
Q = 5
longitudCamino, mejorCamino = algoritmoHormigas(D,cantHormigas,tasaEvaporacion,metodo,maxIteraciones,Q)
if (metodo == 0):
    print("Método global")
elif (metodo == 1):
    print("Método local")
else:
    print("Método uniforme")
print(longitudCamino) #dist: 2819
print(mejorCamino)
# metodo = 1
# longitudCamino = algoritmoHormigasOpt(D,cantHormigas,tasaEvaporacion,metodo,maxIteraciones,Q)
# print("Método uniforme")
# print(longitudCamino) #dist: 2810
# metodo = 2
# longitudCamino = algoritmoHormigasOpt(D,cantHormigas,tasaEvaporacion,metodo,maxIteraciones,Q)
# print("Método local")
# print(longitudCamino)#dist: 4722
