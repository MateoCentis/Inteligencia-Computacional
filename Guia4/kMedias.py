#Ejercicio 2 => K-medias:
def k_medias():
    #---------------------------------------------Inicialización--------------------------------------------
    #1. Inicializarlo al azar (asigno valores cualquiera a los centroides)
    #2. Asignar grupos al azar (vector que dice cada patrón a que centroide va y ahí armo centroides)(no valido para simétricos)
    #3. Otra forma es agarrar patrones al azar (si se puede que este lejos) y esos los uso como centroide 
    
#---------------------------------------------Algoritmo-------------------------------------------------
#-REASIGNAR PATRONES:
    #Paso por todos los patrones y decido cual es el patron más cercano a ese patron, en base a eso defino su grupo

    #matriz donde cada fila es un patron (columnas elementos del patron)
    #vector donde nos dice a que grupo pertenece los patrones 
        #(esto lo asignamos despues de comparar contra la matriz de centroides)

#-REASIGNO CENTROIDES:
    #Una vez tengo definidos todos los patrones calculo los centroides de cada grupo con el promedio

    #Cuando ningun patrón cambie de grupo termina la iteración
    #Matriz donde cada fila es un centroide (las columnas son las dimensiones del centroide, tamaño entrada)
    #Vector que dice cuantos elementos tiene cada centroide (le sumamos cada vez que asignamos uno)

    #Recorro el vector que dice los grupos de patrones y en base al numero de centroide se lo sumamos 
        #a una matriz de centroides incializada en cero 
        #una vez que sumamos todos los patrones tenemos que dividir por el número de patrones que caían en ese grupo
        #con eso obtenemos el promedio y se recalculan los centroides