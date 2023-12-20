import numpy as np
import inspect


def binarioADecimal(individuo,menor,mayor):
    N = len(individuo)
    x = 0
    for k,i in enumerate (individuo[::-1]):
        x += (i * 2**k)
    x = menor + ((mayor-menor)/(2**N-1))*x
    return x

def seleccionTorneo(poblacion, aptitudes, cantParticipantes):
    N = poblacion.shape[0]
    seleccionados = np.zeros((N,poblacion.shape[1]))
    for c in range(N):
        indicesParticipantes = np.random.choice(N, size=cantParticipantes, replace=False)
        aptitudesParticipantes = np.zeros(cantParticipantes)
        cont = 0
        for i in indicesParticipantes:
            aptitudesParticipantes[cont] = aptitudes[i]
            cont += 1
        
        indiceGanador = indicesParticipantes[np.argmax(aptitudesParticipantes)]
        ganador = poblacion[indiceGanador,:].copy()
        
        seleccionados[c,:] = ganador
    return seleccionados

def cruza(individuo1, individuo2, numero):
    if len(individuo1) != len(individuo2):
        assert "Los individuos no tienen la misma longitud para la cruza"
    indice = np.random.randint(0, len(individuo1))
    if numero == 1:
        hijo1 = np.copy(individuo1)
        hijo2 = np.copy(individuo2)
    else:
        hijo1 = np.concatenate((individuo1[:indice],individuo2[indice:]))
        hijo2 = np.concatenate((individuo2[:indice], individuo1[indice:]))
    
    return hijo1, hijo2


#Realiza la mutación de un individuo
def mutacion(individuo):
    indice = np.random.randint(0, len(individuo))
    if individuo[indice] == 0:
        individuo[indice] = 1
    else:
        individuo[indice] = 0

    return individuo

def evaluarPoblacion(poblacion,funcionAptitud,menor,mayor,genotipo):
        N = poblacion.shape[0]
        sizeIndividuo = poblacion.shape[1]
        aptitudes = np.zeros(N)
        cantidadParametros = len((inspect.signature(funcionAptitud)).parameters)
        for i in range(N):
            if genotipo == True:#para ej 2
                aptitudes[i] = funcionAptitud(poblacion[i,:])
            else: 
                if cantidadParametros > 1: #caso 2 parámetros dividimos el individuo a la mitad
                    mitadSize = int(sizeIndividuo/2)
                    individuoDecox = binarioADecimal(poblacion[i,:mitadSize],menor,mayor)
                    individuoDecoy = binarioADecimal(poblacion[i,mitadSize:],menor,mayor)
                    aptitudes[i] = funcionAptitud(individuoDecox,individuoDecoy)
                else:
                    individuoDeco = binarioADecimal(poblacion[i,:],menor,mayor)
                    aptitudes[i] = funcionAptitud(individuoDeco)
        return aptitudes
    
    
def algoritmoGenetico(sizeIndividuo,N,funcionAptitud,generacionesCortar,menor,mayorGlobal,paciencia,genotipo):
    #------------------------------------Inicializar población-------------------------------------------
    #Población inicial de N individuos aleatorios (filas) de tamaño sizeIndividuo (columnas) 
    poblacion = np.random.randint(2, size=(N, sizeIndividuo), dtype=int)
    generaciones = 0
    generacionesSinMejoras = 0
    mejorHistorico = -np.inf
    retornar = 0
    elGOAT = np.zeros(sizeIndividuo)
    historico = []
    while (generacionesSinMejoras < paciencia and generaciones < generacionesCortar):
        #---------------------------Evaluar la aptitud de cada individuo-------------------------------------
        aptitudes = evaluarPoblacion(poblacion,funcionAptitud,menor,mayorGlobal,genotipo)
        #---------------------------------Seleccionar los más aptos------------------------------------------
        masAptos = seleccionTorneo(poblacion,aptitudes,20)
        #---------------------------------------------Cruza--------------------------------------------------
        mejorIndividuo = np.copy(poblacion[np.argmax(aptitudes),:]) #me guardo el mejor individuo
        nuevaPoblacion = np.zeros((N,sizeIndividuo))
        for i in range(int(N/2)):
            numero = np.random.randint(1,10)
            indice1 = np.random.randint(0, N)
            indice2 = np.random.randint(0, N)
            [hijo1,hijo2] = cruza(masAptos[indice1,:],masAptos[indice2,:],numero)
            nuevaPoblacion[i*2,:] = hijo1
            nuevaPoblacion[i*2+1,:] = hijo2
        #---------------------------------------------Mutación-----------------------------------------------
        for i in range(nuevaPoblacion.shape[0]):
            numero = np.random.randint(1,10)
            if numero <  3: #probabilidad 2/10
                individuoMutado = mutacion(nuevaPoblacion[i,:])
                nuevaPoblacion[i,:] = np.copy(individuoMutado)
        #---------------------------------------------Parada-------------------------------------------------
        nuevaPoblacion[-1,:] = mejorIndividuo.copy()
        nuevasAptitudes = evaluarPoblacion(nuevaPoblacion,funcionAptitud,menor,mayorGlobal,genotipo)
        indiceMayorNuevo = np.argmax(nuevasAptitudes)
        mayor = np.copy(nuevasAptitudes[indiceMayorNuevo])
        historico.append(mayor)
        if genotipo and generaciones % 10 == 0 :
            print("Fitness: ", mayor)
        if mayor > mejorHistorico:
            mejorHistorico = np.copy(mayor)
            elGOAT = np.copy(nuevaPoblacion[indiceMayorNuevo,:])
            generacionesSinMejoras = 0
        else:
            generacionesSinMejoras += 1
        poblacion = np.copy(nuevaPoblacion)
        generaciones += 1
    print(" ")
    print("Generaciones totales: ", generaciones)
    print("Generaciones sin mejoras: ", generacionesSinMejoras)
    print("Tamaño de la población", poblacion.shape[0])
    print(elGOAT)
    return elGOAT, historico


