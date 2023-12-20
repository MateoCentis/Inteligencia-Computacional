import numpy as np
import inspect
#si tiene dos ## es un comentario, si tiene # es una anotación a algo a mejorar

def algoritmoEnjambre(cantParticulas,sizeParticula,xMin,xMax,funcionError,cantIteraciones,paciencia):
    
    ##Inicialización al azar
    particulas = np.random.uniform(xMin,xMax,size=(cantParticulas,sizeParticula)) 
    mejoresIndividuos = np.full((cantParticulas,sizeParticula),np.inf) 
    velocidades = np.zeros((cantParticulas,sizeParticula)) #ver si hacerlas aleatorias o como inicializarlas
    mejorIndividuoGlobal = np.full(sizeParticula,np.inf)   #(anduvo en cero) supongo tan bien
    c1 = 2
    c2 = 2
    #variables para criterio de finalización
    iteraciones = 0
    iteracionesSinMejoras = 0
    mejorIndividuoIteracion = np.full(sizeParticula,np.inf)
    historicoMejoresIndividuos = []
    while(iteraciones < cantIteraciones and iteracionesSinMejoras < paciencia):
        
        ##Para cada particula obtengo su mejor posición (conocimiento personal) y el mejor individuo en general (conocimiento social)
        for i in range(len(particulas)):
            individuo = np.copy(particulas[i,:])

            cantidadParametros = len((inspect.signature(funcionError)).parameters)
            if cantidadParametros > 1:
                if (funcionError(individuo[0],individuo[1]) < funcionError(mejoresIndividuos[i,0],mejoresIndividuos[i,1])):
                    mejoresIndividuos[i] = individuo
                
                if (np.all(funcionError(mejoresIndividuos[i,0],mejoresIndividuos[i,1]) < funcionError(mejorIndividuoGlobal[0],mejorIndividuoGlobal[1]))):
                    mejorIndividuoGlobal = mejoresIndividuos[i]
            else:
                if (funcionError(individuo) < funcionError(mejoresIndividuos[i,:])):
                    mejoresIndividuos[i] = individuo
                
                if (np.all(funcionError(mejoresIndividuos[i]) < funcionError(mejorIndividuoGlobal))):
                    mejorIndividuoGlobal = mejoresIndividuos[i]
        
        ##Para cada partícula re-calculo sus posiciones, primero calculando la "velocidad" y luego sumándosela a su posición actual
        for i in range(len(particulas)):
            individuo = np.copy(particulas[i,:])
            
            r1 = np.random.rand()
            r2 = np.random.rand()

            conocimientoPersonal = c1*r1*(mejoresIndividuos[i,:]-individuo)
            conocimientoSocial = c2*r2*(mejorIndividuoGlobal-individuo)
            
            velocidades[i] = velocidades[i] + conocimientoPersonal + conocimientoSocial
            
            particulas[i,:] = particulas[i,:] + velocidades[i]
        
        ##Criterio de finalización
        iteraciones += 1
        historicoMejoresIndividuos.append(mejorIndividuoGlobal)
        cantidadParametros = len((inspect.signature(funcionError)).parameters)
        if cantidadParametros > 1:
            if (funcionError(mejorIndividuoGlobal[0],mejorIndividuoGlobal[1]) < funcionError(mejorIndividuoIteracion[0],mejorIndividuoIteracion[1])):
                iteracionesSinMejoras = 0
                mejorIndividuoIteracion = mejorIndividuoGlobal
            else:
                iteracionesSinMejoras += 1
        else:
            if (funcionError(mejorIndividuoGlobal) < funcionError(mejorIndividuoIteracion)):
                iteracionesSinMejoras = 0
                mejorIndividuoIteracion = mejorIndividuoGlobal
            else:
                iteracionesSinMejoras += 1
    return mejorIndividuoGlobal, historicoMejoresIndividuos