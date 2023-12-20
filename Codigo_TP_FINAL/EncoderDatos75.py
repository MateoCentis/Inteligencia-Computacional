import librosa as lb
import numpy as np
import os
#---------------------------------------------Leer audios y convertirlos a vector-------------------------------------------------
path = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/audios2"
path2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/audiosTobiP"
datosAudioM = []
datosAudioT = []
sr = 16000
# sr = 20000
for archivo in os.listdir(path):
    if archivo.endswith('.wav'):
        pathAudio = os.path.join(path,archivo)
        audio, sr = lb.load(pathAudio, sr=sr)
        datosAudioM.append(audio)

for archivo in os.listdir(path2):
    if archivo.endswith('.wav'):
        pathAudio = os.path.join(path2,archivo)
        audio, sr = lb.load(pathAudio, sr=sr)
        datosAudioT.append(audio)
#---------------------------------------------Obtener coeficientes de mel-------------------------------------------------
sampleRate = sr
cepOrder = 49
framePeriod = 40 #probar con 40 y 200 de frameLength
frameLength = 800 
sizeFFT = 1024*8
print(len(datosAudioM))
print(len(datosAudioT))
coeficientesMelM = []
coeficientesMelT = []
bitrate = 16*sampleRate
for audioM in datosAudioM:
    audioM = np.array(audioM)
    #NORMALIZAR ANTES DE CALCULAR MFCC (en caso de normalización)
    # audioM = (1-1/2**bitrate)/max(audioM)*audioM #normalización del audio
    melCoef = lb.feature.mfcc(y=audioM,sr=sampleRate,n_mfcc=cepOrder,n_fft=sizeFFT,hop_length=framePeriod, win_length=frameLength)
    melCoef = melCoef.T     # (49,N) -> (N,49)
    coeficientesMelM.append(melCoef[:,:25])#win_length=frameLength

for audioT in datosAudioT:
    audioT = np.array(audioT)
    # audioT = (1-1/2**bitrate)/max(audioT)*audioT #normalización del audio
    melCoef = lb.feature.mfcc(y=audioT,sr=sampleRate,n_mfcc=cepOrder,n_fft=sizeFFT,hop_length=framePeriod, win_length=frameLength)
    melCoef = melCoef.T     # (49,N) -> (N,49)
    coeficientesMelT.append(melCoef[:,:25])#win_length=frameLength

#---------------------------------------------Guardar en TXT-------------------------------------------------

archivo1 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesMnorm1.txt"
archivo2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesTnorm1.txt"
# Guardar el vector de coeficientes en el archivo de texto
#Hay N audios y para cada uno hay N ventanas

with open(archivo1, "w") as archivo:
    for audio in coeficientesMelM:#se estaría leyendo [N(ventanas),25]   
        #ventana es de (2000,25)
        for ventana in audio:
            delta_ventana = lb.feature.delta(ventana)
            delta2_ventana = lb.feature.delta(ventana, order=2) 
            linea = np.concatenate((ventana,delta_ventana,delta2_ventana))
            # print(linea.shape)
            # Convertir los elementos a cadenas y unirlos con espacios
            array_str = " ".join(map(str, linea))
            # Escribir la cadena en el archivo
            archivo.write(array_str + "\n")

with open(archivo2, "w") as archivo:
    for audio in coeficientesMelT:
        for ventana in audio:
            delta_ventana = lb.feature.delta(ventana) 
            delta2_ventana = lb.feature.delta(ventana, order=2) 
            linea = np.concatenate((ventana,delta_ventana,delta2_ventana))
            # Convertir los elementos a cadenas y unirlos con espacios
            array_str = " ".join(map(str, linea))
            # Escribir la cadena en el archivo
            archivo.write(array_str + "\n")
