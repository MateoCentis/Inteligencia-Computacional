import librosa as lb
import librosa.display as disp
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scipy.spatial.distance import cdist
# ---------------------------------------Parámetros-------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mel_cepstral_distortion(ref_mfcc, synth_mfcc):
    min_rows = min(ref_mfcc.shape[0], synth_mfcc.shape[0])
    min_cols = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:min_rows, :min_cols]
    synth_mfcc = synth_mfcc[:min_rows, :min_cols]

    dist_matrix = cdist(ref_mfcc, synth_mfcc, metric='euclidean')
    mcd = np.mean(dist_matrix)

    return mcd

def load_checkpoint(model, filename):  # , optimizer, lr
    model.load_state_dict(torch.load(filename, map_location=device))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(90, 128),  # capa de entrada
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),  # primera capa oculta
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),  # segunda capa oculta
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 128),  # tercera capa oculta #512 no mejora
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 128),  # cuarta capa oculta
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 90),
            # nn.Linear(128,90),
        )

    def forward(self, x):
        output = self.model(x)
        return output


path = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/audios2/Audio 1.wav"
path2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/audiosTobiP/audio-20.wav"
audioM, sr = lb.load(path, sr=16000, mono=True)
audioT, sr = lb.load(path2, sr=16000, mono=True)
sampleRate = sr
cepOrder = 60
# lower order :30 , alto orden : los 30 restantes
# framePeriod = 0.005  # (5 ms)
framePeriod = 80  # 80
sizeFFT = 1024 * 8
# frameLength = 0.025 #(25 ms)
frameLength = 800  # 400
# --------------------------------------Obtener coeficientes de mel-------------------------------------------------
# MATEO
melCoefM = lb.feature.mfcc(
    y=audioM,
    sr=sampleRate,
    n_mfcc=cepOrder,
    n_fft=sizeFFT,
    hop_length=framePeriod,
    win_length=frameLength,
)
melCoefM = melCoefM.T
lowOrderMelM = melCoefM[:, :30]
highOrderMelM = melCoefM[:, 30:]
delta_melCoefM = lb.feature.delta(lowOrderMelM)
delta2_melCoefM = lb.feature.delta(lowOrderMelM, order=2)
coefM = np.concatenate((lowOrderMelM, delta_melCoefM, delta2_melCoefM), axis=1)
coefM = torch.from_numpy(coefM)
# TOBIPE08
melCoefT = lb.feature.mfcc(
    y=audioT,
    sr=sampleRate,
    n_mfcc=cepOrder,
    n_fft=sizeFFT,
    hop_length=framePeriod,
    win_length=frameLength,
)
melCoefT = melCoefT.T
lowOrderMelT = melCoefT[:, :30]
highOrderMelT = melCoefT[:, 30:]
delta_melCoefT = lb.feature.delta(lowOrderMelT)
delta2_melCoefT = lb.feature.delta(lowOrderMelT, order=2)
coefT = np.concatenate((lowOrderMelT, delta_melCoefT, delta2_melCoefT), axis=1)
coefT = torch.from_numpy(coefT)

# ----------------------------------------Aplicar modelo a coeficientes-----------------------------------------------

checkpoint_file1 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/geny90N.pth"
checkpoint_file2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/genx90N.pth"
genX = Generator()
genY = Generator()
load_checkpoint(genX, checkpoint_file2)
load_checkpoint(genY, checkpoint_file1)
genX.eval()
genY.eval()
fake_tobias = genX(coefM).detach().cpu().numpy()
fake_mateo = genY(coefT).detach().cpu().numpy()
melCoefTobiasFake = np.concatenate((fake_tobias[:, :30], highOrderMelM), axis=1).T
melCoefMateoFake = np.concatenate((fake_mateo[:, :30], highOrderMelT), axis=1).T

# ---------------------------------------------Síntesis-------------------------------------------------
fakeT = lb.feature.inverse.mfcc_to_audio(
    mfcc=melCoefTobiasFake,
    n_mels=cepOrder,
    sr=sampleRate,
    hop_length=framePeriod,
    win_length=frameLength,
)  # length= modifica el tamaño de salida de y
fakeM = lb.feature.inverse.mfcc_to_audio(
    mfcc=melCoefMateoFake,
    n_mels=cepOrder,
    sr=sampleRate,
    hop_length=framePeriod,
    win_length=frameLength,
)  # length= modifica el tamaño de salida de y

sf.write("C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/Resultados/sintesis90T.wav", fakeT, sampleRate)
sf.write("C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/Resultados/sintesis90M.wav", fakeM, sampleRate)

print("ARCHIVOS ESCRITOS =======>")
#-----------------------------------------Calculo de MCD-------------------------------------------------
MCDTobias = mel_cepstral_distortion(melCoefT,melCoefTobiasFake)
MCDMateo = mel_cepstral_distortion(melCoefM,melCoefMateoFake)
print("MCD Tobias: ", MCDTobias) ##615 ##616 ##
print("MCD Mateo: ", MCDMateo) ##433 ##435 ##

# -------------------------------------------Gráficas-------------------------------------------------
graficas = False
if graficas == True:
    # Espectrograma de fakeT
    fakeT_spec = lb.amplitude_to_db(np.abs(lb.stft(fakeT)), ref=np.max)
    plt.figure(figsize=(10, 4))
    lb.display.specshow(fakeT_spec, y_axis='log', x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma de fakeT')
    # plt.show()

    # Espectrograma de fakeM
    fakeM_spec = lb.amplitude_to_db(np.abs(lb.stft(fakeM)), ref=np.max)
    plt.figure(figsize=(10, 4))
    lb.display.specshow(fakeM_spec, y_axis='log', x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma de fakeM')
    # plt.show()

    # Espectrograma de audioT
    audioT_spec = lb.amplitude_to_db(np.abs(lb.stft(audioT)), ref=np.max)
    plt.figure(figsize=(10, 4))
    lb.display.specshow(audioT_spec, y_axis='log', x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma de audioT')
    # plt.show()

    # Espectrograma de audioM
    audioM_spec = lb.amplitude_to_db(np.abs(lb.stft(audioM)), ref=np.max)
    plt.figure(figsize=(10, 4))
    lb.display.specshow(audioM_spec, y_axis='log', x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma de audioM')
    plt.show()

    # #--------------------------------------Gráficas MFCC-------------------------------------------------

    melCoefMDB = lb.power_to_db(melCoefM.T, ref=np.max)
    plt.figure(figsize=(10, 4))
    disp.specshow(melCoefMDB, x_axis='time', sr=sampleRate, hop_length=framePeriod, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Coefficients Mateo real')  

    melCoefTDB = lb.power_to_db(melCoefT.T, ref=np.max)
    plt.figure(figsize=(10, 4))
    disp.specshow(melCoefTDB, x_axis='time', sr=sampleRate, hop_length=framePeriod, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Coefficients Tobias real')  

    melCoefMateoDB = lb.power_to_db(melCoefMateoFake, ref=np.max)
    plt.figure(figsize=(10, 4))
    disp.specshow(melCoefMateoDB, x_axis='time', sr=sampleRate, hop_length=framePeriod, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Coefficients Mateo Approx ')

    melCoefTobiasDB = lb.power_to_db(melCoefTobiasFake, ref=np.max)
    plt.figure(figsize=(10, 4))
    disp.specshow(melCoefTobiasDB, x_axis='time', sr=sampleRate, hop_length=framePeriod, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Coefficients Tobias Approx ')

    plt.show()
