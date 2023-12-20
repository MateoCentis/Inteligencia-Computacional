import librosa as lb
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
LAMBDA = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GENERATOR_X = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/genx75.pth"
CHECKPOINT_GENERATOR_Y = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/geny75.pth"
CHECKPOINT_DISCRIMINATOR_X = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/discx75.pth"
CHECKPOINT_DISCRIMINATOR_Y = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/discy75.pth"
torch.manual_seed(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#---------------------------------------------Funciones a utilizar-------------------------------------------------
def leer_archivo_txt(archivo):
    with open(archivo, 'r') as file:
        lineas = file.readlines()
        # Dividir cada línea en elementos y convertirlos en float
        vectors = [list(map(float, linea.split())) for linea in lineas]
    return np.array(vectors)

def save_checkpoint(model,filename):
    print("SAVING CHECKPOINT====>")
    torch.save(model.state_dict(), filename)


def load_checkpoint(model, filename): #, optimizer, lr
    print("LOADING CHECKPOINT====>")
    model.load_state_dict(torch.load(filename))

#---------------------------------------------Discriminador-------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Agrega ruido gaussiano a la primera capa lineal
            NoisyLinear(75, 128),#le agregué
            nn.Sigmoid(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),#
            nn.Sigmoid(),
            # nn.Dropout(0.3),
            nn.Linear(128, 75),
            nn.Sigmoid(),
            nn.Linear(75, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output

# Agrega ruido gaussiano para que el discriminador no se aprenda tan fácilmente las salidas del gen
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, noise_std=0.01):
        super(NoisyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.linear(x)
#---------------------------------------------Generador-------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(75, 128),#capa de entrada
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),#primera capa oculta
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),#segunda capa oculta
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 128),#tercera capa oculta #512 no mejora
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128,128), #cuarta capa oculta
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128,75),
            # nn.Linear(128,90),
        )

    def forward(self, x):
        output = self.model(x)
        return output
#código para inicializar los pesos (puede mejorar la convergencia)
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def mel_cepstral_distortion(ref_mfcc, synth_mfcc):
    min_rows = min(ref_mfcc.shape[0], synth_mfcc.shape[0])
    min_cols = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:min_rows, :min_cols]
    synth_mfcc = synth_mfcc[:min_rows, :min_cols]

    dist_matrix = cdist(ref_mfcc, synth_mfcc, metric='euclidean')
    mcd = np.mean(dist_matrix)

    return mcd

#---------------------------------------------Preparar los datos-------------------------------------------------
archivo1 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesMnorm1.txt"
archivo2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesTnorm1.txt"

datosM = leer_archivo_txt(archivo1)
datosT = leer_archivo_txt(archivo2)
sizeDatosM = datosM.shape[0]
sizeDatosT = datosT.shape[0]
datos_combinados = np.concatenate((datosM, datosT), axis=0)
dataset = torch.tensor(datos_combinados) #tensor de datos
train_labels = torch.cat((torch.zeros(sizeDatosM),torch.ones(sizeDatosT)))#0: Mateo, 1: Tobi
# dataset = (dataset - torch.mean(dataset))/torch.std(dataset)
X_train, X_val, y_train, y_val = train_test_split(dataset, train_labels, test_size=0.2) #80% para entrenamiento =>20% validación, testeo a parte

#---------------------------------------------Armar batches-------------------------------------------------
batch_size = 3000 #61 batches con 1146 o 122 con 573 
batch_number = len(X_train)/batch_size
train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# test_set = TensorDataset(X_test,y_test)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

val_set = TensorDataset(X_val,y_val)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

#----------------------------------------Parámetros del entrenamiento------------------------------------------
#x hablante de origen, y hablante de destino
genX = Generator().to(device=device) #mapea x->y
genY = Generator().to(device=device) #mapea y->x                                 
Dx = Discriminator().to(device=device) # distringue X de Xapprox
Dy = Discriminator().to(device=device) # distingue Y de Yapprox
#Inicialización de pesos (PROBAR SIN)
# genX.apply(weights_init)
# genY.apply(weights_init)
# Dx.apply(weights_init)
# Dy.apply(weights_init)

learning_rate = 0.001                                               
num_epochs = 250 #entre 60 y 400
optimizer_discriminatorX = torch.optim.Adam(Dx.parameters(),lr=learning_rate*0.1,betas=(0.5, 0.999)) 
optimizer_discriminatorY = torch.optim.Adam(Dy.parameters(),lr=learning_rate*0.1,betas=(0.5, 0.999))
optimizer_generatorX = torch.optim.Adam(genX.parameters(),lr=learning_rate,betas=(0.5, 0.999)) 
optimizer_generatorY = torch.optim.Adam(genY.parameters(),lr=learning_rate,betas=(0.5, 0.999))
#betas=(0.5, 0.999)

# Añadir el LinearLR Scheduler
# lr_scheduler_discriminatorX = LinearLR(optimizer_discriminatorX, end_lr, total_steps)
# lr_scheduler_discriminatorY = LinearLR(optimizer_discriminatorY, end_lr, total_steps)
# lr_scheduler_generatorX = LinearLR(optimizer_generatorX, end_lr, total_steps)
# lr_scheduler_generatorY = LinearLR(optimizer_generatorY, end_lr, total_steps)

L1 = nn.L1Loss()
mse = nn.MSELoss()
epochs = range(num_epochs)
GenLosses = np.zeros((num_epochs,1))
DiscLosses = np.zeros((num_epochs,1))
epocasSinMejora = 0
paciencia = 10
mejorLoss = np.inf
mejorDiscLoss = np.inf
if LOAD_MODEL:
    load_checkpoint(
        genX,
        CHECKPOINT_GENERATOR_X)
    load_checkpoint(
        genY,
        CHECKPOINT_GENERATOR_Y)
    load_checkpoint(
        Dx,
        CHECKPOINT_DISCRIMINATOR_X)
    load_checkpoint(
        Dy,
        CHECKPOINT_DISCRIMINATOR_Y)
epocaCorte = 0
#---------------------------------------------Entrenamiento-------------------------------------------------
for epoch in tqdm(epochs):#Para cada época  
    for indiceBatch, (samples, labels) in enumerate(train_loader): #Para cada batch
        # Crear máscara booleana para distinguir los hablantes
        mascara0 = labels == 0
        mascara1 = labels == 1
        samplesX = samples[mascara1] #X: Tobias
        samplesY = samples[mascara0] #Y: Mateo
        samplesX = samplesX.to(torch.float) #G horseToZebra
        samplesY = samplesY.to(torch.float) #F zebraToHorse
        X = samplesX.to(device)
        Y = samplesY.to(device)
        #---------------------------------Entrenamiento de discriminadores---------------------------------------
        genX.train()
        genY.train()
        Dy.train()
        Dx.train()
        #Discriminador X-----------------------------------
        fakeX = genX(Y)

        Dx_fake = Dx(fakeX.detach())
        Dx_real = Dx(X)

        Dx_real_loss = mse(Dx_real, torch.ones_like(Dx_real)) #SMOOTHING LABELS -0.1*torch.ones_like(Dx_real)
        Dx_fake_loss = mse(Dx_fake, torch.zeros_like(Dx_fake)) #+0.1*torch.ones_like(Dx_fake)
        Dx_loss = Dx_real_loss + Dx_fake_loss

        #Discriminador Y----------------------------------
        fakeY = genY(X)

        Dy_real = Dy(Y)
        Dy_fake = Dy(fakeY.detach()) #Separa fakeY de la GPU (para no usarla para calcular gradientes)
        Dy_real_loss = mse(Dy_real, torch.ones_like(Dy_real))
        Dy_fake_loss = mse(Dy_fake, torch.zeros_like(Dy_fake))
        Dy_Loss = Dy_real_loss + Dy_fake_loss
        
        D_loss = (Dy_Loss + Dx_loss) / 2

        optimizer_discriminatorX.zero_grad()
        optimizer_discriminatorY.zero_grad()
        D_loss.backward()
        optimizer_discriminatorX.step()
        optimizer_discriminatorY.step()

        #----------------------------------------Entrenamiento de generadores--------------------------------------
        #Durante el entrenamiento de generadores desactivar los discriminadores
        #Adversarial Loss
        Dx_fake = Dx(fakeX)
        Dy_fake = Dy(fakeY)
        lossG_X = mse(Dx_fake, torch.ones_like(Dx_fake))
        lossG_Y = mse(Dy_fake, torch.ones_like(Dy_fake))
        #Cycle Loss
        cycleY = genY(fakeX)
        cycleX = genX(fakeY)
        cycleY_loss = L1(Y,cycleY)
        cycleX_loss = L1(X,cycleX)
        #Identity Loss
        identity_Y = genY(Y)
        identity_X = genX(X)
        identity_Y_loss = L1(Y,identity_Y)
        identity_X_loss = L1(X,identity_X)
        #Total Loss
        G_loss = (lossG_X + lossG_Y + cycleY_loss*LAMBDA + cycleX_loss*LAMBDA + 5*identity_X_loss + 5*identity_Y_loss)
        
        optimizer_generatorX.zero_grad()
        optimizer_generatorY.zero_grad()
        G_loss.backward()
        optimizer_generatorX.step()
        optimizer_generatorY.step()
        
        # lr_scheduler_generatorX.step()
        # lr_scheduler_generatorY.step()
    #---------------------------------------------Evaluación-------------------------------------------------
    genX.eval()
    genY.eval()
    Dx.eval()
    Dy.eval()
    cont = 0
    with torch.no_grad():
        for val_samples, val_labels in val_loader:
            val_samplesX = val_samples[val_labels == 1].to(torch.float).to(device)
            val_samplesY = val_samples[val_labels == 0].to(torch.float).to(device)
            
            fake_valX = genX(val_samplesY)
            val_Dx_fake = Dx(fake_valX)
            val_Dx_real = Dx(val_samplesX)
            val_Dx_loss = mse(val_Dx_real,torch.ones_like(val_Dx_real)) + mse(val_Dx_fake,torch.zeros_like(val_Dx_fake))

            fake_valY = genY(val_samplesX)
            val_Dy_fake = Dy(fake_valY)
            val_Dy_real = Dy(val_samplesY)
            val_Dy_loss = mse(val_Dy_real, torch.ones_like(val_Dy_real)) + mse(val_Dy_fake, torch.zeros_like(val_Dy_fake))
        
            D_loss = (val_Dy_loss + val_Dx_loss) / 2

            #Generadores
            val_lossG_X = mse(val_Dx_fake, torch.ones_like(val_Dx_fake))
            val_lossG_Y = mse(val_Dy_fake, torch.ones_like(val_Dy_fake))

            val_cycleY = genY(fake_valX)
            val_cycleX = genX(fake_valY)
            val_cycleY_loss = L1(val_samplesY, val_cycleY)
            val_cycleX_loss = L1(val_samplesX, val_cycleX)

            val_identity_Y = genY(val_samplesY)
            val_identity_X = genX(val_samplesX)
            val_identity_Y_loss = L1(val_samplesY, val_identity_Y)
            val_identity_X_loss = L1(val_samplesX, val_identity_X)

            G_loss = (val_lossG_X + val_lossG_Y + val_cycleY_loss * LAMBDA + val_cycleX_loss * LAMBDA +
                          5 * val_identity_X_loss + 5 * val_identity_Y_loss)

            if cont % (batch_number//2 + 1) == 0 and cont > 0:#para que solo se tenga en cuenta un batch
                print(f"Epoch: {epoch} Generator loss: {G_loss}")
                GenLosses[epoch] = G_loss.detach().cpu().numpy()
                DiscLosses[epoch] = D_loss.detach().cpu().numpy()
                if GenLosses[epoch] < mejorLoss and DiscLosses[epoch] < mejorDiscLoss:
                    mejorLoss = GenLosses[epoch]
                    mejorDiscLoss = DiscLosses[epoch]
                    epocasSinMejora = 0
                else:
                    epocasSinMejora += 1
                print(f"Epoch: {epoch} Discriminator loss: {D_loss}")            
                MCDX = mel_cepstral_distortion(val_samplesX.detach().cpu().numpy(),fake_valX.detach().cpu().numpy())
                MCDY = mel_cepstral_distortion(val_samplesY.detach().cpu().numpy(),fake_valY.detach().cpu().numpy())
                print(f"Epoch: {epoch} MCDX: {MCDX}")
                print(f"Epoch: {epoch} MCDY: {MCDY}")
            cont += 1
    if epocasSinMejora > paciencia:
        print("Corta por iteraciones sin mejores en la época: ", epoch)
        epocaCorte = epoch
        break
plt.figure()
plt.plot(GenLosses, color="red")
plt.grid(True)
plt.xlim([0, epocaCorte])
plt.ylabel("Losses")
plt.xlabel("Épocas")
plt.title("Loss del generador")
plt.figure()
plt.plot(DiscLosses, color="blue")
plt.xlim([0, epocaCorte])
plt.grid(True)
plt.ylabel("Losses")
plt.xlabel("Épocas")
plt.title("Loss del Discriminador")
plt.show()
if SAVE_MODEL:
    save_checkpoint(genX,filename=CHECKPOINT_GENERATOR_X)
    save_checkpoint(genY,filename=CHECKPOINT_GENERATOR_Y)
    save_checkpoint(Dx, filename=CHECKPOINT_DISCRIMINATOR_X)
    save_checkpoint(Dy,filename=CHECKPOINT_DISCRIMINATOR_Y)

