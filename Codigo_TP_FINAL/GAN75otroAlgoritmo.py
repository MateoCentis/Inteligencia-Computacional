import librosa as lb
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
LAMBDA = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GENERATOR_X = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/genx75o.pth"
CHECKPOINT_GENERATOR_Y = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/geny75o.pth"
CHECKPOINT_DISCRIMINATOR_X = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/discx75o.pth"
CHECKPOINT_DISCRIMINATOR_Y = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/modelos/discy75o.pth"
torch.manual_seed(111)

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
            NoisyLinear(75, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),#
            nn.Sigmoid(),
            nn.Linear(128, 75),
            nn.Sigmoid(),
            nn.Linear(75, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output

# Clase para la capa lineal con ruido gaussiano
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
            nn.Dropout(0.5),
            nn.Linear(256, 256),#segunda capa oculta
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 128),#tercera capa oculta #512 no mejora
            nn.ReLU(),
            nn.Dropout(0.5),
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
archivo1 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesM.txt"
archivo2 = "C:/Users/mateo/OneDrive/Escritorio/[Año 4] 2do CUATRIMESTRE/Inteligencia Computacional/TP FINAL/coeficientesT.txt"

datosM = leer_archivo_txt(archivo1)
datosT = leer_archivo_txt(archivo2)
sizeDatosM = datosM.shape[0]
sizeDatosT = datosT.shape[0]
datos_combinados = np.concatenate((datosM, datosT), axis=0)
dataset = TensorDataset(torch.tensor(datos_combinados),torch.cat((torch.zeros(sizeDatosM),torch.ones(sizeDatosT))))

train_ratio = 0.9
train_size = int(train_ratio*len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
# train_labels = torch.cat((torch.zeros(sizeDatosM),torch.ones(sizeDatosT)))#0: Mateo, 1: Tobi
# dataset = (dataset - torch.mean(dataset))/torch.std(dataset)
# train_labels = train_labels.unsqueeze(1)
# dataset = torch.cat((dataset,train_labels), dim=1)
# X_train, X_test, y_train, y_test = train_test_split(dataset, test_size=0.1) #80% para entrenamiento
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)#10% validación y 10% testeo

#---------------------------------------------Armar batches-------------------------------------------------
batch_size = 3000 #61 batches con 1146 o 122 con 573 
batch_number = train_size/batch_size
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#----------------------------------------Parámetros del entrenamiento------------------------------------------
#x hablante de origen, y hablante de destino
genX = Generator().to(device=device) #mapea x->y
genY = Generator().to(device=device) #mapea y->x                                 
Dx = Discriminator().to(device=device) # distringue X de Xapprox
Dy = Discriminator().to(device=device) # distingue Y de Yapprox
#Inicialización de pesos (puede estancar la convegencia?)
# genX.apply(weights_init)
# genY.apply(weights_init)
# Dx.apply(weights_init)
# Dy.apply(weights_init)

learning_rate = 0.001                                               
num_epochs = 250 #entre 60 y 400
# optimizer_discriminatorX = torch.optim.Adam(Dx.parameters(),lr=learning_rate*0.1) #probar mejorando lr de disc
# optimizer_discriminatorY = torch.optim.Adam(Dy.parameters(),lr=learning_rate*0.1)
# optimizer_generatorX = torch.optim.Adam(genX.parameters(),lr=learning_rate) 
# optimizer_generatorY = torch.optim.Adam(genY.parameters(),lr=learning_rate)
optimizer_generator = torch.optim.Adam(itertools.chain(genY.parameters(),genX.parameters()),lr=learning_rate)
optimizer_discriminator =  torch.optim.Adam(itertools.chain(Dy.parameters(),Dx.parameters()),lr=learning_rate*0.1)
# Ejemplo de gradient clipping
# MAX_GRAD_NORM = 1
# torch.nn.utils.clip_grad_norm_(genX.parameters(), max_norm=MAX_GRAD_NORM)
# torch.nn.utils.clip_grad_norm_(genY.parameters(), max_norm=MAX_GRAD_NORM)
# torch.nn.utils.clip_grad_norm_(Dx.parameters(), max_norm=MAX_GRAD_NORM)
# torch.nn.utils.clip_grad_norm_(Dy.parameters(), max_norm=MAX_GRAD_NORM)

L1 = nn.L1Loss()
# bce = nn.BCELoss()
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
        #----------------------------------------Entrenamiento de generadores--------------------------------------
        #Durante el entrenamiento de generadores desactivar los discriminadores
        # with torch.cuda.amp.autocast():
        genX.train()
        genY.train()
        #NUEVO-------------
        optimizer_generator.zero_grad(set_to_none=True)
        for d_parameters in Dx.parameters():
            d_parameters.requires_grad = False
        for d_parameters in Dy.parameters():
            d_parameters.requires_grad = False
        #---------------
        #Identity Loss
        identity_Y = genY(Y)
        identity_X = genX(X)
        identity_Y_loss = L1(Y,identity_Y)
        identity_X_loss = L1(X,identity_X)
        identity_loss = (identity_X_loss+identity_Y_loss)/2

        #Adversarial Loss (GAN loss)
        fakeX = genX(Y)
        Dx_fake = Dx(fakeX)
        fakeY = genY(X)
        Dy_fake = Dy(fakeY)
        lossG_X = mse(Dx_fake, torch.ones_like(Dx_fake))
        lossG_Y = mse(Dy_fake, torch.ones_like(Dy_fake))
        adversarialLoss = (lossG_X+lossG_Y)/2

        #Cycle Loss
        cycleY = genY(fakeX)
        cycleX = genX(fakeY)
        cycleY_loss = L1(Y,cycleY)
        cycleX_loss = L1(X,cycleX)
        cycle_loss = (cycleX_loss+cycleY_loss)/2
        #Total Loss
        G_loss = (adversarialLoss + cycle_loss*LAMBDA + 5*identity_loss)
        
        G_loss.backward()
        optimizer_generator.step()
        #---------------------------------Entrenamiento de discriminadores---------------------------------------
        #Discriminador X-----------------------------------
        optimizer_discriminator.zero_grad(set_to_none=True)
        for d_parameters in Dx.parameters():
            d_parameters.requires_grad = True
    
        fakeX = genX(Y)
        Dx_fake = Dx(fakeX)
        Dx_real = Dx(X)

        Dx_real_loss = mse(Dx_real, torch.ones_like(Dx_real)) #SMOOTHING LABELS -0.1*torch.ones_like(Dx_real)
        Dx_fake_loss = mse(Dx_fake, torch.zeros_like(Dx_fake)) #+0.1*torch.ones_like(Dx_fake)
        Dx_loss = (Dx_real_loss + Dx_fake_loss)/2

        Dx_loss.backward()

        #Discriminador Y----------------------------------
        optimizer_discriminator.zero_grad(set_to_none=True)
        for d_parameters in Dy.parameters():
            d_parameters.requires_grad = True
        fakeY = genY(X)

        Dy_real = Dy(Y)
        Dy_fake = Dy(fakeY) #Separa fakeY de la GPU (para no usarla para calcular gradientes)
        Dy_real_loss = mse(Dy_real, torch.ones_like(Dy_real))
        Dy_fake_loss = mse(Dy_fake, torch.zeros_like(Dy_fake))
        Dy_Loss = (Dy_real_loss + Dy_fake_loss)/2
        
        Dy_Loss.backward()

        optimizer_discriminator.step()
        
        D_loss = (Dy_Loss + Dx_loss) / 2


        if indiceBatch % (batch_number//2 + 1) == 0 and indiceBatch > 0:#para que solo se tenga en cuenta un batch
            # print("Gradiente DX: ", )
            # print("Gradiente DY: ")
            # print("Gradiente G: ")
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
            # MCDX = mel_cepstral_distortion(X.detach().cpu().numpy(),fakeX.detach().cpu().numpy())
            # MCDY = mel_cepstral_distortion(Y.detach().cpu().numpy(),fakeY.detach().cpu().numpy())
            # print(f"Epoch: {epoch} MCDX: {MCDX}")
            # print(f"Epoch: {epoch} MCDY: {MCDY}")
    if epocasSinMejora > paciencia:
        print("Corta por iteraciones sin mejores en la época: ", epoch)
        epocaCorte = epoch
        break
plt.figure()
plt.plot(GenLosses, color='red')
plt.grid(True)
plt.xlim([0,epocaCorte])
plt.ylabel('Losses')
plt.xlabel('Épocas')
plt.title('Loss del generador')
plt.figure()
plt.plot(DiscLosses, color='blue')
plt.xlim([0,epocaCorte])
plt.grid(True)
plt.ylabel('Losses')
plt.xlabel('Épocas')
plt.title('Loss del Discriminador')
plt.show()
if SAVE_MODEL:
    save_checkpoint(genX,filename=CHECKPOINT_GENERATOR_X)
    save_checkpoint(genY,filename=CHECKPOINT_GENERATOR_Y)
    save_checkpoint(Dx, filename=CHECKPOINT_DISCRIMINATOR_X)
    save_checkpoint(Dy,filename=CHECKPOINT_DISCRIMINATOR_Y)

