import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

xData,yData = datasets.load_digits(return_X_y=True)

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.25, random_state=42)


clasificador = MLPClassifier(hidden_layer_sizes=(20,10),learning_rate_init=0.005, 
                                max_iter=500, activation='logistic',early_stopping=True,
                                validation_fraction=0.15,shuffle=True,random_state=0)#validationF=0.3
clasificador.fit(xTrain,yTrain)
yPredecida = clasificador.predict(xTest)
accuracy1Particion = accuracy_score(yTest,yPredecida)

numeroParticiones1 = 5 
numeroParticiones2 = 10
kf1 = KFold(n_splits=numeroParticiones1, shuffle=True)
kf2 = KFold(n_splits=numeroParticiones2, shuffle=True)

accuracyParticiones = []
for fold, (indiceEntrenamiento, indiceValidacion) in enumerate(kf1.split(xData)):#Split 
    x_train_fold, x_val_fold = xData[indiceEntrenamiento], xData[indiceValidacion]
    y_train_fold, y_val_fold = yData[indiceEntrenamiento], yData[indiceValidacion]
    
    clasificador = MLPClassifier(hidden_layer_sizes=(20,10),learning_rate_init=0.005, 
                                max_iter=500, activation='logistic',early_stopping=True,
                                validation_fraction=0.15,shuffle=True,random_state=0)
    clasificador.fit(x_train_fold,y_train_fold)
    yPredecida = clasificador.predict(x_val_fold)
    accuracyParticiones.append(accuracy_score(y_val_fold,yPredecida))#Guardamos las accuracy

#Media y varianza tasa de acierto 5 particiones 
mediaAccuracy5Particiones = np.mean(accuracyParticiones)
varianzaAccuracy5Particiones = np.var(accuracyParticiones)

accuracyParticiones = []
for fold, (indiceEntrenamiento, indiceValidacion) in enumerate(kf2.split(xData)):#Split 
    x_train_fold, x_val_fold = xData[indiceEntrenamiento], xData[indiceValidacion]
    y_train_fold, y_val_fold = yData[indiceEntrenamiento], yData[indiceValidacion]

    clasificador = MLPClassifier(hidden_layer_sizes=(20,10),learning_rate_init=0.005, 
                            max_iter=500, activation='logistic',early_stopping=True,
                            validation_fraction=0.15,shuffle=True,random_state=0)
    clasificador.fit(x_train_fold,y_train_fold)
    yPredecida = clasificador.predict(x_val_fold)
    accuracyParticiones.append(accuracy_score(y_val_fold,yPredecida))#Guardamos las accuracy

#Media y varianza tasa de acierto 5 particiones    
mediaAccuracy10Particiones = np.mean(accuracyParticiones)
varianzaAccuracy10Particiones = np.var(accuracyParticiones)
#Matriz de confusion
# matrizConfusion = confusion_matrix(yTest,yPredecida, labels=digits.target_names)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrizConfusion,displaxy_labels=digits.target_names)
# fig, ax = plt.subplots(figsize=(8,8))
# disp.plot(ax=ax,cmap="Blues",values_format='',colorbar=None)
# print('Accuracy: ',accuracy)
# plt.show()
print("Accuracy 1 partición: ", accuracy1Particion)
print("Media Accuracy 5 particiones: ", mediaAccuracy5Particiones)
print("Varianza Accuracy 5 particiones: ", varianzaAccuracy5Particiones)
print("Media Accuracy 10 particiones: ", mediaAccuracy10Particiones)
print("Varianza Accuracy 10 particiones: ", varianzaAccuracy10Particiones)