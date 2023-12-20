import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
#Importe de clasificadores
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #Analisis discriminante lineal
from sklearn.neighbors import KNeighborsClassifier #K neighbors
from sklearn.tree import DecisionTreeClassifier  #Arbol de decision
from sklearn import svm #Maquina de vectores
#---------------------------------------------Parámetros-------------------------------------------------
xData,yData = datasets.load_digits(return_X_y=True)
# digits = datasets.load_digits()
#xData 64 columnas y 1797 filas
#yData del 0 al 8
kf = KFold(n_splits=10, shuffle=True)
numeroVecinos = 10
capas = (20,10)
velocidadAprendizaje = 0.005
maxIteraciones = 500
porcionMonitoreo = 0.15
accuracyParticiones = np.zeros((10,6)) #Filas: folds ; Columnas: metodos
for fold, (indiceEntrenamiento, indiceValidacion) in enumerate(kf.split(xData)):
    x_train_fold, x_val_fold = xData[indiceEntrenamiento], xData[indiceValidacion]
    y_train_fold, y_val_fold = yData[indiceEntrenamiento], yData[indiceValidacion]
    #-------------------------------------Perceptron Multicapa----------------------------------------------
    clasificador = MLPClassifier(hidden_layer_sizes=capas,learning_rate_init=velocidadAprendizaje, 
                                max_iter=maxIteraciones, activation='logistic',early_stopping=True,
                                validation_fraction=porcionMonitoreo,shuffle=True,random_state=0)
    clasificador.fit(x_train_fold,y_train_fold)
    yPredecida = clasificador.predict(x_val_fold)
    accuracyParticiones[fold,0] = accuracy_score(y_val_fold,yPredecida)#Guardamos las accuracy
    # ------------------------------------------Naive Bayes--------------------------------------------------
    gnb = GaussianNB(var_smoothing=10e-2)#Parámetros priors: matriz de flotantes para poner probabilidades a priori (sino se asume uniformes)
                        #var_smoothing:número flotante para controlar la suavización en problemas de varianza nula 10e-9 default
    yPredecida = gnb.fit(x_train_fold, y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,1] = accuracy_score(y_val_fold,yPredecida)
    # #------------------------------------Analisis discrimintante lineal-------------------------------------
    lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    #solver elige con que calcular la matriz de dispersion
    #shrinkage controla la estimación de la matriz de dispersion (auto)
    #n_jobs para usar todos los cores de la cpu
    yPredecida = lda.fit(x_train_fold,y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,2] = accuracy_score(y_val_fold,yPredecida)
    # #------------------------------------------K neighbours-------------------------------------------------
    neigh = KNeighborsClassifier(n_neighbors=numeroVecinos, weights="distance",algorithm='auto',
                                    metric='minkowski',p=2)
    neigh.fit(x_train_fold,y_train_fold)
    yPredecida = neigh.predict(x_val_fold) #noqa
    accuracyParticiones[fold,3] = accuracy_score(y_val_fold,yPredecida) 
    # #----------------------------------------Árbol de decisión----------------------------------------------
    dTree = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=30,min_samples_split=10, min_samples_leaf=5,
                                    max_features=40)
    #criterion => gini o entropy
    #splitter => especifa como dividir cada nodo
    #max_depth para prevenir sobreajuste
    #min_samples_split (minimo muestras para dividir) minimo de muestras para que un nodo se divida (SUBIENDO EMPEORA)
    #min_samples_leaf minimo muestras para que una hoja sea considerada hoja
    #max_features 
    yPredecida = dTree.fit(x_train_fold,y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,4] = accuracy_score(y_val_fold,yPredecida)
    # #---------------------------------------------SVM-------------------------------------------------------
    suvm = svm.SVC(C=5,kernel='rbf')
    #degree en caso de ser poly
    yPredecida = suvm.fit(x_train_fold,y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,5] = accuracy_score(y_val_fold,yPredecida)

accuracyMedioPerceptron = np.mean(accuracyParticiones[:,0])
accuracyVarianzaPerceptron = np.var(accuracyParticiones[:,0])

accuracyMedioNB = np.mean(accuracyParticiones[:,1])
accuracyVarianzaNB = np.var(accuracyParticiones[:,1])

accuracyMedioLDA = np.mean(accuracyParticiones[:,2])
accuracyVarianzaLDA = np.var(accuracyParticiones[:,2])

accuracyMedioKn = np.mean(accuracyParticiones[:,3])
accuracyVarianzaKn = np.var(accuracyParticiones[:,3])

accuracyMedioTree = np.mean(accuracyParticiones[:,4])
accuracyVarianzaTree = np.var(accuracyParticiones[:,4])

accuracyMedioSVM = np.mean(accuracyParticiones[:,5])
accuracyVarianzaSVM = np.var(accuracyParticiones[:,5])
#---------------------------------------------Gráficos y tablas-------------------------------------------------
nombres = ['MLPC', 'GaussianNB', 'LDA', 'Kneighbours', 'dTree', 'SVM']
medias = [accuracyMedioPerceptron,accuracyMedioNB,accuracyMedioLDA,
                accuracyMedioKn, accuracyMedioTree, accuracyMedioSVM]
varianzas = [accuracyVarianzaPerceptron,accuracyVarianzaNB,accuracyVarianzaLDA,
                accuracyVarianzaKn, accuracyVarianzaTree, accuracyVarianzaSVM]
# Encabezados de columna
headers = ['Método', 'Media', 'Varianza']

# Datos de ejemplo
# data = [['Alice', 25, 95],['Bob', 30, 87],['Charlie', 35, 92]]
data = [[nombres[0],medias[0],varianzas[0]],[nombres[1],medias[1],varianzas[1]],[nombres[2],medias[2],varianzas[2]],
        [nombres[3],medias[3],varianzas[3]],[nombres[4],medias[4],varianzas[4]],[nombres[5],medias[5],varianzas[5]]]

# Mostrar la tabla
print(tabulate(data, headers, tablefmt='fancy_grid'))

# Datos de ejemplo
plt.figure(1)


# Crear un gráfico de barras con seaborn
sns.barplot(x=nombres, y=medias)
plt.xlabel('Métodos utilizados')
plt.ylabel('Media de exactitud')
plt.title('Accuracy media de cada método')

plt.figure(2)

sns.barplot(x=nombres, y=varianzas)
plt.xlabel('Métodos utilizados')
plt.ylabel('Varianza de exactitud')
plt.title('Varianza de la exactitud de cada método')

plt.show()
