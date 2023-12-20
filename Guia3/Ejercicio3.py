import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn import datasets
from sklearn.model_selection import  KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
#Importe de clasificadores
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #Analisis discriminante lineal
from sklearn.neighbors import KNeighborsClassifier #K neighbors
from sklearn.tree import DecisionTreeClassifier  #Arbol de decision
from sklearn import svm #Maquina de vectores
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
#---------------------------------------------ahora-------------------------------------------------
xData,yData = datasets.load_wine(return_X_y=True)
numeroVecinos = 10
capas = (20,10)
velocidadAprendizaje = 0.005
maxIteraciones = 500
porcionMonitoreo = 0.15
clasificador = MLPClassifier(hidden_layer_sizes=capas,learning_rate_init=velocidadAprendizaje, 
                            max_iter=maxIteraciones, activation='logistic',early_stopping=True,
                            validation_fraction=porcionMonitoreo,shuffle=True,random_state=0)
gnb = GaussianNB(var_smoothing=10e-2)
lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
neigh = KNeighborsClassifier(n_neighbors=numeroVecinos, weights="distance",algorithm='auto',
                                metric='minkowski',p=2)
dTree = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=30,
                                min_samples_split=10, min_samples_leaf=5,max_features=40)
suvm = svm.SVC(C=5,kernel='rbf')

kf = KFold(n_splits=5, shuffle=True, random_state=42)   
accuracyParticiones = np.zeros((5,2))
for fold, (indiceEntrenamiento, indiceValidacion) in enumerate(kf.split(xData)):
    x_train_fold, x_val_fold = xData[indiceEntrenamiento], xData[indiceValidacion]
    y_train_fold, y_val_fold = yData[indiceEntrenamiento], yData[indiceValidacion]

    estimadorBase = dTree
    #max_samples: número de muestras de los datos de entrenamiento para cada clasificador (0,1) en porcentaje
    #max_features: número de caracterisitcas a considerar al ajustar cada clasificador (0,1) para porcentaje
    #bootstrap: booleano indica si se deben muestrear con reemplazo (True) o sin reemplazo (True)
    #---------------------------------------------Bagging-------------------------------------------------
    bagging_classifier = BaggingClassifier(estimator=estimadorBase, n_estimators=50, random_state=42)
    yPredecida = bagging_classifier.fit(x_train_fold,y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,0] = accuracy_score(y_val_fold,yPredecida)
    #---------------------------------------------AdaBoost-------------------------------------------------
    adaboost_classifier = AdaBoostClassifier(estimator=estimadorBase, n_estimators=50, random_state=42)
    yPredecida = adaboost_classifier.fit(x_train_fold,y_train_fold).predict(x_val_fold)
    accuracyParticiones[fold,1] = accuracy_score(y_val_fold,yPredecida)


# Calcular la precisión promedio para cada modelo
bagging_mean_accuracy = np.mean(accuracyParticiones[:,0])
adaboost_mean_accuracy = np.mean(accuracyParticiones[:,1])
bagging_mean_varianza = np.var(accuracyParticiones[:,0])
adaboost_mean_varianza = np.var(accuracyParticiones[:,1])

headers = ['Método', 'Media', 'Varianza']

# Datos de ejemplo
# data = [['Alice', 25, 95],['Bob', 30, 87],['Charlie', 35, 92]]
data = [['Bagging',bagging_mean_accuracy,bagging_mean_varianza],['AdaBoost',adaboost_mean_accuracy,adaboost_mean_varianza]]

# Mostrar la tabla
print(tabulate(data, headers, tablefmt='fancy_grid'))
