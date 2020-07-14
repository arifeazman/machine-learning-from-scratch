## Bulunduğumuz klasörü veri yoluna eklemek gerekmektedir
## Klasör veri yoluna 
## import sys 
## sys.append.path("bulunduğumuz dosyanın konumu") 
## kodları ile eklenebilir
## Veya Spyder 'da sağ üst köşede bulunan dosya yolu
## Bulunduğumuz klasör olarak seçilirse kodlar çalışacaktır


## Lineer regresyon test kodları


import numpy as np
from sklearn import datasets
boston_X, boston_y = datasets.load_boston(return_X_y=True)
boston_X = boston_X[:, np.newaxis, 5]
X_train = boston_X[:-20]
X_test = boston_X[-20:]
y_train = boston_y[:-20]
y_test = boston_y[-20:]

from LinearRegression import *
lin = LinearRegression()
lin.buildModel(X_train, y_train)
lin.evaluateModel(X_test, y_test)
lin.predictValue(5)


###########################################
## KNN test kodları


from sklearn.datasets import load_iris
from sklearn.utils import shuffle
iris_X, iris_y = load_iris(return_X_y=True)
iris_X, iris_y = shuffle(iris_X, iris_y)
X_train = iris_X[:-30]
X_test = iris_X[-30:]
y_train = iris_y[:-30]
y_test = iris_y[-30:] 

from KNNClassifier import *                     
knn = KNNClassifier("eucledean", 10)
knn.buildModel(X_train, y_train)
knn.evaluateModel(X_test, y_test)        
knn.showLabel(X_test[5], load_iris())


#########################################
## Naive bayes test kodları


from sklearn.datasets import load_iris
from sklearn.utils import shuffle
iris_X, iris_y = load_iris(return_X_y=True)
iris_X, iris_y = shuffle(iris_X, iris_y)
X_train = iris_X[:-30]
X_test = iris_X[-30:]
y_train = iris_y[:-30]
y_test = iris_y[-30:]

from NaiveBayesClassifier import *
bayes = NaiveBayesClassifier()
bayes.buildModel(X_train, y_train)
bayes.evaluateModel(X_test, y_test)
bayes.showLabel(X_test[3], load_iris())

print(load_iris().target_names)
print(y_test[3])


##################################################
## Karar ağacı test kodları


from sklearn.datasets import load_iris
from sklearn.utils import shuffle
iris_X, iris_y = load_iris(return_X_y=True)
iris_X, iris_y = shuffle(iris_X, iris_y)
X_train = iris_X[:-30]
X_test = iris_X[-30:]
y_train = iris_y[:-30]
y_test = iris_y[-30:]

from DecisionTreeClassifier import *
dt = DecisionTreeClassifier() 
dt.buildModel(X_train, y_train)  
dt.evaluateModel(X_test, y_test)
dt.showLabel(X_test[9], load_iris())
print(dt.node_num)


####################################################
## MLP test kodları

### CANCER veriseti için

from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle

cancer_X, cancer_y = load_breast_cancer(return_X_y=True)
cancer_X, cancer_y = shuffle(cancer_X, cancer_y)
X_train = cancer_X[:-30]
X_test = cancer_X[-30:]
y_train = cancer_y[:-30]
y_test = cancer_y[-30:]
from sklearn.preprocessing import MaxAbsScaler
max_abs_scaler = MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_test_maxabs = max_abs_scaler.fit_transform(X_test)

from MultiLayerPerceptronClassifier import *
mlp = MultiLayerPerceptronClassifier(0.01, 100, 100)
mlp.buildModel(X_train_maxabs, y_train) 
mlp.evaluateModel(X_test_maxabs, y_test)
mlp.predictClass(X_test_maxabs[0])
mlp.showLabel(X_test_maxabs[0], load_breast_cancer())


### IRIS veriseti için

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
iris_X, iris_y = load_iris(return_X_y=True)
iris_X, iris_y = shuffle(iris_X, iris_y)
X_train = iris_X[:-30]
X_test = iris_X[-30:]
y_train = iris_y[:-30]
y_test = iris_y[-30:]

from MultiLayerPerceptronClassifier import *
mlp = MultiLayerPerceptronClassifier(0.1, 100, 100)
mlp.buildModel(X_train, y_train) 
mlp.evaluateModel(X_test, y_test)
mlp.showLabel(X_test[0], load_iris())
