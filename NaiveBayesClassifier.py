import numpy as np
from math import pi, sqrt, exp

class NaiveBayesClassifier():
    
    def __init__(self):
        self.dataset_statistics = {}      
        
        
    def getDatasetStatisticsForClass(self, X_train, y_train):
        self.classes = np.unique(y_train) 
        y_train_unique = np.unique(y_train)
        row_num, column_num = X_train.shape
        self.class_sizes = {}
        for class_value in y_train_unique:
            self.class_sizes[class_value] = 0
            class_index_list = []
            for i in range(0, y_train.size):
                if(y_train[i] == class_value):
                    class_index_list.append(i)
                    self.class_sizes[class_value] = self.class_sizes[class_value] + 1
            new_X_train = np.take(np.copy(X_train), class_index_list, 0)
            new_y_train = np.take(np.copy(y_train), class_index_list, 0)
            columns_mean = np.mean(new_X_train, axis=0) #along the column
            columns_std = np.std(new_X_train, axis=0, ddof=1)
            self.dataset_statistics[class_value] = [columns_mean, columns_std]
            del new_X_train
            del new_y_train
    
    def calculateGaussPDF(self, x, mean, std): #bir veri noktasının bir özelliği için      
        exponent = exp(-((x-mean)**2 / (2 * std**2 )))
        return (1 / (sqrt(2 * pi) * std)) * exponent 
        
    def buildModel(self, X_train, y_train):
        self.getDatasetStatisticsForClass(X_train, y_train)
      
    def predictClass(self, row):
        self.prob_dict = {}
        for c in self.classes: 
            prob = self.class_sizes[c] / float(sum(self.class_sizes.values()))
            for i in range(0, row.size): #column sayısı kadar
                mean = self.dataset_statistics[c][0][i]
                std = self.dataset_statistics[c][1][i]
                gauss = self.calculateGaussPDF(row[i], mean, std)
                prob *= gauss
            self.prob_dict[c] = prob  
        return max(self.prob_dict, key=self.prob_dict.get)
        
    def evaluateModel(self, X_test, y_test):
        correct_num = 0
        for i in range(0, y_test.size):
            pred = self.predictClass(X_test[i])
            if(pred == y_test[i]):
                correct_num = correct_num + 1
        self.score = correct_num / y_test.size
        print("Score: ", self.score)

    def showLabel(self, row, data):
        target_names = data.target_names
        predicted_class = self.predictClass(row)
        label = target_names[predicted_class]
        print("Label: ", label)
        
        
        