import numpy as np

class LinearRegression:
    
    def __init__(self):                
         pass
        
    def calculateB1(self, X_train, y_train):
        nom = 0
        denum = 0
        for x, y in zip(X_train, y_train):
            nom += ((x - X_train.mean()) * (y - y_train.mean()))
            denum += (x - X_train.mean())**2
        return nom/denum
    
    def buildModel(self, X_train, y_train):       
        self.b1 = self.calculateB1(X_train, y_train)
        self.b0 = y_train.mean() - self.b1 * X_train.mean()  

    def predictValue(self, x):
        print("Predicted value: ", self.b0 + x*self.b1)
        
    def predictAllValues(self, X_test):      
        predict_list = []
        for x in X_test:
            predict_list.append(self.b0 + x*self.b1) 
        y_predicted = np.array(predict_list)
        return y_predicted

    def evaluateModel(self, X_test, y_test):
        sse_sum = 0
        ssr_sum = 0
        y_mean = y_test.mean()           
        y_predicted = self.predictAllValues(X_test)
        for y_i, y_p in zip(y_test, y_predicted):        
            sse_sum += (y_i - y_p)**2
            ssr_sum += (y_p - y_mean)**2
        self.r2 = ssr_sum / (ssr_sum + sse_sum)
        #R2 is determination coefficient
        print("R2 score: ", self.r2)
        
        
        
        