import numpy as np
import random

class MultiLayerPerceptronClassifier():
    
    def __init__(self, learning_rate, epoch, num_hidden ):
        self.lr = learning_rate
        self.epoch = epoch
        self.num_hidden = num_hidden
                
    def toColumnVector(self, row):
        """ y_train 'deki değerler column vector 'e çevriliyor"""
        matrix_row = np.matrix(row)
        return matrix_row.transpose()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def rowToOneHotEncodingMatrix(self, row):
        encod = np.zeros((row.size, np.unique(row).size), dtype=int)
        for i in range(0, row.size):
            val = row[i]
            encod[i][val] = 1
        return np.matrix(encod)

    def buildModel(self, X_train, y_train):
        self.num_class = np.unique(y_train).size
        y_train_encod = self.rowToOneHotEncodingMatrix(y_train)          
    
        # random weight matrix
        self.num_input, self.num_features = X_train.shape
        self.weights_i_h = np.matrix(np.random.uniform(low=0.1, high=0.3, size=(self.num_hidden, self.num_features)))
        self.weights_h_o = np.matrix(np.random.uniform(low=0.1, high=0.3, size=(self.num_class, self.num_hidden)))
        
        #iterate epoch times
        for k in range(0, self.epoch):
            #create shuffled number array from 0 to instance size
            iterate_list = list(range(0, self.num_input))
            random.shuffle(iterate_list)
            
            #iterate through X_train and y_train as rows
            for i in iterate_list:             
                inputs = self.toColumnVector(X_train[i])    
                target_outputs = self.toColumnVector(y_train_encod[i])
                
                # FEED FORWARD PROCESS
                hidden_in = self.weights_i_h * inputs #dim(num_hidden x 1)
                hidden_out = self.sigmoid(hidden_in) 
                output_in = self.weights_h_o * hidden_out #dim(num_class x 1)
                output_out = self.sigmoid(output_in)
                
                #Calculating errors
                error_out = target_outputs - output_out
                error_out = np.matrix(error_out)
                error_hidden = self.weights_h_o.transpose() * error_out #dim(num_hidden x 1)
                error_hidden = np.matrix(error_hidden)
                # BACK PROPAGATION PROCESS
                gradient_h_o = np.multiply(error_out, np.multiply(output_out, (1 - output_out)))
                delta_h_o = self.lr * gradient_h_o * self.toColumnVector(hidden_out)
                gradient_i_h = np.multiply(error_hidden, np.multiply(hidden_out, (1 - hidden_out)))
                delta_i_h =  self.lr *  gradient_i_h * self.toColumnVector(inputs)
                
                #Add errors
                self.weights_h_o = self.weights_h_o + delta_h_o
                self.weights_i_h = self.weights_i_h + delta_i_h
                
    def predictClass(self, row):  
        inputs = self.toColumnVector(row)  
        hidden_in = self.weights_i_h * inputs #dim(num_hidden x 1)
        hidden_out = self.sigmoid(hidden_in) 
        output_in = self.weights_h_o * hidden_out #dim(num_class x 1)
        output_out = self.sigmoid(output_in)        
        predicted_class = np.argmax(output_out)        
        return predicted_class
    
    def evaluateModel(self, X_test, y_test):  
        num_of_success = 0
        for i in range(0, X_test.shape[0]): 
            predicted_class = self.predictClass(X_test[i])
            if(predicted_class == y_test[i]):
                num_of_success = num_of_success + 1
        self.score = num_of_success / y_test.size
        print("Score is:", self.score)
    
    def showLabel(self, row, data):
        target_names = data.target_names
        predicted_class = self.predictClass(row)
        label = target_names[predicted_class]
        print("Label is: ", label)


