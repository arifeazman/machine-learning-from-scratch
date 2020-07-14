import numpy as np

class KNNClassifier:

    def __init__(self, dim, k):
        self.dim = dim  
        self.k = k
        
    def buildModel(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train 
          
    def calculateMinkowskiDist(self, vector1, vector2):  
        if(self.dim == "eucledean"):
            self.dim = 2
        elif(self.dim == "manhattan"):
            self.dim = 1             
        sumVal = 0    
        for i in range(0, vector1.size):
            sumVal = (vector1[i]-vector2[i])**self.dim
        return sumVal ** (1. / self.dim)    
 
    def predictClass(self, test_vector):
        ## Train deki bütün sıraların test verisi ile uzaklık verilerini listeye at
        train_set_size = self.X_train.shape[0]
        dist_and_label = np.zeros([train_set_size, 2], dtype=float)
        for i in range(0, train_set_size): ##Her eğitim verisindeki bir satır indisi
            dist = self.calculateMinkowskiDist(test_vector, self.X_train[i])
            label = self.y_train[i]
            dist_and_label[i] = dist, label

        ## Array dist verilerine göre sıralanıyor yeni bir array a atılıyor    
        sorted_dist_and_label = dist_and_label[dist_and_label[:,0].argsort()]
        ## Bu sıralanmış array de en küçük k kadar satır alınıyor
        k_sorted_dist_and_label = sorted_dist_and_label[:self.k,:]
        ## Etiket integer değerlerini ayırdım
        label_array = k_sorted_dist_and_label[:, -1]
        label_array_int = label_array.astype('int')
        counts = np.bincount(label_array_int)
        ## En sık çıkan label return ile döndürdüm
        self.result_class = np.argmax(counts)
        return self.result_class
    
    def evaluateModel(self, X_test, y_test):
        set_size, _ = X_test.shape
        result_classes = np.zeros(set_size)
        for i in range(0, set_size):
            result_classes[i] = self.predictClass(X_test[i])
        
        correct_num = 0
        for i in range(0, set_size):
            if(y_test[i] == result_classes[i]):
                correct_num = correct_num + 1        
        self.score = correct_num / set_size
        print("Score: ", self.score)

    def showLabel(self, row, data):
        target_names = data.target_names
        predicted_class = self.predictClass(row)
        label = target_names[predicted_class]
        print("Label: ", label)
        
        
        
        
        