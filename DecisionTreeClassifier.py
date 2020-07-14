import numpy as np
from Node import *

class DecisionTreeClassifier():
    
    def __init__(self):
        self.node_num = 0
        
    def calculateGini(self, class_hist_list):
        class_total_sum = sum(class_hist_list)
        square_sum = 0
        if(class_total_sum == 0):
            square_sum = 0
        else:
            for i in range(0, len(class_hist_list)):
                ex = class_hist_list[i] / class_total_sum
                square_sum = square_sum + pow(ex, 2)
        return 1.0 - square_sum
              
    def calculateGiniForDataset(self, X_train, y_train):
        row_num, col_num = X_train.shape
        y_unique = np.unique(y_train) #bunlar class değerleri
        #[col_num, row_num, min_gini_val]
        min_values=[0, 0, 1]
        for col in range(0, col_num):
            column = X_train[:, col]
            for i in range(0, column.size): #Mesela 3. satırdaki değere göre split et
                #buradaki split değerim column[i]
                class_list_g1 = [0 for x in range(y_unique.size)]
                class_list_g2 = [0 for x in range(y_unique.size)]
                for j in range(0, column.size): #Bütün satırları gez
                    if(column[i] > column[j]): #group1                   
                        for k in range(0, y_unique.size): #class değerlerini gez uygun olana göre hist i arttır
                            if(y_train[j] == y_unique[k]):
                                class_list_g1[k] = class_list_g1[k] + 1
                    else: #group2                    
                        for k in range(0, y_unique.size):
                            if(y_train[j] == y_unique[k]):
                                class_list_g2[k] = class_list_g2[k] + 1
                gini_1 = self.calculateGini(class_list_g1)
                gini_2 = self.calculateGini(class_list_g2)
                prop_1 = sum(class_list_g1) / column.size
                prop_2 = sum(class_list_g2) / column.size
                gini = prop_1 * gini_1 + prop_2 * gini_2 #proportioned gini index
                if(gini < min_values[2]):
                    min_values = col, column[i], gini # min gini değerini güncelle               
        return min_values
 
    def splitDataset(self, X_train, y_train, attr, val, nod):
        #Find left and right
        self.node_num = self.node_num + 1
        left_list_X = []
        left_list_y = []
        right_list_X = []
        right_list_y = []
        split_attr, split_val, _ = self.calculateGiniForDataset(X_train, y_train)
        for i in range(X_train.shape[0]): #satırları gez
            if(X_train[i][split_attr] < split_val):
                left_list_X.append(X_train[i])
                left_list_y.append(y_train[i])
            else:
                right_list_X.append(X_train[i])
                right_list_y.append(y_train[i])
        if(len(left_list_y) == 0 or len(right_list_y) == 0):
            cl = np.argmax(np.bincount(y_train.astype(int)))
            nod.class_val = cl                
        else:
            left_X_train = np.array(left_list_X)
            left_y_train = np.array(left_list_y)
            right_X_train = np.array(right_list_X)
            right_y_train = np.array(right_list_y)
            if(left_y_train.size <2): #bölünen verilerde bir veri var
                cl = np.argmax(np.bincount(left_y_train.astype(int)))
                nod.insertNode(None, None, "left", cl) #leaf node
            elif(left_y_train.size >= 2):
                a1, v1, _ = self.calculateGiniForDataset(left_X_train, left_y_train)
                cl = np.argmax(np.bincount(left_y_train.astype(int)))
                nod.insertNode(a1, v1, "left", cl)
                self.splitDataset(left_X_train, left_y_train, a1, v1, nod.left)
                
            if(right_y_train.size <2): #bölünen verilerde bir veri var
                cl = np.argmax(np.bincount(right_y_train.astype(int)))
                nod.insertNode(None, None, "right", cl) #leaf node
            elif(right_y_train.size >= 2):
                a2, v2, _ = self.calculateGiniForDataset(right_X_train, right_y_train)
                cl = np.argmax(np.bincount(right_y_train.astype(int)))
                nod.insertNode(a2, v2, "right",cl)
                self.splitDataset(right_X_train, right_y_train, a2, v2, nod.right)     
            
    def buildModel(self, X_train, y_train):
        #Tüm verisetinin ginisini bul
        attr, val, _ = self.calculateGiniForDataset(X_train, y_train)
        #kök düğümü oluştur
        cl = np.argmax(np.bincount(y_train.astype(int)))
        self.root = Node(attr, val, cl)
        self.splitDataset(X_train, y_train, attr, val, self.root)
        
    def searchTree(self, row, node):
        if(node.rule_attr == None or node.rule_val == None): #if leaf node
            self.result_class = node.class_val
        elif(row[node.rule_attr] < node.rule_val):
            if(node.left == None):
                self.result_class = node.class_val
            else:
                self.searchTree(row, node.left)
        else:
            if(node.right == None):
                self.result_class = node.class_val                
            else:
                self.searchTree(row, node.right)
        return self.result_class      
      
    def evaluateModel(self, X_test, y_test):
        correct_num = 0
        for i in range(0, y_test.size):
            pred = self.searchTree(X_test[i], self.root)
            if(pred == y_test[i]):
                correct_num = correct_num + 1
        self.score = correct_num / y_test.size
        print("Score:", self.score)

    def showLabel(self, row, data):
        target_names = data.target_names
        predicted_class = self.searchTree(row, self.root)
        label = target_names[predicted_class]
        print("Label: ", label)
        
        
        