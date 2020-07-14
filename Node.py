class Node:
    
    def __init__(self, rule_attr, rule_val, class_val):
        self.left = None
        self.right = None
        self.rule_attr = rule_attr
        self.rule_val = rule_val
        self.class_val = None

    def insertNode(self, rule_attr, rule_val, choice, class_val):
        if choice == 'left':
            if self.left is None:
                self.left = Node(rule_attr, rule_val, class_val) #solda eleman yoksa oluştur
        elif(choice == 'right'):
            if self.right is None:
                self.right = Node(rule_attr, rule_val, class_val) #solda eleman yoksa oluştur    
    
    def printTree(self):
        print(self.rule_attr, self.rule_val, self.class_val)
        if self.left:
            print("left: ")
            self.left.printTree()
        
        if self.right:
            print("right: ")
            self.right.printTree()