class Node:
    def __init__(self, label, edge_list = [], boundary=False):
        self.label = label 
        self.edge_list = edge_list 
        self.boundary = boundary 
        self.value = None

    def getLabel(self):
        return self.label 

    def setEdgeList(self, edge_list):
        self.edge_list = edge_list
    
    def getEdgeList(self):
        return self.edge_list 
    
    def isBoundary(self):
        return self.boundary

    def getDegree(self):
        return float(len(self.edge_list))

    def setValue(self, val):
        self.value = val 

    def getValue(self):
        return self.value 