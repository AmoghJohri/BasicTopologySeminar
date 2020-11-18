from node import Node
import numpy as np 
import matplotlib.pyplot as plt 

def getArrPosition(arr, node):
    i = 0
    while i < len(arr):
        if arr[i].getLabel() == node.getLabel():
            return i
        i += 1
    return -1

def strictlyConvexMapping(G):
    # input is a Graph G, which is a list of Nodes 
    boundaryNodes = []
    interiorNodes = []

    for each in G:
        if each.isBoundary():
            boundaryNodes.append(each)
        else:
            interiorNodes.append(each) 
    
    Ax = [[0. for i in range(len(interiorNodes))] for j in range(len(interiorNodes))]
    bx = [0. for i in range(len(interiorNodes))]
    Ay = [[0. for i in range(len(interiorNodes))] for j in range(len(interiorNodes))]
    by = [0. for i in range(len(interiorNodes))]
    for each in interiorNodes:
        pos = getArrPosition(interiorNodes, each)
        di = each.getDegree()
        rhsx = 0.
        rhsy = 0.
        for neighbor in each.getEdgeList():
            if neighbor.isBoundary():
                rhsx += (1./di)*np.cos(2*np.pi*(getArrPosition(boundaryNodes, neighbor)+1.)/float(len(boundaryNodes)))
                rhsy += (1./di)*np.sin(2*np.pi*(getArrPosition(boundaryNodes, neighbor)+1.)/float(len(boundaryNodes)))
            else:
                Ax[pos][getArrPosition(interiorNodes, neighbor)] = (-1./di)
                Ay[pos][getArrPosition(interiorNodes, neighbor)] = (-1./di)
        Ax[pos][pos] = (1.)
        Ay[pos][pos] = (1.) 
        bx[pos] = rhsx 
        by[pos] = rhsy 

    X = (np.linalg.inv(np.asarray(Ax)).dot(np.asarray(bx))).tolist()
    Y = (np.linalg.inv(np.asarray(Ay)).dot(np.asarray(by))).tolist()

    for i in range(len(interiorNodes)):
        interiorNodes[i].setValue((X[i],Y[i]))
    for i in range(len(boundaryNodes)):
        boundaryNodes[i].setValue((np.cos(2*np.pi*(i+1)/float(len(boundaryNodes))), (np.sin(2*np.pi*(i+1)/float(len(boundaryNodes))))))
    
    return

if __name__ == "__main__":
    A = Node("A",boundary=True)
    B = Node("B",boundary=False)
    C = Node("C",boundary=True)
    D = Node("D",boundary=True)
    E = Node("E", boundary=False)
    F = Node("F", boundary=False)
    A.setEdgeList([B, C, D, E])
    B.setEdgeList([A, C, E, F])
    C.setEdgeList([A, B, D, F])
    D.setEdgeList([A, E, C, F])
    E.setEdgeList([A, B, D, F])
    F.setEdgeList([B, C, D, E])
    G = [A, B, C, D, E, F]
    strictlyConvexMapping(G)
    for each in G:
        plt.scatter(each.getValue()[0], each.getValue()[1], 100)
    plt.legend([each.getLabel() for each in G], loc = 'upper right')
    for each in G:
        for node in each.getEdgeList():
            plt.plot([each.getValue()[0], node.getValue()[0]], [each.getValue()[1], node.getValue()[1]])
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), linewidth=1,  color='k')
    plt.grid()
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Straight Line Embedding")
    plt.show()
    