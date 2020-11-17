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
        Ax[pos][pos] = (-1./di)
        Ay[pos][pos] = (-1./di) 
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
    A.setEdgeList([B, C, D])
    A_edge = [1, 2, 3]
    B.setEdgeList([A, C, D])
    B_edge = [1, 2, 3]
    C.setEdgeList([A, B, D])
    C_edge = [0, 1, 3]
    D.setEdgeList([A, B, C])
    D_edge = [0, 1, 2]
    G = [A, B, C, D]
    strictlyConvexMapping(G)
    X = []
    Y = []
    colors = ['c', 'b', 'g', 'r']
    k = 0
    for each in G:
        X.append(each.getValue()[0])
        Y.append(each.getValue()[1])
        plt.scatter(X[-1], Y[-1], 100, color = colors[k])
        k += 1
    plt.legend(["A", "B", "C", "D"])
    for each in A_edge:
        plt.plot([X[0], X[each]], [Y[0], Y[each]], color='k')
    for each in B_edge:
        plt.plot([X[1], X[each]], [Y[1], Y[each]],  color='k')
    for each in C_edge:
        plt.plot([X[2], X[each]], [Y[2], Y[each]],  color='k')
    for each in D_edge:
        plt.plot([X[3], X[each]], [Y[3], Y[each]],  color='k')
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), linewidth=1,  color='k')
    plt.grid()
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Straight Line Embedding")
    plt.show()
    