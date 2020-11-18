# importing libraries
import numpy             as np 
import matplotlib.pyplot as plt 
# import node to define the graph
from   node import Node
# function for straight line embedding algorithm
def strictlyConvexMapping(G):
    # input is a Graph G, which is a list of Nodes 
    boundaryNodes = []
    interiorNodes = []
    # separating boundary vertices and interior vertices
    for each in G:
        if each.isBoundary():
            boundaryNodes.append(each)
        else:
            interiorNodes.append(each) 
    # defining placeholders for system of linear equations for X and Y co-ordinates
    Ax = [[0. for i in range(len(interiorNodes))] for j in range(len(interiorNodes))]
    bx = [0.  for i in range(len(interiorNodes))]
    Ay = [[0. for i in range(len(interiorNodes))] for j in range(len(interiorNodes))]
    by = [0.  for i in range(len(interiorNodes))]
    for each in interiorNodes:
        pos  = interiorNodes.index(each)
        di   = each.getDegree()
        rhsx = 0.
        rhsy = 0.
        for neighbor in each.getEdgeList():
            if neighbor.isBoundary():
                rhsx += (1./di)*np.cos(2*np.pi*(boundaryNodes.index(neighbor)+1.)/float(len(boundaryNodes)))
                rhsy += (1./di)*np.sin(2*np.pi*(boundaryNodes.index(neighbor)+1.)/float(len(boundaryNodes)))
            else:
                Ax[pos][interiorNodes.index(neighbor)] = (-1./di)
                Ay[pos][interiorNodes.index(neighbor)] = (-1./di)
        Ax[pos][pos] = (1.)
        Ay[pos][pos] = (1.) 
        bx[pos]      = rhsx 
        by[pos]      = rhsy 
    # solving the system of linear equations
    X = (np.linalg.inv(np.asarray(Ax)).dot(np.asarray(bx))).tolist()
    Y = (np.linalg.inv(np.asarray(Ay)).dot(np.asarray(by))).tolist()
    # setting the value for interior vertices
    for i in range(len(interiorNodes)):
        interiorNodes[i].setValue((X[i],Y[i]))
    # setting the value for boundary vertices
    for i in range(len(boundaryNodes)):
        boundaryNodes[i].setValue((np.cos(2*np.pi*(i+1)/float(len(boundaryNodes))), (np.sin(2*np.pi*(i+1)/float(len(boundaryNodes))))))
    # done
    return
# main function
if __name__ == "__main__":
    # defining the vertices
    A = Node("A",boundary=True)
    B = Node("B",boundary=False)
    C = Node("C",boundary=True)
    D = Node("D",boundary=True)
    E = Node("E", boundary=False)
    F = Node("F", boundary=False)
    # defining the edges
    A.setEdgeList([B, C, D, E])
    B.setEdgeList([A, C, E, F])
    C.setEdgeList([A, B, D, F])
    D.setEdgeList([A, E, C, F])
    E.setEdgeList([A, B, D, F])
    F.setEdgeList([B, C, D, E])
    # defining the graph
    G = [A, B, C, D, E, F]
    # getting the straight line embedding
    strictlyConvexMapping(G)
    # plotting the vertices
    for each in G:
        plt.scatter(each.getValue()[0], each.getValue()[1], 100)
    plt.legend([each.getLabel() for each in G], loc = 'upper right')
    # plotting the edges
    for each in G:
        for node in each.getEdgeList():
            plt.plot([each.getValue()[0], node.getValue()[0]], [each.getValue()[1], node.getValue()[1]])
    # plotting a circle of unit radius
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    plt.grid()
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Straight Line Embedding")
    plt.show()
    