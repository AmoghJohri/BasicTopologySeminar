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
# if __name__ == "__main__":
#     # defining the vertices
#     A = Node("A",boundary=True)
#     B = Node("B",boundary=False)
#     C = Node("C",boundary=True)
#     D = Node("D",boundary=True)
#     E = Node("E", boundary=False)
#     F = Node("F", boundary=False)
#     # defining the edges
#     A.setEdgeList([B, C, D, E])
#     B.setEdgeList([A, C, E, F])
#     C.setEdgeList([A, B, D, F])
#     D.setEdgeList([A, E, C, F])
#     E.setEdgeList([A, B, D, F])
#     F.setEdgeList([B, C, D, E])
#     # defining the graph
#     G = [A, B, C, D, E, F]
#     # getting the straight line embedding
#     strictlyConvexMapping(G)
#     # plotting the vertices
#     for each in G:
#         plt.scatter(each.getValue()[0], each.getValue()[1], 100)
#     plt.legend([each.getLabel() for each in G], loc = 'upper right')
#     # plotting the edges
#     for each in G:
#         for node in each.getEdgeList():
#             plt.plot([each.getValue()[0], node.getValue()[0]], [each.getValue()[1], node.getValue()[1]])
#     # plotting a circle of unit radius
#     t = np.linspace(0,np.pi*2,100)
#     plt.plot(np.cos(t), np.sin(t), linewidth=1)
#     plt.grid()
#     plt.xlabel("X-Axis")
#     plt.ylabel("Y-Axis")
#     plt.title("Straight Line Embedding")
#     plt.show()
# main
if __name__ == "__main__":
    # defining the vertices
    A = Node("A", boundary=True)
    B = Node("B", boundary=True)
    C = Node("C", boundary=True)
    D = Node("D", boundary=True)
    E = Node("E", boundary=True)
    F = Node("F", boundary=True)
    G = Node("G", boundary=True)
    H = Node("H", boundary=True)
    I = Node("I", boundary=True)
    J = Node("J", boundary=True)
    K = Node("K", boundary=True)
    L = Node("L", boundary=True)
    M = Node("M", boundary=False)
    N = Node("N", boundary=False)
    O = Node("O", boundary=False)
    P = Node("P", boundary=False)
    Q = Node("Q", boundary=False)
    R = Node("R", boundary=False)
    S = Node("S", boundary=False)
    T = Node("T", boundary=False)
    U = Node("U", boundary=False)
    V = Node("V", boundary=False)
    W = Node("W", boundary=False)
    X = Node("X", boundary=False)
    Y = Node("Y", boundary=False)
    Z = Node("Z", boundary=False)
    # defining the edges
    A.setEdgeList([B, L, V, M])
    B.setEdgeList([A, C, M, N])
    C.setEdgeList([B, N, O, D])
    D.setEdgeList([C, O, P, E])
    E.setEdgeList([D, P, F])
    F.setEdgeList([E, P, Q, G])
    G.setEdgeList([F, Q, R, H])
    H.setEdgeList([G, R, S, I])
    I.setEdgeList([H, S, T, J])
    J.setEdgeList([I, T, U, K])
    K.setEdgeList([J, U, L])
    L.setEdgeList([K, U, V, A])
    M.setEdgeList([A, B, V, W, X, N])
    N.setEdgeList([B, C, M, X, O])
    O.setEdgeList([C, D, P, X, N])
    P.setEdgeList([D, E, F, Q, X, O])
    Q.setEdgeList([P, F, G, R, S, X])
    R.setEdgeList([G, H, Q, S])
    S.setEdgeList([Q, R, H, I, T, X])
    T.setEdgeList([X, S, I, J, U, V, Z, Y])
    U.setEdgeList([T, J, K, L, V])
    V.setEdgeList([T, U, L, A, M, W, Z])
    W.setEdgeList([M, X, Y, Z, V])
    X.setEdgeList([M, N, O, P, Q, S, T, Y, W])
    Y.setEdgeList([W, Z, X, T])
    Z.setEdgeList([Y, T, V, W])
    # defining the graph
    G = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
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
    plt.plot(np.cos(t), np.sin(t), linewidth=1, color='k')
    plt.grid()
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Straight Line Embedding")
    plt.show()
    