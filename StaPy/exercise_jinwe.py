import networkx as nx
import numpy as np

#  the given undirected network graph is a 6 by 6 grid
g = nx.grid_2d_graph(6, 6, create_using=nx.DiGraph)
g = nx.convert_node_labels_to_integers(g)
# you can visualize this network and the labels
planar = nx.planar_layout(g)
nx.draw_networkx(g, layout=planar)
#  we supply a simple demand matrix
od_matrix = np.zeros(shape=(36, 36))
# lets say that the cells on the corners send traffic to one another
tmp = [0, 5, 30, 35]
for i in tmp:
    for j in tmp:
        if i != j:
            od_matrix[i][j] = 10

# the cost on a link is a simple BPR function, free flow travel times are set at 1 capacity is set at 5 for all (i,j)
capacity = 5

def cost(flow):
    return 1.0 + np.multiply(0.15, pow(flow / capacity, 4))
