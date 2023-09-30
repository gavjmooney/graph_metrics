import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from parse_graph import read_graphml, write_graphml
import os
from polygon_layout import polygon_layout


def load_graph(filename):
    G = nx.read_graphml(filename)
    G = G.to_undirected()
    return G

    # for node in G.nodes:
    #     try:
    #         # Assign node attrbiutes for coordinate position of nodes
    #         G.nodes[node]['x'] = float(G.nodes[node]['x'])
    #         G.nodes[node]['y'] = float(G.nodes[node]['y'])

    #     except KeyError:
    #         # Graph doesn't have positional attributes
    #         #print("Graph does not contain positional attributes. Assigning them randomly.")
    #         pos = nx.random_layout(G)
    #         for k,v in pos.items():
    #             pos[k] = {"x":v[0]*G.number_of_nodes()*20, "y":v[1]*G.number_of_nodes()*20}

    #         nx.set_node_attributes(G, pos)

#G = read_graphml("data-10-2.graphml")

# G = nx.read_graphml("data-10-2.graphml")


# pos = nx.spring_layout(G)

# for node,(x,y) in pos.items():
#     G.nodes[node]['x'] = float(x) * 750
#     G.nodes[node]['y'] = float(y) * 750


# write_graphml(G, "LFRout.graphml")
# quit()


graphs_dir = "..\\Graphs\\"
drawings_dir = "..\\Graph Drawings\\"


algs = [
    #(nx.circular_layout, "Circular"),
    #(nx.kamada_kawai_layout, "Kamada-Kawai"),
    #(nx.spring_layout, "Fruchterman-Reingold"),
    (nx.random_layout, "Random"),
    #(nx.spiral_layout, "Spiral"),
    #(nx.spectral_layout, "Spectral"),
]

#HOLA, POLYGON, GRID?, BIGANGLE, GRADIENTDESCENT?,

# extra = [
#     (nx.spectral_layout, "Spectral"),
#     (),
# ]

graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Newman-Watts-Strogatz", "Geometric", "Stochastic-Block-Model"]
graph_types = ["Rome", "North"]
graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Newman-Watts-Strogatz", "Geometric", "Stochastic-Block-Model", "Rome", "North"]

i = 0



for graph_type in os.listdir(graphs_dir):
    if graph_type not in graph_types:
        continue
    for graph in os.listdir(graphs_dir + graph_type):
        if graph.endswith("gml") or graph.endswith('txt') or graph.endswith('vna'):
            continue
        #G = load_graph(graphs_dir + graph_type + "\\" + graph)

        # avg_degree = 0
        # #print(nx.density(G))
        # for n, deg in G.degree:
        #     avg_degree += deg
        # avg_degree = avg_degree / G.number_of_nodes()
        # print(graph)
        # print(avg_degree)
        # continue

        G = nx.read_graphml(graphs_dir + graph_type + "\\" + graph)
        # G = polygon_layout(G, 5)
        # write_graphml(G, drawings_dir + graph_type + "\\" + "Polygon5" + "\\poly5_" + graph)
        # continue

        for alg in algs:
            i += 1
            #print(i)

            pos = alg[0](G)

            for node,(x,y) in pos.items():
                G.nodes[node]['x'] = float(x) * 750
                G.nodes[node]['y'] = float(y) * 750

            if not os.path.exists(drawings_dir + graph_type + "\\" + alg[1] + "\\"):
                os.makedirs(drawings_dir + graph_type + "\\" + alg[1] + "\\")


            write_graphml(G, drawings_dir + graph_type + "\\" + alg[1] + "\\" + str.lower(alg[1]) + "_" + graph)

            # if i == 5:
            #     quit()
