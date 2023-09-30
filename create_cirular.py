import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from parse_graph import read_graphml, write_graphml
import os
from metrics_suite import MetricsSuite
from simulated_annealing import SimulatedAnnealing

graphs_dir = "..\\Graphs\\"
drawings_dir = "..\\Graph Drawings\\"



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
        
        i += 1
        print(i, graph)
        G = nx.read_graphml(graphs_dir + graph_type + "\\" + graph)



        pos = nx.circular_layout(G)

        for node,(x,y) in pos.items():
            G.nodes[node]['x'] = float(x) * 750
            G.nodes[node]['y'] = float(y) * 750

        # nx.draw(G, pos=pos)
        # plt.show()

        write_graphml(G, "test1.graphml")

        if not os.path.exists(drawings_dir + graph_type + "\\" + "Circular" + "\\"):
            os.makedirs(drawings_dir + graph_type + "\\" + "Circular" + "\\")


        write_graphml(G, drawings_dir + graph_type + "\\" + "Circular" + "\\" + "circular_j0" + "_" + graph)