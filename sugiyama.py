from grandalf.graphs import Vertex,Edge,Graph,graph_core
from grandalf.utils.nx import *
import networkx as nx
import parse_graph
import matplotlib.pyplot as plt
from grandalf.layouts import SugiyamaLayout
import os

class defaultview(object):
    w,h = 25, 25

graphs_dir = "..\\Graphs\\"
drawings_dir = "..\\Graph Drawings\\"


graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Newman-Watts-Strogatz", "Geometric", "Stochastic-Block-Model", "Rome", "North"]
#graph_types = ["Rome", "North"]


for graph_type in os.listdir(graphs_dir):
    if graph_type not in graph_types:
        continue
    for graph in os.listdir(graphs_dir + graph_type):
        if graph.endswith("gml") or graph.endswith('txt') or graph.endswith('vna'):
            continue

        nx_graph = nx.read_graphml(graphs_dir + graph_type + "\\" + graph)
        print(graph)

        nx_nodes_1 = nx_graph.nodes()
        nx_edges = nx_graph.edges()

        degrees = [val for (node, val) in nx_graph.degree()]

        num_nodes = len(nx_nodes_1)


        V = [Vertex(v) for v in nx_nodes_1]
        vertex_strings = [v.data for v in V]

        E = list()
        for e in nx_edges:
            i1 = vertex_strings.index(e[0])
            i2 = vertex_strings.index(e[1])
            E.append(Edge(V[i1], V[i2]))

        sugi_graph = Graph(V,E)

        for v in V:
            v.view = defaultview()

        # for v in sugi_graph.E():
        #     print(v.v[0].data)    how to access edge values (v[1] for target node of an edge)
        #     print(v.data)         how to access node values

        sug = SugiyamaLayout(sugi_graph.C[0])
        sug.init_all()
        sug.draw()

        new_graph = nx.Graph()    

        try:
            for v in sugi_graph.V():
                #print(v.view.xy[0])
                new_graph.add_node(v.data)
                new_graph.nodes[v.data][u'LabelGraphics'] = {u'text': v.data, u'fontName': u'Dialog', u'fontSize': 12, u'anchor': u'c'}
                new_graph.nodes[v.data][u'graphics'] = {u'outline': u'#000000', u'h': 100.0, u'raisedBorder': 0, u'w': 100.0, u'y': v.view.xy[1], u'x': v.view.xy[0], u'type': u'rectangle', u'fill': u'#FFCC00'}
                
            for e in sugi_graph.E():
                new_graph.add_edge(e.v[0].data, e.v[1].data)
        except:
            print(f"{graph} fail")
            continue
        
    

        parse_graph.write_graphml(new_graph, drawings_dir + graph_type + "\\" + "Sugiyama" + "\\sugi_" + graph, True)