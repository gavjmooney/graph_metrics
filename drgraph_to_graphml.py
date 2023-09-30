import os
import networkx as nx
import parse_graph as pg



def main():

    graphs_dir = "..\\Graphs\\"
    drawings_dir = "..\\Graph Drawings\\"

    for graph_type in os.listdir(graphs_dir):
        if graph_type == "Nathan":
            continue
        for graph in os.listdir(graphs_dir + graph_type):
            if graph.endswith("gml") or graph.endswith('graphml') or graph.endswith("vna"):
                continue
            
            print(graph)
            G = nx.Graph()

            with open(graphs_dir + graph_type + "\\" + graph, "r") as f:

                for line in f.readlines()[1:]:
                    u, v = line.split(" ")[0:2]
                    G.add_edge(u, v)

                

                with open(drawings_dir + graph_type +"\\DRGraph\\" + graph, "r") as f2:
                    i = 0
                    for line in f2.readlines()[1:]:
                        x, y = line.split(" ")
                        y = y[0:-1]

                        x, y = float(x), float(y)
                        x *= 1000
                        y *= 1000
                        x, y = str(x), str(y)

                        G.nodes[str(i)]['x'] = x
                        G.nodes[str(i)]['y'] = y
                        i += 1

            pg.write_graphml(G, drawings_dir + graph_type +"\\DRGraph\\" + graph[0:-3] + "graphml")
            #quit()

if __name__ == "__main__":
    main()