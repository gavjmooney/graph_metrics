import networkx as nx
from networkx.algorithms import approximation as ap
import os
import pandas as pd
from parse_graph import read_graphml
import numpy
from metrics_suite import  MetricsSuite
import re


def main(stop=None):
    graphs_dir = "..\\Graphs\\"
    drawings_dir = "..\\Graph Drawings\\"
    output_file = "Graph_Drawing_Metrics.csv"

    metric_weights = {"edge_crossing": 1,
                "edge_orthogonality": 1,
                "node_orthogonality": 0,
                "angular_resolution": 1,
                "symmetry": 0,
                "node_resolution": 1,
                "edge_length": 1,
                "gabriel_ratio": 1,
                "crossing_angle": 1,
                "stress": 1,
                "neighbourhood_preservation": 1,
                "aspect_ratio": 1,
                "node_uniformity": 1
    }

    metric_weights_hola = {"edge_crossing": 0,
                "edge_orthogonality": 0,
                "node_orthogonality": 0,
                "angular_resolution": 0,
                "symmetry": 0,
                "node_resolution": 1,
                "edge_length": 0,
                "gabriel_ratio": 0,
                "crossing_angle": 0,
                "stress": 0,
                "neighbourhood_preservation": 1,
                "aspect_ratio": 0,
                "node_uniformity": 1
    }


    metric_weights_hola_bp = {"edge_crossing": 1,
                "edge_orthogonality": 1,
                "node_orthogonality": 0,
                "angular_resolution": 1,
                "symmetry": 0,
                "node_resolution": 0,
                "edge_length": 1,
                "gabriel_ratio": 1,
                "crossing_angle": 1,
                "stress": 1,
                "neighbourhood_preservation": 0,
                "aspect_ratio": 1,
                "node_uniformity": 0
    }




    #graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Newman-Watts-Strogatz", "Geometric", "Stochastic-Block-Model", "North", "Rome"]
    graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Geometric", "Newman-Watts-Strogatz", "Stochastic-Block-Model"] #,"North", "Rome"] 
    layouts = ["Fruchterman-Reingold", "Random", "Kamada-Kawai", "Sugiyama", "DRGraph", "HOLA"]
    #layouts = ["HOLA"]

    total = 0
    for graph_type in os.listdir(drawings_dir):
        if graph_type not in graph_types:
            continue

        for alg in os.listdir(drawings_dir + graph_type):
            if alg not in layouts:
                continue

            for graph in os.listdir(drawings_dir + graph_type + "\\" + alg):
                if graph.endswith("txt"):
                    continue

                if alg == "HOLA":
                    pattern = r'_j(\d+(\.\d+)?)_'
                    match = re.search(pattern, graph)
                    j_val = int(match.group(1))
                    if j_val != 0:
                        continue
                
                total += 1

    i = 0
    # with open(output_file, "w") as out_f:
    with open(output_file, "r+") as out_f:
        done = []
        # legend = "filename,generator,layout,n,m,num_crossings,edge_crossings,centred_edge_crossings,edge_orthogonality,angular_resolution,node_resolution,edge_length,gabriel_ratio,crossing_angle,stress,neighbourhood_preservation,aspect_ratio,node_uniformity\n"
        # out_f.write(legend)

        for line in out_f:
            graph_name = line.strip().split(",")[0]
            done.append(graph_name)

        for graph_type in os.listdir(drawings_dir):
            if graph_type not in graph_types:
                continue

            for alg in os.listdir(drawings_dir + graph_type):
                if alg not in layouts:
                    continue

                for graph in os.listdir(drawings_dir + graph_type + "\\" + alg):
                    if graph.endswith("txt"):
                        continue
                    if graph in done:
                        i += 1
                        done.remove(graph)
                        continue

                    print(f"{i}/{total}:\t {graph}")
                    
                    if i == stop: break
                    
                    #G = read_graphml(drawings_dir + graph_type + "\\" + alg + "\\" + graph)

                    if alg == "HOLA":
                        
                        pattern = r'_j(\d+(\.\d+)?)_'
                        match = re.search(pattern, graph)
                        j_val = int(match.group(1))
                        if j_val != 0:
                            continue
                        
                        i += 1
                        G = nx.read_gml(drawings_dir + graph_type + "\\" + alg + "\\" + graph, label=None)

                        for n in G.nodes():
                            G.nodes[n]['x'] = G.nodes[n]['graphics']['x']
                            G.nodes[n]['y'] = G.nodes[n]['graphics']['y']

                        ms1 = MetricsSuite(G, metric_weights=metric_weights_hola)
                        ms1.calculate_metrics()

                        from bends_promotion import new_bends_promotion
                        H = new_bends_promotion(G)

                        mapping = {}
                        ind = 0
                        for n in H.nodes():
                            mapping[n] = ind
                            H.nodes[n]['x'] = H.nodes[n]['graphics']['x']
                            H.nodes[n]['y'] = H.nodes[n]['graphics']['y']
                            ind += 1

                        H = nx.relabel_nodes(H, mapping)

                        ms2 = MetricsSuite(H, metric_weights=metric_weights_hola_bp)
                        ms2.calculate_metrics()
                        #ms.pretty_print_metrics()

                        ecm = ms2.metrics["edge_crossing"]["value"]
                        ecc = ((ecm - 0.762) * (0.05 / 0.028)) + 0.5
                        
                        values = [  
                                    G.number_of_nodes(),
                                    G.number_of_edges(),
                                    ms2.metrics["edge_crossing"]["num_crossings"],
                                    ecm,
                                    ecc,
                                    ms2.metrics["edge_orthogonality"]["value"],
                                    ms2.metrics["angular_resolution"]["value"],
                                    ms1.metrics["node_resolution"]["value"],
                                    ms2.metrics["edge_length"]["value"],
                                    ms2.metrics["gabriel_ratio"]["value"],
                                    ms2.metrics["crossing_angle"]["value"],
                                    ms2.metrics["stress"]["value"],
                                    ms1.metrics["neighbourhood_preservation"]["value"],
                                    ms2.metrics["aspect_ratio"]["value"],
                                    ms1.metrics["node_uniformity"]["value"],
                                    
                        ]

                        line = graph + "," + graph_type + "," + alg + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in values) + "\n"

                        out_f.write(line)

                    else:
                        i += 1
                        G = read_graphml(drawings_dir + graph_type + "\\" + alg + "\\" + graph)

                        ms = MetricsSuite(G, metric_weights=metric_weights)
                        ms.calculate_metrics()

                        ecm = ms.metrics["edge_crossing"]["value"]
                        ecc = ((ecm - 0.762) * (0.05 / 0.028)) + 0.5

                        values = [  
                                    G.number_of_nodes(),
                                    G.number_of_edges(),
                                    ms.metrics["edge_crossing"]["num_crossings"],
                                    ecm,
                                    ecc,
                                    ms.metrics["edge_orthogonality"]["value"],
                                    ms.metrics["angular_resolution"]["value"],
                                    ms.metrics["node_resolution"]["value"],
                                    ms.metrics["edge_length"]["value"],
                                    ms.metrics["gabriel_ratio"]["value"],
                                    ms.metrics["crossing_angle"]["value"],
                                    ms.metrics["stress"]["value"],
                                    ms.metrics["neighbourhood_preservation"]["value"],
                                    ms.metrics["aspect_ratio"]["value"],
                                    ms.metrics["node_uniformity"]["value"],
                                    
                        ]

                        line = graph + "," + graph_type + "," + alg + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in values) + "\n"

                        out_f.write(line)


                if i == stop: break     
            if i == stop: break
        


if __name__ == "__main__":
    main()

    # G = read_graphml("test.graphml")
    # spectral_radius(G)
    # Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    # G0 = G.subgraph(Gcc[0])
    # print(apl(G0))
    # print(nx.average_shortest_path_length(G0))
    # print(apl2(G))
    # print(apl3(G))