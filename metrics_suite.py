from logging import exception
#from msilib.schema import Error
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import random as rand
import math
from write_graph import write_graphml_pos
import time
import crosses_promotion

from scipy.spatial import KDTree

class MetricsSuite():
    """A suite for calculating several metrics for graph drawing aesthetics, as well as methods for combining these into a single cost function.
    Takes as an argument a path to a GML or GraphML file, or a NetworkX Graph object. Also takes as an arhument a dictionary of metric:weight key/values.
    Note: to prevent unnecessary calculations, omit metrics with weight 0."""

    def __init__(self, graph=None, metric_weights=None, mcdat="weighted_sum", sym_threshold=2, sym_tolerance=3, file_type="GraphML"):

        self.metrics = {"edge_crossing": {"func":self.edge_crossing, "value":None, "num_crossings":None, "weight":0},
                        "edge_orthogonality": {"func":self.edge_orthogonality, "value":None, "weight":0},
                        "node_orthogonality": {"func":self.node_orthogonality, "value":None, "weight":0},
                        "angular_resolution": {"func":self.angular_resolution, "value":None, "weight":0},
                        "symmetry": {"func":self.symmetry, "value":None, "weight":0},
                        "node_resolution": {"func":self.node_resolution, "value":None, "weight":0},
                        "edge_length": {"func":self.edge_length, "value":None, "weight":0},
                        "gabriel_ratio": {"func":self.gabriel_ratio, "value":None, "weight":0},
                        "crossing_angle": {"func":self.crossing_angle, "value":None, "weight":0},
                        "stress": {"func":self.get_stress, "value":None, "weight":0},
                        "neighbourhood_preservation": {"func":self.neighbourhood_preservation, "value":None, "weight":0},
                        "aspect_ratio": {"func":self.aspect_ratio, "value":None, "weight":0},
                        "node_uniformity": {"func":self.node_uniformity, "value":None, "weight":0},
        } 

        self.mcdat_dict = {"weighted_sum":self._weighted_sum,
                           "weighted_prod":self._weighted_prod,
        }

        self.graph_cross_promoted = None

        # Check all metrics given are valid and assign weights
        if metric_weights:
            metrics_to_remove = []
            for metric in metric_weights:
                assert metric in self.metrics, f"Unknown metric: {metric}. Available metrics: {list(self.metrics.keys())}"

                if type(metric_weights[metric]) != int and type(metric_weights[metric]) != float:
                    raise TypeError(f"Metric weights must be a number, not {type(metric_weights[metric])}")

                if metric_weights[metric] < 0:
                    raise ValueError(f"Metric weights must be positive.")

                self.metrics[metric]["weight"] = metric_weights[metric]

                if metric_weights[metric] == 0:
                    metrics_to_remove.append(metric)
            
            # Remove 0 weighted metrics
            for metric in metrics_to_remove:
                metric_weights.pop(metric)

            self.initial_weights = metric_weights

        else:
            self.initial_weights = {"edge_crossing": 1}

        # Check metric combination strategy is valid
        assert mcdat in self.mcdat_dict, f"Unknown mcdat: {mcdat}. Available mcats: {list(self.mcdat_dict.keys())}"
        self.mcdat = mcdat
        
        if graph is None:
            self.graph = self.load_graph_test()
        elif isinstance(graph, str):
            self.fname = graph
            self.graph = self.load_graph(graph, file_type=file_type)
        elif isinstance(graph, nx.Graph):
            self.fname = ""
            self.graph = graph
        else:
            raise TypeError(f"'graph' must be a string representing a path to a GML or GraphML file, or a NetworkX Graph object, not {type(graph)}")

        # Check symmetry parameters
        if type(sym_tolerance) != int and type(sym_tolerance) != float:
            raise TypeError(f"sym_tolerance must be a number, not {type(sym_tolerance)}")

        if sym_tolerance < 0:
            raise ValueError(f"sym_tolerance must be positive.")

        self.sym_tolerance = sym_tolerance


        if type(sym_threshold) != int and type(sym_threshold) != float:
            raise TypeError(f"sym_threshold must be a number, not {type(sym_threshold)}")

        if sym_threshold < 0:
            raise ValueError(f"sym_threshold must be positive.")
        
        self.sym_threshold = sym_threshold



    def _weighted_prod(self):
        """Returns the weighted product of all metrics. Should NOT be used as a cost function - may be useful for comparing graphs."""
        return math.prod(self.metrics[metric]["value"] * self.metrics[metric]["weight"] for metric in self.initial_weights)


    def _weighted_sum(self):
        """Returns the weighted sum of all metrics. Can be used as a cost function."""
        total_weight = sum(self.metrics[metric]["weight"] for metric in self.metrics)
        return sum(self.metrics[metric]["value"] * self.metrics[metric]["weight"] for metric in self.initial_weights) / total_weight
    

    def load_graph_test(self, nxg=nx.sedgewick_maze_graph):
        """Loads a test graph with a random layout."""
        G = nxg()
        pos = nx.random_layout(G)
        for k,v in pos.items():
            pos[k] = {"x":v[0], "y":v[1]}

        nx.set_node_attributes(G, pos)
        return G


    def load_graph(self, filename, file_type="GraphML"):
        """Loads a graph from a file."""

        if not (filename.lower().endswith('gml') or filename.lower().endswith('graphml')):
            raise Exception("Filetype must be GraphML.")

        # Accounts for some files which are actually GML files, but have the GraphML extension
        with open(filename) as f:
            first_line = f.readline()
            if first_line.startswith("graph"):
                file_type = "GML"
        
        if file_type == "GML":
            G = nx.read_gml(filename)
            for node in G.nodes:
                try:
                    # Assign node attrbiutes for coordinate position of nodes
                    G.nodes[node]['x'] = float(G.nodes[node]['graphics']['x'])
                    G.nodes[node]['y'] = float(G.nodes[node]['graphics']['y'])

                except KeyError:
                    # Graph doesn't have positional attributes
                    #print("Graph does not contain positional attributes. Assigning them randomly.")
                    pos = nx.random_layout(G)
                    for k,v in pos.items():
                        pos[k] = {"x":v[0]*G.number_of_nodes()*20, "y":v[1]*G.number_of_nodes()*20}

                    nx.set_node_attributes(G, pos)

        
        elif file_type == "GraphML":

            G = nx.read_graphml(filename)
            G = G.to_undirected()

            for node in G.nodes:
                try:
                    # Assign node attrbiutes for coordinate position of nodes
                    G.nodes[node]['x'] = float(G.nodes[node]['x'])
                    G.nodes[node]['y'] = float(G.nodes[node]['y'])

                except KeyError:
                    # Graph doesn't have positional attributes
                    #print("Graph does not contain positional attributes. Assigning them randomly.")
                    pos = nx.random_layout(G)
                    for k,v in pos.items():
                        pos[k] = {"x":v[0]*G.number_of_nodes()*20, "y":v[1]*G.number_of_nodes()*20}

                    nx.set_node_attributes(G, pos)


        return G


    def write_graph_no_pos(self, filename, graph=None):
        """Writes a graph without preserving any information about node position."""
        if graph is None:
            graph = self.graph

        nx.write_graphml(graph, filename, named_key_ids=True)


    def write_graph(self, filename, graph=None, scale=False):
        """Writes a graph to GraphML format. May not preserve ALL attributes of a graph loaded from GraphML, but will save position of nodes."""
        if graph is None:
            graph = self.graph

        # If specified, scale the size of the graph to make it more suited to graphml format
        if scale:
            coords = []
            for node in graph:
                coords.append(abs(float((graph.nodes[node]['x']))))
                coords.append(abs(float((graph.nodes[node]['y']))))

            avg_dist_origin = sum(coords) / len(coords)
            
            # Note values are arbritrary
            if avg_dist_origin < 100:
                for node in graph:
                    graph.nodes[node]["x"] *= 750
                    graph.nodes[node]["y"] *= 750

        write_graphml_pos(graph, filename)


    def calculate_metric(self, metric):
        """Calculate the value of the given metric by calling the associated function."""
        self.metrics[metric]["value"] = self.metrics[metric]["func"]()


    def calculate_metrics(self):
        """Calculates the values of all metrics with non-zero weights."""
        # for metric in self.metrics:
        #     if self.metrics[metric]["weight"] != 0:
        #         print(metric, end=" ")
        #         t1 = time.time()
        #         self.calculate_metric(metric)
        #         t2 = time.time()
        #         print(f"Took: {t2-t1}")

        t1 = time.time()
        for metric in self.metrics:
            if self.metrics[metric]["weight"] != 0:
                self.calculate_metric(metric)
        t2 = time.time()
        print(f"Took: {t2-t1}")


    def combine_metrics(self):
        """Combine several metrics based on the given multiple criteria descision analysis technique."""
        # Important to loop over initial weights to avoid checking the weight of all metrics when they are not needed
        for metric in self.initial_weights:
            self.calculate_metric(metric)

        return self.mcdat_dict[self.mcdat]()


    def draw_graph(self, graph=None, flip=True):
        """Draws the graph using standard NetworkX methods with matplotlib. Due to the nature of the coordinate systems used,
        graphs will be flipped on the X axis. To see the graph the way it would be drawn in yEd, set flip to True (default=True)."""
        if graph is None:
            graph = self.graph

        if flip:
            pos={k:np.array((v["x"], 0-float(v["y"])),dtype=np.float32) for (k, v) in[u for u in graph.nodes(data=True)]}
        else:
            pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in graph.nodes(data=True)]}

        nx.draw(graph, pos=pos, with_labels=True)
        plt.show()


    def pretty_print_metrics(self):
        """Prints all metrics and their values in an easily digestible view."""
        self.calculate_metrics()
        print("-"*40)
        print("{:<20s}Value\tWeight".format("Metric"))
        print("-"*40)
        for k,v in self.metrics.items():
            
            if v['value']:
                val_str = f"{v['value']:.3f}"
                print(f"{k:<20s}{val_str:<5s}\t{v['weight']}")
            else:
                print(f"{k:<20s}{str(v['value']):<5s}\t{v['weight']}")
        print("-"*40)
        print(f"Evaluation using {self.mcdat}: {self.combine_metrics():.5f}")
        print("-"*40)
        

    def pretty_print_nodes(self, graph=None):
        """Prints the nodes in the graph and their attributes"""
        if graph is None:
            graph = self.graph
        
        for n in graph.nodes(data=True):
            print(n)


    def _on_opposite_sides(self, a, b, line):
        """Check if two lines pass the on opposite sides test. Return True if they do."""
        g = (line[1][0] - line[0][0]) * (a[1] - line[0][1]) - (line[1][1] - line[0][1]) * (a[0] - line[0][0])
        h = (line[1][0] - line[0][0]) * (b[1] - line[0][1]) - (line[1][1] - line[0][1]) * (b[0] - line[0][0])
        return g * h <= 0.0 and (a != line[1] and b != line[0] and a != line[0] and b != line[1])


    def _bounding_box(self, line_a, line_b):
        """Check if two lines pass the bounding box test. Return True if they do."""
        x1 = min(line_a[0][0], line_a[1][0])
        x2 = max(line_a[0][0], line_a[1][0])
        x3 = min(line_b[0][0], line_b[1][0])
        x4 = max(line_b[0][0], line_b[1][0])

        y1 = min(line_a[0][1], line_a[1][1])
        y2 = max(line_a[0][1], line_a[1][1])
        y3 = min(line_b[0][1], line_b[1][1])
        y4 = max(line_b[0][1], line_b[1][1])

        return x4 >= x1 and y4 >= y1 and x2 >= x3 and y2 >= y3


    def _intersect(self, line_a, line_b):
        """Check if two lines intersect by checking the on opposite sides and bounding box 
        tests. Return True if they do."""
        return (self._on_opposite_sides(line_a[0], line_a[1], line_b) and 
                self._on_opposite_sides(line_b[0], line_b[1], line_a) and 
                self._bounding_box(line_a, line_b))

    def count_impossible_triangle_crossings(self, G):

        triangles = []
        for u,v in G.edges():                    
            for t in G.neighbors(u):
                if v in G.neighbors(t) and {u,v,t} not in triangles:
                    triangles.append({u,v,t})

        triangle_edges = []
        for u, v, t in triangles:
            if {u, v} not in triangle_edges:
                triangle_edges.append({u, v})
            if {v, t} not in triangle_edges:
                triangle_edges.append({v, t})
            if {t, u} not in triangle_edges:
                triangle_edges.append({t, u})


        total_impossible = 0
        for u, v, t in triangles:
            bubble = []
            bubble.extend(G.edges(u))
            bubble.extend(G.edges(v))
            bubble.extend(G.edges(t))

            subG = nx.Graph(bubble)

            for a, b in G.edges():
                if (a, b) in subG.edges() or (b, a) in subG.edges():
                    continue

                if {a, b} in triangle_edges:
                    continue

                total_impossible += 1

        covered_triangles = []
        for u, v, t in triangles:
            for a, b, c in triangles:
                if {u, v, t} in covered_triangles or {a, b, c} in covered_triangles:
                    continue

                if {u, v, t} == {a, b, c}:
                    continue
                
                covered_triangles.append({u, v, t})
                # Triangles share an edge
                if (({u, v} == {a, b} or {u, v} == {b, c} or {u, v} == {c, a}) or 
                    ({v, t} == {a, b} or {v, t} == {b, c} or {v, t} == {c, a}) or
                    ({t, u} == {a, b} or {t, u} == {b, c} or {t, u} == {c, a})):
                
                    total_impossible += 1
                    continue
                
                # Triangles share a node
                if ((u == a or u == b or u == c) or (v == a or v == b or v == c) or (t == a or t == b or t == c)):
                    total_impossible += 2
                    continue
                
                
                total_impossible += 3

        num_4_cycles = 0
        for u, v in G.edges():
            for t in G.neighbors(u):
                if t == v:
                    continue

                for w in G.neighbors(v):
                    if w == t or w == u:
                        continue

                    if w in G.neighbors(t):
                        square = G.subgraph([u,v,t,w])
                        num_adj = 0

                        for su, sv in square.edges():
                            if {su, sv} in triangle_edges:
                                num_adj += 1
                        
                        if num_adj < 2:
                            num_4_cycles += 1

        return total_impossible + (num_4_cycles // 4)
    

    def edge_crossing(self):
        """Calculate the metric for the number of edge_crossing, scaled against the total
        number of possible crossings."""

        import edge_crossing_metric as ec

        new_ec, c = ec.edge_crossing(self.graph)
        self.metrics["edge_crossing"]["num_crossings"] = c
        return new_ec

        # Estimate for the upper bound for the number of edge crossings
        # m = self.graph.number_of_edges()
        # c_all = (m * (m - 1))/2
        
        # c_impossible = sum([(self.graph.degree[u] * (self.graph.degree[u] - 1)) for u in self.graph])/2
        
        # c_mx = c_all - c_impossible - self.count_impossible_triangle_crossings(self.graph)

        # c_tri = self.count_impossible_triangle_crossings2()

        # c_mx_no_tri = c_all - c_tri

        # c_mx_no_tri_no_deg = c_all - c_impossible - c_tri


        
        # print(f"Total Upper bound: {c_all}")
        # print(f"Impossible by degree: {c_impossible}")
        # print(f"Impossible by triangle: {c_tri}")
        # print(f"Upper bound removing degree: {c_mx}")
        # print(f"Upper bound removing triangles: {c_mx_no_tri}")
        # print(f"Upper bound removing degree and triangles: {c_mx_no_tri_no_deg}")

        

        # covered = []
        # c = 0
        # # Iterate over all pairs of edges, checking if they intersect
        # for e in self.graph.edges:
            
        #     a_p1 = (self.graph.nodes[e[0]]["x"], self.graph.nodes[e[0]]["y"]) # Position of source node of e
        #     a_p2 = (self.graph.nodes[e[1]]["x"], self.graph.nodes[e[1]]["y"]) # Position of target node of e
        #     line_a = (a_p1, a_p2)
            
        #     for e2 in self.graph.edges:
        #         if e == e2:
        #             continue
                
        #         b_p1 = (self.graph.nodes[e2[0]]["x"], self.graph.nodes[e2[0]]["y"]) # Position of source node of e2
        #         b_p2 = (self.graph.nodes[e2[1]]["x"], self.graph.nodes[e2[1]]["y"]) # Position of target node of e2
        #         line_b = (b_p1, b_p2)
                
        #         if self._intersect(line_a, line_b) and (line_a, line_b) not in covered:
        #             covered.append((line_b, line_a))                  
        #             c += 1

        # print(f"Num Crossings: {c}")
        # print(f"Original EC: {1 - (c / c_mx) if c_mx > 0 else 1}")
        # print(f"EC without triangles: {1 - (c / c_mx_no_tri) if c_mx_no_tri > 0 else 1}")
        # print(f"EC without triangles and degrees: {1 - (c / c_mx_no_tri_no_deg) if c_mx_no_tri_no_deg > 0 else 1}")

        # self.metrics["edge_crossing"]["num_crossings"] = c
        # return 1 - (c / c_mx) if c_mx > 0 else 1

    # NOTE: Replaced by self.metrics["edge_Crossing"]["num_crossings"] in edge_Crossing function
    def count_crossings(self, G=None):
        """Calculate the  number of edge crossings """
        if G is None:
            G = self.graph

        covered = []
        c = 0
        for e in G.edges:
            
            a_p1 = (G.nodes[e[0]]["x"], G.nodes[e[0]]["y"]) # Position of source node of e
            a_p2 = (G.nodes[e[1]]["x"], G.nodes[e[1]]["y"]) # Position of target node of e
            line_a = (a_p1, a_p2)
            
            for e2 in G.edges:
                if e == e2:
                    continue
                
                b_p1 = (G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]) # Position of source node of e2
                b_p2 = (G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]) # Position of target node of e2
                line_b = (b_p1, b_p2)
                
                if self._intersect(line_a, line_b) and (line_a, line_b) not in covered:
                    covered.append((line_b, line_a))                  
                    c += 1
        return c


    def edge_orthogonality(self):
        """Calculate the metric for edge orthogonality."""
        ortho_list = []

        # Iterate over each edge and get it's minimum angle relative to the orthogonal grid
        for e in self.graph.edges:
            source = e[0]
            target = e[1]

            x1, y1 = self.graph.nodes[source]["x"], self.graph.nodes[source]["y"]
            x2, y2 = self.graph.nodes[target]["x"], self.graph.nodes[target]["y"]

            try:
                gradient = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                gradient = 0

            angle = math.degrees(math.atan(abs(gradient)))

            edge_ortho = min(angle, abs(90-angle), 180-angle) /45
            ortho_list.append(edge_ortho)

        # Return 1 minus the average of minimum angles
        return 1 - (sum(ortho_list) / self.graph.number_of_edges())


    def angular_resolution(self, all_nodes=False):
        """Calculate the metric for angular resolution. If all_nodes is True, include nodes with degree 1, for which the angle will always be perfect."""
        angles_sum = 0
        nodes_count = 0
        for node in self.graph.nodes:
            if self.graph.degree[node] <= 1:
                continue

            nodes_count += 1
            ideal = 360 / self.graph.degree[node] # Each node has an ideal angle for adjacent edges, based on the number of adjacent edges

            x1, y1 = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
            actual_min = 360

            # Iterate over adjacent edges and calculate the difference of the minimum angle from the ideal angle
            for adj in self.graph.neighbors(node):
                x2, y2 = self.graph.nodes[adj]['x'], self.graph.nodes[adj]['y']
                angle1 = math.degrees(math.atan2((y2 - y1), (x2 - x1)))

                for adj2 in self.graph.neighbors(node):
                    if adj == adj2:
                        continue
                    
                    x3, y3 = self.graph.nodes[adj2]['x'], self.graph.nodes[adj2]['y']
                    angle2 = math.degrees(math.atan2((y3 - y1), (x3 - x1)))

                    diff = abs(angle2 - angle1)

                    if diff < actual_min:
                        actual_min = diff

            angles_sum += abs((ideal - actual_min) / ideal)

        # Return 1 minus the average of minimum angles
        return 1 - (angles_sum / self.graph.number_of_nodes()) if all_nodes else 1 - (angles_sum / nodes_count)


    def crossing_angle(self, crossing_limit=999999):
        """Calculate the metric for the edge crossings angle. crossing_limit specifies the maximum number of crossings allowed, 
        which is limited due to long execution times."""
        if self.metrics["edge_crossing"]["num_crossings"] is None:
            self.calculate_metric("edge_crossing")

        # if self.metrics["edge_crossing"]["num_crossings"] > crossing_limit:
        #     return 0

        if self.graph_cross_promoted is None:
            G = self.crosses_promotion()
        else:
            G = self.graph_cross_promoted

        angles_sum = 0
        num_minor_nodes = 0
        for node in G.nodes:
            # Only crosses promoted nodes should be counted
            if not self._is_minor(node, G):
                continue
            
            num_minor_nodes += 1
            ideal = 360 / G.degree[node] # This should always be 90 degrees, except in rare cases where multiple edges intersect at the exact same point
            

            x1, y1 = G.nodes[node]['x'], G.nodes[node]['y']
            actual_min = 360

            # Iterate over adjacent edges and calculate the difference of the minimum angle from the ideal angle
            for adj in G.neighbors(node):
                x2, y2 = G.nodes[adj]['x'], G.nodes[adj]['y']
                angle1 = math.degrees(math.atan2((y2 - y1), (x2 - x1)))

                for adj2 in G.neighbors(node):
                    if adj == adj2:
                        continue
                    
                    x3, y3 = G.nodes[adj2]['x'], G.nodes[adj2]['y']
                    angle2 = math.degrees(math.atan2((y3 - y1), (x3 - x1)))

                    diff = abs(angle1 - angle2)

                    if diff < actual_min:
                        actual_min = diff

            angles_sum += abs((ideal - actual_min) / ideal)

        # Return 1 minus the average of minimum angles
        return 1 - (angles_sum / num_minor_nodes) if num_minor_nodes > 0 else 1


    def node_orthogonality(self):
        """Calculate the metric for node orthogonality."""
        coord_set =[]

        # Start with random node
        first_node = rand.sample(list(self.graph.nodes), 1)[0]        
        min_x, min_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]

        # Find minimum x and y positions
        for node in self.graph.nodes:
            x = self.graph.nodes[node]["x"]
            y = self.graph.nodes[node]["y"]
            
            if x < min_x:
                min_x = x
            elif y < min_y:
                min_y = y

        x_distance = abs(0 - float(min_x))
        y_distance = abs(0 - float(min_y))

        # Adjust graph so node with minimum coordinates is at 0,0
        for node in self.graph.nodes:
            self.graph.nodes[node]["x"] = float(self.graph.nodes[node]["x"]) - x_distance
            self.graph.nodes[node]["y"] = float(self.graph.nodes[node]["y"]) - y_distance


        # Start with random node
        first_node = rand.sample(list(self.graph.nodes), 1)[0]
        
        min_x, min_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]
        max_x, max_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]

        for node in self.graph.nodes:
            x, y = self.graph.nodes[node]["x"], self.graph.nodes[node]["y"]

            coord_set.append(x)
            coord_set.append(y)
            
            # Get GCD of node positions
            gcd = int(float(coord_set[0]))
            for coord in coord_set[1:]:
                gcd = math.gcd(int(float(gcd)), int(float(coord)))

            # Get maximum and minimum coordinates
            if x > max_x:
                max_x = x
            elif x < min_x:
                min_x = x           

            if y > max_y:
                max_y = y            
            elif y < min_y:
                min_y = y 

        # Get size of unit grid
        h = abs(max_y - min_y)
        w = abs(max_x - min_x)

        reduced_h = h / gcd
        reduced_w = w / gcd

        A = ((reduced_w+1) * (reduced_h+1))

        # Return number of nodes on the unit grid weighted against the number of positions on the unit grid
        return len(self.graph.nodes) / A


    # def _add_crossing_node(self, l1, l2, G, e, e2):
    #     """Helper function for crosses promotion which adds a new node to graph G where e and e2 intersect."""

    #     x_diff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
    #     y_diff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

    #     def det(a, b):
    #         return a[0] * b[1] - a[1] * b[0]

    #     div = det(x_diff, y_diff)

    #     if div == 0:
    #         return G

    #     # Get position of intersection
    #     d = (det(*l1), det(*l2))
    #     x = det(d, x_diff) / div
    #     y = det(d, y_diff) / div

    #     label = "c" + str(len(G.nodes())) # Must be a unique name, ensure graph doesn't already have nodes with the name 'c + some number'

    #     G.add_node(label)

    #     # Add some default attributes to the node
    #     G.nodes[label]["label"] = '\n'
    #     G.nodes[label]["shape_type"] = "ellipse"
    #     G.nodes[label]["x"] = x
    #     G.nodes[label]["y"] = y
    #     G.nodes[label]["type"] = "minor" # Only nodes added in crosses promotion are minor
    #     G.nodes[label]["color"] = "#3BC6E5"

    #     # Add edges between the new node and end points of e and e2, then remove e and e2
    #     G.add_edge(e[0], label)
    #     G.add_edge(label, e[1])

    #     G.add_edge(e2[0], label)
    #     G.add_edge(label, e2[1])

    #     G.remove_edge(e[0], e[1])
    #     G.remove_edge(e2[0], e2[1])

    #     return G


    def crosses_promotion(self):
        """Perform crosses promotion on the graph, adding nodes where edges cross."""
        
        crosses_promoted_G = crosses_promotion.crosses_promotion(self.graph)
        self.graph_cross_promoted = crosses_promoted_G
        return crosses_promoted_G
        
        # crosses_promoted_G = self.graph.copy() # Maintain original graph by creating copy
        # for node in crosses_promoted_G:
        #     crosses_promoted_G.nodes[node]["type"] = "major"

        # if self.metrics["edge_crossing"]["num_crossings"] is None:
        #     self.calculate_metric("edge_crossing")


        # num_crossings = self.metrics["edge_crossing"]["num_crossings"] 

        # second_covered = list()
        # crossing_count = 0
        # crossing_found = False

        # # Until all crossings have been covered
        # while crossing_count != num_crossings:

        #     crossing_found = False

        #     edges = crosses_promoted_G.edges

        #     # Iterate over each pair of edges
        #     for e in edges:
        #         # Don't need to check already covered crossings for second edge
        #         if e in second_covered:
        #             continue
                    
        #         # Create line segment represnting first edge
        #         source_node = crosses_promoted_G.nodes[e[0]]
        #         target_node = crosses_promoted_G.nodes[e[1]]
                
        #         l1_p1_x = source_node["x"]
        #         l1_p1_y = source_node["y"]

        #         l1_p2_x = target_node["x"]
        #         l1_p2_y = target_node["y"]

        #         l1_p1 = (l1_p1_x, l1_p1_y)
        #         l1_p2 = (l1_p2_x, l1_p2_y)
        #         l1 = (l1_p1, l1_p2)

        #         for e2 in edges:
        #             if e == e2:
        #                 continue
                    
        #             # Create line segment represnting second edge
        #             source2_node = crosses_promoted_G.nodes[e2[0]]
        #             target2_node = crosses_promoted_G.nodes[e2[1]]

        #             l2_p1_x = source2_node["x"]
        #             l2_p1_y = source2_node["y"]

        #             l2_p2_x = target2_node["x"]
        #             l2_p2_y = target2_node["y"]

        #             l2_p1 = (l2_p1_x, l2_p1_y)
        #             l2_p2 = (l2_p2_x, l2_p2_y)
        #             l2 = (l2_p1, l2_p2)

        #             # Check if line segments intersect
        #             if self._intersect(l1, l2):# and (l1, l2) not in second_covered:                    
        #                 crossing_count += 1
        #                 second_covered.append(e)
        #                 crosses_promoted_G = self._add_crossing_node(l1, l2, crosses_promoted_G, e, e2)
        #                 crossing_found = True
        #                 break

        #         if crossing_found:
        #             break
            
        #     # Debug info
        #     if not crossing_found:
        #         print("$"*20)
        #         print(f"{crossing_count}/{num_crossings}")
        #         print("$"*20)
        #         break
        
        # self.graph_cross_promoted = crosses_promoted_G
        # return crosses_promoted_G


    def _find_bisectors(self, G):
        """Returns the set of perpendicular bisectors between every pair of nodes"""
        bisectors = []
        covered = []

        # For each pair of nodes
        for n1 in G.nodes:
            n1_x, n1_y = G.nodes[n1]["x"], G.nodes[n1]["y"]

            for n2 in G.nodes:
                if n1 == n2 or (n1, n2) in covered:
                    continue

                n2_x, n2_y = G.nodes[n2]["x"], G.nodes[n2]["y"]

                # Get the midpoint between the two nodes
                midpoint_x = (n2_x + n1_x) / 2
                midpoint_y = (n2_y + n1_y) / 2

                # Get the gradient of perpendicualr bisector
                try:
                    initial_gradient = (n2_y - n1_y) / (n2_x - n1_x)
                    perp_gradient = (1 / initial_gradient) * -1
                    c = midpoint_y - (perp_gradient * midpoint_x)

                except:
                    if (n2_x == n1_x):
                        perp_gradient = "x"
                        c = midpoint_y

                    elif (n2_y == n1_y):
                        perp_gradient = "y"
                        c = midpoint_x

                grad_c = (perp_gradient, c)

                bisectors.append(grad_c)
                covered.append((n2, n1))

        return set(bisectors) # Set removes duplicates


    def _is_minor(self, node, G):
        """Returns True if a node was created by crosses promotion."""
        return G.nodes[node]["type"] == "minor"


    def _rel_point_line_dist(self, gradient, y_intercept, x ,y):
        """Helper function to get the relative distance between a bisector and a point."""
        gradient *= -1
        y_intercept *= -1

        x = gradient * float(x)
        denom = math.sqrt(gradient**2 + 1)
        return (x + float(y) + float(y_intercept)) / denom


    def _same_position(self, n1, n2, G, tolerance=0):
        """Helper function to determine if two nodes are in the same postion, with some tolerance."""
        x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
        x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']

        if tolerance == 0:
            return (x1 == x2 and y1 == y2)

        return self._in_circle(x1, y1, x2, y2, tolerance)


    def _is_positive(self, x):
        """Return true if x is postive."""
        return x > 0


    def _are_collinear(self, a, b, c, G):
        """Returns true if the three points are collinear, by checking if the determinant is 0."""
        return ((G.nodes[a]['x']*G.nodes[b]['y']) + (G.nodes[b]['x']*G.nodes[c]['y']) + (G.nodes[c]['x']*G.nodes[a]['y'])
         - (G.nodes[a]['x']*G.nodes[c]['y']) - (G.nodes[b]['x']*G.nodes[a]['y']) - (G.nodes[c]['x']*G.nodes[b]['y'])) == 0


    def _mirror(self, axis, e1, e2, G, tolerance=0):
        """Helper function to determine if two edges are mirrored about a bisector."""
        e1_p1_x, e1_p1_y = G.nodes[e1[0]]["x"], G.nodes[e1[0]]["y"]
        e1_p2_x, e1_p2_y = G.nodes[e1[1]]["x"], G.nodes[e1[1]]["y"]

        e2_p1_x, e2_p1_y = G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]
        e2_p2_x, e2_p2_y = G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]

        # The end nodes of edge1 are P and Q
        # The end nodes of edge2 are X and Y
        P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

        if axis[0] == "x":
            p = axis[1] - e1_p1_y
            q = axis[1] - e1_p2_y
            x = axis[1] - e2_p1_y
            y = axis[1] - e2_p2_y
        elif axis[0] == "y":
            p = axis[1] - e1_p1_x
            q = axis[1] - e1_p2_x
            x = axis[1] - e2_p1_x
            y = axis[1] - e2_p2_x
        else:
            p = self._rel_point_line_dist(axis[0], axis[1], e1_p1_x, e1_p1_y)
            q = self._rel_point_line_dist(axis[0], axis[1], e1_p2_x, e1_p2_y)
            x = self._rel_point_line_dist(axis[0], axis[1], e2_p1_x, e2_p1_y)
            y = self._rel_point_line_dist(axis[0], axis[1], e2_p2_x, e2_p2_y)

        if e1 == e2:
            # Same edge
            return 0
        elif p == 0 and q == 0:
            # Edge on axis
            return 0
        elif y == 0 and x == 0:
            # Edge on other axis
            return 0
        elif self._same_position(P, X, G, tolerance) and (self._same_rel_position(p, 0, tolerance) and self._same_rel_position(x, 0, tolerance)):
            if self._same_rel_position(q, y, tolerance) and (self._is_positive(q) != self._is_positive(y)):
                if not self._are_collinear(Q, P, Y, G):
                    # Shared node on axis but symmetric
                    return 1
        elif self._same_position(P, Y, G, tolerance) and (self._same_rel_position(p, 0, tolerance) and self._same_rel_position(y, 0, tolerance)):
            if self._same_rel_position(q, x, tolerance) and (self._is_positive(q) != self._is_positive(x)):
                if not self._are_collinear(Q, P, X, G): 
                    # Shared node on axis but symmetric
                    return 1
        elif self._same_position(Q, Y, G, tolerance) and (self._same_rel_position(q, 0, tolerance) and self._same_rel_position(y, 0, tolerance)):
            if self._same_rel_position(p, x, tolerance) and (self._is_positive(x) != self._is_positive(p)):
                if not self._are_collinear(P, Q, X, G):
                    # Shared node on axis but symmetric
                    return 1
        elif self._same_position(Q, X, G, tolerance) and (self._same_rel_position(q, 0, tolerance) and self._same_rel_position(x, 0, tolerance)):
            if self._same_rel_position(p, y, tolerance) and (self._is_positive(p) != self._is_positive(y)):
                if not self._are_collinear(P, Q, Y, G):
                    # Shared node on axis but symmetric
                    return 1
        elif self._is_positive(p) != self._is_positive(q):
            # Edge crosses axis
            return 0
        elif self._is_positive(x) != self._is_positive(y):
            # Other edge crosses axis
            return 0
        elif (self._same_rel_position(p, x, tolerance) and self._same_rel_position(q, y, tolerance) ) and (self._is_positive(p) != self._is_positive(x)) and (self._is_positive(q) != self._is_positive(y)):
            # Distances are equal and signs are different
            x1, y1 = G.nodes[P]["x"], G.nodes[P]["y"]
            x2, y2 = G.nodes[X]["x"], G.nodes[X]["y"]
            x3, y3 = G.nodes[Q]["x"], G.nodes[Q]["y"]
            x4, y4 = G.nodes[Y]["x"], G.nodes[Y]["y"]

            dist1 = self._euclidean_distance((x1,y1), (x2,y2))
            dist2 = self._euclidean_distance((x3,y3), (x4,y4))
            axis_dist1 = abs(p) * 2
            axis_dist2 = abs(q) * 2
            if self._same_distance(axis_dist1, dist1) and self._same_distance(axis_dist2, dist2):
                return 1

        elif (self._same_rel_position(p, y, tolerance)  and self._same_rel_position(x, q, tolerance) ) and (self._is_positive(p) != self._is_positive(y)) and (self._is_positive(x) != self._is_positive(q)):
            # Distances are equal and signs are different
            x1, y1 = G.nodes[P]["x"], G.nodes[P]["y"]
            x2, y2 = G.nodes[Y]["x"], G.nodes[Y]["y"]
            x3, y3 = G.nodes[Q]["x"], G.nodes[Q]["y"]
            x4, y4 = G.nodes[X]["x"], G.nodes[X]["y"]

            dist1 = self._euclidean_distance((x1,y1), (x2,y2))
            dist2 = self._euclidean_distance((x3,y3), (x4,y4))
            axis_dist1 = abs(p) * 2
            axis_dist2 = abs(q) * 2
            if self._same_distance(axis_dist1, dist1) and self._same_distance(axis_dist2, dist2):
                return 1
        else:
            return 0


    def _same_rel_position(self, a, b, tolerance=0):
        """Helper function to determine if two nodes are in the same postion (regardless of sign compared to the bisector), with some tolerance."""
        if tolerance == 0:
            return abs(a) == abs(b)
        else:
            return abs(abs(a)-abs(b)) <= tolerance


    def _same_distance(self, a, b, tolerance=0.5):
        """Helper function to determine if two distances are the same, with some tolerance."""
        return abs(abs(a)-abs(b)) <= tolerance


    def _sym_value(self, e1, e2, G):
        """Helper function to calcualte the level of symmetry between two edges, based on whoch nodes were crosses promoted."""
        # The end nodes of edge1 are P and Q
        # The end nodes of edge2 are X and Y
        P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

        
        if self._is_minor(P, G) == self._is_minor(X, G) and self._is_minor(Q, G) == self._is_minor(Y, G):
            # P=X and Q=Y
            return 1
        elif self._is_minor(P, G) == self._is_minor(Y, G) and self._is_minor(Q, G) == self._is_minor(X, G):
            # P=Y and X=Q
            return 1
        elif self._is_minor(P, G) == self._is_minor(X, G) and self._is_minor(Q, G) != self._is_minor(Y, G):
            # P=X but Q!=Y
            return 0.5
        elif self._is_minor(P, G) == self._is_minor(Y, G) and self._is_minor(Q, G) != self._is_minor(X, G):
            # P=Y but Q!=X
            return 0.5
        elif self._is_minor(P, G) != self._is_minor(X, G) and self._is_minor(Q, G) == self._is_minor(Y, G):
            # P!=X but Q==Y
            return 0.5
        elif self._is_minor(P, G) != self._is_minor(Y, G) and self._is_minor(Q, G) == self._is_minor(X, G):
            # P!=Y but Q==X
            return 0.5
        elif self._is_minor(P, G) != self._is_minor(X, G) and self._is_minor(Q, G) != self._is_minor(Y, G):
            # P!=X and Q!=Y
            return 0.25
        elif self._is_minor(P, G) != self._is_minor(Y, G) and self._is_minor(Q, G) != self._is_minor(X, G):
            # P!=Y and Q!=X
            return 0.25


    def _graph_to_points(self, G, edges=None):
        """Helper function for convex hulls which converts a graph's nodes to a list of points."""
        points = []

        if edges is None:
            for n in G.nodes:
                p1_x, p1_y = G.nodes[n]["x"], G.nodes[n]["y"]
                points.append((p1_x, p1_y))

        else:
            for e in edges:
                p1_x, p1_y = G.nodes[e[0]]["x"], G.nodes[e[0]]["y"]
                p2_x, p2_y = G.nodes[e[1]]["x"], G.nodes[e[1]]["y"]
                points.append((p1_x, p1_y))
                points.append((p2_x, p2_y))

        return points


    def symmetry(self, G=None, show_sym=False, crosses_limit=999999):
        """Calcualte the symmetry metric. """
        self.calculate_metric("edge_crossing")
        if self.metrics["edge_crossing"]["num_crossings"] > crosses_limit:
            return 0
        threshold = self.sym_threshold
        tolerance = self.sym_tolerance

        if G is None:
            if self.graph_cross_promoted is None:
                G = self.crosses_promotion()
            else:
                G = self.graph_cross_promoted
            
        
        axes = self._find_bisectors(G)

        total_area = 0
        total_sym = 0

        for a in axes:

            num_mirror = 0
            sym_val = 0
            subgraph = []
            covered = []

            for e1 in G.edges:
                for e2 in G.edges:
                    if e1 == e2 or (e1,e2) in covered:
                        continue
                    
                    if self._mirror(a, e1, e2, G, tolerance) == 1:
                        num_mirror += 1
                        sym_val += self._sym_value(e1, e2, G)
                        subgraph.append(e1)
                        subgraph.append(e2)

                    covered.append((e2,e1))

            # Compare number of mirrored edges to specified threshold
            if num_mirror >= threshold:

                points = self._graph_to_points(G, subgraph)

                if len(points) <= 2:
                    break
                
                # Add area of local symmetry to total area and add to total symmetry
                conv_hull = ConvexHull(points, qhull_options="QJ")
                sub_area = conv_hull.volume
                total_area += sub_area

                total_sym += (sym_val * sub_area) / (len(subgraph)/2)

                # Debug info
                if show_sym:
                    ag = nx.Graph()
                    ag.add_edges_from(subgraph)

                    for node in ag:
                        if node in G:
                            ag.nodes[node]["x"] = G.nodes[node]["x"]
                            ag.nodes[node]["y"] = G.nodes[node]["y"]
                    self.draw_graph(ag)

                
        # Get the are of the convex hull of the graph
        whole_area_points = self._graph_to_points(G)

        whole_hull = ConvexHull(whole_area_points)
        whole_area = whole_hull.volume

        # Return the symmetry weighted against either the area of the convex hull of the graph or the combined area of all local symmetries
        return total_sym / max(whole_area, total_area)


    def get_bounding_box(self, G=None):
        """Helper function to get the bounding box of the graph."""
        if G is None:
            G = self.graph

        # Start with a random node
        first_node = rand.sample(list(G.nodes), 1)[0]
        min_x, min_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]
        max_x, max_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]

        # Get the maximum and minimum x and y positions.
        for node in G.nodes:
            x = G.nodes[node]["x"]
            y = G.nodes[node]["y"]

            if x > max_x:
                max_x = x

            if x < min_x:
                min_x = x           

            if y > max_y:
                max_y = y        

            if y < min_y:
                min_y = y 

        return ((min_x, min_y), (max_x, max_y))


    def _euclidean_distance(self, a, b):
        """Helper function to get the euclidean distance between two points a and b."""
        return math.sqrt(((b[0] - a[0])**2) + ((b[1] - a[1])**2))


    def node_resolution(self):
        """Calulate the metric for node resolution, which is the ratio of the smallest and largest distance between any pair of nodes."""

        # Start with two random nodes
        first_node, second_node = rand.sample(list(self.graph.nodes), 2)
        a = self.graph.nodes[first_node]['x'], self.graph.nodes[first_node]['y']
        b = self.graph.nodes[second_node]['x'], self.graph.nodes[second_node]['y']

        min_dist = self._euclidean_distance(a, b)
        max_dist = min_dist

        # Iterate over every pair of nodes, keeping track of the maximum and minimum distances between them
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if i == j:
                    continue
                
                a = self.graph.nodes[i]['x'], self.graph.nodes[i]['y']
                b = self.graph.nodes[j]['x'], self.graph.nodes[j]['y']

                d = self._euclidean_distance(a, b)

                if d < min_dist:
                    min_dist = d

                if d > max_dist:
                    max_dist = d

        return min_dist / max_dist
        

    def edge_length(self, ideal=None):
        """Calculate the edge length metric by comparing the edge lengths to an ideal length. Default ideal is average of all edge lengths."""

        ideal_edge_length = 0
        for edge in self.graph.edges:
            a = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            b = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']
            
            ideal_edge_length += self._euclidean_distance(a, b)

        if not ideal:
            # For unweighted graphs, set the ideal edge length to the average edge length
            ideal_edge_length = ideal_edge_length / self.graph.number_of_edges()
        else:
            ideal_edge_length = ideal
        
        edge_length_sum = 0
        for edge in self.graph.edges:
            a = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            b = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']
            edge_length_sum += abs(ideal_edge_length - self._euclidean_distance(a, b)) / ideal_edge_length

        # Remove negatives
        if edge_length_sum > self.graph.number_of_edges():
            return 1 - abs(1 - (edge_length_sum / self.graph.number_of_edges()))

        return 1 - (edge_length_sum / self.graph.number_of_edges())


    def _midpoint(self, a, b, G=None):
        """Given two nodes and the graph they are in, return the midpoint between them"""
        if G is None:
            G = self.graph

        x1, y1 = G.nodes[a]['x'], G.nodes[a]['y']
        x2, y2 = G.nodes[b]['x'], G.nodes[b]['y']

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        return (mid_x, mid_y)


    def _in_circle(self, x, y, center_x, center_y, r):
        """Return true if the point x, y is inside or on the perimiter of the circle with center center_x, center_y and radius r"""
        return ((x - center_x)**2 + (y - center_y)**2) <= r**2


    def _circles_intersect(self, x1, y1, x2, y2, r1, r2):
        """Returns true if two circles touch or intersect."""
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) <= (r1 + r2) * (r1 + r2)


    def gabriel_ratio(self):
        """Calculate the metric for the gabriel ratio. A graph is a Gabriel graph if no node falls within the area of any circles constructed using each edge as its diameter."""
        
        # Initial upper bound on number of nodes which could potentially be violating nodes
        possible_non_conforming = (self.graph.number_of_edges() * self.graph.number_of_nodes()) - (self.graph.number_of_edges() * 2)

        
        num_non_conforming = 0
        
        # Iterate over each edge
        for edge in self.graph.edges:
            
            # Get the equation of the circle with the edge as its diameter
            a = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            b = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']

            r = self._euclidean_distance(a, b) / 2
            center_x, center_y = self._midpoint(edge[0], edge[1])

            # Check if any nodes fall with within the circle and increment the counter if they do
            for node in self.graph.nodes:
                if edge[0] == node or edge[1] == node:
                    continue
                
                x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']

                if self._in_circle(x, y, center_x, center_y, r):
                    num_non_conforming += 1
                    # If the node is adjacent to either node in the current edge reduce total by 1,
                    # since the nodes cannot both simultaneously be in each others circle
                    if node in self.graph.neighbors(edge[0]):
                        possible_non_conforming -= 1
                    if node in self.graph.neighbors(edge[1]):
                        possible_non_conforming -= 1 
                    

        # Return 1 minus the ratio of non conforming nodes to the upper bound on possible non conforming nodes.
        return 1 - (num_non_conforming / possible_non_conforming) if possible_non_conforming > 0 else 1


    # def stress(self):

    #     stress = 0
    #     covered = []
    #     for u in self.graph.nodes():
    #         for v in self.graph.nodes():
    #             if u == v:
    #                 continue
    #             if {u, v} in covered:
    #                 continue
                
    #             ax, ay = self.graph.nodes[u]['x'], self.graph.nodes[u]['y']
    #             bx, by = self.graph.nodes[v]['x'], self.graph.nodes[v]['y']
                
    #             d_ij = nx.shortest_path_length(self.graph, u, v)
    #             euc_d_ij = self._euclidean_distance((ax, ay), (bx, by))
    #             w_ij = 1 / (d_ij ** 2)
                
    #             stress += (w_ij * ((euc_d_ij - d_ij) ** 2))

    #             covered.append({u, v})

    #     return stress


    def get_stress(self):

        # X = np.array([[float(self.graph.nodes[n]['x']), float(self.graph.nodes[n]['y'])] for n in self.graph.nodes()])

        # apsp = dict(nx.all_pairs_shortest_path_length(self.graph))
        # apsp = dict(sorted(apsp.items()))
        
        # d = [[] for _ in range(self.graph.number_of_nodes())]

        # for i, k in enumerate(apsp):
        #     apsp[k] = dict(sorted(apsp[k].items()))
        #     d[i] = [float(v) for v in apsp[k].values()]

        # d = np.array(d)


        # N = len(X)
        # ss = (X * X).sum(axis=1)
        # diff = np.sqrt(abs(ss.reshape((N,1)) + ss.reshape((1,N)) - 2 * np.dot(X,X.T)))
        # np.fill_diagonal(diff, 0)
        # stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) )

        # from scipy.optimize import minimize_scalar
        # min_a = minimize_scalar(stress)

        # ss = 0 
        # for i in range(N):
        #     for j in range(i):
        #         ss += (d[i,j] * d[i,j])
        # return 1 - stress(a=min_a.x) / ss

        G = self.graph

        X = np.array([[float(G.nodes[n]['x']), float(G.nodes[n]['y'])] for n in G.nodes()])

        apsp = dict(nx.all_pairs_shortest_path_length(G))
        apsp = dict(sorted(apsp.items()))
        
        d = [[] for _ in range(G.number_of_nodes())]

        for i, k in enumerate(apsp):
            apsp[k] = dict(sorted(apsp[k].items()))
            d[i] = [float(v) for v in apsp[k].values()]

        d = np.array(d)

        from math import comb
        N = len(X)
        ss = (X * X).sum(axis=1)

        diff = np.sqrt(abs(ss.reshape((N,1)) + ss.reshape((1,N)) - 2 * np.dot(X,X.T)))

        np.fill_diagonal(diff, 0)

        stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) ) / comb(N,2)

        from scipy.optimize import minimize_scalar
        min_a = minimize_scalar(stress)

        return stress(a=min_a.x)
    

    def stress_not_normal(self, G):

        X = np.array([[float(self.graph.nodes[n]['x']), float(self.graph.nodes[n]['y'])] for n in self.graph.nodes()])

        apsp = dict(nx.all_pairs_shortest_path_length(self.graph))
        apsp = dict(sorted(apsp.items()))
        
        d = [[] for _ in range(self.graph.number_of_nodes())]

        for i, k in enumerate(apsp):
            apsp[k] = dict(sorted(apsp[k].items()))
            d[i] = [float(v) for v in apsp[k].values()]

        d = np.array(d)

        from math import comb
        N = len(X)
        ss = (X * X).sum(axis=1)

        diff = np.sqrt( abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)) )

        np.fill_diagonal(diff,0)
        stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) ) / comb(N,2)

        from scipy.optimize import minimize_scalar
        min_a = minimize_scalar(stress)
        #print("a is ",min_a.x)
        return stress(a=min_a.x)


    def aspect_ratio(self):

        points = [(self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph.nodes()]

        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)
        x_max = max(point[0] for point in points)
        y_max = max(point[1] for point in points)
        width = x_max - x_min
        height = y_max - y_min

        if height > width:
            return width / height
        else:
            return height / width
        
    
    # def node_uniformity(self):

    #     points = [(self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph.nodes()]
    #     x_min = min(point[0] for point in points)
    #     y_min = min(point[1] for point in points)
    #     x_max = max(point[0] for point in points)
    #     y_max = max(point[1] for point in points)

    #     # Calculate the number of cells based on the square root of the number of points
    #     num_points = len(points)
    #     num_cells = int(math.sqrt(num_points))

    #     # Create an empty grid with the specified number of cells
    #     grid = [[0 for _ in range(num_cells)] for _ in range(num_cells)]

    #     # Calculate the cell size based on the range of x and y coordinates
    #     cell_width = (x_max - x_min) / num_cells
    #     cell_height = (y_max - y_min) / num_cells

    #     # Count the number of points in each grid cell
    #     for point in points:
    #         x_index = int((point[0] - x_min) // cell_width)
    #         y_index = int((point[1] - y_min) // cell_height)
    #         grid[x_index][y_index] += 1

    #     # Calculate the evenness metric
    #     total_cells = num_cells * num_cells
    #     average_points_per_cell = num_points / total_cells
    #     evenness = sum(abs(cell - average_points_per_cell) for row in grid for cell in row) / (2 * total_cells)
    #     return 1 - evenness


    def node_uniformity(self):

        points = [(self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph.nodes()]
        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)
        x_max = max(point[0] for point in points)
        y_max = max(point[1] for point in points)


        num_points = len(points)
        num_cells = int(math.sqrt(num_points))


        cell_width = (x_max - x_min) / num_cells
        cell_height = (y_max - y_min) / num_cells


        grid = [[0 for _ in range(num_cells)] for _ in range(num_cells)]


        for i in range(num_cells):
            for j in range(num_cells):
                for point in points:
                    square = ((x_min + (i * cell_width)), (y_min + (j * cell_height))), ((x_min + ((i + 1) * cell_width)), (y_min + ((j + 1) * cell_height))) 
                    #print(square)
                    if self._is_point_inside_square(point, (square[0][0], square[0][1]), (square[1][0], square[1][1])):
                        grid[i][j] += 1


        total_cells = num_cells * num_cells
        average_points_per_cell = num_points / total_cells
        evenness = sum(abs(cell - average_points_per_cell) for row in grid for cell in row) / (2 * total_cells)
        return 1 - evenness if evenness < 1 else 0

    def _is_point_inside_square(self, point, square_bottom_left, square_top_right):
        x, y = point
        x1, y1 = square_bottom_left
        x2, y2 = square_top_right

        return x1 <= x <= x2 and y1 <= y <= y2


   
    def avg_degree(self, G):
        degs = []
        for n in G.nodes():
            degs.append(G.degree(n))
        
        return sum(degs)/G.number_of_nodes()



    def neighbourhood_preservation(self, k=None):

        if k == None:
            k = math.floor(self.avg_degree(self.graph))

        adj = nx.to_numpy_array(self.graph)

        K = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))

        points = [(self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph.nodes()]

        for i, u in enumerate(self.graph.nodes()):
            for j, v in enumerate(self.graph.nodes()):
                
                # shortest_paths = nx.shortest_path_length(G, source=u)

                # if shortest_paths[v] <= k:
                #     K[i][j] = 1
                
                p = (self.graph.nodes[u]['x'], self.graph.nodes[u]['y'])
                nearest = self._find_k_nearest_points(p, points, k+1)

                q = (self.graph.nodes[v]['x'], self.graph.nodes[v]['y'])

                if q in nearest:
                    K[i][j] = 1

        np.fill_diagonal(K, 0)
        # print(K)
        intersection = np.logical_and(adj, K)
        union = np.logical_or(adj, K)
        return intersection.sum() / union.sum()


    def _find_k_nearest_points(self, p, points, k):
        tree = KDTree(points)
        distances, indices = tree.query(p, k=k)
        return [points[i] for i in indices]

    
   
            

if __name__ == "__main__":
    pass
    #ms = MetricsSuite("../Graph Drawings/Barabasi-Albert/Fruchterman-Reingold/fruchterman-reingold_j0_BBA_i0_n70_m136.graphml")
    #ms = MetricsSuite("test2.graphml")
    #ms = MetricsSuite("../Graph Drawings/Barabasi-Albert/Fruchterman-Reingold/fruchterman-reingold_j0_BBA_i35_n10_m24.graphml")
    #print(ms.neighbourhood_preservation(2))
    #ms.new_EC()
    # print(ms.edge_crossing())
    # ms.count_shared_triangles()
    
    #ms.count_impossible_triangle_crossings2()#
    #ms.node_uniformity()
    pass
    #ms.draw_graph()
    #ms.draw_graph()
    #print(ms.edge_crossing())
    # print(ms.count_shared_triangles())
    # print(ms.count_shared_triangles2())
    #ms.draw_graph(ms.graph)
    # print(ms.stress())
    # print(ms.stress2())
    # print(f"EC: {ms.edge_crossing()}")
    # print(f"NEW EC: {ms.ec()}")
    #ms.draw_graph()

    # H = ms.resolve_crossings(ms.graph)
    # ms.draw_graph(H)

    # write_graphml_pos(H, "crosses_out.graphml")
