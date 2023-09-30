import networkx as nx
import os
from parse_graph import read_graphml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def count_impossible_triangle_crossings(G):
    nx.draw(G, with_labels=True)
    plt.show()
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

    number_of_triangles = 0
    num_not_part_tri = 0
    edges_not_tri = []
    total_impossible = 0
    for u, v, t in triangles:
        number_of_triangles += 1
        bubble = []
        bubble.extend(G.edges(u))
        bubble.extend(G.edges(v))
        bubble.extend(G.edges(t))

        subG = nx.Graph(bubble)
        # nx.draw(subG)
        # plt.show()

        for a, b in G.edges():
            if (a, b) in subG.edges() or (b, a) in subG.edges():
                continue

            if {a, b} in triangle_edges:
                continue

            if {a,b} not in edges_not_tri:
                edges_not_tri.append({a,b})

            total_impossible += 1
    num_not_part_tri = len(edges_not_tri)
    print(edges_not_tri)

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
                        

    print(f"Num Triangles: {number_of_triangles}")
    print(f"Num Not in Triangles: {num_not_part_tri}")
    return (total_impossible, (num_4_cycles // 4))


def count_impossible_triangle_crossings2(G):
    #nx.draw(G, with_labels=True)
    #plt.show()
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

    triangle_nodes = list(set([e for s in triangles for e in s]))
    #print(triangles)
    #print(triangle_nodes)
    edges_not_adj_to_triangles = []

    for u, v in G.edges():
        if u in triangle_nodes or v in triangle_nodes:
            continue

        edges_not_adj_to_triangles.append({u,v})
        
    #print(edges_not_adj_to_triangles)

    def share_edge(triangle1, triangle2):
        return len(triangle1.intersection(triangle2)) >= 2

    triangles_with_no_shared_edge = []

    # Iterate through triangles
    for triangle in triangles:
        has_shared_edge = False
        for other_triangle in triangles:
            if triangle != other_triangle and share_edge(triangle, other_triangle):
                has_shared_edge = True
                break
        
        if not has_shared_edge:
            triangles_with_no_shared_edge.append(triangle)

    #print(triangles_with_no_shared_edge)


    from itertools import combinations
    # Initialize a dictionary to store shared edges
    shared_edges = {}

    # Iterate through combinations of two triangles
    for triangle1, triangle2 in combinations(triangles, 2):
        shared_edge = triangle1.intersection(triangle2)
        
        # If there's a shared edge, add it to the dictionary
        if len(shared_edge) == 2:
            shared_edges.setdefault(tuple(shared_edge), []).extend([triangle1, triangle2])

    # Display triangles with shared edges
    for edge, ts in shared_edges.items():
        shared_edges[edge] = list(set(frozenset(s) for s in ts))
    
    # for edge, ts in shared_edges.items():
    #     print("Shared edge:", edge)
    #     print("Triangles:", ts)
    #     print()

    # print(len(shared_edges.keys()))
    # print(len(triangles))
    #total_impossible += (len(edges_not_adj_to_triangles) * len(triangles)) # NOT QUITE CORRECT, HAVE TO LOOK FOR SHARED EDGES
    #total_impossible += (len(edges_not_adj_to_triangles) * len(shared_edges.keys())) # NOT QUITE CORRECT, HAVE TO LOOK FOR SHARED EDGES
    total_impossible += (len(edges_not_adj_to_triangles) * len(triangles_with_no_shared_edge))


    #some way to find unique triangle strucutres required here

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
                        

    return (total_impossible, (num_4_cycles // 4))
    

def edge_crossing(G):
    """Calculate the metric for the number of edge_crossing, scaled against the total
    number of possible crossings."""

    # Estimate for the upper bound for the number of edge crossings
    m = G.number_of_edges()
    c_all = (m * (m - 1))/2
    
    c_deg = sum([(G.degree[u] * (G.degree[u] - 1)) for u in G])/2
    
    c_tri, c_4 = count_impossible_triangle_crossings2(G)

    #c_mx_old = c_all - c_deg

    c_mx_new = c_all - c_deg - c_tri - c_4

    

    covered = []
    c = 0
    # Iterate over all pairs of edges, checking if they intersect
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
            
            if intersect(line_a, line_b) and (line_a, line_b) not in covered:
                covered.append((line_b, line_a))                  
                c += 1


    #old_ec = 1 - (c / c_mx_old) if c_mx_old > 0 else 1

    new_ec = 1 - (c / c_mx_new) if c_mx_new > 0 else 1

    # output_vals = [
    #     c,
    #     c_all,
    #     c_deg,
    #     c_tri,
    #     c_4,
    #     c_mx_old,
    #     c_mx_new,
    #     old_ec,
    #     new_ec,
    # ]

    #return output_vals
    return new_ec, c


def count_4_cycles(G):
    

    num_4_cycles = 0
    #four_cycles = []

    for u, v in G.edges():
        for t in G.neighbors(u):
            if t == v:
                continue

            for w in G.neighbors(v):
                if w == t or w == u:
                    continue

                if w in G.neighbors(t):
                    # if {u,v,t,w} not in four_cycles:
                    #     four_cycles.append({u,v,t,w})
                    num_4_cycles += 1

    #return len(four_cycles)
    return num_4_cycles // 4


def on_opposite_sides(a, b, line):
    """Check if two lines pass the on opposite sides test. Return True if they do."""
    g = (line[1][0] - line[0][0]) * (a[1] - line[0][1]) - (line[1][1] - line[0][1]) * (a[0] - line[0][0])
    h = (line[1][0] - line[0][0]) * (b[1] - line[0][1]) - (line[1][1] - line[0][1]) * (b[0] - line[0][0])
    return g * h <= 0.0 and (a != line[1] and b != line[0] and a != line[0] and b != line[1])


def bounding_box(line_a, line_b):
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


def intersect(line_a, line_b):
    """Check if two lines intersect by checking the on opposite sides and bounding box 
    tests. Return True if they do."""
    return (on_opposite_sides(line_a[0], line_a[1], line_b) and 
            on_opposite_sides(line_b[0], line_b[1], line_a) and 
            bounding_box(line_a, line_b))



def main(stop=None):
    graphs_dir = "..\\Graphs\\"
    drawings_dir = "..\\Graph Drawings\\"
    output_file = "..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv"
    graph_types = ["Lancichinetti-Fortunato-Radicchi", "Barabasi-Albert", "Erdos-Renyi", "Newman-Watts-Strogatz", "Geometric", "Stochastic-Block-Model", "Rome", "North"]
    layouts = ["Fruchterman-Reingold", "Kamada-Kawai", "Sugiyama", "DRGraph", "Random"] #, "HOLA"]
    layouts = ["Fruchterman-Reingold", "Sugiyama", "Random"] #, "HOLA"]
    

    i = 0
    with open(output_file, "w") as out_f:

        legend = "filename,n,m,density,num_crossings,initial_max,c_deg,c_tri,c_4,old_final_max,new_final_max,old_ec,new_ec\n"
        out_f.write(legend)

        for graph_type in os.listdir(drawings_dir):
            if graph_type not in graph_types:
                continue

            for alg in os.listdir(drawings_dir + graph_type):
                if alg not in layouts:
                    continue

                for graph in os.listdir(drawings_dir + graph_type + "\\" + alg):
                    if graph.endswith("txt") or graph.endswith("vna"):
                        continue

                    print(f"{i} {graph}")
                    

                    if i == stop: break
                    i += 1

                    if alg == "HOLA":
                        G = nx.read_gml(drawings_dir + graph_type + "\\" + alg + "\\" + graph, label=None)

                        for n in G.nodes():
                            G.nodes[n]['x'] = G.nodes[n]['graphics']['x']
                            G.nodes[n]['y'] = G.nodes[n]['graphics']['y']

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

                        data = edge_crossing(H)

                        line = graph + "," + str(G.number_of_nodes()) + "," + str(G.number_of_edges()) + "," + str(nx.density(G)) + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in data) + "\n"

                        out_f.write(line)

                    else:

                        G = read_graphml(drawings_dir + graph_type + "\\" + alg + "\\" + graph)

                        data = edge_crossing(G)

                        line = graph + "," + str(G.number_of_nodes()) + "," + str(G.number_of_edges()) + "," + str(nx.density(G)) + "," +  ",".join(str(round(v, 3)) if v != None else "None" for v in data) + "\n"

                        out_f.write(line)

def main2(stop=None):
    drawings_dir = "Random_ER\\"
    output_file = "Random_ER_Edge_Crossings_Metric.csv"

    i = 0
    with open(output_file, "w") as out_f:

        legend = "filename,n,m,density,num_crossings,initial_max,c_deg,c_tri,c_4,old_final_max,new_final_max,old_ec,new_ec\n"
        out_f.write(legend)

        for graph in os.listdir(drawings_dir):

            print(f"{i} {graph}")
            

            if i == stop: break
            i += 1

            G = read_graphml(drawings_dir + graph)

            data = edge_crossing(G)

            line = graph + "," + str(G.number_of_nodes()) + "," + str(G.number_of_edges()) + "," + str(nx.density(G)) + "," +  ",".join(str(round(v, 3)) if v != None else "None" for v in data) + "\n"

            out_f.write(line)



def random_distributions():
    # df = pd.read_csv("..\\Data\\Edge_Crossings.csv")
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    df2 = df.copy()
    df2['random'] = ["" for _ in range(len(df2))]




    for i in range(int(len(df2))):
        if "random" in df2.loc[i, 'filename']:
            df2.loc[i, 'random'] = "Random"
        else:

            df2.loc[i, 'random'] = "Not Random"

    df2 = df2[df2['random'] == 'Random']
    new_df = pd.concat([df2, df2], ignore_index=True)

    

    new_df['Original/Improved'] = ["Improved EC" for _ in range(len(new_df))]

    new_df['blank'] = ["" for _ in range(len(new_df))]


    for i in range(int(len(new_df)/2)):
        new_df.loc[i, 'new_ec'] = new_df.loc[i, 'old_ec']
        new_df.loc[i, 'Original/Improved'] = "Original EC"


    #print(new_df)


    
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Generate violin plots
    #ax.violinplot([df['old_ec'], df['new_ec']], showmedians=True)


    colors = ["tab:blue", "tab:orange"]

    #data = [new_df['old_ec'], new_df['new_ec']]
    # Create a violin plot for the current layout and label
    ax = sns.violinplot(data=new_df, ax=ax, saturation=0.5, hue='Original/Improved', x="new_ec", y="blank", split=True, inner="quartile", linewidth=0.1, palette=colors)

    for l in ax.lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('red')
        l.set_alpha(0.8)
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)

    

    ax.set_xlabel("")
    ax.set_ylabel("")
    #ax.legend_.remove()
    ax.set_xlim(0, 1)
    ax.set_title('')

    # Rotate the x-axis labels vertically
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', length=0)  # Remove horizontal tick marks

    # Show tick marks only for the outer top and right plots
    ax.set_xticklabels([], ha='center')  # Align center

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '', '1'])
    for tick_label, tick_pos in zip(ax.get_xticklabels(), ax.get_xticks()):
        if tick_pos == 0:
            tick_label.set_ha('left')
        elif tick_pos == 0.5:
            tick_label.set_ha('center')
        elif tick_pos == 1:
            tick_label.set_ha('right')

    ax.set_yticks([])

    ax.set_title('Edge Crossing Metric Value Distributions')
    ax.set_xlabel('Edge Crossing Metric Value')
    #ax.set_ylabel('# Of Drawings')

    # Show the plot
    plt.savefig('ec_dist_ran.pdf', format="pdf") 
    plt.show()


def distributions():
    # df = pd.read_csv("..\\Data\\Edge_Crossings.csv")
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    df2 = df.copy()



    # df2['random'] = ["" for _ in range(len(df2))]

    # for i in range(int(len(df2))):
    #     if "random" in df2.loc[i, 'filename']:

    #         df2.loc[i, 'random'] = "Random"
    #     else:

    #         df2.loc[i, 'random'] = "Not Random"


    # df2 = df2[df2['random'] != 'Random']

    new_df = pd.concat([df2, df2], ignore_index=True)

    

    new_df['Original/Improved'] = ["Improved EC" for _ in range(len(new_df))]

    new_df['blank'] = ["" for _ in range(len(new_df))]


    for i in range(int(len(new_df)/2)):
        new_df.loc[i, 'new_ec'] = new_df.loc[i, 'old_ec']
        new_df.loc[i, 'Original/Improved'] = "Original EC"


    #print(new_df)


    
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Generate violin plots
    #ax.violinplot([df['old_ec'], df['new_ec']], showmedians=True)


    colors = ["tab:blue", "tab:orange"]

    #data = [new_df['old_ec'], new_df['new_ec']]
    # Create a violin plot for the current layout and label
    ax = sns.violinplot(data=new_df, ax=ax, saturation=0.5, hue='Original/Improved', x="new_ec", y="blank", split=True, inner="quartile", linewidth=0.1, palette=colors)

    for l in ax.lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('red')
        l.set_alpha(0.8)
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)

    

    ax.set_xlabel("")
    ax.set_ylabel("")
    #ax.legend_.remove()
    ax.set_xlim(0, 1)
    ax.set_title('')

    # Rotate the x-axis labels vertically
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', length=0)  # Remove horizontal tick marks

    # Show tick marks only for the outer top and right plots
    ax.set_xticklabels([], ha='center')  # Align center

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '', '1'])
    for tick_label, tick_pos in zip(ax.get_xticklabels(), ax.get_xticks()):
        if tick_pos == 0:
            tick_label.set_ha('left')
        elif tick_pos == 0.5:
            tick_label.set_ha('center')
        elif tick_pos == 1:
            tick_label.set_ha('right')

    ax.set_yticks([])

    ax.set_title('Edge Crossing Metric Value Distributions')
    ax.set_xlabel('Edge Crossing Metric Value')
    #ax.set_ylabel('# Of Drawings')

    # Show the plot
    plt.savefig('ec_dist_all.pdf', format="pdf") 
    plt.show()



def corr():
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))

    ax1.scatter(df['m'], df['new_ec'], color="tab:orange")
    ax2.scatter(df['m'], df['old_ec'], color="tab:blue")

    ax1.set_xlabel('Number of Edges')
    ax1.set_ylabel('Improved Edge Crossing')

    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Original Edge Crossing')

    trendline1_coeffs = np.polyfit(df['m'], df['new_ec'], 1)
    trendline1_x = np.linspace(df['m'].min(), df['m'].max(), 100)
    trendline1_y = np.polyval(trendline1_coeffs, trendline1_x)
    ax1.plot(trendline1_x, trendline1_y, color='black', label='Trendline')

    trendline2_coeffs = np.polyfit(df['m'], df['old_ec'], 1)
    trendline2_x = np.linspace(df['m'].min(), df['m'].max(), 100)
    trendline2_y = np.polyval(trendline2_coeffs, trendline2_x)
    ax2.plot(trendline2_x, trendline2_y, color='black', label='Trendline')

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle('Correlation between m and each metric')

    
    plt.tight_layout()
    plt.savefig('ec_m.png', format="png") 
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))

    ax1.scatter(df['n'], df['new_ec'], color="tab:orange")
    ax2.scatter(df['n'], df['old_ec'], color="tab:blue")

    ax1.set_xlabel('Number of Edges')
    ax1.set_ylabel('Improved Edge Crossing')

    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Original Edge Crossing')

    trendline1_coeffs = np.polyfit(df['n'], df['new_ec'], 1)
    trendline1_x = np.linspace(df['n'].min(), df['n'].max(), 100)
    trendline1_y = np.polyval(trendline1_coeffs, trendline1_x)
    ax1.plot(trendline1_x, trendline1_y, color='black', label='Trendline')

    trendline2_coeffs = np.polyfit(df['n'], df['old_ec'], 1)
    trendline2_x = np.linspace(df['n'].min(), df['n'].max(), 100)
    trendline2_y = np.polyval(trendline2_coeffs, trendline2_x)
    ax2.plot(trendline2_x, trendline2_y, color='black', label='Trendline')

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle('Correlation between n and each metric')

    
    plt.tight_layout()
    plt.savefig('ec_n.png', format="png") 
    plt.show()


def numeric():
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    data = df['new_ec']

    # Range
    range_val = np.max(data) - np.min(data)

    # Variance
    variance_val = np.var(data)

    # Standard Deviation
    std_deviation_val = np.std(data)

    # Interquartile Range (IQR)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(data - np.mean(data)))

    # Using pandas
    data_series = pd.Series(data)
    range_val = data_series.max() - data_series.min()
    variance_val = data_series.var()
    std_deviation_val = data_series.std()
    iqr = data_series.quantile(0.75) - data_series.quantile(0.25)
    mad = data_series.mad()

    print("Range:", range_val)
    print("Variance:", variance_val)
    print("Standard Deviation:", std_deviation_val)
    print("Interquartile Range (IQR):", iqr)
    print("Mean Absolute Deviation (MAD):", mad)



def distributions_sep():
    # df = pd.read_csv("..\\Data\\Edge_Crossings.csv")
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    df2 = df.copy()


    df2['layout'] = ["" for _ in range(len(df2))]

    for i in range(int(len(df2))):
        if "random" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "ran"
        elif "sugi" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "sugi"
        elif "fruchterman" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "fr"
        else:
            df2.loc[i, 'layout'] = "other"
    
    new_df = pd.concat([df2, df2], ignore_index=True)

    new_df['Original/Improved'] = ["Improved EC" for _ in range(len(new_df))]
    new_df['blank'] = ["" for _ in range(len(new_df))]


    for i in range(int(len(new_df)/2)):
        new_df.loc[i, 'new_ec'] = new_df.loc[i, 'old_ec']
        new_df.loc[i, 'Original/Improved'] = "Original EC"

    fr_df = new_df[new_df['layout'] == 'fr']
    sugi_df = new_df[new_df['layout'] == 'sugi']
    ran_df = new_df[new_df['layout'] == 'ran']
    

    dfs = {"Fruchterman-Reingold": fr_df,
           "Sugiyama": sugi_df,
           "Random": ran_df}
    

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    colors = ["tab:blue", "tab:orange"]


    for i, (label, df) in enumerate(dfs.items()):
        ax = axs[i]
        ax = sns.violinplot(data=df, ax=ax, saturation=0.5, hue='Original/Improved', x="new_ec", y="blank", split=True, inner="quartile", linewidth=0.1, palette=colors)

        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('red')
            l.set_alpha(0.8)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        ax.set_title('')
        #ax.tick_params(axis='x', rotation=0)
        #ax.tick_params(axis='y', length=0)
        #ax.set_xticklabels([], ha='center')
        #ax.set_xticks([0, 0.5, 1])
        #ax.set_xticklabels(['0', '', '1'])
        # for tick_label, tick_pos in zip(ax.get_xticklabels(), ax.get_xticks()):
        #     if tick_pos == 0:
        #         tick_label.set_ha('left')
        #     elif tick_pos == 0.5:
        #         tick_label.set_ha('center')
        #     elif tick_pos == 1:
        #         tick_label.set_ha('right')

        ax.set_yticks([])
        ax.set_title(label)
        ax.set_xlabel('Edge Crossing Metric Value')

    
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left') 

    # Adjust spacing between subplots
    #plt.subplots_adjust(wspace=0.4)

    # Save the plot
    plt.savefig('ec_dist_sep_no_lim.pdf', format="pdf")
    plt.show()



def numeric2():
    # df = pd.read_csv("..\\Data\\Edge_Crossings.csv")
    df = pd.read_csv("..\\Data\\crossings\\Edge_Crossings_Metric_Fixed.csv")

    df2 = df.copy()


    df2['layout'] = ["" for _ in range(len(df2))]

    for i in range(int(len(df2))):
        if "random" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "ran"
        elif "sugi" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "sugi"
        elif "fruchterman" in df2.loc[i, 'filename']:
            df2.loc[i, 'layout'] = "fr"
        else:
            df2.loc[i, 'layout'] = "other"
    

    fr_df = df2[df2['layout'] == 'fr']
    sugi_df = df2[df2['layout'] == 'sugi']
    ran_df = df2[df2['layout'] == 'ran']
    

    dfs = {"Fruchterman-Reingold": fr_df,
           "Sugiyama": sugi_df,
           "Random": ran_df}
    
    print("New")
    for label, df in dfs.items():
        median = np.median(df['new_ec'])
        q1 = np.percentile(df['new_ec'], 25)
        q3 = np.percentile(df['new_ec'], 75)
        range_val = df['new_ec'].max() - df['new_ec'].min()
        
        print(f"Dataframe: {label}")
        print(f"Median: {median:.3f}")
        print(f"Q1: {q1:.3f}")
        print(f"Q3: {q3:.3f}")
        print(f'Range: {range_val:.3f}')
        print()

    print("\nOld")
    for label, df in dfs.items():
        median = np.median(df['old_ec'])
        q1 = np.percentile(df['old_ec'], 25)
        q3 = np.percentile(df['old_ec'], 75)
        range_val = df['old_ec'].max() - df['old_ec'].min()
        
        print(f"Dataframe: {label}")
        print(f"Median: {median:.3f}")
        print(f"Q1: {q1:.3f}")
        print(f"Q3: {q3:.3f}")
        print(f'Range: {range_val:.3f}')
        print()


def test():
    #G = read_graphml("random_GEO_i53_n10_m18.graphml")
    G = read_graphml("test.graphml")
    print("\n"*10)
    print(edge_crossing(G))
    
    nx.draw(G, with_labels=True)
    plt.show()
    # print()
    # G = read_graphml("random_GEO_i53_n10_m18_2.graphml")
    # print(edge_crossing(G))

    #count_impossible_triangle_crossings2(G)
    #print(count_4_cycles(G))
    #count_impossible_triangle_crossings6(G)

    
   

if __name__ == "__main__":
    main2()
    # if len(sys.argv) > 1:
    #     first_argument = sys.argv[1]
    #     print(first_argument)
    # else:
    #     print("Usage: python edge_crossing_metric.py [graphml_file]")

    # G = read_graphml(first_argument)
    # ec_vals = edge_crossing(G)
    # #     output_vals = [
    # #     c,
    # #     c_all,
    # #     c_deg,
    # #     c_tri,
    # #     c_4,
    # #     c_mx_old,
    # #     c_mx_new,
    # #     old_ec,
    # #     new_ec,
    # # ]
    # print(f"Number of crossings: {ec_vals[0]}")
    # print(f"Total possible crossings (assuming every edge crosses every other edge): {ec_vals[1]}")
    # print(f"Number of impossible crossings due to adjacency: {ec_vals[2]}")
    # print(f"Max possible crossings after considering adjacency: {ec_vals[1]} - {ec_vals[2]} = {ec_vals[1]-ec_vals[2]}")
    # print()
    # print(f"Number of impossible crossings due to triangles: {ec_vals[3]}")
    # print(f"Number of impossible crossings due to 4-cycles: {ec_vals[4]}")
    # print(f"Max possible crossings after considering adjaceny, triangles, 4-cycles: {ec_vals[1]} - {ec_vals[2]} - {ec_vals[3]} - {ec_vals[4]} = {ec_vals[6]}")
    # print(f"Edge crossing metric: {ec_vals[-1]:.3f}")


    #test()
    #main()
    #main()

    #numeric2()
    #distributions_sep()
    #distributions()
    #random_distributions()
    #corr()
    #numeric()