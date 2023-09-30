import networkx as nx
#from write_graph import write_graphml_pos
import parse_graph
from pygraphml import GraphMLParser

def bends_promotion(G): 

    edge_list1 = [e for e in G.edges(data = True) if e in G.edges(data=True)]   
    for e in edge_list1:   
        prev = e[0]
        try:
            points = e[2]['graphics']   
            if len(points) < 2:            
                continue       
            else:                
                points = points['Line']['point']
                G.remove_edge(e[0], e[1])
                for i in range(1, len(points)-1):
                    node_index = len(G.nodes) + 1
                    node_index = str(node_index)                    
                    x = float(str(points[i]['x']))
                    y = float(str(points[i]['y']))
                    G.add_node(node_index)                    
                    G.nodes[node_index][u'LabelGraphics'] = {u'text': node_index, u'fontName': u'Dialog', u'fontSize': 12, u'anchor': u'c', 'bends_node':1}
                    G.nodes[node_index][u'graphics'] = {u'outline': u'#000000', u'h': 100.0, u'raisedBorder': 0, u'w': 100.0, u'y': y, u'x': x, u'type': u'rectangle', u'fill': u'#FFCC00'}
                    G.add_edge(prev, node_index)
                    prev = node_index
                G.add_edge(prev,e[1])
        except ValueError:
            print("no bends")
            continue
    # nx.write_gml(G,"bends_promoted_" + str(len(G.nodes())))
    
    return G


def remove_duplicates(G):

    nodes_to_fix = {}
    deleted_nodes ={}
    nodes_checked = []
    for n in G.nodes:
        if n in nodes_checked:
            continue
        has_dupes = False
        duped_nodes = []
        for u in G.neighbors(n):
            if G.nodes[n]['graphics']['x'] == G.nodes[u]['graphics']['x'] and G.nodes[n]['graphics']['y'] == G.nodes[u]['graphics']['y']:
                has_dupes = True
                duped_nodes.append(u)
        
        if duped_nodes != []:
            nodes_to_fix[n] = {}
            nodes_to_fix[n]['pos'] = (G.nodes[n]['graphics']['x'], G.nodes[n]['graphics']['y'])
            nodes_to_fix[n]['nodes_to_remove'] = duped_nodes
            edges_to_keep = []

            for u in duped_nodes:
                deleted_nodes[u] = n
                for v in G.neighbors(u):
                    if v not in duped_nodes and n != v:
                        edges_to_keep.append((n, v))
            nodes_to_fix[n]['edges_to_add'] = edges_to_keep

        
        nodes_checked.extend(duped_nodes)
    

    for k in nodes_to_fix.keys():
        for n in nodes_to_fix[k]['nodes_to_remove']:
            G.remove_node(n)
        for e in nodes_to_fix[k]['edges_to_add']:
            link = deleted_nodes.get(e[1])
            if link is None:
                G.add_edge(e[0], e[1])
            else:
                G.add_edge(e[0], link)

    print()
    all_removed = []
    for k in nodes_to_fix.keys():
        print(nodes_to_fix[k]['edges_to_add'])
        all_removed.extend(nodes_to_fix[k]['nodes_to_remove'])

    print()
    print(sorted(all_removed))

    return G


def new_bends_promotion(G):

    H = G.copy()
    # for e in H.edges(data=True):
    #     print(e)
    #     print()
    # #print(H.edges(data=True))

    # # print()
    # for n in H.nodes(data=True):
    #     print(n)

    # existing_node_positions = []

    # for n in H.nodes():
    #     existing_node_positions.append((H.nodes[n]['graphics']['x'], H.nodes[n]['graphics']['y']))

    # print(existing_node_positions)

    for n in H.nodes():
        H.nodes[n]['bend_node'] = False
        #H.nodes[n]['bend_id'] = (None, None)

    j = 1
    bend_nodes = []
    edges_to_remove = []
    for e in H.edges(data=True):
        if 'Line' not in e[2]['graphics']:
            continue
        
        e1 = e[0]
        e2 = e[1]

        edges_to_remove.append((e1,e2))

        points = e[2]['graphics']['Line']['point']
        #print(points)
        
        
        for i in range(1, len(points)-1):
            #print(i)
            bend_nodes.append({'name':"bend-"+str(j)+"-"+str(i), 'original_edge':(e1,e2), 'bend_id':(j,i), 'x':points[i]['x'], 'y':points[i]['y']})
            

        j += 1
    for n in bend_nodes:
        #graphics = {'x': n['x'], 'y': n['y'], u'outline': u'#000000', u'h': 100.0, u'raisedBorder': 0, u'w': 100.0, u'type': u'rectangle', u'fill': u'#FFCC00'}
        graphics = {'x': n['x'], 'y': n['y']}
        H.add_node(n['name'], original_edge=n['original_edge'], bend_node=True, bend_id=n['bend_id'], graphics=graphics)
        #H.nodes[n['name']]["color"] = "#097969"

    # for k, v in H.nodes(data=True):
    #     if "bend" in str(k):
    #         H.nodes[k]["color"] = "#097969"
            
            

    # return H    

    for e in edges_to_remove:
        H.remove_edge(e[0], e[1])

    # for n in H.nodes(data=True):
    #     print(n)
    
    # for n in H.nodes():
    #     if not H.nodes[n]['bend_node']:
    #         continue
        
    #     bend_num = 1
    #     while bend_num <= H.nodes[n]['bend_id'][0]:
    #         #print(bend_num)
    #         bend_part_num = 1
    #         while bend_part_num <= H.nodes[n]['bend_id'][1]:
    #             print(bend_num, bend_part_num)
    #             bend_part_num += 1

    #         bend_num += 1
        
    #     print()
    bend_nodes = [H.nodes[n] for n in H.nodes() if H.nodes[n]['bend_node']]
    # for n in bend_nodes:
    #     print(n)
    #print(bend_nodes)
    num_bends = max(bend_nodes, key=lambda x:x['bend_id'][0])['bend_id'][0]
    #print()
    # print(num_bends)
    # print()

    i = 1
    while i <= num_bends:
        nodes_in_this_bend = []
        bend_index = -1
        for n in bend_nodes:
            if n['bend_id'][0] == i:
                nodes_in_this_bend.append(n)
                bend_index = bend_nodes.index(n)
        # print("-----")
        # print(bend_index)
        # print("-----")
            
        nodes_in_this_bend.insert(0, bend_nodes[bend_index]['original_edge'][0])
        nodes_in_this_bend.append(bend_nodes[bend_index]['original_edge'][1])

        # print('#####')
        # print(nodes_in_this_bend)
        # print('#####')
        
        #print(i)
        #for n in nodes_in_this_bend:
        #     print(n)
        # print()

        j = 0
        k = 1
        while j <= len(nodes_in_this_bend) - 2:
            if j == 0:
                e1 = nodes_in_this_bend[j]
                e2 = 'bend-' + str(nodes_in_this_bend[k]['bend_id'][0]) + '-' + str(nodes_in_this_bend[k]['bend_id'][1])

            elif j == len(nodes_in_this_bend) - 2:
                e1 = 'bend-' + str(nodes_in_this_bend[j]['bend_id'][0]) + '-' + str(nodes_in_this_bend[j]['bend_id'][1])
                e2 = nodes_in_this_bend[k]
                
            else:
                e1 = 'bend-' + str(nodes_in_this_bend[j]['bend_id'][0]) + '-' + str(nodes_in_this_bend[j]['bend_id'][1])
                e2 = 'bend-' + str(nodes_in_this_bend[k]['bend_id'][0]) + '-' + str(nodes_in_this_bend[k]['bend_id'][1])
            
            H.add_edge(e1, e2)
            #print(j,k)
            j += 1
            k += 1


        i += 1


    return H
      

# G = nx.read_gml("hola_in.gml", label=None)
# H = G.copy()
# for n in H.edges(data=True):
#     print(n)
# nx.write_gml(H, "hola_out.gml")



#G = nx.read_gml("..\Graph Drawings\Barabasi-Albert\HOLA\HOLA_BBA_i0_n10_m21.gml", label=None)
# G = nx.read_gml("holatest.gml", label=None)
# #G = parse_graph.read_graphml("..\Graph Drawings\Barabasi-Albert\HOLA\HOLA_BBA_i0_n100_m196.graphml")

# H = new_bends_promotion(G)

# # for n in H.nodes():
# #     print(H.nodes[n])

# #print(H.nodes(data=True))
# #nx.write_gml(H, "hola_out.gml")
# nx.write_gml(H, "hola_out.gml")

# from crosses_promotion import crosses_promotion, crosses_promotion2, crosses_promotion3

# for n in H.nodes():

#     H.nodes[n]['x'] = H.nodes[n]['graphics']['x']
#     H.nodes[n]['y'] = H.nodes[n]['graphics']['y']

# I = crosses_promotion(H)

# for n in H.nodes():
#     H.nodes[n]['graphics']['x'] = H.nodes[n]['x'] 
#     H.nodes[n]['graphics']['y'] = H.nodes[n]['y'] 

# J = crosses_promotion2(H)

# for n in H.nodes():
#     J.nodes[n]['graphics']['x'] = J.nodes[n]['x'] 
#     J.nodes[n]['graphics']['y'] = J.nodes[n]['y'] 

# K = crosses_promotion3(H)

# for n in H.nodes():
#     K.nodes[n]['graphics']['x'] = K.nodes[n]['x'] 
#     K.nodes[n]['graphics']['y'] = K.nodes[n]['y'] 



# nx.write_gml(I, "hola_out_cp.gml")
# nx.write_gml(J, "hola_out_cp2.gml")
# nx.write_gml(K, "hola_out_cp3.gml")

# I = G.copy()
# I = remove_duplicates(I)
# H = G.copy()
# H = remove_duplicates(H)

# nx.write_gml(G, "new_mine.gml")
# nx.write_gml(H, "hola_out.gml")
# write_graphml_pos(H, "hola_out.graphml")

# nx.write_gml(I, "new_mine_extra.gml")
# nx.write_gml(H, "new_nathan.gml")
#nx.write_gml(H, "fixed.gml")
#write_graphml_pos(G, "bp.graphml")