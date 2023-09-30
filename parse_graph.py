import networkx as nx
import xml.etree.ElementTree as ET
from xml.dom import minidom

def read_graphml(filename):

    G = nx.Graph()

    tree = ET.parse(filename)
    root = tree.getroot()

    node_id = "d1"
    edge_id = "d2"

    for data_elm in root:
        if data_elm.get("yfiles.type") == "nodegraphics":
            node_id = data_elm.get("id")
        if data_elm.get("yfiles.type") == "edgegraphics":
            edge_id = data_elm.get("id")


    #print(root)
    for node in root.findall(".//{http://graphml.graphdrawing.org/xmlns}node"):
        #print(node.get("id"))
        for data in node:
            if data.get("key") != node_id:
                continue

            for shape_node in data:
                for elm in shape_node:
                    #print(elm.tag)
                    if elm.tag == "{http://www.yworks.com/xml/graphml}Geometry":
                        h = float(elm.get("height"))
                        w = float(elm.get("width"))
                        x = float(elm.get("x"))
                        y = float(elm.get("y"))
                    
                    if elm.tag == "{http://www.yworks.com/xml/graphml}Fill":
                        color = elm.get("color")

                    if elm.tag == "{http://www.yworks.com/xml/graphml}Shape":
                        shape = elm.get("type")

            

        G.add_node(node.get("id"), x=x, y=y, w=w, h=h, color=color, shape=shape)


    for edge in root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"):
        #print(edge.tag)
        source = edge.get("source")
        target = edge.get("target")
        polyline = False
        bends = []

        #print(source, target)

        for data in edge:

            if data.get("key") != edge_id:
                continue
            
            
            for poly_line_edge in data:
                for path in poly_line_edge:
                    for point in path:
                        bends.append((point.get("x"), point.get("y")))
        
        if bends != []:
            polyline = True

        G.add_edge(source, target, polyline=polyline, bends=bends)


    return G


# Adapted from: https://github.com/hadim/pygraphml/blob/master/pygraphml/graphml_parser.py
def write_graphml(G, filename, gml_format=False):

    doc = minidom.Document()
    # create root elements
    root = doc.createElement('graphml')
    root.setAttribute("xmlns", "http://graphml.graphdrawing.org/xmlns")
    root.setAttribute("xmlns:y", "http://www.yworks.com/xml/graphml")
    root.setAttribute("xmlns:yed", "http://www.yworks.com/xml/yed/3")

    doc.appendChild(root)

    # create key attribute for nodegraphics
    attr_node = doc.createElement('key')
    attr_node.setAttribute('id', 'd1')
    attr_node.setAttribute('yfiles.type', "nodegraphics")
    attr_node.setAttribute('for', "node")
    root.appendChild(attr_node)

    # create key attribute for edgegraphics
    attr_node = doc.createElement('key')
    attr_node.setAttribute('id', 'd2')
    attr_node.setAttribute('yfiles.type', "edgegraphics")
    attr_node.setAttribute('for', "edge")
    root.appendChild(attr_node)

    # create graph attribute for edges and nodes to be added to
    graph_node = doc.createElement('graph')
    graph_node.setAttribute('id', 'G')
    graph_node.setAttribute('edgedefault', 'undirected')
    root.appendChild(graph_node)

    # Add nodes
    for n in G.nodes():
        
        node = doc.createElement('node')
        if gml_format:
            node.setAttribute('id', "n"+str(n))
        else:
            node.setAttribute('id', str(n))
        data = doc.createElement('data')
        data.setAttribute('key', 'd1')

        # Adding node that allows styles and attributes to be added to nodes
        shapeElement = doc.createElement('y:ShapeNode')

        # Set shape of node 
        nodeShape = doc.createElement('y:Shape')
        shape = G.nodes[n].get("shape_type", 'ellipse')
        nodeShape.setAttribute('type', shape)

        # adding label to node
        nodeLabel = doc.createElement('y:NodeLabel')
        nodeLabel.setAttribute('textColor', '#000000')
        nodeLabel.setAttribute('fontSize', '6')
        label = doc.createTextNode(str(G.nodes[n].get("label", '\n')))
        nodeLabel.appendChild(label)

        # assign colours to nodes
        nodeColour = doc.createElement('y:Fill')
        nodeColour.setAttribute('transparent', 'false')

        nodeColour.setAttribute('color', str(G.nodes[n].get("color", "#FFCC00")))

        # set size of nodes
        pos = doc.createElement('y:Geometry')
        pos.setAttribute('height', '30.0')
        pos.setAttribute('width', '30.0')

        # set x and y coordinates for each point
        if gml_format:
            pos.setAttribute('x', str(float(G.nodes[n]['graphics'].get("x", '0')) - 15))
            pos.setAttribute('y', str(float(G.nodes[n]['graphics'].get("y", '0')) -15))
        else:
            pos.setAttribute('x', str(G.nodes[n].get("x", '0')))
            pos.setAttribute('y', str(G.nodes[n].get("y", '0')))

        # adding styling attributes to each node
        shapeElement.appendChild(nodeColour)
        shapeElement.appendChild(pos)
        shapeElement.appendChild(nodeShape)
        shapeElement.appendChild(nodeLabel)
        data.appendChild(shapeElement)
        node.appendChild(data)
        graph_node.appendChild(node)

    # Add edges between nodes
    for e in G.edges():
        edge = doc.createElement('edge')
        if gml_format:
            edge.setAttribute('source', "n"+str(e[0]))
            edge.setAttribute('target', "n"+str(e[1]))
        else:
            edge.setAttribute('source', str(e[0]))
            edge.setAttribute('target', str(e[1]))
        graph_node.appendChild(edge)

        data = doc.createElement('data')
        data.setAttribute('key', 'd2')
        edge.appendChild(data)

        polyline_edge = doc.createElement('y:PolyLineEdge')
        data.appendChild(polyline_edge)

        path = doc.createElement("y:Path")
        path.setAttribute('sx', "0.0")
        path.setAttribute('sy', "0.0")
        path.setAttribute('tx', "0.0")
        path.setAttribute('ty', "0.0")
        polyline_edge.appendChild(path)

        try:
            if gml_format:
                for bend in G.edges[e]['graphics']['Line']['point']:
                    point = doc.createElement("y:Point")
                    point.setAttribute("x", str(bend["x"]))
                    point.setAttribute("y", str(bend["y"]))
                    path.appendChild(point)
            else:
                for bend in G.edges[e]["bends"]:
                    point = doc.createElement("y:Point")
                    point.setAttribute("x", str(bend[0]))
                    point.setAttribute("y", str(bend[1]))
                    path.appendChild(point)
        except KeyError:
            continue
        
        linestyle = doc.createElement("y:LineStyle")
        linestyle.setAttribute("color", "#000000")
        linestyle.setAttribute("type", "line")
        linestyle.setAttribute("width", "1.0")
        polyline_edge.appendChild(linestyle)

        arrows = doc.createElement("y:Arrows")
        arrows.setAttribute("source", "none")
        arrows.setAttribute("target", "none")
        polyline_edge.appendChild(arrows)

        bendstyle = doc.createElement("y:BendStyle")
        bendstyle.setAttribute("smoothed", "true")
        polyline_edge.appendChild(bendstyle)

        

    with open(filename, 'w') as f:
        f.write(doc.toprettyxml(indent='    '))


def test_graph_read_write():
    G = read_graphml("test.graphml")

    write_graphml(G, "wtest.graphml")

    H = read_graphml("wtest.graphml")

    G_nodes = []
    G_edges = []

    H_nodes = []
    H_edges = []

    print(G.nodes(data=True) == H.nodes(data=True))

    for n in G.nodes(data=True):
        G_nodes.append(n)

    for e in G.edges(data=True):
        G_edges.append(e)
        
    for n in H.nodes(data=True):
        H_nodes.append(n)

    for e in H.edges(data=True):
        H_edges.append(e)

    print(G_nodes == H_nodes)
    print(G_edges == H_edges)

    print(len(G_nodes) == len(H_nodes))
    print(len(G_edges) == len(H_edges))

    for i in range(len(G_nodes)):
        print(G_nodes[i])
        print(H_nodes[i])
        print()


def convert_gml_to_graphml(fname_gml, fname_graphml):
    G = nx.read_gml(fname_gml, label=None)
    write_graphml(G, fname_graphml, True)

def convert_graphml_to_gml(fname_graphml, fname_gml, with_nx=False):
    """ IMPORTANT: DOES NOT PRESERVE NODE POSITIONS OR EDGE ATTRIBUTES! USE ONLY ON GRAPHS, NOT DRAWINGS."""
    if with_nx:
        G = nx.read_graphml(fname_graphml)
    else:
        G = read_graphml(fname_graphml)
    nx.write_gml(G, fname_gml)


# G = read_graphml(".new_output.graphml")

# write_graphml(G, "hola_out.graphml")

# H = read_graphml("hola_out.graphml")

# G = nx.read_gml(".new_output.gml", label=None)

# H = read_graphml("hola_out.graphml")

# G_nodes = []
# G_edges = []

# H_nodes = []
# H_edges = []

# for n in G.nodes(data=True):
#     G_nodes.append(n)

# for e in G.edges(data=True):
#     G_edges.append(e)
    
# for n in H.nodes(data=True):
#     H_nodes.append(n)

# for e in H.edges(data=True):
#     H_edges.append(e)

# print(G_nodes == H_nodes)
# print(G_edges == H_edges)

# print(len(G_nodes) == len(H_nodes))
# print(len(G_edges) == len(H_edges))

# for i in range(len(G_nodes)):
#     print(G_nodes[i])
#     print(H_nodes[i])
#     print()

# for i in range(len(G_edges)):
#     print(G_edges[i])
#     print(H_edges[i])
#     print()


# G = nx.read_gml(".new_output.gml", label=None)

# write_graphml(G, "wgmltest.graphml", True)

#convert_graphml_to_gml("test.graphml", "test.gml")
#convert_gml_to_graphml("hola_in.gml", "hola_out.graphml")