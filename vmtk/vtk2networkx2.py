#convert a centerlines object to one that can segmented into sections
#currently able to create lines, but not poly lines yet

import vtk
import networkx as nx
import sys
import copy

from writeNodesEdges import writeObjects, writePolyLine

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("/Users/sansomk/caseFiles/ultrasound/tcd/case2/vmtk/case2_vmtk_decimate_hole_ctrlines.vtp")
reader.Update()

clean = vtk.vtkCleanPolyData()
clean.ConvertPolysToLinesOff()
clean.ConvertStripsToPolysOff()
clean.PointMergingOn()
clean.SetInputConnection(reader.GetOutputPort())
clean.Update()

pd = clean.GetOutput()

# create a new graph
G = nx.Graph()
# add each point as node
for i in range(pd.GetNumberOfPoints()):
    G.add_node(i, vertex=pd.GetPoint(i))
    for j in range(pd.GetPointData().GetNumberOfArrays()):
        n_comp = pd.GetPointData().GetArray(j).GetNumberOfComponents()
        if (n_comp == 1):
            G.nodes[i][pd.GetPointData().GetArrayName(j)] = pd.GetPointData().GetArray(j).GetTuple(i)[0]
        else:
            G.nodes[i][pd.GetPointData().GetArrayName(j)] = list(pd.GetPointData().GetArray(j).GetTuple(i))

for i in range(pd.GetNumberOfCells()):
    cell = pd.GetCell(i)
    for j in range(cell.GetNumberOfPoints()-1):
        vt1 = cell.GetPointId(j)
        vt2 = cell.GetPointId(j+1)
        #Find the squared distance between the points.
        squaredDistance = vtk.vtkMath.Distance2BetweenPoints(pd.GetPoint(vt1), pd.GetPoint(vt2))

        G.add_edge(vt1, vt2, weight=squaredDistance**0.5)

writeObjects(G,
    node_scalar_list=["MaximumInscribedSphereRadius"],
    edge_scalar_list=["weight"],
    fileout="/Users/sansomk/caseFiles/ultrasound/tcd/case2/vmtk/case2_vmtk_decimate_hole_ctrlines_graph")

d1 = 0
d3 = 0
terminal = []
interior = []
for node in G.nodes():
    d = G.degree[node]
    if (d > 2):
        interior.append(node)

paths = []
for node in interior:
    deg = G.degree(node)
    cur = copy.deepcopy(node)
    for n in G.neighbors(node):
        paths.append([node])
        tmp = copy.deepcopy(n)
        while G.degree(tmp) == 2:
            paths[-1].append(tmp)
            new_ = [ nei for nei in G.neighbors(tmp) if nei not in paths[-1]]
            assert len(new_) == 1
            paths[-1].append(new_[0])
            tmp = copy.deepcopy(new_[0])

c = [ (s, frozenset(s)) for s in paths]
shift = []
while c:
    new = []
    item = c.pop()
    shift.append(item[0])
    for idx, n in enumerate(c):
        if(item[1] == n[1]):
            new.append(idx)
    for i in new:
        c.pop(i)

writePolyLine(G, shift,
    node_scalar_list=["MaximumInscribedSphereRadius"],
    edge_scalar_list=["weight"],
    fileout="/Users/sansomk/caseFiles/ultrasound/tcd/case2/vmtk/case2_vmtk_decimate_hole_ctrlines_graph_test")

#H = nx.Graph()
#test_ = dict([ (k, G.nodes(data=True)[k]) for k in key_pair if k in G.nodes])

"""

#print(d1, d3)#G.degree[node])
#print(terminal, interior)

H = nx.Graph()
for term_node in terminal:
    min_length = sys.float_info.max
    min_node = None

    for inter_node in interior:
    length = nx.shortest_path_length(G, term_node, inter_node, weight="weight")
    if (length < min_length):
    min_node = inter_node
    min_length = length
    key_pair = [term_node, min_node]

    test_ = dict([ (k, G.nodes(data=True)[k]) for k in key_pair if k in G.nodes])

    H.add_nodes_from(test_)
    nx.set_node_attributes(H, test_)

    H.add_edge(term_node, min_node, weight = min_length)

#print(H.edges(data=True))
I = nx.convert_node_labels_to_integers(H, first_label=0, ordering='default', label_attribute="global")
#print(H.edges(data=True))

writeObjects(I,
    node_scalar_list=["MaximumInscribedSphereRadius"],
    edge_scalar_list=["weight"],
    fileout="/Users/sansomk/caseFiles/ultrasound/tcd/case2/vmtk/case2_vmtk_decimate_hole_ctrlines_graph_test")

J = nx.Graph()
interior_cp = interior.copy()
for inter_node in interior:
    min_length = sys.float_info.max
    min_node = None
    for inter_node2 in interior_cp:
    if (inter_node != inter_node2 ):
    length = nx.shortest_path_length(G, inter_node, inter_node2, weight="weight")
    if (length < min_length):
    min_node = inter_node2
    min_length = length
    key_pair = [inter_node, min_node]

    test_ = dict([ (k, G.nodes(data=True)[k]) for k in key_pair if k in G.nodes])
    #print(test_)
    J.add_nodes_from(test_)
    nx.set_node_attributes(J, test_)

    J.add_edge(inter_node, min_node, weight = min_length)

# d1 = 0
# term = []
# inside = []
# for node in J.nodes():
#     d = J.degree[node]
#     if (d == 1):
#         term.append(node)
#     if ( d == 2):
#         inside.append(node)
#
# K = nx.Graph()
# term_cp = interior.copy()
# for term_node in term:
#     min_length = sys.float_info.max
#     min_node = None
#     for term_node2 in term_cp:
#         if (term_node != term_node2 ):
#             length = nx.shortest_path_length(G, term_node, term_node2, weight="weight")
#             if (length < min_length):
#                 min_node = term_node2
#                 min_length = length
#     key_pair = [term_node, min_node]
#
#     test_ = dict([ (k, G.nodes(data=True)[k]) for k in key_pair if k in G.nodes])
#     #print(test_)
#     K.add_nodes_from(test_)
#     nx.set_node_attributes(K, test_)
#
#     K.add_edge(term_node, min_node, weight = min_length)
#
# #print(K.edges(data=True))
# L = nx.convert_node_labels_to_integers(K, first_label=0, ordering='default', label_attribute="global")
# #print(K.edges(data=True), K.nodes(data=True))

writeObjects(L,
    node_scalar_list=["MaximumInscribedSphereRadius"],
    edge_scalar_list=["weight"],
    fileout="/Users/sansomk/caseFiles/ultrasound/tcd/case2/vmtk/case2_vmtk_decimate_hole_ctrlines_graph_test2")
"""
