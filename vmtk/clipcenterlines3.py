#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os
import sys

import networkx as nx
from writeNodesEdges import writeObjects, writePolyLine

def addPolyLine(pts_ids):
    new_line = vtk.vtkPolyLine()
    new_line.GetPointIds().SetNumberOfIds(pts_ids.GetNumberOfIds())
    for  i in range(pts_ids.GetNumberOfIds()):
        new_line.GetPointIds().SetId(i, pts_ids.GetId(i))
    return new_line

# clip centerlines for accurate segment analysis


def centerline2graph(pd):
    # create a new graph
    G = nx.Graph()
    # add each point as node
    for i in range(pd.GetNumberOfPoints()):
        G.add_node(i, vertex=pd.GetPoint(i))
        for j in range(pd.GetPointData().GetNumberOfArrays()):
            n_comp = pd.GetPointData().GetArray(j).GetNumberOfComponents()
            if (n_comp == 1):
                G.nodes[i][pd.GetPointData().GetArrayName(j)] = pd.GetPointData().GetArray(j).GetTuple(i)
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
            
    return G
    

def Execute(args):
    print("clip centerlines")
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    
    #print(args.clean_ctr)
    if(args.clean_ctr):
        print("default cleaning")
        cleaner = vtk.vtkCleanPolyData()
        cleaner.PointMergingOn()
        cleaner.ConvertPolysToLinesOff()
        cleaner.SetInputData(reader_ctr.Surface)
        cleaner.Update()
        centerlines = cleaner.GetOutput()
    else:
        centerlines = reader_ctr.Surface
    
    centerlines.BuildLinks()
    centerlines.BuildCells()
    
    reader_br = vmtkscripts.vmtkSurfaceReader()
    reader_br.InputFileName = args.boundary_file
    reader_br.Execute()
    boundary_reference = reader_br.Surface
        
    
    #print(pt1, pt2)
    #v =  pt2 - pt1 #pt1 - pt2
    #v_mag = np.linalg.norm(v)
    #n = v / v_mag
    #print("should be 1.0", np.linalg.norm(n), n)

    #https://en.wikipedia.org/wiki/Vector_projection
    # get starting point from centroid by projecting centroid onto normal direction
    #neck_projection = np.dot(neck_centroid-pt1, n)*n
    #neck_start_pt = pt1 + neck_projection
    new_ctr = vtk.vtkPolyData()
    new_ctr.DeepCopy(centerlines)


    locator = vtk.vtkPointLocator()
    locator.SetDataSet(new_ctr)
    locator.BuildLocator()
    
    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(new_ctr)
    cell_loc.BuildLocator()

    clip_ids = []
    new_points = vtk.vtkPoints()
    new_cell_array = vtk.vtkCellArray()
    scalar = vtk.vtkIntArray()
    scalar.SetNumberOfComponents(1)
    scalar.SetNumberOfTuples(new_ctr.GetNumberOfPoints()) 
    scalar.SetName("clipper")
    scalar.Fill(0)
    clip_points = []
    clip_ids = []
    for br_pt_i  in range(boundary_reference.GetNumberOfPoints()):
        pt = boundary_reference.GetPoint(br_pt_i) #B
        pt_b = np.array(pt)
        #print(pt)
        #ctr_ptId = locator.FindClosestPoint(pt)
        id_list = vtk.vtkIdList()
        locator.FindClosestNPoints(10, pt, id_list)
        
        n_br = np.array(boundary_reference.GetPointData().GetArray("BoundaryNormals").GetTuple(br_pt_i))
        
        found_back = False
        found_front = False
        for p in range(id_list.GetNumberOfIds()):
            ctr = np.array(new_ctr.GetPoint(id_list.GetId(p)))
            n_s = np.dot(pt_b - ctr, n_br)
            if (n_s < 0.0 and  found_back == False):
                proj_start = ctr
                start_id = id_list.GetId(p)
                found_back = True
            if (n_s > 0.0 and  found_front == False):
                proj_stop = ctr
                stop_id = id_list.GetId(p)
                found_front = True
                
            if(found_front == True and found_back == True):
                break

        if(found_front == False or found_back == False):
            print("didn't find points on either side")
            

        #Get each vector normal to outlet
        n_ctr = np.array(new_ctr.GetPointData().GetArray("FrenetTangent").GetTuple(start_id))

        if ( np.dot(n_br, n_ctr) < 0.0):
            n_ctr = -1.0 * n_ctr

        #outlet centroid projected onto centerline based on FrenetTangent
        proj_vec = np.dot(n_br, pt_b - proj_start) * n_ctr
        proj_end = proj_vec + proj_start

        new_ctr.GetPoints().SetPoint(stop_id, tuple(proj_end))
        clip_points.append(proj_end)
        clip_ids.append((start_id,stop_id))
       
       
    #clean = vtk.vtkCleanPolyData()
    #clean.ConvertPolysToLinesOff()
    #clean.ConvertStripsToPolysOff()
    #clean.PointMergingOn()
    #clean.SetInputData(new_ctr)
    #clean.Update()

    #pd = clean.GetOutput()
    pd = new_ctr

    # create a new graph
    G = centerline2graph(pd)
    
    #print(len(G))

    #print(clip_ids)
    remove_nodes = []
    for start_node, end_node in clip_ids:
        join =  [ n for n in G[end_node]  if n != start_node]
        assert len(join) == 1, "Should only have one node left"
        
        G.remove_edge(end_node, join[0])
        remove_nodes.append(end_node)#end_diff[0])

    #print("yay")
    remove_list = [] 
    for c in nx.connected_components(G):
        for rn in remove_nodes:
            if rn in c:
                remove_list.extend(list(c))
                #print(len(c))
                #G.remove_nodes_from(list(c))
                break
    G.remove_nodes_from(remove_list)
    
    
    mapping_reset = { i:idx for idx, i in enumerate(G.nodes)}
    G_relabel = nx.relabel_nodes(G, mapping_reset, copy=True)
    #G_relabel = G
    interior = []
    degree_dict = {}
    for node, deg in G_relabel.degree():
        #print(node, deg)
        degree_dict[node]  = tuple((int(deg),)) # needs to be a tuple or list
        if (deg > 2):
            interior.append(node)
            
    nx.set_node_attributes(G_relabel, degree_dict, name="degree")
    
    #print(G_relabel.nodes(data=True))


    path_dict = {}
    length_dict = {}
    path_labels = {}
    visited = []
    #print(len(interior))
    for node in interior:
        if node in visited:
            pass
        else:
            visited.append(node)
        for n in G_relabel.neighbors(node):
            tmp = n
            new_path = [node]
            while G_relabel.degree(tmp) == 2:
                new_path.append(tmp)
                tmp = [ nd for nd in G_relabel.neighbors(tmp) if nd not in new_path]
                assert len(tmp) == 1, "only should have one meaningful neighbor"
                tmp = tmp[0]
            # add last point
            if (G_relabel.degree(tmp) != 2):
                new_path.append(tmp)
                new_key = frozenset([node, tmp])
                length_dict[new_key] = nx.shortest_path_length(G_relabel, node, tmp, weight='weight')
                path_dict[new_key] = new_path
                path_labels[new_key] = "centerline_{0}_{1}".format(node, tmp)
            else:
                print("there is a problem")

            if tmp in visited:
                pass
            else:
                visited.append(tmp)
            
    #print(length_dict)
    
    #print(G.nodes.keys())
    #print(G.nodes(data=True))
    
    #writeObjects(G_relabel,
    #    node_scalar_list=["MaximumInscribedSphereRadius"],
    #    node_vector_list = ["FrenetTangent"],
    #    edge_scalar_list=["weight"],
    #    fileout="test")
    

    writePolyLine(G_relabel, list(path_dict.values()),
        node_scalar_list = ["MaximumInscribedSphereRadius", "degree"],
        node_vector_list = ["FrenetTangent"],
        edge_scalar_dict = {"length": list(length_dict.values())},
        edge_label = ("labels", list(path_labels.values())),
        fileout = os.path.splitext(args.out_file)[0])
    
    
    #writer = vmtkscripts.vmtkSurfaceWriter()
    #writer.OutputFileName = args.out_file

    #if(args.clean_ctr):
        #cleaner2 = vtk.vtkCleanPolyData()
        #cleaner2.PointMergingOn()
        #cleaner.ConvertPolysToLinesOff()
        #cleaner2.SetInputConnection(pass_arrays.GetOutputPort())
        #cleaner2.Update()
        #writer.Input = cleaner2.GetOutput()
    #else:
        #writer.Input = pass_arrays.GetOutput()

    #writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-b", dest="boundary_file", required=True, help="boundary reference file", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="dome centerlines", metavar="FILE")
    parser.add_argument("--noclean", dest="clean_ctr", action='store_false', help="bool clean the poly data before and after")
    parser.add_argument("-o", dest="out_file", required=True, help="output filename for clipped centerlines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




