#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os
import sys


def addPolyLine(pts_ids):
    new_line = vtk.vtkPolyLine()
    new_line.GetPointIds().SetNumberOfIds(pts_ids.GetNumberOfIds())
    for  i in range(pts_ids.GetNumberOfIds()):
        new_line.GetPointIds().SetId(i, pts_ids.GetId(i))
    return new_line

# clip centerlines for accurate segment analysis

def Execute(args):
    print("clip centerlines")
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    
    print(args.clean_ctr)
    if(args.clean_ctr):
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
    for i in range(boundary_reference.GetNumberOfPoints()):
        pt = boundary_reference.GetPoint(i) #B
        pt_b = np.array(pt)
        #print(pt)
        #ctr_ptId = locator.FindClosestPoint(pt)
        id_list = vtk.vtkIdList()
        locator.FindClosestNPoints(2, pt, id_list)
        
        
        ctr1 = np.array(new_ctr.GetPoint(id_list.GetId(0))) # A
        ctr2 = np.array(new_ctr.GetPoint(id_list.GetId(1))) 
        #ctr3 = np.array(new_ctr.GetPoint(ctr_ptId + 1))
        
        
        n_br = np.array(boundary_reference.GetPointData().GetArray("BoundaryNormals").GetTuple(i))
        
        n_s_2 = np.dot(pt_b - ctr2, n_br)
        n_s_1 = np.dot(pt_b - ctr1, n_br)
        if ( n_s_1 < 0.0):
            proj_start = ctr2
            start_id = id_list.GetId(0)
        elif (n_s_2 < 0.0):
            proj_start = ctr1
            start_id = id_list.GetId(1)
        else:
            print("two closest points are on same side")
        #Get each vector normal to outlet
        n_ctr = np.array(new_ctr.GetPointData().GetArray("FrenetTangent").GetTuple(start_id))
        
        if ( np.dot(n_br, n_ctr) < 0.0):
            n_ctr = -1.0 * n_ctr

        #outlet centroid projected onto centerline based on FrenetTangent
        proj_vec = np.dot(n_br, pt_b - proj_start) * n_ctr
        proj_end = proj_vec + proj_start

        two_closest = vtk.vtkIdList()
        locator.FindClosestNPoints(2, proj_end, two_closest)
        
        vec_closest = np.array(new_ctr.GetPoints().GetPoint(two_closest.GetId(0))) - proj_end
        point_furthest = proj_end - vec_closest

        new_ctr.GetPoints().SetPoint(two_closest.GetId(1), tuple(point_furthest))
        #new_ctr.GetPoints().SetPoint(two_closest.GetId(0), proj_end_other)
        #new_ctr.GetPointData().GetArray("FrenetTangent").SetTuple(closest_to, tuple(n_br))
        
        cell_id_list = vtk.vtkIdList()
        new_ctr.GetPointCells(two_closest.GetId(0), cell_id_list)
        print("haller") 
        print(cell_id_list.GetNumberOfIds())
        ctr_cell = new_ctr.GetCell(cell_id_list.GetId(0))
        #print(ctr_cell)

        print(n_s)
        if (n_s < -np.finfo(float).eps):
            start = cell_id_match+1
            stop = ctr_cell.GetNumberOfPoints()
            step = int(1)
        else:
            start = cell_id_match-1 
            stop = int(-1)
            step = int(-1)

        new_poly_line = vtk.vtkPolyLine()

        for k in range(start, stop, step):
            old_pt_id = ctr_cell.GetPointIds().GetId(k)
            scalar.SetTuple(old_pt_id, [1])
            #new_pt_d = new_points.InsertNextPoint(new_ctr.GetPonts().GetPoint(old_pt_id)
            #new_poly_line.GetPointIds().InsertNextId(old_pt_id)


    new_ctr.GetPointData().AddArray(scalar)
    new_ctr.GetPointData().SetActiveScalars("clipper")
    
    pass_arrays = vtk.vtkPassArrays()
    pass_arrays.SetInputData(new_ctr)
    pass_arrays.UseFieldTypesOn()
    pass_arrays.AddArray(vtk.vtkDataObject.POINT, "clipper")
    pass_arrays.GetOutput().GetPointData().SetActiveScalars("clipper")
    pass_arrays.AddFieldType(vtk.vtkDataObject.POINT)
    pass_arrays.AddFieldType(vtk.vtkDataObject.CELL)
    pass_arrays.Update() 
    
    clip = vtk.vtkClipPolyData()
    clip.SetValue(0.5)
    clip.SetInputConnection(pass_arrays.GetOutputPort())
    clip.InsideOutOn()
    #clip.GetOutput().GetPointData().CopyScalarsOff()
    #clip.GetOutput().GetPointData().CopyVectorsOff()
    #clip.GetOutput().GetCellData().CopyScalarsOff()
    #clip.GetOutput().GetCellData().CopyVectorsOff()
    clip.Update()
    
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file

    if(args.clean_ctr):
        cleaner2 = vtk.vtkCleanPolyData()
        cleaner2.PointMergingOn()
        cleaner.ConvertPolysToLinesOff()
        cleaner2.SetInputConnection(clip.GetOutputPort())
        cleaner2.Update()
        writer.Input = cleaner2.GetOutput()
    else:
        writer.Input = clip.GetOutput()

    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-b", dest="boundary_file", required=True, help="boundary reference file", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="dome centerlines", metavar="FILE")
    parser.add_argument("--clean", dest="clean_ctr", action='store_true', help="bool clean the poly data before and after")
    parser.add_argument("-o", dest="out_file", required=True, help="output filename for clipped centerlines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




