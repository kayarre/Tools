#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os


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

    clip_ids = []
    for i in range(boundary_reference.GetNumberOfPoints()):
        pt = boundary_reference.GetPoint(i) #B
        pt_b = np.array(pt)
        #print(pt)
        ctr_ptId = locator.FindClosestPoint(pt)
        
        ctr1 = np.array(new_ctr.GetPoint(ctr_ptId)) # A
        ctr2 = np.array(new_ctr.GetPoint(ctr_ptId - 1)) 
        ctr3 = np.array(new_ctr.GetPoint(ctr_ptId + 1))
        
        vec_b = pt_b - ctr1
        
        #Get each vector normal to outlet
        n_ctr = np.array(new_ctr.GetPointData().GetArray("FrenetTangent").GetTuple(ctr_ptId))
        n_br = np.array(boundary_reference.GetPointData().GetArray("BoundaryNormals").GetTuple(i))
        
        n_s = np.dot(n_ctr, n_br)
        if ( n_s < 0.0):
            n_ctr *= -1.0
        #print(n_s)
        #np.dot(n_br, vec_b),
        #
        n_dot_b = np.dot(n_ctr, vec_b)

        #print(n_dot_b,  n_dot_b*n_ctr+ctr1)
        # outlet centroid projected onto centerline based on FrenetTangent
        projected_pt = n_dot_b*n_ctr+ctr1
        
        # compare previous and next point
        vec_nm1 = ctr2 - projected_pt
        vec_np1 = ctr3 - projected_pt
        
        n_dot_nm1 = np.dot(n_ctr, vec_nm1)
        n_dot_np1 = np.dot(n_ctr, vec_np1)
        #print(n_dot_nm1, n_dot_np1)
        # one is always positive and one is always negative
        assert n_dot_nm1/n_dot_np1 < 0.0
        
        #index info
        n_out_idx = 0
        if (n_dot_nm1 < 0.0 and n_dot_np1 > 0.0):
            n_out_idx =  1
            #print("nm1 is opposite direction of outlet", mvidx)
        else: 
            n_out_idx = -1
            #print("nm1 is same direction of outlet", mvidx)
        
        # is projection vector in opposite direction of normal?
        if( n_dot_b < 0.0): # move closest point
            mvidx = ctr_ptId
        else: #move  next outside point
            mvidx = ctr_ptId +  n_out_idx

        #move point
        #print(ctr_ptId, mvidx, n_out_idx)
        new_ctr.GetPoints().SetPoint(mvidx, projected_pt)
        cell_ids_list = vtk.vtkIdList()
        new_ctr.GetPointCells (mvidx, cell_ids_list)
        if(cell_ids_list.GetNumberOfIds() > 1):
            print("something has gone wrong, this is a bifurcation point")
        else:
            cell = new_ctr.GetCell(cell_ids_list.GetId(0))
            for i in range(cell.GetNumberOfPoints()):
                if ( cell.GetPointIds().GetId(i) == mvidx):
                    # cell_id, point_id, numberOfPoints
                    clip_ids.append((cell_ids_list.GetId(0), i, cell.GetNumberOfPoints(), n_out_idx))
            
    print(clip_ids)
    outputLines = vtk.vtkCellArray()
    output = vtk.vtkPolyData()
    
    lengthArray = vtk.vtkDoubleArray()
    lengthArray.SetName("length")
    lengthArray.SetNumberOfComponents(1)
    
    cell_Ids = vtk.vtkIdList()
    new_points = vtk.vtkPoints()
    points_list = []
    for i in range(new_ctr.GetNumberOfCells()):
        cell = new_ctr.GetCell(i)
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
             continue
        edge = []
        cell_ends = [0, cell.GetNumberOfPoints()-1]
        edge.append(cell.GetPointIds().GetId(cell_ends[0]))
        edge.append(cell.GetPointIds().GetId(cell_ends[1]))
        start_pt = []
        for e, idx in zip(edge, cell_ends):
            new_ctr.GetPointCells(e, cell_Ids)
            start_pt.append((cell_Ids.GetNumberOfIds(), idx))
        
        
        cell_bounds = [ t[1] for t in clip_ids if t[0] == i] # get cell point bounds
        cell_max_idx = [ i[1] for i in start_pt if i[0] == 1]
        
        
        start_pt.sort(key=lambda tup: tup[0])
        print(start_pt, cell_bounds, cell_max_idx)
        
        if (len(cell_bounds) == 2):
            # starts and ends outside
            bnd = sorted(cell_bounds)
        elif (len(cell_bounds) == 0):
            #starts and ends inside
            bnd = cell_ends
        elif (len(cell_bounds) == 1):
            #one point inside and one outside
            bnds = sorted([ cell_ends[0], cell_bounds[0], cell_ends[1]])
            if(start_pt[0][0] == 1 and start_pt[0][1] != 0):
                #print("start out")
                bnd = bnds[:-1]
            elif(start_pt[0][0] == 1 and start_pt[0][1] == 0):
                bnd = bnds[1:]
            else:
                print("whoops")
            
        #print(bnd)
        #if (args.clean_ctr):
        cell_branch_pt = []
        #cell_Ids = vtk.vtkIdList()
        for j in range(cell.GetNumberOfPoints()):
            new_ctr.GetPointCells(cell.GetPointIds().GetId(j), cell_Ids)
            if (cell_Ids.GetNumberOfIds() > 1 and j not in cell_ends ):
                cell_branch_pt.append(j)
                #print(cell.GetPointIds().GetId(j))
        
        for j in cell_branch_pt:
            bnd.append(j)
        bnd.sort()
        #print("yeay")
        #print(bnd)
        
        for j in range(len(bnd)-1):
            start_ = bnd[j]
            end_ = bnd[j+1]
            pts_ids = vtk.vtkIdList()
            length = 0.0
            prevPoint = cell.GetPoints().GetPoint(0)
            for k in range(start_, end_+1):
                pt_id = cell.GetPointIds().GetId(k)
                pts_ids.InsertNextId(pt_id)
                point = cell.GetPoints().GetPoint(k)
                length += vtk.vtkMath.Distance2BetweenPoints(prevPoint,point)**0.5
                if (pt_id not in points_list):
                    points_list.append(pt_id)
                prevPoint = point
            lengthArray.InsertNextTuple([length])
            new_polyline = addPolyLine(pts_ids)
            outputLines.InsertNextCell(new_polyline)
            #if(k == j):
            
            

        
    #boundary_reference


    
    output.SetPoints(new_ctr.GetPoints())
    output.SetLines(outputLines)
    output.GetCellData().AddArray(lengthArray)
    for i in range(new_ctr.GetPointData().GetNumberOfArrays()):
        output.GetPointData().AddArray(new_ctr.GetPointData().GetArray(i))
    
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file

    if(args.clean_ctr):
        cleaner2 = vtk.vtkCleanPolyData()
        cleaner2.PointMergingOn()
        cleaner.ConvertPolysToLinesOff()
        cleaner2.SetInputData(output)
        cleaner2.Update()
        writer.Input = cleaner2.GetOutput()
    else:
        writer.Input = output

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




