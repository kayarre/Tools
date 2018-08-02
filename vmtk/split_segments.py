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
    print("split centerlines")
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    
    print(args.clean_ctr)
    if(args.clean_ctr):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.PointMergingOn()
        cleaner.ConvertLinesToPointsOff()
        #cleaner.ToleranceIsOn()
        cleaner.SetTolerance(args.clean_tol) # assumes mm
        #print(cleaner.GetTolerance())
        cleaner.SetInputData(reader_ctr.Surface)
        cleaner.Update()
        centerlines = cleaner.GetOutput()
    else:
        centerlines = reader_ctr.Surface
    
    centerlines.BuildLinks()
    centerlines.BuildCells()
    
    #if(args.clean_ctr):
        #for i in range(centerlines.GetNumberOfCells()):
            #if centerlines.GetCell(i).GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
                #centerlines.DeleteCell(i)
            
        #centerlines.RemoveDeletedCells()

    
    cell_Ids = vtk.vtkIdList()
    outputLines = vtk.vtkCellArray()
    output = vtk.vtkPolyData()
    
    lengthArray = vtk.vtkDoubleArray()
    lengthArray.SetName("length")
    lengthArray.SetNumberOfComponents(1)
    
    pts_ids = vtk.vtkIdList()
    
    for i in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(i)
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
             continue

        n_cell_pts = cell.GetNumberOfPoints()
        prevPoint = centerlines.GetPoint(cell.GetPointIds().GetId(0))
        length = 0.0
        start_pt_idx = 0 
        for j in range(n_cell_pts):
            centerlines.GetPointCells(cell.GetPointIds().GetId(j), cell_Ids)
            n_pt_neighbors = cell_Ids.GetNumberOfIds()

            pt_id = cell.GetPointIds().GetId(j)
            pts_ids.InsertNextId(pt_id)
            point = centerlines.GetPoint(cell.GetPointIds().GetId(j))
            length += vtk.vtkMath.Distance2BetweenPoints(prevPoint,point)**0.5
            prevPoint = point
            if((j > start_pt_idx  and n_pt_neighbors > 1) or  (j == n_cell_pts-1)):
                #close

                new_polyline = addPolyLine(pts_ids)
                # weird issue with duplicate points if they are not removed
                if(length > 0.0):
                    outputLines.InsertNextCell(new_polyline)
                    lengthArray.InsertNextTuple([length])
                start_pt_idx = j
                if(n_pt_neighbors > 1):
                    pts_ids.Reset()
                    pts_ids.InsertNextId(pt_id)
                    length = 0.0
        pts_ids.Reset()

    output.SetPoints(centerlines.GetPoints())
    output.SetLines(outputLines)
    output.GetCellData().AddArray(lengthArray)
    for i in range(centerlines.GetPointData().GetNumberOfArrays()):
        output.GetPointData().AddArray(centerlines.GetPointData().GetArray(i))
    
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file

    if(args.clean_ctr):
        cleaner2 = vtk.vtkCleanPolyData()
        cleaner2.PointMergingOn()
        cleaner2.ConvertLinesToPointsOn()
        cleaner2.SetInputData(output)
        cleaner2.Update()
        writer.Input = cleaner2.GetOutput()
    else:
        writer.Input = output

    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-c", dest="centerlines", required=True, help="centerlines", metavar="FILE")
    parser.add_argument("--clean", dest="clean_ctr", action='store_true', help=" clean centerlines after")
    parser.add_argument("--tolerance", dest="clean_tol",  type=float, help='absolute tolerance for point merging', default=0.0003)
    parser.add_argument("-o", dest="out_file", required=True, help="output filename for split centerlines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




