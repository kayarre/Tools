#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
from vmtk import vtkvmtk
import argparse
import itertools
import os
import copy


# map the average crossectional area to the boundary reference info
def Execute(args):
    print("evaluate centerlines")
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    centerlines = reader_ctr.Surface
    
    locator_cell = vtk.vtkPointLocator()
    locator_cell.SetDataSet(centerlines)
    locator_cell.BuildLocator()
    
    reader_br = vmtkscripts.vmtkSurfaceReader()
    reader_br.InputFileName = args.boundary_reference
    reader_br.Execute()
    
    boundary_reference = reader_br.Surface

    terminal_pts = boundary_reference.GetNumberOfPoints()
    
    avg_area = vtk.vtkDoubleArray()
    avg_area.SetName("avg_crosssection")
    avg_area.SetNumberOfComponents(1)
    avg_area.SetNumberOfTuples(terminal_pts)
    
    #for j in range(centerlines.GetCellData().GetNumberOfArrays()):
        #new_array = centerlines.GetCellData().GetArray(j)
        #array_name = centerlines.GetCellData().GetArrayName(j)
        ##new_array.SetNumberOfTuples(terminal_pts)
        #boundary_reference.GetPointData().AddArray(new_array)
        ##boundary_reference.GetPointData().GetArray(array_name).SetNumberOfTuples(terminal_pts)

    #writer = vmtkscripts.vmtkSurfaceWriter()
    #writer.OutputFileName = "test_out.vtp"
    #writer.Input = boundary_reference
    #writer.Execute()

    for i in range(terminal_pts):
        pt = boundary_reference.GetPoint(i) #B
        ctr_ptId = locator_cell.FindClosestPoint(pt)
        
        cell_ids_list = vtk.vtkIdList()
        centerlines.GetPointCells(ctr_ptId, cell_ids_list)
        
        if(cell_ids_list.GetNumberOfIds() > 1):
            print("something has gone wrong, this is a bifurcation point")
        else:
            new_tuple = centerlines.GetCellData().GetArray("avg_crosssection").GetTuple(cell_ids_list.GetId(0))
            avg_area.SetTuple(i, new_tuple)
            #for j in range(centerlines.GetCellData().GetNumberOfArrays()):
                #print(cell_ids_list.GetId(0))
                #new_tuple = centerlines.GetCellData().GetArray(j).GetTuple(cell_ids_list.GetId(0))
                #array_name = centerlines.GetCellData().GetArrayName(j)
                ##print(array_name)
                #boundary_reference.GetPointData().GetArray(array_name).SetTuple(i, new_tuple)


    boundary_reference.GetPointData().AddArray(avg_area)
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file
    writer.Input = boundary_reference
    writer.Execute()
    
            



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='map outlet average crossectional area to boundary reference information')
    parser.add_argument("-b", dest="boundary_reference", required=True, help="input bounadry reference ", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="centerlines", metavar="FILE")
    parser.add_argument("-o", dest="out_file", required=True, help="boundary reference information", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




