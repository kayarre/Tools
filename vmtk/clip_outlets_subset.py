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
    
    reader_surf = vmtkscripts.vmtkSurfaceReader()
    reader_surf.InputFileName = args.surface
    reader_surf.Execute()
    surface = reader_surf.Surface
    
    locator_cell = vtk.vtkPointLocator()
    locator_cell.SetDataSet(surface)
    locator_cell.BuildLocator()
    
    reader_br = vmtkscripts.vmtkSurfaceReader()
    reader_br.InputFileName = args.boundary_reference
    reader_br.Execute()
    
    boundary_reference = reader_br.Surface

    terminal_pts = boundary_reference.GetNumberOfPoints()
    
    main_body_id = 695041
    

    for i in range(terminal_pts):
        pt = boundary_reference.GetPoint(i) #B
        ctr_ptId = locator_cell.FindClosestPoint(pt)
        
        vmtkscripts.vmtkSurfaceSlipperCenterline2


    boundary_reference.GetPointData().AddArray(avg_area)
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file
    writer.Input = boundary_reference
    writer.Execute()
    
            



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='map outlet average crossectional area to boundary reference information')
    parser.add_argument("-b", dest="boundary_reference", required=True, help="input bounadry reference ", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="input centerlines", metavar="FILE")
    parser.add_argument("-s", dest="surface", required=True, help="surface to clip", metavar="FILE")
    parser.add_argument("-o", dest="out_file", required=True, help="clipped surface  reference information", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




