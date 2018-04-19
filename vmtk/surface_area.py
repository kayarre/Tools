#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse

#extimate the points required for an average triangle size
def Execute(args):

    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = args.surface
    reader.Execute()
    Surface = reader.Surface
    # estimates surface area to estimate the point density

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(Surface)
    cleaner.Update()

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputConnection(cleaner.GetOutputPort())
    triangleFilter.Update()

    massProps = vtk.vtkMassProperties()
    massProps.SetInputConnection(triangleFilter.GetOutputPort())
    massProps.Update()

    print(massProps.GetSurfaceArea())

    area = massProps.GetSurfaceArea()

    target_area = 3.0**0.5/4.0*args.edge_length**2.0

    print ("target number of cells: {0}".format(area / target_area)) # A_total = N*(area_equilateral_triangle)

    print ("target number of points: {0}".format(area / target_area / 2.0)) #in the limit of equilateral triangles ratio ,Ncells/Npoints = 2


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument('--edge', dest="edge_length",  type=float, help='edge length float', default=0.08)
    args = parser.parse_args()
    #print(args)
    Execute(args)
    
