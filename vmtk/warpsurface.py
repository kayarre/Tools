
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import copy

# creates lines normal to surface for evaluation in the probe image with surface
def Execute(args):
    print("get lines along normal of surface")
    
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = args.surface
    reader.Execute()
    Surface = reader.Surface    

    narrays = Surface.GetPointData().GetNumberOfArrays()
    has_normals = False
    for i in range(narrays):
        if (  Surface.GetPointData().GetArrayName(i) == "Normals"):
            has_normals = True
            break

    if(has_normals):
        normals = Surface
    else:
        get_normals = vtk.vtkPolyDataNormals()
        get_normals.SetInputData(Surface)
        get_normals.SetFeatureAngle(30.0) # default
        get_normals.SetSplitting(True)
        get_normals.Update()
        get_normals.GetOutput().GetPointData().SetActiveVectors("Normals")
        normals = get_normals.GetOutput()

    dx=args.slice_thickness
    n_pts = normals.GetNumberOfPoints()
    # Create a vtkCellArray container and store the lines in it
    lines = vtk.vtkCellArray()


    #Create a vtkPoints container and store the points in it
    pts = vtk.vtkPoints()

    count = 0
    sublayer = args.sublayers # no visual difference between 2 and 3 (5 layers vs 7 layers)
    subdx = dx/sublayer
    pts_tot = 2*sublayer + 1
    for i in range(n_pts):
        pt = np.array(normals.GetPoint(i))
        vec = np.array(normals.GetPointData().GetArray("Normals").GetTuple(i))
        pt1 = pt + dx*vec 
        
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(pts_tot)

        for j in range(pts_tot):
            pt2 = pt1 - j*subdx*vec
            pts.InsertNextPoint(pt2)
            polyLine.GetPointIds().SetId(j, count)
        
            count +=1
        lines.InsertNextCell(polyLine)
        
        
    linesPolyData = vtk.vtkPolyData()
    # Add the points to the polydata container
    linesPolyData.SetPoints(pts)
    # Add the lines to the polydata container
    linesPolyData.SetLines(lines)

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = linesPolyData
    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output surface file", metavar="FILE")
    parser.add_argument("-t", '--thickness', dest="slice_thickness",  type=float, help='half thickness of lines ', default=0.5625)
    parser.add_argument("-l", '--sublayers', dest="sublayers",  type=int, help='number of sublayers for lines', default=2)
    args = parser.parse_args()
    #print(args)
    Execute(args)


    #surface_out = vtk.vtkPolyData()
    #surface_out.DeepCopy(normals.GetOutput())
    ##surface_out.GetPointData().SetActiveVectors("Normals")

    #surface_in = vtk.vtkPolyData()
    #surface_in.DeepCopy(normals.GetOutput())
    ##surface_in.GetPointData().SetActiveVectors("Normals")


    #warp1 = vtk.vtkWarpVector()
    #warp1.SetScaleFactor(0.5625)
    #warp1.SetInputData(surface_out)
    #warp1.Update()

    #out_largest = vtk.vtkPolyDataConnectivityFilter()
    #out_largest.SetInputConnection(warp1.SetOutputPort()
    #out_largest.SetExtractionModeToLargestRegion()


    #warp2 = vtk.vtkWarpVector()
    #warp2.SetScaleFactor(-0.5625)
    #warp2.SetInputData(surface_in)
    #warp2.Update()


    #in_largest = vtk.vtkPolyDataConnectivityFilter()
    #in_largest.SetInputConnection(warp2.SetOutputPort()
    #in_largest.SetExtractionModeToLargestRegion()
