
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import copy

# evaluate the probed surface to get average values mapped to original segmentation
def Execute(args):
    print("get average along line probes")
    
    reader_lines = vmtkscripts.vmtkSurfaceReader()
    reader_lines.InputFileName = args.lines_file
    reader_lines.Execute()
    lines_surface = reader_lines.Surface
    
    n_cells = lines_surface.GetNumberOfCells()
    n_pts = lines_surface.GetCell(0).GetNumberOfPoints()

    lines = np.empty((n_cells, n_pts))
    pts = np.empty((n_cells, 3))
    da = lines_surface.GetPointData().GetArray("NRRDImage")
    
    for i in range(n_cells):
        cellids = lines_surface.GetCell(i).GetPointIds()
        #n_pts = cell.GetNumberOfPoints()
        for j in range(n_pts):
            if(j == n_pts // 2):
                pts[i,:] = np.array(lines_surface.GetPoint(cellids.GetId(j)))
            
            lines[i, j] = lines_surface.GetPointData().GetArray("NRRDImage").GetTuple(cellids.GetId(j))[0]
        

    ln_avg = np.average(lines, axis=1)

    ln_avg_norm = ln_avg / ln_avg.max()
    
    
    reader_surface = vmtkscripts.vmtkSurfaceReader()
    reader_surface.InputFileName = args.surface_file
    reader_surface.Execute()
    Surface = reader_surface.Surface  
    
    #Create the tree
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(Surface)
    pointLocator.BuildLocator()

    array = vtk.vtkDoubleArray()
    array.SetNumberOfComponents(n_pts)
    array.SetName("raw")
    array.SetNumberOfTuples(n_cells)

    avg = vtk.vtkDoubleArray()
    avg.SetNumberOfComponents(1)
    avg.SetName("avg")
    avg.SetNumberOfTuples(n_cells)

    avg_norm = vtk.vtkDoubleArray()
    avg_norm.SetNumberOfComponents(1)
    avg_norm.SetName("normalized")
    avg_norm.SetNumberOfTuples(n_cells)

    for i in range(n_cells):
        surf_id = pointLocator.FindClosestPoint(pts[i])
        #print(ln_avg.shape)
        avg.SetValue(surf_id, ln_avg[i])
        array.SetTuple(surf_id, list(lines[i,:]))
        avg_norm.SetValue(surf_id, ln_avg_norm[i])
    

    Surface.GetPointData().AddArray(avg)
    Surface.GetPointData().AddArray(array)
    Surface.GetPointData().AddArray(avg_norm)
    

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = Surface
    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface_file", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-l", dest="lines_file", required=True, help="input file with probed lines", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output file with averages probed lines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)
