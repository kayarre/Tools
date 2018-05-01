
#!/usr/bin/env python

import vtk
import numpy as np

from vmtk import vmtkscripts
from scipy.stats import skew
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
    ln_std = np.std(lines, axis=1, ddof=1)
    ln_skew = skew(lines, axis=1, bias=False)

    avg_min = ln_avg.min()
    ln_avg_norm = (ln_avg + avg_min) / (ln_avg.max() + avg_min)
    
    # get weighted average
    x = np.linspace(-args.slice_thickness, args.slice_thickness, lines.shape[1])
    std = args.slice_thickness/2.0
    mean = 0.0
    dist = 1.0/np.sqrt(2.0*np.pi*std**2)*np.exp(-(x-mean)**2/(2.0*std**2))
    ln_avg_weight = np.average(lines, axis=1, weights = dist)
    
    
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
    array.SetName("rawImageSamples")
    array.SetNumberOfTuples(n_cells)

    avg = vtk.vtkDoubleArray()
    avg.SetNumberOfComponents(1)
    avg.SetName("avgSample")
    avg.SetNumberOfTuples(n_cells)

    avg_norm = vtk.vtkDoubleArray()
    avg_norm.SetNumberOfComponents(1)
    avg_norm.SetName("normalized")
    avg_norm.SetNumberOfTuples(n_cells)
    
    stddev = vtk.vtkDoubleArray()
    stddev.SetNumberOfComponents(1)
    stddev.SetName("stddev")
    stddev.SetNumberOfTuples(n_cells)
    
    skewness = vtk.vtkDoubleArray()
    skewness.SetNumberOfComponents(1)
    skewness.SetName("skewness")
    skewness.SetNumberOfTuples(n_cells)
    
    weighted_avg = vtk.vtkDoubleArray()
    weighted_avg.SetNumberOfComponents(1)
    weighted_avg.SetName("weighted_average")
    weighted_avg.SetNumberOfTuples(n_cells)
    

    for i in range(n_cells):
        surf_id = pointLocator.FindClosestPoint(pts[i])
        #print(ln_avg.shape)
        avg.SetValue(surf_id, ln_avg[i])
        array.SetTuple(surf_id, list(lines[i,:]))
        avg_norm.SetValue(surf_id, ln_avg_norm[i])
        stddev.SetValue(surf_id, ln_std[i])
        skewness.SetValue(surf_id, ln_skew[i])
        weighted_avg.SetValue(surf_id, ln_avg_weight[i])
    

    Surface.GetPointData().AddArray(avg)
    Surface.GetPointData().AddArray(array)
    Surface.GetPointData().AddArray(avg_norm)
    Surface.GetPointData().AddArray(stddev)
    Surface.GetPointData().AddArray(skewness)
    Surface.GetPointData().AddArray(weighted_avg)
    

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = Surface
    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='average probed information along lines')
    parser.add_argument("-i", dest="surface_file", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-l", dest="lines_file", required=True, help="input file with probed lines", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output file with averages probed lines", metavar="FILE")
    parser.add_argument("-t", '--thickness', dest="slice_thickness",  type=float, help='half thickness of lines ', default=0.5625)
    args = parser.parse_args()
    #print(args)
    Execute(args)
