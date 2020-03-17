#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os
import sys


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
    
    attribute = vtk.vtkDoubleArray()
    attribute.SetName("CenterlineSectionAreaAvg")
    attribute.SetNumberOfComponents(1)
    attribute.SetNumberOfTuples(centerlines.GetNumberOfCells())

    attribute2 = vtk.vtkDoubleArray()
    attribute2.SetName("MaximumInscribedSphereRadiusAvgArea")
    attribute2.SetNumberOfComponents(1)
    attribute2.SetNumberOfTuples(centerlines.GetNumberOfCells())

    for i in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(i)
        cell_avg = 0.0
        cell_avg2 = 0.0
        cell_n_pts = cell.GetNumberOfPoints()
        for j in range(cell_n_pts):
            pt_id = cell.GetPointIds().GetId(j)
            area = centerlines.GetPointData().GetArray("CenterlineSectionArea").GetTuple(pt_id)[0]
            cell_avg += area
            rad = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius").GetTuple(pt_id)[0]
            cell_avg2 += np.pi*rad**2.0
            #print(area)
        avg = cell_avg / float(cell_n_pts)
        attribute.SetTuple(i, (avg,))
        avg2 = cell_avg2 / float(cell_n_pts)
        attribute2.SetTuple(i, (avg2,))


    centerlines.GetCellData().AddArray(attribute)
    centerlines.GetCellData().AddArray(attribute2)

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file

    if(args.clean_ctr):
        cleaner2 = vtk.vtkCleanPolyData()
        cleaner2.PointMergingOn()
        cleaner.ConvertPolysToLinesOff()
        cleaner2.SetInputData(centerlines)
        cleaner2.Update()
        writer.Input = cleaner2.GetOutput()
    else:
        writer.Input = centerlines

    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate area property from vesesl')
    parser.add_argument("-c", dest="centerlines", required=True, help="surface centerlines", metavar="FILE")
    parser.add_argument("--noclean", dest="clean_ctr", action='store_false', help="bool clean the poly data before and after")
    parser.add_argument("-o", dest="out_file", required=True, help="output filename for averaged centerlines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)
