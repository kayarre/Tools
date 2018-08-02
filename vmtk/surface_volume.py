#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse

#estimate the volume of closed surface 
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

    print(" surface area: {0}".format(massProps.GetSurfaceArea()))
    print(" volume: {0}".format(massProps.GetVolume()))


    vol = massProps.GetVolume()
    area = massProps.GetSurfaceArea()
    
    centerofmass = vtk.vtkCenterOfMass()
    centerofmass.SetInputConnection(triangleFilter.GetOutputPort())
    centerofmass.Update()
    com = centerofmass.GetCenter()
    
    print(" center of mass : {0}".format(com))
    

    target_area = 3.0**0.5/4.0*args.edge_length**2.0

    print ("target number of cells: {0}".format(area / target_area)) # A_total = N*(area_equilateral_triangle)

    print ("target number of points: {0}".format(area / target_area / 2.0)) #in the limit of equilateral triangles ratio ,Ncells/Npoints = 2
    
    
    #NCells = triangleFilter.GetOutput().GetNumberOfCells()
    #cross = [0.0, 0.0, 0.0]
    #vol = 0.0
    #for i in range(NCells):
         #cell = triangleFilter.GetOutput().GetCell(i)
         #if(cell.GetNumberOfPoints() > 3):
             #print("houston we have a problem")
         #pt1 = cell.GetPoints().GetPoint(0)
         #pt2 = cell.GetPoints().GetPoint(1)
         #pt3 = cell.GetPoints().GetPoint(2)
         
         #vtk.vtkMath.Cross(pt1, pt2, cross)
         #vol += 1.0/6.0*vtk.vtkMath.Dot(cross, pt3)
        
    print ("volume of closed mesh {0}".format(vol))
    
    print ("volume of closed mesh {0} cm^3".format(vol*10**6))
    # sphere packing for random packing 0.64
    # http://mathworld.wolfram.com/SpherePacking.html
    print( "sphere packing : {0}".format(vol*.64))
    sphere_vol = 4.0/3.0*np.pi*args.radius**3
    print("number of particles random: {0}".format(vol*.64/sphere_vol))
    print("number of particles ideal: {0}".format(vol*np.pi/(3.0*2.0**(0.5)*sphere_vol)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument('--edge', dest="edge_length",  type=float, help='edge length float', default=0.08)
    parser.add_argument('--radius', dest="radius",  type=float, help='radius sphere', default=0.08)
    args = parser.parse_args()
    #print(args)
    Execute(args)
    
