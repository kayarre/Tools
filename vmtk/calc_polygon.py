
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
from vmtk import vtkvmtk
import argparse
import copy

# creates lines normal to surface for evaluation in the probe image with surface
def warp_surface(args):
    print("warp the surface ")
    
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = args.surface
    reader.Execute()
    Surface = reader.Surface    

    boundaries = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundaries.SetInputData(Surface)
    boundaries.Update()



    boundaryReferenceSystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    boundaryReferenceSystems.SetInputData(Surface)
    boundaryReferenceSystems.SetBoundaryRadiusArrayName('BoundaryRadius')
    boundaryReferenceSystems.SetBoundaryNormalsArrayName('BoundaryNormals')
    boundaryReferenceSystems.SetPoint1ArrayName('Point1')
    boundaryReferenceSystems.SetPoint2ArrayName('Point2')
    boundaryReferenceSystems.Update()

    ReferenceSystems = boundaryReferenceSystems.GetOutput()



    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = warp.GetOutput()
    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    #parser.add_argument("-o", dest="file_out", required=True, help="output surface file", metavar="FILE")
    #parser.add_argument("-s", '--scale', dest="fuzz_scale",  type=float, help='how much to fuzz surface ', default=0.08)
    args = parser.parse_args()
    #print(args)
    warp_surface(args)

