
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import copy

# creates lines normal to surface for evaluation in the probe image with surface
def warp_surface(args):
    print("warp the surface ")
    
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
        print("already have")
    else:
        get_normals = vtk.vtkPolyDataNormals()
        get_normals.SetInputData(Surface)
        get_normals.SetFeatureAngle(30.0) # default
        get_normals.SetSplitting(True)
        get_normals.Update()
        get_normals.GetOutput().GetPointData().SetActiveVectors("Normals")
        normals = get_normals.GetOutput()
        print("normals generated")

    random = vtk.vtkRandomAttributeGenerator()
    random.SetInputData(normals)
    random.SetDataTypeToDouble()
    random.GeneratePointScalarsOn	()
    random.SetComponentRange(-0.5, 0.5)
    random.Update()
    
    #n = random.GetOutput().GetPointData().GetNumberOfArrays()
    #for i in range(n):
        #print(random.GetOutput().GetPointData().GetArrayName(i))
    
    calc = vtk.vtkArrayCalculator()
    calc.SetInputConnection(random.GetOutputPort())
    calc.AddScalarArrayName("RandomPointScalars", 0)
    calc.AddVectorArrayName("Normals", 0, 1, 2)
    calc.SetFunction("Normals * RandomPointScalars")
    calc.SetResultArrayName("RandomLengthNormalVectors")
    calc.Update()
    
    
    warp = vtk.vtkWarpVector()
    warp.SetInputConnection(calc.GetOutputPort())
    warp.SetInputArrayToProcess(0, 0, 0,
                                vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                "RandomLengthNormalVectors");
    warp.SetScaleFactor(args.fuzz_scale)
    warp.Update()


    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = warp.GetOutput()
    writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output surface file", metavar="FILE")
    parser.add_argument("-s", '--scale', dest="fuzz_scale",  type=float, help='how much to fuzz surface ', default=0.08)
    args = parser.parse_args()
    #print(args)
    warp_surface(args)

