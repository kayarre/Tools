
#!/usr/bin/env python

import vtk
import numpy as np

from vmtk import vmtkscripts
from scipy.stats import skew
import argparse
import copy

# evaluate the inlet of the each case 
def Execute(args):
    print("get average along line probes")
    cell_type = "point"
    
    if(cell_type == "cell"):
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        vtk_data_type = vtk.vtkDataObject.CELL
    else:
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        vtk_data_type = vtk.vtkDataObject.POINT
    
    reader = vmtkscripts.vmtkMeshReader()
    reader.InputFileName = args.mesh_file
    reader.Execute()
    mesh = reader.Mesh
    
    pass_filt = vtk.vtkPassArrays()
    pass_filt.SetInputData(mesh)
    pass_filt.AddArray(vtk_data_type, "velocity")
    pass_filt.Update()
    
    surf = vmtkscripts.vmtkMeshToSurface()
    surf.Mesh = pass_filt.GetOutput()
    surf.Execute()
    
    normals = vmtkscripts.vmtkSurfaceNormals()
    normals.Surface = surf.Surface
    #accept defaults
    normals.Execute()
    
    calc1 = vtk.vtkArrayCalculator()
    calc1.SetFunction("velocity_X*-Normals_X+velocity_Y*-Normals_Y+velocity_Z*-Normals_Z")
    calc1.AddScalarVariable("velocity_X", "velocity_X",0)
    calc1.AddScalarVariable("velocity_Y", "velocity_Y",0)
    calc1.AddScalarVariable("velocity_Z", "velocity_Z",0)
    calc1.SetResultArrayName("vdotn")
    calc1.SetInputData(normals.Surface)
    if(cell_type == "cell"):
        calc1.SetAttributeModeToUseCellData()

    else:
        calc1.SetAttributeModeToUsePointData()
    calc1.SetResultArrayType(vtk.VTK_DOUBLE)
    
    integrate_attrs = vtk.vtkIntegrateAttributes()
    integrate_attrs.SetInputConnection(calc1.GetOutputPort())
    
    integrate_attrs.UpdateData()
    
    area = integrate_attrs.GetCellData().GetArray(0).GetValue(0)

    D = 2.0*np.sqrt(area/np.pi)
    
    calc2= vtk.vtkArrayCalculator()
    calc2.SetFunction("vdotn*10**6*60")
    calc2.AddScalarVariable("vdotn", "vdotn",0)
    calc2.SetResultArrayName("Q")
    calc2.SetInputConnection(integrate_attrs.GetOutputPort())
    if(cell_type == "cell"):
        calc2.SetAttributeModeToUseCellData()
    else:
        calc2.SetAttributeModeToUsePointData()
    calc2.SetResultArrayType(vtk.VTK_DOUBLE)
    calc2.UpdateData()
    
    calc3= vtk.vtkArrayCalculator()
    calc3.SetFunction("vdotn/{0}*1050.0/0.0035*{1}".format(area, D))
    calc3.AddScalarVariable("vdotn", "vdotn",0)
    calc3.SetResultArrayName("Re")
    calc3.SetInputConnection(integrate_attrs.GetOutputPort())
    if(cell_type == "cell"):
        calc3.SetAttributeModeToUseCellData()
    else:
        calc3.SetAttributeModeToUsePointData()
    calc3.SetResultArrayType(vtk.VTK_DOUBLE)
    
    calc3.UpdateData()
    
    over_time = vtk.vtkExtractDataArraysOverTime()
    over_time.SetInputConnection(calc3.GetOutputPort())
    
    if(cell_type == "cell"):
        over_time.SetFieldAssociation(vtk_data_type)
    else:
        over_time.SetFieldAssociation(vtk_data_type)
    over_time.UpdateData()
    
    writer = vtk.vtkDelimitedTextWriter()
    writer.SetInputConnection(over_time.GetOutputPort())
    writer.SetFileName(args.file_out)
    writer.Write()
    
    #writer = vmtkscripts.vmtkSurfaceWriter()
    #writer.OutputFileName = args.file_out
    #writer.Input = Surface
    #writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='average probed information along lines')
    parser.add_argument("-m", dest="mesh_file", required=True, help="input mesh of inlet", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output surface file with averages probed lines", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)
