
import vtk
import numpy as np
import os


def post_proc_cfd_diff(parameter_list):
    
    dir_path = parameter_list[0]
    vtu_input = parameter_list[1]
    cell_type = parameter_list[1]
    vtu_output_1 = parameter_list[3]
    vtu_output_2 = parameter_list[4]
    N_peak = parameter_list[5]
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.join(dir_path, vtu_input))
    reader.Update()
    N = reader.GetNumberOfTimeSteps()

    print(N)
    
    if(cell_type == "cell"):
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        vtk_data_type = vtk.vtkDataObject.CELL
    else:
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        vtk_data_type = vtk.vtkDataObject.POINT
    
    pass_arr = vtk.vtkPassArrays()
    pass_arr.SetInputConnection(reader.GetOutputPort())
    pass_arr.AddArray(vtk_data_type, "absolute_pressure")
    pass_arr.AddArray(vtk_data_type, "x_wall_shear")
    pass_arr.AddArray(vtk_data_type, "y_wall_shear")
    pass_arr.AddArray(vtk_data_type, "z_wall_shear")

    calc1 = vtk.vtkArrayCalculator()
    calc1.SetFunction("sqrt(x_wall_shear^2+y_wall_shear^2+z_wall_shear^2)")
    calc1.AddScalarVariable("x_wall_shear", "x_wall_shear",0)
    calc1.AddScalarVariable("y_wall_shear", "y_wall_shear",0)
    calc1.AddScalarVariable("z_wall_shear", "z_wall_shear",0)
    calc1.SetResultArrayName("WSS")
    calc1.SetInputConnection(pass_arr.GetOutputPort())
    if(cell_type == "cell"):
        calc1.SetAttributeModeToUseCellData()
    else:
        calc1.SetAttributeModeToUsePointData()
    calc1.SetResultArrayType(vtk.VTK_DOUBLE)

    x_WSS_grad = vtk.vtkGradientFilter()
    x_WSS_grad.SetInputConnection(calc1.GetOutputPort())
    x_WSS_grad.ComputeGradientOn()
    x_WSS_grad.FasterApproximationOff()
    x_WSS_grad.ComputeDivergenceOff()
    x_WSS_grad.ComputeVorticityOff()
    x_WSS_grad.ComputeQCriterionOff()
    x_WSS_grad.SetResultArrayName("x_WSS_grad")
    x_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk_process, "x_wall_shear")

    y_WSS_grad = vtk.vtkGradientFilter()
    y_WSS_grad.SetInputConnection(x_WSS_grad.GetOutputPort())
    y_WSS_grad.ComputeGradientOn()
    y_WSS_grad.FasterApproximationOff()
    y_WSS_grad.ComputeDivergenceOff()
    y_WSS_grad.ComputeVorticityOff()
    y_WSS_grad.ComputeQCriterionOff()
    y_WSS_grad.SetResultArrayName("y_WSS_grad")
    y_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk_process, "y_wall_shear")

    z_WSS_grad = vtk.vtkGradientFilter()
    z_WSS_grad.SetInputConnection(y_WSS_grad.GetOutputPort())
    z_WSS_grad.ComputeGradientOn()
    z_WSS_grad.FasterApproximationOff()
    z_WSS_grad.ComputeDivergenceOff()
    z_WSS_grad.ComputeVorticityOff()
    z_WSS_grad.ComputeQCriterionOff()
    z_WSS_grad.SetResultArrayName("z_WSS_grad")
    z_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk_process, "z_wall_shear")

    calc2 = vtk.vtkArrayCalculator()
    calc2.AddScalarVariable("x_component", "x_WSS_grad",0)
    calc2.AddScalarVariable("y_component", "y_WSS_grad",1)
    calc2.AddScalarVariable("z_component", "z_WSS_grad",2)
    calc2.SetFunction("sqrt(x_component^2+y_component^2+z_component^2)")
    calc2.SetResultArrayName("WSSG")
    calc2.SetInputConnection(z_WSS_grad.GetOutputPort())
    if(cell_type == "cell"):
        calc2.SetAttributeModeToUseCellData()
    else:
        calc2.SetAttributeModeToUsePointData()
    calc2.SetResultArrayType(vtk.VTK_DOUBLE)

    # initialize the output to include the peak values
    grid = vtk.vtkUnstructuredGrid()
    #N_peak = 3
    reader.SetTimeStep(N_peak)
    calc2.Update()
    print("loading peak: {0} timestep to copy data".format(reader.GetTimeStep()))
    grid.DeepCopy(calc2.GetOutput())

    reader.SetTimeStep(0)
    #reader.Update()
    calc2.Update()
    print("loading {0}th timestep for averaging initialization".format(reader.GetTimeStep()))
    
    stats = vtk.vtkTemporalStatistics()
    
    stats.SetInputConnection(calc2.GetOutputPort())
    stats.ComputeMaximumOff()
    stats.ComputeMinimumOff()
    stats.ComputeStandardDeviationOff()
    stats.ComputeAverageOn()
    stats.Update()

    print("what's the time step after stats :{0}".format(reader.GetTimeStep()))
    grid_out = vtk.vtkUnstructuredGrid()
    grid_out.DeepCopy(stats.GetOutput())


    if(cell_type == "cell"):
        out_data = grid_out.GetCellData()
        grid_data = grid.GetCellData()
    else:
        out_data = grid_out.GetPointData()
        grid_data = grid.GetPointData()

    print("update names")
    out_data.AddArray(grid_data.GetArray("WSS"))
    out_data.GetArray("WSS").SetName("WSS_peak")
    out_data.AddArray(grid_data.GetArray("WSSG"))
    out_data.GetArray("WSSG").SetName("WSSG_peak")
    out_data.AddArray(grid_data.GetArray("absolute_pressure"))
    out_data.GetArray("absolute_pressure").SetName("pressure_peak")

    out_data.GetArray("WSS_average").SetName("TAWSS")
    out_data.GetArray("WSSG_average").SetName("TAWSSG")

    print("TAWSSVector")
    calc3 = vtk.vtkArrayCalculator()
    calc3.AddScalarVariable("x_wall_shear_average", "x_wall_shear_average",0)
    calc3.AddScalarVariable("y_wall_shear_average", "y_wall_shear_average",0)
    calc3.AddScalarVariable("z_wall_shear_average", "z_wall_shear_average",0)
    calc3.SetFunction("sqrt(x_wall_shear_average^2+y_wall_shear_average^2+z_wall_shear_average^2)")
    calc3.SetResultArrayName("TAWSSVector")
    calc3.SetInputData(grid_out)
    if(cell_type == "cell"):
        calc3.SetAttributeModeToUseCellData()
    else:
        calc3.SetAttributeModeToUsePointData()
    calc3.SetResultArrayType(vtk.VTK_DOUBLE)
    calc3.Update()
    
    print("OSI")
    calc4 = vtk.vtkArrayCalculator()
    calc4.AddScalarVariable("TAWSSVector", "TAWSSVector",0)
    calc4.AddScalarVariable("TAWSS", "TAWSS",0)
    calc4.SetFunction("0.5*(1.0-(TAWSSVector/(TAWSS)))")
    calc4.SetResultArrayName("OSI")
    calc4.SetInputConnection(calc3.GetOutputPort())
    if(cell_type == "cell"):
        calc4.SetAttributeModeToUseCellData()
    else:
        calc4.SetAttributeModeToUsePointData()
    calc4.SetResultArrayType(vtk.VTK_DOUBLE)
    #calc4.Update()
    
    # peak ratios
    calc5 = vtk.vtkArrayCalculator()
    calc5.AddScalarVariable("WSS_peak", "WSS_peak",0)
    calc5.AddScalarVariable("TAWSS", "TAWSS",0)
    calc5.SetFunction("WSS_peak/TAWSS")
    calc5.SetResultArrayName("WSS_peak_q_TAWSS")
    calc5.SetInputConnection(calc4.GetOutputPort())
    if(cell_type == "cell"):
        calc5.SetAttributeModeToUseCellData()
    else:
        calc5.SetAttributeModeToUsePointData()
    calc5.SetResultArrayType(vtk.VTK_DOUBLE)
    #calc5.Update()
    
    calc6 = vtk.vtkArrayCalculator()
    calc6.AddScalarVariable("WSSG_peak", "WSSG_peak",0)
    calc6.AddScalarVariable("TAWSSG", "TAWSSG",0)
    calc6.SetFunction("WSSG_peak/TAWSSG")
    calc6.SetResultArrayName("WSSG_peak_q_TAWSSG")
    calc6.SetInputConnection(calc5.GetOutputPort())
    if(cell_type == "cell"):
        calc6.SetAttributeModeToUseCellData()
    else:
        calc6.SetAttributeModeToUsePointData()
    calc6.SetResultArrayType(vtk.VTK_DOUBLE)
    
    calc7 = vtk.vtkArrayCalculator()
    calc7.AddScalarVariable("pressure_peak", "pressure_peak",0)
    calc7.AddScalarVariable("absolute_pressure_average", "absolute_pressure_average",0)
    calc7.SetFunction("pressure_peak/absolute_pressure_average")
    calc7.SetResultArrayName("pressure_peak_q_pressure_average")
    calc7.SetInputConnection(calc6.GetOutputPort())
    if(cell_type == "cell"):
        calc7.SetAttributeModeToUseCellData()
    else:
        calc7.SetAttributeModeToUsePointData()
    calc7.SetResultArrayType(vtk.VTK_DOUBLE)

    pass_filt = vtk.vtkPassArrays()
    pass_filt.SetInputConnection(calc7.GetOutputPort())
    pass_filt.AddArray(vtk_data_type, "WSS")
    pass_filt.AddArray(vtk_data_type, "WSSG")
    pass_filt.AddArray(vtk_data_type, "pressure_peak")
    pass_filt.AddArray(vtk_data_type, "TAWSS")
    pass_filt.AddArray(vtk_data_type, "TAWSSG")
    pass_filt.AddArray(vtk_data_type, "OSI")
    pass_filt.AddArray(vtk_data_type, "WSS_peak_q_TAWSS")
    pass_filt.AddArray(vtk_data_type, "WSSG_peak_q_TAWSSG")
    pass_filt.AddArray(vtk_data_type, "pressure_peak_q_pressure_average")

    pass_filt.Update()
    #if(cell_type == "cell"):
    #    print(pass_filt.GetOutput().GetCellData().GetArray("OSI").GetValue(0))
    #else:
    #    print(pass_filt.GetOutput().GetPointData().GetArray("OSI").GetValue(0))

    writer2 = vtk.vtkXMLUnstructuredGridWriter()
    writer2.SetFileName(os.path.join(dir_path, vtu_output_2))
    writer2.SetInputConnection(pass_filt.GetOutputPort())
    writer2.Update()


if ( __name__ == '__main__' ):

    #dir_path = "/raid/sansomk/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/"
    dir_path = "/raid/sansomk/caseFiles/tcd/case1/fluent/vtk_out"
    vtu_input = "wall_outfile_node.vtu"
    post_proc_cfd_diff([dir_path, vtu_input, "point", 17])
    
