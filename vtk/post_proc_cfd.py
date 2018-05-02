
import vtk
import numpy as np
import os


def post_proc_cfd(dir_path, vtu_input, cell_type="point",
                  vtu_output_1="calc_test_node.vtu",
                  vtu_output_2="calc_test_node_stats.vtu", N_peak=3):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.join(dir_path, vtu_input))
    reader.Update()
    N = reader.GetNumberOfTimeSteps()

    print(N)
    #N = test.GetNumberOfBlocks()ls
    #block = test.GetBlock(0)

    #for i in range(N):
    #    print(i, test.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()))


    #grid = reader.GetOutput()
    #wallshear = grid.GetCellData().GetArray("x_wall_shear")
    #print(wallshear)
    calc1 = vtk.vtkArrayCalculator()
    calc1.SetFunction("sqrt(x_wall_shear^2+y_wall_shear^2+z_wall_shear^2)")
    calc1.AddScalarVariable("x_wall_shear", "x_wall_shear",0)
    calc1.AddScalarVariable("y_wall_shear", "y_wall_shear",0)
    calc1.AddScalarVariable("z_wall_shear", "z_wall_shear",0)
    calc1.SetResultArrayName("WSS")
    calc1.SetInputConnection(reader.GetOutputPort())
    if(cell_type == "cell"):
        calc1.SetAttributeModeToUseCellData()
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        vtk_data_type = vtk.vtkDataObject.CELL
    else:
        calc1.SetAttributeModeToUsePointData()
        vtk_process = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        vtk_data_type = vtk.vtkDataObject.POINT
    calc1.SetResultArrayType(vtk.VTK_DOUBLE)

    x_WSS_grad = vtk.vtkGradientFilter()
    x_WSS_grad.SetInputConnection(calc1.GetOutputPort())
    x_WSS_grad.ComputeGradientOn()
    x_WSS_grad.FasterApproximationOff()
    x_WSS_grad.SetResultArrayName("x_WSS_grad")
    x_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk_process, "x_wall_shear")

    y_WSS_grad = vtk.vtkGradientFilter()
    y_WSS_grad.SetInputConnection(x_WSS_grad.GetOutputPort())
    y_WSS_grad.ComputeGradientOn()
    y_WSS_grad.FasterApproximationOff()
    y_WSS_grad.SetResultArrayName("y_WSS_grad")
    x_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk_process, "y_wall_shear")

    z_WSS_grad = vtk.vtkGradientFilter()
    z_WSS_grad.SetInputConnection(y_WSS_grad.GetOutputPort())
    z_WSS_grad.ComputeGradientOn()
    z_WSS_grad.FasterApproximationOff()
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
    print("loading {0}th timestep to copy data".format(N_peak))
    calc2.Update()
    grid.DeepCopy(calc2.GetOutput())
    #grid.SetNumberOfTimeSteps(1)
    #grid.SetTimeStep(0)
    #grid.Update()

    #sqrt((ddx({Wall shear-1}))**2 + (ddy({Wall shear-2}))**2 + (ddz({Wall shear-3}))**2)'
    def init_zero(in_array, sz_array):
        for i in range(sz_array):
            in_array.SetValue(i,0.0)

    def array_sum(out_array, in_array, sz_array):
        for i in range(sz_array):
            out_array.SetValue(i, out_array.GetValue(i) + in_array.GetValue(i))

    def array_division(out_array, in_array, sz_array):
        for i in range(sz_array):
            out_array.SetValue(i, out_array.GetValue(i) / in_array.GetValue(i))

    def array_avg(out_array, N):
        float_N = float(N)
        for i in range(N):
            out_array.SetValue(i, out_array.GetValue(i) / float_N)

    reader.SetTimeStep(0)


    print("loading {0}th timestep for averaging initialization".format(0))
    reader.Update()
    calc2.Update()

    if(cell_type == "cell"):
        calc_data = calc2.GetOutput().GetCellData()
        grid_data = grid.GetCellData()
        n_sz = grid.GetNumberOfCells()
    else:
        calc_data = calc2.GetOutput().GetPointData()
        grid_data = grid.GetPointData()
        n_sz = grid.GetNumberOfPoints()

    TAWSS = vtk.vtkDoubleArray()
    TAWSS.DeepCopy(calc_data.GetArray("WSS"))
    TAWSS.SetName("TAWSS")

    TAWSSG = vtk.vtkDoubleArray()
    TAWSSG.DeepCopy(calc_data.GetArray("WSSG"))
    TAWSSG.SetName("TAWSSG")

    x_shear_avg = vtk.vtkDoubleArray()
    x_shear_avg.DeepCopy(calc_data.GetArray("x_wall_shear"))
    x_shear_avg.SetName("x_shear_avg")

    y_shear_avg = vtk.vtkDoubleArray()
    y_shear_avg.DeepCopy(calc_data.GetArray("y_wall_shear"))
    y_shear_avg.SetName("y_shear_avg")

    z_shear_avg = vtk.vtkDoubleArray()
    z_shear_avg.DeepCopy(calc_data.GetArray("z_wall_shear"))
    z_shear_avg.SetName("z_shear_avg")

    #TAWSSVector = vtk.vtkDoubleArray()
    #TAWSSVector.DeepCopy(calc_data.GetArray("z_wall_shear"))
    #TAWSSVector.SetName("TAWSSVector")
    #grid_data.AddArray(TAWSSVector)

    # def get_array_names(input):
    #     N_point_array = input.GetOutput().GetPointData().GetNumberOfArrays()
    #     N_WSS = 9999999
    #     for i in range(N_point_array):
    #         name_WSS = input.GetOutput().GetPointData().GetArrayName(i)
    #         if (name_WSS == "WSS"):
    #             N_WSS = i
    #         print(name_WSS)
    #
    # def array_sum(output, input_calc, N):
    #     for i in range(N):
    #         calc = output.GetValue(i) + input_calc.GetValue(i)
    #         output.SetValue(i, calc)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    #writer.SetFileName(os.path.join(out_dir,'test_outfile.vtu'))
    writer.SetFileName(os.path.join(dir_path, vtu_output_1))
    writer.SetNumberOfTimeSteps(N)
    #writer.SetTimeStepRange(0,len(filelist)-1)
    writer.SetInputConnection(calc2.GetOutputPort())
    writer.Start()
    #avg_map = {"TAWSS":"WSS", "TAWSSG": "WSSG", "x_shear_avg":"x_wall_shear",
    #            "y_shear_avg":"y_wall_shear" , "z_shear_avg":"z_wall_shear"}

    for i in range(1,N):
        reader.SetTimeStep(i)
        print("Time step {0} for average calc".format(i))
        reader.Update()
        calc2.Update()

        if(cell_type == "cell"):
            calc_data = calc2.GetOutput().GetCellData()
        else:
            calc_data = calc2.GetOutput().GetPointData()

        #get_array_names(calc2)
        array_sum(TAWSS, calc_data.GetArray("WSS"), n_sz)
        array_sum(TAWSSG, calc_data.GetArray("WSSG"), n_sz)
        array_sum(x_shear_avg, calc_data.GetArray("x_wall_shear"), n_sz)
        array_sum(y_shear_avg, calc_data.GetArray("y_wall_shear"), n_sz)
        array_sum(z_shear_avg, calc_data.GetArray("z_wall_shear"), n_sz)

        writer.WriteNextTime(reader.GetTimeStep())

    writer.Stop()

    array_avg(TAWSS, N)
    array_avg(TAWSSG, N)
    array_avg(x_shear_avg, N)
    array_avg(y_shear_avg, N)
    array_avg(z_shear_avg, N)

    WSS_peak2mean = vtk.vtkDoubleArray()
    WSS_peak2mean.DeepCopy(grid_data.GetArray("WSS"))
    WSS_peak2mean.SetName("WSS_peak2mean")
    array_division(WSS_peak2mean, TAWSS, n_sz)

    WSSG_peak2mean = vtk.vtkDoubleArray()
    WSSG_peak2mean.DeepCopy(grid_data.GetArray("WSSG"))
    WSSG_peak2mean.SetName("WSSG_peak2mean")
    array_division(WSSG_peak2mean, TAWSSG, n_sz)

    grid_data.AddArray(TAWSS)
    grid_data.AddArray(TAWSSG)
    grid_data.AddArray(x_shear_avg)
    grid_data.AddArray(y_shear_avg)
    grid_data.AddArray(z_shear_avg)
    grid_data.AddArray(WSS_peak2mean)
    grid_data.AddArray(WSSG_peak2mean)

    print("got here")
    calc3 = vtk.vtkArrayCalculator()
    calc3.AddScalarVariable("x_shear_avg", "x_shear_avg",0)
    calc3.AddScalarVariable("y_shear_avg", "y_shear_avg",0)
    calc3.AddScalarVariable("z_shear_avg", "z_shear_avg",0)
    calc3.SetFunction("sqrt(x_shear_avg^2+y_shear_avg^2+z_shear_avg^2)")
    calc3.SetResultArrayName("TAWSSVector")
    calc3.SetInputData(grid)
    if(cell_type == "cell"):
        calc3.SetAttributeModeToUseCellData()
    else:
        calc3.SetAttributeModeToUsePointData()
    calc3.SetResultArrayType(vtk.VTK_DOUBLE)
    calc3.Update()

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
    calc4.Update()

    pass_filt = vtk.vtkPassArrays()
    pass_filt.SetInputConnection(calc4.GetOutputPort())
    pass_filt.AddArray(vtk_data_type, "WSS")
    pass_filt.AddArray(vtk_data_type, "WSSG")
    pass_filt.AddArray(vtk_data_type, "absolute_pressure")
    pass_filt.AddArray(vtk_data_type, "TAWSS")
    pass_filt.AddArray(vtk_data_type, "TAWSSG")
    pass_filt.AddArray(vtk_data_type, "OSI")
    pass_filt.AddArray(vtk_data_type, "WSS_peak2mean")
    pass_filt.AddArray(vtk_data_type, "WSSG_peak2mean")

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
    post_proc_cfd(dir_path, vtu_input, cell_type="point", N_peak=17)
