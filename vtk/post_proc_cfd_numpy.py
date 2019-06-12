
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/wall_outfile_node.vtu")
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
calc1.SetFunction("sqrt(x_wall_shear^2+y_wall_shear^2+y_wall_shear^2)")
calc1.AddScalarVariable("x_wall_shear", "x_wall_shear",0)
calc1.AddScalarVariable("y_wall_shear", "y_wall_shear",0)
calc1.AddScalarVariable("z_wall_shear", "z_wall_shear",0)
calc1.SetResultArrayName("WSS")
calc1.SetInputConnection(reader.GetOutputPort())
calc1.SetAttributeModeToUsePointData()
#calc1.SetAttributeModeToUseCellData()
calc1.SetResultArrayType(vtk.VTK_DOUBLE)

x_WSS_grad = vtk.vtkGradientFilter()
x_WSS_grad.SetInputConnection(calc1.GetOutputPort())
x_WSS_grad.ComputeGradientOn()
x_WSS_grad.FasterApproximationOff()
x_WSS_grad.SetResultArrayName("x_WSS_grad")
x_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "x_wall_shear")

y_WSS_grad = vtk.vtkGradientFilter()
y_WSS_grad.SetInputConnection(x_WSS_grad.GetOutputPort())
y_WSS_grad.ComputeGradientOn()
y_WSS_grad.FasterApproximationOff()
y_WSS_grad.SetResultArrayName("y_WSS_grad")
x_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "y_wall_shear")

z_WSS_grad = vtk.vtkGradientFilter()
z_WSS_grad.SetInputConnection(y_WSS_grad.GetOutputPort())
z_WSS_grad.ComputeGradientOn()
z_WSS_grad.FasterApproximationOff()
z_WSS_grad.SetResultArrayName("z_WSS_grad")
z_WSS_grad.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "y_wall_shear")

calc2 = vtk.vtkArrayCalculator()
calc2.AddScalarVariable("x_component", "x_WSS_grad",0)
calc2.AddScalarVariable("y_component", "y_WSS_grad",1)
calc2.AddScalarVariable("z_component", "z_WSS_grad",2)
calc2.SetFunction("sqrt(x_component^2+y_component^2+z_component^2)")
calc2.SetResultArrayName("WSSG")
calc2.SetInputConnection(z_WSS_grad.GetOutputPort())
calc2.SetAttributeModeToUsePointData()
calc2.SetResultArrayType(vtk.VTK_DOUBLE)


grid = vtk.vtkUnstructuredGrid()
N_peak = 3
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

def array_sum(in_array, out_array, sz_array):
    for i in range(sz_array):
        in_array.SetValue(i, out_array.GetValue(i) + in_array.GetValue(i))

reader.SetTimeStep(0)
print("loading {0}th timestep for averaging initialization".format(0))
reader.Update()
calc2.Update()
TAWSS = vtk.vtkDoubleArray()
TAWSS.DeepCopy(calc2.GetOutput().GetPointData().GetArray("WSS"))
TAWSS.SetName("TAWSS")
grid.GetPointData().AddArray(TAWSS)

TAWSSG = vtk.vtkDoubleArray()
TAWSSG.DeepCopy(calc2.GetOutput().GetPointData().GetArray("WSSG"))
TAWSSG.SetName("TAWSSG")
grid.GetPointData().AddArray(TAWSSG)

x_shear_avg = vtk.vtkDoubleArray()
x_shear_avg.DeepCopy(calc2.GetOutput().GetPointData().GetArray("x_wall_shear"))
x_shear_avg.SetName("x_shear_avg")
grid.GetPointData().AddArray(x_shear_avg)

y_shear_avg = vtk.vtkDoubleArray()
y_shear_avg.DeepCopy(calc2.GetOutput().GetPointData().GetArray("y_wall_shear"))
y_shear_avg.SetName("y_shear_avg")
grid.GetPointData().AddArray(y_shear_avg)

z_shear_avg = vtk.vtkDoubleArray()
z_shear_avg.DeepCopy(calc2.GetOutput().GetPointData().GetArray("z_wall_shear"))
z_shear_avg.SetName("z_shear_avg")
grid.GetPointData().AddArray(z_shear_avg)

#TAWSSVector = vtk.vtkDoubleArray()
#TAWSSVector.DeepCopy(calc2.GetOutput().GetPointData().GetArray("z_wall_shear"))
#TAWSSVector.SetName("TAWSSVector")
#grid.GetPointData().AddArray(TAWSSVector)



def get_array_names(input):
    N_point_array = input.GetOutput().GetPointData().GetNumberOfArrays()
    N_WSS = 9999999
    for i in range(N_point_array):
        name_WSS = input.GetOutput().GetPointData().GetArrayName(i)
        if (name_WSS == "WSS"):
            N_WSS = i
        print(name_WSS)
#
# def array_sum(output, input_calc, N):
#     for i in range(N):
#         calc = output.GetValue(i) + input_calc.GetValue(i)
#         output.SetValue(i, calc)

def array_avg(out_array, N):
    for i in range(N):
        out_array.SetValue(i, out_array.GetValue(i) / N)

writer = vtk.vtkXMLUnstructuredGridWriter()
#writer.SetFileName(os.path.join(out_dir,'test_outfile.vtu'))
writer.SetFileName("/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node2.vtu")
writer.SetNumberOfTimeSteps(N)
#writer.SetTimeStepRange(0,len(filelist)-1)
writer.SetInputConnection(calc2.GetOutputPort())
writer.Start()
#avg_map = {"TAWSS":"WSS", "TAWSSG": "WSSG", "x_shear_avg":"x_wall_shear",
#            "y_shear_avg":"y_wall_shear" , "z_shear_avg":"z_wall_shear"}

grid_wrap = dsa.WrapDataObject(grid)
calc2_wrap = dsa.WrapDataObject(calc2.GetOutput())


for i in range(N):
    reader.SetTimeStep(i)
    print("Time step {0} for average calc".format(i))
    reader.Update()
    calc2.Update()
    #get_array_names(calc2)
    if( i > 0):
        array_sum(grid.GetPointData().GetArray("TAWSS"),
                  calc2.GetOutput().GetPointData().GetArray("WSS"),
                  grid.GetNumberOfPoints())
        array_sum(grid.GetPointData().GetArray("TAWSSG"),
                  calc2.GetOutput().GetPointData().GetArray("WSSG"),
                  grid.GetNumberOfPoints())
        array_sum(grid.GetPointData().GetArray("x_shear_avg"),
                  calc2.GetOutput().GetPointData().GetArray("x_wall_shear"),
                  grid.GetNumberOfPoints())
        array_sum(grid.GetPointData().GetArray("y_shear_avg"),
                  calc2.GetOutput().GetPointData().GetArray("y_wall_shear"),
                  grid.GetNumberOfPoints())
        array_sum(grid.GetPointData().GetArray("z_shear_avg"),
                  calc2.GetOutput().GetPointData().GetArray("z_wall_shear"),
                  grid.GetNumberOfPoints())

    writer.WriteNextTime(reader.GetTimeStep())

array_avg(grid.GetPointData().GetArray("TAWSS"), N)
array_avg(grid.GetPointData().GetArray("TAWSSG"), N)
array_avg(grid.GetPointData().GetArray("x_shear_avg"), N)
array_avg(grid.GetPointData().GetArray("y_shear_avg"), N)
array_avg(grid.GetPointData().GetArray("z_shear_avg"), N)

writer.Stop()

calc3 = vtk.vtkArrayCalculator()
calc3.AddScalarVariable("x_shear_avg", "x_shear_avg",0)
calc3.AddScalarVariable("y_shear_avg", "y_shear_avg",0)
calc3.AddScalarVariable("z_shear_avg", "z_shear_avg",0)
calc3.SetFunction("sqrt(x_shear_avg^2+y_shear_avg^2+z_shear_avg^2)")
calc3.SetResultArrayName("TAWSSVector")
calc3.SetInputData(grid)
calc3.SetAttributeModeToUsePointData()
calc3.SetResultArrayType(vtk.VTK_DOUBLE)
calc3.Update()

calc4 = vtk.vtkArrayCalculator()
calc4.AddScalarVariable("TAWSSVector", "TAWSSVector",0)
calc4.AddScalarVariable("TAWSS", "TAWSS",0)
calc4.SetFunction("0.5*(1.0-(TAWSSVector/(TAWSS)))")
calc4.SetResultArrayName("OSI")
calc4.SetInputConnection(calc3.GetOutputPort())
calc4.SetAttributeModeToUsePointData()
calc4.SetResultArrayType(vtk.VTK_DOUBLE)
calc4.Update()

pass_filt = vtk.vtkPassArrays()
pass_filt.SetInputConnection(calc4.GetOutputPort())
pass_filt.AddArray(vtk.vtkDataObject.POINT, "WSS")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "WSSG")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "absolute_pressure")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "TAWSS")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "TAWSSG")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "OSI")
pass_filt.AddArray(vtk.vtkDataObject.POINT, "velocity")
pass_filt.Update()
print(pass_filt.GetOutput().GetPointData().GetArray("OSI").GetValue(0))

writer2 = vtk.vtkXMLUnstructuredGridWriter()
writer2.SetFileName("/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node_stats.vtu")
writer2.SetInputConnection(pass_filt.GetOutputPort())
writer2.Update()
