import vtk

def ReadPolyData(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(input,filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(input)
    writer.Write()


file_path = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/case1_VCG.ply"

out_path = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/case1_VCG_smooth.ply"

reader = vtk.vtkPLYReader()
reader.SetFileName(file_path)
reader.Update()

smooth = vtk.vtkSmoothPolyDataFilter()
smooth.SetInputConnection(reader.GetOutputPort())
smooth.SetNumberOfIterations(10)
smooth.BoundarySmoothingOff()
smooth.SetFeatureAngle(120)
smooth.SetEdgeAngle(90)
smooth.SetRelaxationFactor(.05)


writer = vtk.vtkPLYWriter()
writer.SetFileName(out_path)
writer.SetInputConnection(smooth.GetOutputPort())
writer.Write()
