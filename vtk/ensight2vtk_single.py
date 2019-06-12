import vtk
from glob import glob
import os

file_path = "/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent3/ensight"

out_dir = "/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent3/vtk_out"

file_pattern = "TH_0_PATIENT7100_M-3-?.????.dat.encas"

file_index = 15

filelist = sorted(glob(os.path.join(file_path, file_pattern)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file_name_path = os.path.splitext(filelist[file_index])[0]
print(file_name_path)
append = vtk.vtkAppendFilter()
append.MergePointsOn()
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(os.path.join(out_dir,'{0}.vtu'.format(file_name_path)))
writer.SetNumberOfTimeSteps(1)
writer.SetInputConnection(append.GetOutputPort())
writer.Start()

reader = vtk.vtkEnSightGoldBinaryReader()

path, file_name = os.path.split(filelist[file_index])
print(path, file_name)
split_name = file_name.split('-')
split_ext = split_name[-1].split('.')
time = float('.'.join(split_ext[0:2]))
print("file time: {0:.4f}".format(time))
print(file_name)
reader.SetFilePath(file_path)
reader.SetCaseFileName(file_name)
reader.Update()
#N = reader.GetNumberOfCellArrays()
N = reader.GetOutput().GetNumberOfBlocks()
for i in range(0, N):
    name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
    append.AddInputData(reader.GetOutput().GetBlock(i))
writer.WriteNextTime(time)
writer.Stop()



"""
reader = vtk.vtkEnSightGoldBinaryReader()
reader.SetFilePath("/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa2/ensight")
reader.SetCaseFileName("case1_dsa-5-6.0000.dat.encas")
reader.Update()
#N = reader.GetNumberOfCellArrays()
N = reader.GetNumberOfVariables()

append = vtk.vtkAppendFilter()
append.MergePointsOn()
for i in range(0, N):
    append.AddInputData(reader.GetOutput().GetBlock(i))

append.Update()
umesh = vtk.vtkUnstructuredGrid()
umesh = append.GetOutput()

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("test.vtu")
writer.SetInputData(umesh)
writer.Update()
"""
