import vtk
from glob import glob
import os

file_path = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa2/ensight"

out_dir = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa2/vtk_out"

file_pattern = "case1_dsa-5-?.????.dat.encas"

filelist = sorted(glob(os.path.join(file_path, file_pattern)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

append = vtk.vtkAppendFilter()
append.MergePointsOn()
reader = vtk.vtkEnSightGoldBinaryReader()
writer = vtk.vtkXMLPUnstructuredGridWriter()
writer.SetFileName(os.path.join(out_dir,'test_outfile.pvtu'))
writer.SetNumberOfTimeSteps(len(filelist))
#writer.SetTimeStepRange(0,len(filelist)-1)
writer.SetInputConnection(append.GetOutputPort())

writer.Start()

for file_p in filelist:
    path, file_name = os.path.split(file_p)
    split_name = file_name.split('-')
    split_ext = split_name[-1].split('.')
    time = float('.'.join(split_ext[0:2]))
    print("file time: {0:.4f}".format(time))
    print(file_name)
    reader.SetFilePath(file_path)
    reader.SetCaseFileName(file_name)
    reader.Update()
    #N = reader.GetNumberOfCellArrays()
    N = reader.GetNumberOfVariables()
    for i in range(0, N):
        append.AddInputData(reader.GetOutput().GetBlock(i))
    writer.WriteNextTime(time)
    for i in range(0, N):
        append.RemoveInputData(reader.GetOutput().GetBlock(i))
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
