import vtk
from glob import glob
import os
### THIS DOESN"T WORK!
file_path = "/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent3/"

out_dir = "/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent3/vtk_out"

wall = True
surface = True
file_pattern = "TH_0_PATIENT7100_M-3-?.????.dat"

filelist = sorted(glob(os.path.join(file_path, file_pattern)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

append = vtk.vtkAppendFilter()
append.MergePointsOn()

append2 = vtk.vtkAppendFilter()
append2.MergePointsOn()
reader = vtk.vtkFLUENTReader() #vtk.vtkEnSightGoldBinaryReader()

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(os.path.join(out_dir,'wall_outfile_node.vtu'))
writer.SetNumberOfTimeSteps(len(filelist))
writer.SetInputConnection(append.GetOutputPort())
writer.Start()

writer2 = vtk.vtkXMLUnstructuredGridWriter()
writer2.SetFileName(os.path.join(out_dir,'surface_outfile_node.vtu'))
writer2.SetNumberOfTimeSteps(len(filelist))
writer2.SetInputConnection(append2.GetOutputPort())
writer2.Start()

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
    N = reader.GetOutput().GetNumberOfBlocks()
    for i in range(0, N):
        name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
        if (wall):
            if (name == "wall"):
                append.AddInputData(reader.GetOutput().GetBlock(i))
                print("saving just the {0} in block {1}".format(name, i))
        if(surface):
            if (name == "wall" or name.split('_')[-1] in ['out', 'in']):
                append2.AddInputData(reader.GetOutput().GetBlock(i))
                print("saving just the {0} in block {1}".format(name, i))
        else:
            append2.AddInputData(reader.GetOutput().GetBlock(i))

    writer.WriteNextTime(time)
    writer2.WriteNextTime(time)
    if (file_p == filelist[-1]):
        continue
    else:
        for i in range(0, N):
            name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
            if (wall):
                if (name == "wall" or name.split('_')[-1] in ['out', 'in']):
                    append.RemoveInputData(reader.GetOutput().GetBlock(i))
            if(surface):
                if (name == "wall" or name.split('_')[-1] in ['out', 'in']):
                    append2.AddInputData(reader.GetOutput().GetBlock(i))
                    print("saving just the {0} in block {1}".format(name, i))
            else:
                append2.RemoveInputData(reader.GetOutput().GetBlock(i))
writer.Stop()
writer2.Stop()




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
