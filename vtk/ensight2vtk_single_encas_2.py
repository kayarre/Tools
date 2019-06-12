import vtk
from glob import glob
import os

file_path = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/ensight/"

out_dir = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out"

wall = True
surface = True
file_pattern = "case4_dsa_0.15_fluent.encas"

#filelist = sorted(glob(os.path.join(file_path, file_pattern)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

reader = vtk.vtkEnSightGoldBinaryReader()
reader.SetFilePath(file_path)
reader.SetCaseFileName(file_pattern)
reader.Update()

time_sets = reader.GetTimeSets()
time_array = time_sets.GetItem(0)
#current_time = reader.GetTimeValue()


N = reader.GetOutput().GetNumberOfBlocks()
block_list_wall = []
block_list = []
for i in range(0, N):
    name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
    print(name)
    if (wall):
        if (name == "wall"):
            block_list_wall.append(i)
    if(surface):
        if (name == "wall" or name.split('_')[0] in ['outlet', 'inlet']):
            block_list.append(i)
    else:
        block_list.append(i)

print(block_list_wall)
if (wall):
    # extract a block
    wall_block = vtk.vtkExtractBlock()
    wall_block.SetInputConnection(reader.GetOutputPort())
    for b in block_list_wall:
        print(reader.GetOutput().GetMetaData(b).Get(vtk.vtkCompositeDataSet.NAME()))
        wall_block.AddIndex(b)
    wall_block.Update()
    print(wall_block.GetOutput().GetNumberOfBlocks(), " yo")
    print(wall_block.GetOutput().GetMetaData(0).Get(vtk.vtkCompositeDataSet.NAME()))
    append = vtk.vtkAppendFilter()
    append.MergePointsOn()
    for i in range(wall_block.GetOutput().GetNumberOfBlocks()):
        append.AddInputData(wall_block.GetOutput().GetBlock(i))
    append.Update()
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(os.path.join(out_dir,'wall_outfile_node_test.vtu'))
    #writer.SetNumberOfTimeSteps(int(time_array.GetNumberOfTuples()))
    writer.SetInputConnection(append.GetOutputPort())
    writer.Update()

"""
other_block = vtk.vtkExtractBlock()
other_block.SetInputConnection(reader.GetOutputPort())
for b in block_list:
    other_block.AddIndex(b)
other_block.Update()
print(block_list)
print(other_block.GetOutput().GetNumberOfBlocks(), " yo")
append2 = vtk.vtkAppendFilter()
append2.MergePointsOn()
for i in range(other_block.GetOutput().GetNumberOfBlocks()):
    append2.AddInputData(other_block.GetOutput().GetBlock(i))
append2.Update()

writer2 = vtk.vtkXMLUnstructuredGridWriter()
writer2.SetFileName(os.path.join(out_dir,'surface_outfile_node_test.vtu'))
#writer2.SetNumberOfTimeSteps(int(time_array.GetNumberOfTuples()))
writer2.SetInputConnection(append2.GetOutputPort())
writer2.Update()
"""
