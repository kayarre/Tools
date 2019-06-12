import vtk
import os


def ensight2vtk(file_path, out_dir, file_name,
                vtu_out_1="wall_outfile_node.vtu",
                vtu_out_2="inlet_outfile_node.vtu", vtu_out_3="interior_outfile_node.vtu",
                wall=True, inlet=True, interior=True, interior_name="default_interior-1"):

    print(wall, inlet, interior)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    reader = vtk.vtkEnSightGoldBinaryReader()
    reader.SetFilePath(file_path)
    reader.SetCaseFileName(file_name)
    reader.Update()

    # solution_writer = vtk.vtkXMLMultiBlockDataWriter()
    # solution_writer.SetFileName(os.path.join(out_dir, vtu_out_3))
    # solution_writer.SetInputData(reader.GetOutput())
    # solution_writer.Write()

    append = vtk.vtkAppendFilter()
    append.MergePointsOn()

    append2 = vtk.vtkAppendFilter()
    append2.MergePointsOn()

    append3 = vtk.vtkAppendFilter()
    append3.MergePointsOn()

    time_sets = reader.GetTimeSets()
    time_array = time_sets.GetItem(0)
    current_time = reader.GetTimeValue()
    print(current_time)

    if (wall):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(os.path.join(out_dir, vtu_out_1))
        writer.SetNumberOfTimeSteps(int(time_array.GetNumberOfTuples()))
        writer.SetInputConnection(append.GetOutputPort())
        writer.Start()

    if (inlet):
        writer2 = vtk.vtkXMLUnstructuredGridWriter()
        writer2.SetFileName(os.path.join(out_dir,vtu_out_2))
        writer2.SetNumberOfTimeSteps(int(time_array.GetNumberOfTuples()))
        writer2.SetInputConnection(append2.GetOutputPort())
        writer2.Start()
    if(interior):
        writer3 = vtk.vtkXMLUnstructuredGridWriter()
        writer3.SetFileName(os.path.join(out_dir,vtu_out_3))
        writer3.SetNumberOfTimeSteps(int(time_array.GetNumberOfTuples()))
        writer3.SetInputConnection(append3.GetOutputPort())
        writer3.Start()

    print("Number of Blocks: {0}".format(time_array.GetNumberOfTuples()))
    for i in range(time_array.GetNumberOfTuples()):
        next_time = time_array.GetTuple(i)[0]

        print(next_time)
        if( current_time == next_time):
            print("first time")
            pass
        else:
            # update the reader
            reader.SetTimeValue(next_time)
            current_time = next_time
            reader.Update()
            print("success")
        #N = reader.GetNumberOfCellArrays()
        N = reader.GetOutput().GetNumberOfBlocks()
        for i in range(0, N):
            name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
            if (wall):
                if (name.split(':')[-1] == "wall"):
                    append.AddInputData(reader.GetOutput().GetBlock(i))
                    print("saving just the {0} in block {1}".format(name, i))
            if(inlet):
                if (name.split(':')[-1].split('_')[0] in ["inlet", "ica"]):
                    append2.AddInputData(reader.GetOutput().GetBlock(i))
                    print("saving just the {0} in block {1}".format(name, i))
            if(interior):
                if (name == interior_name):
                    append3.AddInputData(reader.GetOutput().GetBlock(i))
                    print("saving just the {0} in block {1}".format(name, i))

        if(wall):
            writer.WriteNextTime(current_time)
        if(inlet):
            writer2.WriteNextTime(current_time)
        if(interior):
            writer3.WriteNextTime(current_time)

        if (current_time == reader.GetMaximumTimeValue()):
            pass
        else:
            for i in range(0, N):
                name = reader.GetOutput().GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
                if (wall):
                    if (name.split(':')[-1] == "wall"):
                        append.RemoveInputData(reader.GetOutput().GetBlock(i))
                        #print("removing the {0} in block {1}".format(name, i))
                if(inlet):
                    if (name.split(':')[-1].split('_')[0] in ["inlet", "ica"]):
                        append2.RemoveInputData(reader.GetOutput().GetBlock(i))
                        #print("removing the {0} in block {1}".format(name, i))
                if(interior):
                    if (name == interior_name):
                        append3.RemoveInputData(reader.GetOutput().GetBlock(i))
                        #print("removing the {0} in block {1}".format(name, i))
    if(wall):
        writer.Stop()
    if(inlet):
        writer2.Stop()
    if(interior):
        writer3.Stop()

if ( __name__ == '__main__' ):

    file_path = "/raid/home/ksansom/caseFiles/tcd/case1/fluent/ensight/"
    out_dir = "/raid/home/ksansom/caseFiles/tcd/case1/fluent/vtk_out"
    file_pattern = "case1-ensight.encas"
    #ensight2vtk(file_path, out_dir, file_pattern, interior=False)
    ensight2vtk(file_path, out_dir, file_pattern, vtu_out_3="interior_vol_outfile_node.vtu", wall=False, inlet=False, interior=True, interior_name="case1_vmtk_decimate2_fill_trim_ext2")



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
