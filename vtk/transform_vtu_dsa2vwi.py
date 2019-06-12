import vtk
import h5py

"""
    read the transform from slicer and apply it to a surface mesh
"""

file_path1 = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node_stats.vtu"
file_path2 = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node.vtu"
file_path3 = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/interior_outfile_node.vtu"

out_path = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node_stats_dsa2vwi.vtu"
out_path2 = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node_dsa2vwi.vtu"
out_path3 = "/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/interior_outfile_node_dsa2vwi.vtu"



print('Reading vtu mesh file.')
reader1 = vtk.vtkXMLUnstructuredGridReader()
reader1.SetFileName(file_path1)
reader1.Update()

reader2 = vtk.vtkXMLUnstructuredGridReader()
reader2.SetFileName(file_path2)
reader2.Update()

reader3 = vtk.vtkXMLUnstructuredGridReader()
reader3.SetFileName(file_path3)
reader3.Update()

#reader1 = vtk.vtkXMLPolyDataReader()
#reader1.SetFileName(file_path1)

trans_file = h5py.File("/home/ksansom/caseFiles/mri/VWI_proj/case1/registration_2/Transform.h5", 'r')
trans_data = trans_file['/TransformGroup/0/TranformParameters'].value
trans_type = trans_file['/TransformGroup/0/TransformType'].value
trans_fixed = trans_file['/TransformGroup/0/TranformFixedParameters'].value


def list_trans_2_4x4matrix(trans_list, scale_trans):
    trans = vtk.vtkMatrix4x4()
    # set rotation
    for i in range(3):
        for j in range(3):
            trans.SetElement(i,j,trans_list[i*3 + j])
    # set translation
    for i in range(3):
        trans.SetElement(i,3,trans_list[9 + i]/scale_trans)
    # not sure what this sets
    for i in range(3):
        trans.SetElement(3,i,0.0)
    #set global scale
    trans.SetElement(3,3,1.0)

    return trans

input_units = "mm"
if(input_units == "mm"):
    scale_translation = 1000.0
else:
    scale_translation = 1.0

trans_m = list_trans_2_4x4matrix(trans_data, scale_translation)

 # convert from itk format to vtk format
lps2ras = vtk.vtkMatrix4x4()
lps2ras.SetElement(0,0,-1)
lps2ras.SetElement(1,1,-1)
ras2lps = vtk.vtkMatrix4x4()
ras2lps.DeepCopy(lps2ras) # lps2ras is diagonal therefore the inverse is identical
vtkmat = vtk.vtkMatrix4x4()

# https://www.slicer.org/wiki/Documentation/Nightly/Modules/Transforms
vtk.vtkMatrix4x4.Multiply4x4(lps2ras, trans_m, vtkmat)
vtk.vtkMatrix4x4.Multiply4x4(vtkmat, ras2lps, vtkmat)

# Convert from LPS (ITK) to RAS (Slicer)
#vtk.vtkMatrix4x4.Multiply4x4(ras2lps, trans_m, vtkmat)
#tk.vtkMatrix4x4.Multiply4x4(vtkmat, lps2ras, vtkmat)

# Convert the sense of the transform (from ITK resampling to Slicer modeling transform)
invert = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(vtkmat, invert)
#print(invert)


# linear transform matrix
invert_lt = vtk.vtkMatrixToLinearTransform()
invert_lt.SetInput(invert)

#pre = vtk.vtkTransform()
#pre.RotateZ(0)#180)

print(invert_lt.GetMatrix())
trans_1 = vtk.vtkTransform()
trans_1.SetInput(invert_lt)
#trans_1.Concatenate(pre)
#trans_1.Concatenate(second_lt)
#trans_1.PostMultiply() # Does it do the matrix order = resize * trans_1 * pre
trans_1.Update()


second_trans = [0.989624049914536, -0.01820165778214229, 0.14252346994349668, -2.755470746170803, 0.019607903388433528, 0.9997718820638946, -0.008468409480471466, -0.3766689731697975, -0.142336818692364, 0.011175128115635211, 0.9897551952660507, 1.4619829512121851, 0.0, 0.0, 0.0, 1.0]

for i in range(3):
    second_trans[3*(i+1)+i] = second_trans[3*(i+1)+i] / (scale_translation)

trans_m_2 = vtk.vtkTransform()
trans_m_2.SetMatrix(second_trans)
#trans_m_2.Scale([1.0/scale_translation,1.0/scale_translation,1.0/scale_translation] )
#trans_m_2.GetMatrix().Invert()

print(trans_m_2.GetMatrix())
#pre = vtk.vtkTransform()
#pre.RotateZ(0)#180)

poly_filt = vtk.vtkTransformFilter()
poly_filt.SetInputConnection(reader1.GetOutputPort())
poly_filt.SetTransform(trans_1)

poly_filt2 = vtk.vtkTransformFilter()
poly_filt2.SetInputConnection(poly_filt.GetOutputPort())
poly_filt2.SetTransform(trans_m_2)

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetInputConnection(poly_filt2.GetOutputPort())
writer.SetFileName(out_path)
writer.Update()

time_set_range = reader2.GetTimeStepRange()
current_time = reader2.GetTimeStep()
print(current_time)

poly_filt3 = vtk.vtkTransformFilter()
poly_filt3.SetInputConnection(reader2.GetOutputPort())
poly_filt3.SetTransform(trans_1)

poly_filt4 = vtk.vtkTransformFilter()
poly_filt4.SetInputConnection(poly_filt3.GetOutputPort())
poly_filt4.SetTransform(trans_m_2)


writer2 = vtk.vtkXMLUnstructuredGridWriter()
writer2.SetInputConnection(poly_filt4.GetOutputPort())
writer2.SetFileName(out_path2)
writer2.SetNumberOfTimeSteps(int(time_set_range[1] - time_set_range[0]))
writer2.Start()

poly_filt5 = vtk.vtkTransformFilter()
poly_filt5.SetInputConnection(reader3.GetOutputPort())
poly_filt5.SetTransform(trans_1)

poly_filt6 = vtk.vtkTransformFilter()
poly_filt6.SetInputConnection(poly_filt5.GetOutputPort())
poly_filt6.SetTransform(trans_m_2)

writer3 = vtk.vtkXMLUnstructuredGridWriter()
writer3.SetInputConnection(poly_filt6.GetOutputPort())
writer3.SetFileName(out_path3)
writer3.SetNumberOfTimeSteps(int(time_set_range[1] - time_set_range[0]))
writer3.Start()


print("Number of Times: {0}".format(time_set_range[1]))
for i in range(time_set_range[0], time_set_range[1]):
    next_time = i

    print( "write : {0}".format(next_time))
    if( current_time == next_time):
        print("first time")
        pass
    else:
        # update the reader
        reader2.SetTimeStep(next_time)
        reader2.Update()
        poly_filt3.Update()
        poly_filt4.Update()

        reader3.SetTimeStep(next_time)
        reader3.Update()
        poly_filt5.Update()
        poly_filt6.Update()
        current_time = next_time
    writer2.WriteNextTime(current_time)
    writer3.WriteNextTime(current_time)

writer2.Stop()
writer3.Stop()
