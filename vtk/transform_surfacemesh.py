import vtk
import h5py

"""
    read the transform from slicer and apply it to a surface mesh
"""

file_path1 = "/home/ksansom/caseFiles/mri/VWI_proj/case1/dsa/case1_vmtk_decimate_trim.ply"

out_path = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/dsa2mra_trans.vtp"



print('Reading PLY surface file.')
reader1 = vtk.vtkPLYReader()
reader1.SetFileName(file_path1)
reader1.Update()

#reader1 = vtk.vtkXMLPolyDataReader()
#reader1.SetFileName(file_path1)

trans_file = h5py.File("/home/ksansom/caseFiles/mri/VWI_proj/case1/registration/combined_trans.h5", 'r')
trans_data = trans_file['/TransformGroup/0/TranformParameters'].value
trans_type = trans_file['/TransformGroup/0/TransformType'].value
trans_fixed = trans_file['/TransformGroup/0/TranformFixedParameters'].value



input_units = "m"
if(input_units == "mm"):
    scale_translation = 1000.0
else:
    scale_translation = 1.0

trans_m = vtk.vtkMatrix4x4()
# set rotation
for i in range(3):
    for j in range(3):
        trans_m.SetElement(i,j,trans_data[i*3 + j])
# set translation
for i in range(3):
    trans_m.SetElement(i,3,trans_data[9 + i]/scale_translation)
# not sure what this sets
for i in range(3):
    trans_m.SetElement(3,i,0.0)
#set global scale
trans_m.SetElement(3,3,1.0)

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


pre = vtk.vtkTransform()
pre.RotateZ(180)

trans_1 = vtk.vtkTransform()
trans_1.SetInput(invert_lt)
trans_1.Concatenate(pre)
trans_1.PreMultiply() # Does it do the matrix order = resize * trans_1 * pre
trans_1.Update()

print(trans_1.GetMatrix())

poly_filt = vtk.vtkTransformPolyDataFilter()
poly_filt.SetInputConnection(reader1.GetOutputPort())
poly_filt.SetTransform(trans_1)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(poly_filt.GetOutputPort())
writer.SetFileName(out_path)
writer.Update()
