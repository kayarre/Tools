import vtk
"""
    attempt to get images mapped to the same space, but it'ts not working.
"""
reader_path = "/home/ksansom/caseFiles/mri/VWI_proj/case2/vmtk/7_AX_3D_MRA_COW.mha"
reader = vtk.vtkMetaImageReader()
reader.SetFileName(reader_path)
reader.Update()

reader_path2 = "/home/ksansom/caseFiles/mri/VWI_proj/case2/vmtk/mra_crop.mha"
reader2 = vtk.vtkMetaImageReader()
reader2.SetFileName(reader_path2)
reader2.Update()

image = vtk.vtkImageData()
image = reader.GetOutput()
origin = image.GetOrigin()


image2 = vtk.vtkImageData()
image2 = reader2.GetOutput()
origin2 = image2.GetOrigin()

"""
pre = vtk.vtkTransform()
pre.RotateZ(180)

#Reslice does all of the work
reslice = vtk.vtkImageReslice()
reslice.SetInputConnection(reader.GetOutputPort())
reslice.SetResliceTransform(pre)
reslice.SetInterpolationModeToCubic()
reslice.SetOutputSpacing(
                        image.GetSpacing()[0],
                        image.GetSpacing()[1],
                        image.GetSpacing()[2])
reslice.SetOutputOrigin(
                        image.GetOrigin()[0],
                        image.GetOrigin()[1],
                        image.GetOrigin()[2])

reslice.SetOutputExtent(image.GetExtent())
"""

image.SetOrigin(
                origin[0] - origin2[0],
                origin[1] - origin2[1],
                origin[2] - origin2[2])

writer = vtk.vtkMetaImageWriter()
writer.SetFileName("/home/ksansom/caseFiles/mri/VWI_proj/case2/vmtk/7_AX_3D_MRA_COW_rot.mha")
writer.SetInputData(image)
writer.Write()
