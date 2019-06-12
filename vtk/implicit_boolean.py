import vtk
#import numpy as np
import os
import sys
vmtk_avail = True
try:
    from vmtk import vtkvmtk
except ImportError:
    print("unable to import vmtk module")
    vmtk_avail = False

file1 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M_clean_0.vtp"
file2 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M_clean_3.vtp"
file3 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M.stl"

path_str = os.path.split(file1)
split_ext = os.path.splitext(path_str[-1])

print('Reading vtp surface file.')
reader1 = vtk.vtkXMLPolyDataReader()
reader1.SetFileName(file1)
reader1.Update()

reader2 = vtk.vtkXMLPolyDataReader()
reader2.SetFileName(file2)
reader2.Update()

reader3 = vtk.vtkSTLReader()
reader3.SetFileName(file3)
reader3.Update()
bounds = reader3.GetOutput().GetBounds()
print(bounds)

# Compute the range to select a reasonable contour value
x_range = bounds[1] - bounds[0]
print(x_range)

buffer_dist = x_range*0.005 #estimate the buffer
bounds = [i + (-1.0)**(idx+1)*buffer_dist for idx, i in enumerate(bounds)]
print(bounds)
cubeForBounds = vtk.vtkCubeSource()
cubeForBounds.SetBounds(bounds)
cubeForBounds.Update()
#sys.exit()
print("implicitModeller")
implicitModeller = vtk.vtkImplicitModeller()
implicitModeller.SetSampleDimensions(320,720,128)
implicitModeller.ComputeModelBounds(cubeForBounds.GetOutput())
implicitModeller.SetInputData(reader1.GetOutput())
#implicitModeller.AdjustBoundsOn()
#implicitModeller.SetAdjustDistance(.05) # Adjust by 10%
implicitModeller.SetMaximumDistance(0.01)
implicitModeller.SetProcessModeToPerVoxel()
implicitModeller.SetNumberOfThreads(20)
implicitModeller.StartAppend()
implicitModeller.Append(reader2.GetOutput())
implicitModeller.EndAppend()
#implicitModeller.Update()

# write just a piece (extracted piece) as well as the whole thing
idWriter = vtk.vtkXMLImageDataWriter()
file_name = os.path.join(path_str[0], "{0}_image_test.vti".format(split_ext[0]))
idWriter.SetFileName(file_name)
idWriter.SetDataModeToBinary()
# idWriter.SetDataModeToAscii()
idWriter.SetInputConnection(implicitModeller.GetOutputPort())
idWriter.Write()

#Create the 0 isosurface
print("contour filter")
contourFilter = vtk.vtkContourFilter()
contourFilter.SetInputConnection(implicitModeller.GetOutputPort())
contourFilter.SetValue(0, 0.1)
#contourFilter.Update()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(contourFilter.GetOutputPort())
normals.FlipNormalsOn()


writer3 = vtk.vtkXMLPolyDataWriter()
writer3.SetInputConnection(normals.GetOutputPort())
file_name = os.path.join(path_str[0], "{0}_implicit_test.vtp".format(split_ext[0]))
writer3.SetFileName(file_name)
writer3.Write()
