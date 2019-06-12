import vtk
#import numpy as np
import os

vmtk_avail = True
try:
    from vmtk import vtkvmtk
except ImportError:
    print("unable to import vmtk module")
    vmtk_avail = False

file1 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M_clean_0.vtp"
file2 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M_clean_3.vtp"

path_str = os.path.split(file1)
split_ext = os.path.splitext(path_str[-1])

print('Reading vtp surface file.')
reader1 = vtk.vtkXMLPolyDataReader()
reader1.SetFileName(file1)
reader1.Update()

reader2 = vtk.vtkXMLPolyDataReader()
reader2.SetFileName(file2)
reader2.Update()


intersectionPolyDataFilter = vtk.vtkIntersectionPolyDataFilter()
intersectionPolyDataFilter.SetInputData(0, reader1.GetOutput())
intersectionPolyDataFilter.SetInputData(1, reader2.GetOutput())
intersectionPolyDataFilter.Update()
intersect_result = intersectionPolyDataFilter.GetStatus()
print("intersection status: {0}".format(intersect_result))

writeinter = vtk.vtkXMLPolyDataWriter()
writeinter.SetInputConnection(intersectionPolyDataFilter.GetOutputPort())
file_name = os.path.join(path_str[0], "{0}_intersect.vtp".format(split_ext[0]))
writeinter.SetFileName(file_name)
writeinter.Write()
#distance = vtk.vtkDistancePolyDataFilter()
#distance.SetInputConnection( 0, intersection.GetOutputPort( 1 ) )
#distance.SetInputConnection( 1, intersection.GetOutputPort( 2 ) )


booleanOperationFilter = vtk.vtkLoopBooleanPolyDataFilter()
booleanOperationFilter.SetInputData(0, reader1.GetOutput())
booleanOperationFilter.SetInputData(1, reader2.GetOutput())
booleanOperationFilter.SetOperationToUnion()
#booleanOperationFilter.SetTolerance(2.0e-3)
booleanOperationFilter.Update()
bool_result = booleanOperationFilter.GetStatus()
print("boolean status: {0}".format(bool_result))
'''
cleaner1 = vtk.vtkCleanPolyData()
cleaner1.SetInputConnection(booleanOperationFilter.GetOutputPort())
#cleaner1.SetTolerance(1.0e-7)
cleaner1.Update()

if(vmtk_avail):
    #creates round outlet region caps
    capper = vtkvmtk.vtkvmtkSmoothCapPolyData()
    capper.SetInputConnection(cleaner1.GetOutputPort())
    capper.SetConstraintFactor(1.0)
    capper.SetNumberOfRings(8)
else:
    capper = vtk.vtkFillHolesFilter()
    capper.SetInputConnection(cleaner1.GetOutputPort())
    capper.SetHoleSize(10.0);

cleaner = vtk.vtkCleanPolyData()
cleaner.SetInputConnection(capper.GetOutputPort())
#cleaner.SetTolerance(1.0e-7)
cleaner.Update()
'''
#
writer3 = vtk.vtkXMLPolyDataWriter()
writer3.SetInputConnection(booleanOperationFilter.GetOutputPort())
file_name = os.path.join(path_str[0], "{0}_bool.vtp".format(split_ext[0]))
writer3.SetFileName(file_name)
writer3.Write()
