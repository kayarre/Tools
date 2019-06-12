import vtk
import h5py

file_path1 = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/dsa2mra_trans.vtp"
file_path2 = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/smooth_case1_vmtk_decimate.vtp"

out_path = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/dsa2mra_icp.vtp"

reader1 = vtk.vtkXMLPolyDataReader()
reader1.SetFileName(file_path1)

reader2 = vtk.vtkXMLPolyDataReader()
reader2.SetFileName(file_path2)

pass_filt = vtk.vtkPassArrays()
pass_filt.SetInputConnection(reader1.GetOutputPort())
pass_filt.Update()

# don't need this now
trans_target = vtk.vtkTransform()
trans_target.Scale(0.001, 0.001, 0.001)

trans_target_filt = vtk.vtkTransformPolyDataFilter()
trans_target_filt.SetInputConnection(reader2.GetOutputPort())
trans_target_filt.SetTransform(trans_target)
trans_target_filt.Update()


source = vtk.vtkPolyData()
target = vtk.vtkPolyData()
source.ShallowCopy(pass_filt.GetOutput())
target.ShallowCopy(trans_target_filt.GetOutput())

icp = vtk.vtkIterativeClosestPointTransform()
icp.SetSource(source)
icp.SetTarget(target)
icp.SetMaximumNumberOfLandmarks(source.GetNumberOfPoints())
#icp.SetCheckMeanDistance(1)
#icp.SetMaximumMeanDistance( 0.5)
icp.SetMaximumNumberOfIterations(400)
#icp.SetMaximumNumberOfLandmarks(500)
icp.GetLandmarkTransform().SetModeToRigidBody()

icp.Modified()
icp.Update()
mat = vtk.vtkMatrix4x4()
mat = icp.GetMatrix()
print(mat)

transform = vtk.vtkTransformPolyDataFilter()
transform.SetInputConnection(reader1.GetOutputPort())
transform.SetTransform(icp)
transform.Update()


writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(transform.GetOutputPort())
writer.SetFileName(out_path)
writer.Update()
