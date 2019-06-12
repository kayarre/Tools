import vtk
#import numpy as np
import copy
import pandas as pd

file_path1 = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/smooth_case1_vmtk_decimate_preclip.ply"

out_path = "/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/smooth_case1_vmtk_decimate_preclip_clip.vtp"

print('Reading PLY surface file.')
reader1 = vtk.vtkPLYReader()
reader1.SetFileName(file_path1)
reader1.Update()

df = pd.read_csv("/home/ksansom/caseFiles/mri/VWI_proj/case1/vmtk/dsa2mra_trans2_br.dat", sep=' ')
polyData = vtk.vtkPolyData()
polyData = reader1.GetOutput()

#if z is the normal direction out of the face then
# scale1 is the x & y scaling and scale2 is the z scaling of the clipping box
scale1 = 3.0
scale2 = 5.0
for index, row in df.iterrows():
    cent = [row['X'], row['Y'], row['Z']]
    normal = [row["BoundaryNormals0"], row["BoundaryNormals1"], row["BoundaryNormals2"]]
    radius = row["BoundaryRadius"]
    pt1 = [row["Point10"], row["Point11"], row["Point12"]]
    pt2 = [row["Point20"], row["Point21"], row["Point22"]]

    points = vtk.vtkPoints()
    points2 = vtk.vtkPoints()
    normals = vtk.vtkFloatArray()
    normals.SetNumberOfComponents(3)

    points.InsertPoint(0, cent)
    normals.InsertTuple3(0, -normal[0], -normal[1], -normal[2])


    pt2 = copy.deepcopy(cent)
    normal2 = copy.deepcopy(normal)
    normal_temp = copy.deepcopy(normal2)
    vtk.vtkMath.MultiplyScalar(normal_temp, scale2*radius)
    vtk.vtkMath.Add(pt2, normal_temp, pt2)
    points.InsertPoint(1, pt2)
    normals.InsertTuple3(1, normal2[0], normal2[1], normal2[2])

    pt3 = copy.deepcopy(cent)
    normal3 = copy.deepcopy(normal)
    vtk.vtkMath.Subtract(pt1, cent, normal3)
    normal_mag = vtk.vtkMath.Normalize(normal3)
    print("should be 1.0 = {0}".format(vtk.vtkMath.Normalize(normal3)))
    normal_temp = copy.deepcopy(normal3)
    vtk.vtkMath.MultiplyScalar(normal_temp, scale1*radius)
    vtk.vtkMath.Add(pt3, normal_temp, pt3)
    points.InsertPoint(2, pt3)
    normals.InsertTuple3(2, normal3[0], normal3[1], normal3[2])

    pt4 = copy.deepcopy(cent)
    normal4 = copy.deepcopy(normal3)
    vtk.vtkMath.Subtract(pt4, normal_temp, pt4)
    points.InsertPoint(3, pt4)
    normals.InsertTuple3(3, -normal4[0], -normal4[1], -normal4[2])

    pt5 = copy.deepcopy(cent)
    normal5 = copy.deepcopy(normal)
    vtk.vtkMath.Cross(normal, normal3, normal5)
    normal_mag = vtk.vtkMath.Normalize(normal5)
    print("should be 1.0 = {0}".format(vtk.vtkMath.Normalize(normal5)))
    normal_temp = copy.deepcopy(normal5)
    vtk.vtkMath.MultiplyScalar(normal_temp, scale1*radius)
    vtk.vtkMath.Add(pt5, normal_temp, pt5)
    points.InsertPoint(4, pt5)
    normals.InsertTuple3(4, normal5[0], normal5[1], normal5[2])

    pt6 = copy.deepcopy(cent)
    normal6 = copy.deepcopy(normal5)
    vtk.vtkMath.Subtract(pt6, normal_temp, pt6)
    points.InsertPoint(5, pt6)
    normals.InsertTuple3(5, -normal6[0], -normal6[1], -normal6[2])

    planes = vtk.vtkPlanes()
    planes.SetPoints(points)
    planes.SetNormals(normals)


    clipPolyData = vtk.vtkClipPolyData()
    #clipPolyData.SetOutputPointsPrecision(outputPointsPrecision)
    clipPolyData.SetClipFunction(planes)
    clipPolyData.SetInputData(polyData)
    clipPolyData.Update()

    polyData = clipPolyData.GetOutput()

connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
connectivityFilter.SetInputData(polyData)
connectivityFilter.SetExtractionModeToLargestRegion()
connectivityFilter.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(connectivityFilter.GetOutputPort())
writer.SetFileName(out_path)
writer.Update()
