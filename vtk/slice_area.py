import vtk
import numpy as np

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName('slice.vtp')
reader.Update()
data = reader.GetOutput()


data.GetLines().InitTraversal()
idList = vtk.vtkIdList()
aPolygon = vtk.vtkPolygon()
aPolygon.GetPointIds().SetNumberOfIds(data.GetNumberOfPoints())

for i in range(data.GetNumberOfPoints()):
    aPolygon.GetPointIds().SetId(i, i)

#print(aPolygon)
#print(aPolygon.ComputeArea())


#Compute the center of mass
centerOfMassFilter = vtk.vtkCenterOfMass()
centerOfMassFilter.SetInputData(data)
centerOfMassFilter.SetUseScalarsAsWeights(False)
centerOfMassFilter.Update()

trianglePoints = [None] * 3
center = centerOfMassFilter.GetCenter()
print(center)

newPoints = vtk.vtkPoints()
newPoints.DeepCopy(data.GetPoints())
nlines = data.GetNumberOfLines()

barycenterId = newPoints.InsertNextPoint(center)

triangles = vtk.vtkCellArray()
for i in range(nlines):
    boundary = data.GetCell(i)

    trianglePoints[0]  = data.GetPoint(boundary.GetPointId(0))
    trianglePoints[1]  = barycenterId
    trianglePoints[2]  = data.GetPoint( boundary.GetPointId(1))

    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId( 0, boundary.GetPointId(0) )
    triangle.GetPointIds().SetId( 1, barycenterId )
    triangle.GetPointIds().SetId( 2, boundary.GetPointId(1) )

    triangles.InsertNextCell ( triangle )

#Create a polydata object
trianglePolyData = vtk.vtkPolyData()
# Add the geometry and topology to the polydata
trianglePolyData.SetPoints ( newPoints )
trianglePolyData.SetPolys ( triangles )

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("triangulated.vtp")
writer.SetInputData(trianglePolyData)
writer.Update()
