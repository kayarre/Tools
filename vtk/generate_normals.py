import vtk
import numpy as np

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName('DSI007LERb_vmtk_trim_decimate_ctrlines_smooth.vtp')
reader.Update()
data = reader.GetOutput()
radius = reader.GetOutput().GetPointData().GetArray("MaximumInscribedSphereRadius")

lines = vtk.vtkPolyData()
lines.DeepCopy(reader.GetOutput())

vectors = vtk.vtkFloatArray()
vectors.SetName("Normals")
vectors.SetNumberOfComponents(3)
#vectors.InsertNextTuple(tuple)

data.GetLines().InitTraversal()
idList = vtk.vtkIdList()
while(data.GetLines().GetNextCell(idList)):
    print("Line has {0} points ".format(idList.GetNumberOfIds()))

    for pointId in range(idList.GetNumberOfIds()):
        #print("{0} ".format(idList.GetId(pointId)), end=''
        if(pointId >= idList.GetNumberOfIds()-1 ):
            back = pointId - 1
            for_ = pointId
        else:
            back = pointId
            for_ = pointId + 1
        pt1 = data.GetPoint(idList.GetId(back))
        pt2 = data.GetPoint(idList.GetId(for_))

        pt_new = tuple( p2 - p1 for p2,p1 in zip(pt2,pt1))
        norm = np.linalg.norm(pt_new)
        pt_new = tuple(p/norm for p in pt_new)
        vectors.InsertNextTuple(pt_new)
    break


lines.GetPointData().AddArray(vectors)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('DSI007LERb_vmtk_trim_decimate_ctrlines_smooth_tangents.vtp')
writer.SetInputData(lines)
writer.Update()

'''
    #print('')

N = data.GetNumberOfLines()
lines = data.GetLines()
for i in range(N):
  cell = data.GetCell(i)
  n_pts = cell.GetNumberOfPoints()
  pts = cell.GetPoints()
  #print(cell.GetCellType())
  spline = vtk.vtkParametricSpline()
  spline.SetPoints(cell.GetPoints())
  #Fit a spline to the points
  functionSource = vtk.vtkParametricFunctionSource()
  functionSource.SetParametricFunction(spline)
  functionSource.SetUResolution(cell.GetPoints().GetNumberOfPoints())
  functionSource.Update()

  #Interpolate the scalars
  interpolatedRadius = vtk.vtkTupleInterpolator()
  interpolatedRadius.SetInterpolationTypeToLinear()
  interpolatedRadius.SetNumberOfComponents(1)
  for i in range(n_pts):
    pt_id = cell.GetPointId(i)
    interpolatedRadius.AddTuple(i, radius.GetValue(pt_id))
    if( i == 0):
        vectors.InsertNextTuple([0.0,0.0,0.0])
    elif(i > n_pts-1):
        vectors.InsertNextTuple([0.0,0.0,0.0])
    else:
        pt1 = pts.GetPoint(pt_id-1)
        pt2 = pts.GetPoint(pt_id)
        pt3 = pts.GetPoint(pt_id)

  # Generate the normals scalars
  tubeRadius = vtk.vtkDoubleArray()
  n = functionSource.GetOutput().GetNumberOfPoints()
  tubeRadius.SetNumberOfTuples(n);
  tubeRadius.SetName("TubeRadius");
  tMin = interpolatedRadius.GetMinimumT()
  tMax = interpolatedRadius.GetMaximumT()
  double r;
  for (unsigned int i = 0; i < n; ++i)
    {
    double t = (tMax - tMin) / (n - 1) * i + tMin;
    interpolatedRadius->InterpolateTuple(t, &r);
    tubeRadius->SetTuple1(i, r);
    }

'''
