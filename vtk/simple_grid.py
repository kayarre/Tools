
import numpy as np #from numpy import mgrid, empty, sin, pi
import vtk


def simple_grid():
  # Generate some points.
  x, y, z = np.mgrid[1:6:11j, 0:4:13j, 0:3:6j]
  base = x[..., 0] + y[..., 0]
  # Some interesting z values.
  for i in range(z.shape[2]):
    z[..., i] = base * 0.25 * i
  return x, y, z

def uniform_grid(bounds, dims):
  # Generate some points.
  x, y, z = np.mgrid[bounds[0]:bounds[1]:(dims[0] * 1j),
                     bounds[2]:bounds[3]:(dims[1] * 1j),
                     bounds[4]:bounds[5]:(dims[2] * 1j)
                    ]
  base = x[..., 0] + y[..., 0]
  # Some interesting z values.
  # for i in range(z.shape[2]):
  #   z[..., i] = base * 0.25 * i
  return x, y, z

def reshape_pts(x,y,z):
  # The actual points.
  pts = np.empty(z.shape + (3,), dtype=float)
  pts[..., 0] = x
  pts[..., 1] = y
  pts[..., 2] = z
  # We reorder the points, scalars and vectors so this is as per VTK's
  # requirement of x first, y next and z last.
  pts = pts.transpose(2, 1, 0, 3).copy()
  pts.shape = pts.size // 3, 3
  return pts

def gen_data(x,y,z):
  # Simple scalars.
  scalars = x * x + y * y + z * z
  # Some vectors
  vectors = np.empty(z.shape + (3,), dtype=float)
  vectors[..., 0] = (4 - y * 2)
  vectors[..., 1] = (x * 3 - 12)
  vectors[..., 2] = np.sin(z * np.pi)

  scalars = scalars.T.copy()

  vectors = vectors.transpose(2, 1, 0, 3).copy()
  vectors.shape = vectors.size // 3, 3

  return scalars, vectors


# def uniform_grid(x,y,z):
#   # The actual points.
#   pts = np.empty(z.shape + (3,), dtype=float)
#   pts[..., 0] = x
#   pts[..., 1] = y
#   pts[..., 2] = z

#   # Simple scalars.
#   scalars = x * x + y * y + z * z
#   # Some vectors
#   vectors = np.empty(z.shape + (3,), dtype=float)
#   vectors[..., 0] = (4 - y * 2)
#   vectors[..., 1] = (x * 3 - 12)
#   vectors[..., 2] = np.sin(z * np.pi)

#   # We reorder the points, scalars and vectors so this is as per VTK's
#   # requirement of x first, y next and z last.
#   pts = pts.transpose(2, 1, 0, 3).copy()
#   #print(pts.shape)

#   pts.shape = pts.size // 3, 3
#   scalars = scalars.T.copy()
#   vectors = vectors.transpose(2, 1, 0, 3).copy()
#   print(vectors.shape)
#   vectors.shape = vectors.size // 3, 3
#   print(vectors.shape)

#   return pts, scalars, vectors

def test_simple_grid():
  x,y,z = simple_grid()

  pts = reshape_pts(x,y,z)

  scalars, vectors = gen_data(x,y,z)
  print(pts.shape, scalars.shape, vectors.shape)
  vtk_pts = vtk.vtkPoints()
  for pt in pts:
    #print(pt)
    vtk_pts.InsertNextPoint(pt)
  # Uncomment the following if you want to add some noise to the data.
  #pts += np.random.randn(dims[0]*dims[1]*dims[2], 3)*0.04

  sgrid = vtk.vtkStructuredGrid()
  sgrid.SetDimensions(x.shape)

  sgrid.SetPoints(vtk_pts)
  scalar_arr = vtk.vtkDoubleArray()
  scalar_arr.SetNumberOfComponents(1)
  scalar_arr.SetName("distance")
  vec_arr = vtk.vtkDoubleArray()
  vec_arr.SetNumberOfComponents(3)
  vec_arr.SetName("vector")
  #s = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
  for idx, s_ in enumerate(scalars.ravel()):
    scalar_arr.InsertNextTuple([s_])
    vec_arr.InsertNextTuple(vectors[idx])
  #print(s.shape)
  sgrid.GetPointData().AddArray(scalar_arr)
  sgrid.GetPointData().AddArray(vec_arr)
  # sgrid.point_data.scalars.name = 'scalars'

  # Uncomment the next two lines to save the dataset to a VTK XML file.
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName("test_2.vts")
  writer.SetInputData(sgrid)
  writer.Write()
  print("success")

def test_uniform_grid():
  bounds = [-10., 20., 20., 40., 0., 60.]
  dims = (37, 23, 65)
  x,y,z = uniform_grid(bounds, dims)

  pts = reshape_pts(x,y,z)

  scalars, vectors = gen_data(x,y,z)
  #print(pts.shape, scalars.shape, vectors.shape)
  vtk_pts = vtk.vtkPoints()
  for pt in pts:
    #print(pt)
    vtk_pts.InsertNextPoint(pt)
  # Uncomment the following if you want to add some noise to the data.
  #pts += np.random.randn(dims[0]*dims[1]*dims[2], 3)*0.04

  sgrid = vtk.vtkStructuredGrid()
  sgrid.SetDimensions(x.shape)

  sgrid.SetPoints(vtk_pts)
  scalar_arr = vtk.vtkDoubleArray()
  scalar_arr.SetNumberOfComponents(1)
  scalar_arr.SetName("distance")
  vec_arr = vtk.vtkDoubleArray()
  vec_arr.SetNumberOfComponents(3)
  vec_arr.SetName("vector")

  for idx, s_ in enumerate(scalars.ravel()):
    scalar_arr.InsertNextTuple([s_])
    vec_arr.InsertNextTuple(vectors[idx])
  #print(s.shape)
  sgrid.GetPointData().AddArray(scalar_arr)
  sgrid.GetPointData().AddArray(vec_arr)

  centers = vtk.vtkCellCenters()
  centers.SetInputData(sgrid)
  centers.VertexCellsOn()
  centers.Update()

  # sgrid.point_data.scalars.name = 'scalars'

  # Uncomment the next two lines to save the dataset to a VTK XML file.
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName("test_uniform.vts")
  writer.SetInputData(sgrid)
  writer.Write()

  writer2 = vtk.vtkXMLPolyDataWriter()
  writer2.SetFileName("test_uniform_centers.vtp")
  writer2.SetInputConnection(centers.GetOutputPort())
  writer2.Write()
  print("success")


def main():

  test_simple_grid()

  test_uniform_grid()

if __name__ == '__main__':
  main()