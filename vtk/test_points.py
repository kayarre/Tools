
import numpy as np #from numpy import mgrid, empty, sin, pi
import vtk

import pandas as pd
from scipy import special


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
  #base = x[..., 0] + y[..., 0]
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


def test_uniform_grid(bounds, dims):
  x,y,z = uniform_grid(bounds, dims)
  pts = reshape_pts(x,y,z)
  #print(pts.shape)
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

  return sgrid, centers

  # sgrid.point_data.scalars.name = 'scalars'

  # Uncomment the next two lines to save the dataset to a VTK XML file.
  # writer = vtk.vtkXMLStructuredGridWriter()
  # writer.SetFileName("test_uniform.vts")
  # writer.SetInputData(sgrid)
  # writer.Write()

  # writer2 = vtk.vtkXMLPolyDataWriter()
  # writer2.SetFileName("test_uniform_centers.vtp")
  # writer2.SetInputConnection(centers.GetOutputPort())
  # writer2.Write()
  # print("success")

def read_file ():
  path = "/home/krs/code/python/Tools/vtk/c109-20001.anm"
  with open(path, mode="r") as f :
    data = pd.read_csv(f, sep='\s+', names=["n", "m", "a", "aj" ])
    #print(data.head())
  n = data["n"].to_numpy()
  m = data["m"].to_numpy()
  #print(n.shape[0])
  coeff = np.empty((n.shape[0]), dtype=complex)
  coeff.real = data["a"].to_numpy()
  coeff.imag = data["aj"].to_numpy()
  #print(coeff[0])
  # with open(path, mode="r") as f :
  #   data = np.loadtxt(f, sep='\s+', names=["n", "m", "a", "aj" ])
  #   print(data)
  
  return n, m, coeff
  
class Sphere(object):

  def __init__(self, res=10):
    res = (4 if res < 4 else res) # ternary
    self.radius = 0.5
    self.center = [0.0, 0.0, 0.0]
    self.thetaResolution = int(res)
    self.phiResolution = int(res)
    self.startTheta = 0.0
    self.endTheta = 360.0
    self.startPhi = 0.0
    self.endPhi = 180.0
    self.LatLongTessellation = False
    self.output = vtk.vtkPolyData()

  def do_stuff(self):
    x = [0.0, 0.0, 0.0]
    n = [0.0, 0.0, 0.0]
    pts = [0, 0, 0, 0]
    numPoles = 0
    localThetaResolution = self.thetaResolution
    localStartTheta = self.startTheta
    localEndTheta = self.endTheta

    numPieces = self.thetaResolution

    while(localEndTheta < localStartTheta):
      localEndTheta += 360.0
    
    deltaTheta = (localEndTheta - localStartTheta) / localThetaResolution
    
    # if you eant to split this up into pieces this part here allow that
    start = 0 #piece * localThetaResolution / numPieces
    end = numPieces #1   #localThetaResolution / numPieces

    localEndTheta = localStartTheta + float(end)*deltaTheta
    localStartTheta = localStartTheta + float(start)*deltaTheta

    localThetaIndx = int(end - start)

    numPts = self.phiResolution * localThetaIndx + 2
    numPolys = self.phiResolution * 2 * localThetaIndx

    newPoints = vtk.vtkPoints()
    newPoints.Allocate(numPts)

    newPolys = vtk.vtkCellArray()
    #newPolys.AllocateEstimate(numPolys, 3)
    
    newNormals = vtk.vtkDoubleArray()
    newNormals.SetNumberOfComponents(3)
    newNormals.Allocate(3 * numPts)
    newNormals.SetName("Normals")

    # Create sphere
    # Create north pole if needed
    if (self.startPhi <= 0.0):
      x[0] = self.center[0]
      x[1] = self.center[1]
      x[2] = self.center[2] + self.radius

      newPoints.InsertPoint(numPoles, x)

      x[0] = 0.0
      x[1] = 0.0
      x[2] = 1.0
      newNormals.InsertTuple(numPoles, x)
      numPoles += 1

    # Create south pole if needed
    if (self.endPhi >= 180.0):
      x[0] = self.center[0]
      x[1] = self.center[1]
      x[2] = self.center[2] - self.radius
      
      newPoints.InsertPoint(numPoles, x)
      
      x[0] = 0.0
      x[1] = 0.0
      x[2] = -1.0

      newNormals.InsertTuple(numPoles, x)
      numPoles += 1

    # Check data, determine increments, and convert to radians
    startTheta = (localStartTheta if localStartTheta < localEndTheta else localEndTheta) 
    startTheta *= vtk.vtkMath.Pi() / 180.0
    
    endTheta = (localEndTheta if localEndTheta > localStartTheta else localStartTheta)
    endTheta *= vtk.vtkMath.Pi() / 180.0

    startPhi = (self.startPhi if self.startPhi < self.endPhi else self.endPhi)
    startPhi *= vtk.vtkMath.Pi() / 180.0
    endPhi = (self.endPhi if self.endPhi > self.startPhi else self.startPhi)
    endPhi *= vtk.vtkMath.Pi() / 180.0

    phiResolution = self.phiResolution - numPoles
    deltaPhi = (endPhi - startPhi) / (self.phiResolution - 1)
    thetaResolution = localThetaResolution
    # check that it should return float versus int
    if (abs(localStartTheta - localEndTheta) < 360.0):
      localThetaResolution += 1
    deltaTheta = (endTheta - startTheta) / thetaResolution

    jStart = (1 if self.startPhi <= 0.0 else 0)
    jEnd = (self.phiResolution - 1  if self.endPhi >= 180.0 else self.phiResolution)

    # Create intermediate points
    for i in range(localThetaResolution):
      theta = localStartTheta * vtk.vtkMath.Pi() / 180.0 + i * deltaTheta

      for j in range(jStart, jEnd):
        phi = startPhi + j * deltaPhi
        radius = self.radius * np.sin(phi)
        
        n[0] = radius * np.cos(theta)
        n[1] = radius * np.sin(theta)
        n[2] = self.radius * np.cos(phi)
        
        x[0] = n[0] + self.center[0]
        x[1] = n[1] + self.center[1]
        x[2] = n[2] + self.center[2]
        newPoints.InsertNextPoint(x)

        norm = vtk.vtkMath.Norm(n)
        if (norm == 0.0):
          norm = 1.0
        n[0] /= norm
        n[1] /= norm
        n[2] /= norm
        newNormals.InsertNextTuple(n)

    # Generate mesh connectivity
    base = phiResolution * localThetaResolution

    # check if fabs is required
    if (abs(localStartTheta - localEndTheta) < 360.0):
        localThetaResolution -= 1
    if (self.startPhi <= 0.0): # around north pole
      for i in range(localThetaResolution):
        pts[0] = (phiResolution * i + numPoles)
        pts[1] = ((phiResolution * (i + 1) % base) + numPoles)
        pts[2] = 0
        newPolys.InsertNextCell(3, pts[:3])
  

    if (self.endPhi >= 180.0): # around south pole
      numOffset = phiResolution - 1 + numPoles
      
      for i in range(localThetaResolution):
        pts[0] = phiResolution * i + numOffset
        pts[2] = ((phiResolution * (i + 1)) % base) + numOffset
        pts[1] = numPoles - 1
      
        newPolys.InsertNextCell(3, pts[:3])

    # bands in-between poles
    for i in range(localThetaResolution):
      for j in range(phiResolution - 1):
        pts[0] = phiResolution * i + j + numPoles
        pts[1] = pts[0] + 1
        pts[2] = ((phiResolution * (i + 1) + j) % base) + numPoles + 1
        if (self.LatLongTessellation == True):
          newPolys.InsertNextCell(3, pts[:3])
          pts[1] = pts[2]
          pts[2] = pts[1] - 1
          newPolys.InsertNextCell(3, pts[:3])
        else:
          pts[3] = pts[2] - 1
          newPolys.InsertNextCell(4, pts)

    # Update ourselves and release memory
    #
    newPoints.Squeeze()
    self.output.SetPoints(newPoints)
    #newPoints.Delete()
    newNormals.Squeeze()
    self.output.GetPointData().SetNormals(newNormals)
    #newNormals.Delete()
    newPolys.Squeeze()
    self.output.SetPolys(newPolys)
    #newPolys.Delete()

    writer2 = vtk.vtkXMLPolyDataWriter()
    writer2.SetFileName("test_sphere.vtp")
    writer2.SetInputData(self.output)
    writer2.Write()
    print("success")


def gen_surface(n, m, coef):
  theta = np.linspace(0.0, 2.0*np.pi, num=20, endpoint=False) # don't repeat the last part
  phi = np.linspace(0.0, np.pi, num=20, endpoint=True)

  T, P = np.meshgrid(theta, phi) # thete is 0-2pi, and phi is 0-pi
  r = np.zeros(T.shape, dtype=complex)
  for idx in range(n.shape[0]):
    r += coef[idx] * special.sph_harm(m[idx], n[idx], T, P)
  
  return r, T, P

def test_sphere_in_box():

  bounds = [-10., 20., 20., 40., 0., 60.]
  dims = (37, 23, 65)
  sgrid, centers = test_uniform_grid(bounds, dims)
  
  
  box_centroid = [ (bounds[j+1]+ bounds[j]) / 2.0 for j in range(0,6,2)]
  box_extents = [ (bounds[j+1] - bounds[j])  for j in range(0,6,2)]

  test = Sphere(res=20)
  test.center = box_centroid
  test.radius = np.array(box_extents).min() / 4.0
  test.LatLongTessellation = False
  test.do_stuff()


  in_out = vtk.vtkUnsignedCharArray()
  in_out.SetNumberOfComponents(1)
  in_out.SetNumberOfTuples(centers.GetOutput().GetNumberOfCells())
  in_out.Fill(0)
  in_out.SetName("Inside")


  tree = vtk.vtkModifiedBSPTree()
  tree.SetDataSet(test.output)
  tree.BuildLocator()
  #intersect the locator with the line
  tolerance = 0.0000001
  IntersectPoints = vtk.vtkPoints()
  IntersectCells = vtk.vtkIdList()

  hex_cen = vtk.vtkDoubleArray()
  hex_cen.SetNumberOfComponents(3)
  hex_cen.SetNumberOfTuples(centers.GetOutput().GetNumberOfCells())
  hex_cen.SetName("centroid")

  #pts_from_grid = sgrid.GetPoints()
  for idx in  range(centers.GetOutput().GetNumberOfCells()):
    grid_pt = centers.GetOutput().GetPoint(idx)
    hex_cen.SetTuple(idx, grid_pt)

    code = tree.IntersectWithLine(box_centroid, grid_pt,
                                  tolerance, IntersectPoints,
                                  IntersectCells)
    if (code == 0):
      # no intersection
      in_out.SetTuple(idx, [1])
  

  
  sgrid.GetCellData().AddArray(in_out)
  sgrid.GetCellData().AddArray(hex_cen)

  # Uncomment the next two lines to save the dataset to a VTK XML file.
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName("test_inside.vts")
  writer.SetInputData(sgrid)
  writer.Write()

def main():
  # n, m, coeff = read_file()
  # print(n[0], m[0], coeff[0])
  # r, T, P = gen_surface(n, m, coeff)
  # print(r.shape)#, T, P)

  test_sphere_in_box()
  

if __name__ == '__main__':
  main()